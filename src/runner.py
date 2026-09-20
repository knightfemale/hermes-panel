"""Running the panel script and reading its reply.

The script speaks one JSON value in, one JSON object out, over stdin/stdout. Keeping the
panel text off the command line matters: a rendered panel must never be able to
come back as a flag or an argument.

Two deliberate departures from the shell-hook runner in Hermes core:

* stdout is decoded strictly. A panel is persisted and replayed forever, so
  mojibake would be permanent damage rather than a cosmetic glitch.
* the child is given ``PYTHONUTF8=1``. The service runs under systemd without
  ``LANG``, where Python's stdout would otherwise default to ASCII and raise on
  the first non-English panel.

The rest of the environment is inherited untouched. The script is the operator's
own code, configured on purpose, and scrubbing is aimed at multiplexed tenants
sharing one credential-holding process — not at this case — while breaking
legitimate use of proxies. The accepted asymmetry is that a script can print its
own environment into the panel and so persist it into the conversation.
"""

from __future__ import annotations

import json
import logging
import os
import selectors
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

logger = logging.getLogger(__name__)

# Hard ceilings. The script is the operator's own code, so these guard against
# bugs rather than attacks — but the panel is persisted and replayed on every
# later request, and the state is fed back as the next turn's input, so both cost
# more than they look like they do.
MAX_STDOUT_BYTES = 64 * 1024
MAX_STATE_BYTES = 4 * 1024

# Enough context to debug from the service log without dumping a whole payload.
_LOG_PREVIEW_CHARS = 200

_READ_CHUNK = 64 * 1024

# How long to wait for a script that has already closed its pipes. It should be
# instant; the bound exists so a leaked descendant cannot stall the turn.
_REAP_TIMEOUT_SECONDS = 1.0


class ScriptFailure(Exception):
    """The script produced no usable panel. The message is safe to show the user."""


@dataclass(frozen=True)
class ScriptOutput:
    """A script's reply. An empty ``panel`` means "nothing to say this turn"."""

    panel: str
    state: dict[str, Any] | None


def run_script(
    script: Path, payload: dict[str, Any] | None, *, timeout: float, max_chars: int
) -> ScriptOutput:
    """Run ``script`` and parse its reply, or raise ``ScriptFailure``.

    ``payload`` is serialised as-is; ``None`` becomes a JSON ``null`` request, which is how
    a script is told there is no previous reply to work from.
    """
    process = subprocess.Popen(
        [sys.executable, str(script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_child_env(),
        # Its own session, so a timeout can take the script's whole process tree
        # with it; a script that spawns helpers must not leave them behind.
        start_new_session=True,
    )

    request = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        stdout, stderr = _exchange(process, request, timeout=timeout)
    except ScriptFailure:
        _retire(process, force=True)
        raise
    except Exception as exc:
        # Whatever went wrong, the process must not be left running, and callers
        # only have to handle one exception type. The traceback is logged so a bug
        # in here is not mistaken for a misbehaving script.
        _retire(process, force=True)
        logger.exception("panel runner failed unexpectedly")
        raise ScriptFailure(f"could not talk to the script: {exc}") from exc
    finally:
        _close_pipes(process)

    status = _retire(process, force=False)
    if status is None:
        # The reply is complete, so keep it; only the process needed reclaiming.
        logger.warning("panel script kept running after replying; killed it")
    elif status != 0:
        raise ScriptFailure(f"script exited with status {status}")

    if stderr:
        # Diagnostics stay in the service log; only the panel may reach the model.
        logger.warning("panel script wrote to stderr: %r", stderr)

    return _parse(stdout, max_chars=max_chars)


def _exchange(
    process: subprocess.Popen, request: bytes, *, timeout: float
) -> tuple[bytes, bytes]:
    """Send the request and collect the reply, with one deadline over everything.

    Both directions run under that deadline. Writing the request up front would
    block for good when the payload is larger than the pipe buffer and the script
    never reads stdin; reading stdout to the cap only after EOF would let a
    runaway script fill memory first. So the request is pushed as the pipe accepts
    it, and the cap is enforced as stdout arrives.
    """
    deadline = time.monotonic() + timeout
    stdout: bytearray = bytearray()
    stderr: bytearray = bytearray()
    pending = memoryview(request)

    selector = selectors.DefaultSelector()
    try:
        if process.stdin is not None:
            os.set_blocking(process.stdin.fileno(), False)
            selector.register(process.stdin, selectors.EVENT_WRITE)
        if process.stdout is not None:
            selector.register(process.stdout, selectors.EVENT_READ)
        if process.stderr is not None:
            selector.register(process.stderr, selectors.EVENT_READ)
    except BaseException:
        # Registration can only fail on a broken descriptor, but the selector owns
        # an epoll fd that the garbage collector should not be left to reclaim.
        selector.close()
        raise

    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            # A zero timeout still reports what is already readable, so a reply
            # that arrived just as the deadline passed is not discarded.
            for key, _ in selector.select(max(remaining, 0.0)):
                stream = key.fileobj
                if stream is process.stdin:
                    pending = pending[_push(stream, pending) :]
                    if not pending:
                        _close_stream(stream)
                        selector.unregister(stream)
                    continue
                chunk = os.read(stream.fileno(), _READ_CHUNK)
                if not chunk:
                    selector.unregister(stream)
                elif stream is process.stdout:
                    stdout += chunk
                    if len(stdout) > MAX_STDOUT_BYTES:
                        raise ScriptFailure(
                            f"script wrote more than {MAX_STDOUT_BYTES} bytes of panel output"
                        )
                elif len(stderr) < _LOG_PREVIEW_CHARS:
                    # Keep draining stderr but remember only enough to debug with.
                    stderr += chunk[: _LOG_PREVIEW_CHARS - len(stderr)]
            if remaining <= 0 and selector.get_map():
                raise ScriptFailure(f"script timed out after {timeout:g}s")
    finally:
        selector.close()

    return bytes(stdout), bytes(stderr)


def _push(stream: Any, pending: memoryview) -> int:
    """Write as much of the request as the pipe accepts right now."""
    try:
        return os.write(stream.fileno(), pending)
    except BlockingIOError:
        return 0
    except BrokenPipeError:
        # The script exited without reading stdin; its exit status says why.
        return len(pending)


def _retire(process: subprocess.Popen, *, force: bool) -> int | None:
    """Wait briefly for the script to exit, killing it if it lingers.

    ``force`` is for the failure paths, where the script is already known to be
    unwanted and waiting would only delay the turn. Returns the exit status, or
    ``None`` when the script had to be killed.
    """
    if not force:
        try:
            return process.wait(timeout=_REAP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            pass
    _kill_tree(process)
    try:
        process.wait(timeout=_REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        logger.warning("panel script survived SIGKILL; abandoning it")
    # A killed script has no meaningful exit status, and its reply was already
    # complete before we decided to reclaim it.
    return None


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    return env


def _kill_tree(process: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except OSError:
        # Already gone, or we lost the race to signal it; killing the child alone
        # is the best remaining effort, and even that can lose the same race.
        try:
            process.kill()
        except OSError:
            pass


def _close_pipes(process: subprocess.Popen) -> None:
    """Release the pipe file objects; Popen does not close them on wait().

    A long-lived service runs this once per turn, so leaving them to the garbage
    collector would drift toward the process's file-descriptor limit.
    """
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            _close_stream(stream)


def _close_stream(stream: Any) -> None:
    try:
        stream.close()
    except OSError:
        pass


def _parse(stdout: bytes, *, max_chars: int) -> ScriptOutput:
    try:
        raw = stdout.decode("utf-8")
    except UnicodeDecodeError:
        _fail("script output was not valid UTF-8")

    text = raw.strip()
    if not text:
        # Silence is a valid answer: a script that printed nothing has nothing to
        # say, and a crash cannot be mistaken for it because the exit code differs.
        return ScriptOutput(panel="", state=None)

    try:
        reply = json.loads(text, parse_constant=_reject_constant)
    except ValueError as exc:
        logger.warning(
            "panel script stdout was not valid JSON: %r", text[:_LOG_PREVIEW_CHARS]
        )
        raise ScriptFailure("script output was not valid JSON") from exc

    if not isinstance(reply, dict):
        _fail("script output must be a JSON object")

    panel = reply.get("panel", "")
    if not isinstance(panel, str):
        _fail("'panel' must be a string")

    state = reply.get("state")
    if state is not None:
        if not isinstance(state, dict):
            _fail("'state' must be an object or null")
        size = len(json.dumps(state, ensure_ascii=False).encode("utf-8"))
        if size > MAX_STATE_BYTES:
            # The state is echoed back as the next turn's input, so an oversized
            # one is a cost paid on every turn, not once.
            _fail(f"state is {size} bytes, over the {MAX_STATE_BYTES} limit")

    panel = panel.strip()
    if len(panel) > max_chars:
        # Discard rather than truncate: a shortened panel would state something
        # the script did not, and it would be replayed that way forever.
        _fail(f"panel is {len(panel)} characters, over the {max_chars} limit")

    return ScriptOutput(panel=panel, state=state)


def _reject_constant(name: str) -> NoReturn:
    # json.loads accepts NaN/Infinity unless told otherwise. They are not JSON, and
    # this value is destined for a database column, so reject them at the door.
    raise ValueError(f"non-standard JSON constant {name!r}")


def _fail(message: str) -> NoReturn:
    raise ScriptFailure(message)
