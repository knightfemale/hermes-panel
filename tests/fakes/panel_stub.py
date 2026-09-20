"""A scripted panel script, for the offline tests.

Behaviour is chosen with ``PANEL_STUB_MODE``: the runner passes the environment
through, so a test can steer the stub without inventing a test-only field in the
protocol. Each mode pins exactly one branch of the runner's contract.

``echo`` reports what it read on stdin, which is how the tests prove the request
arrived intact rather than merely that something arrived.
"""

from __future__ import annotations

import json
import os
import sys
import time

MODE = os.environ.get("PANEL_STUB_MODE", "echo")

# `deaf` must not touch stdin at all: it exists to prove the plugin never blocks
# on a request the script refuses to consume.
request: object = None
if MODE != "deaf":
    request = json.load(sys.stdin)


def previous_state() -> object:
    """The state inside the record the plugin sent, or ``None`` when there is none."""
    if not isinstance(request, dict):
        return None
    return request.get("state")


def emit(value: object) -> None:
    json.dump(value, sys.stdout, ensure_ascii=False)


if MODE == "echo":
    emit(
        {
            "panel": f"prev={json.dumps(request, ensure_ascii=False)}",
            "state": {"echoed": True},
        }
    )
elif MODE == "silent":
    pass
elif MODE == "empty":
    emit({"panel": "   ", "state": {"d": "2026-09-20"}})
elif MODE == "utf8":
    emit({"panel": "今天 09:35", "state": {"t": "09:35"}})
elif MODE == "noisy":
    sys.stderr.write("warning: noisy stub\n")
    emit({"panel": "clean", "state": {"seen": previous_state()}})
elif MODE == "count":
    # A deterministic speaker, for proving that a silent turn records nothing. Each value
    # has exactly one successor; the poison inside the silent reply sits *after* the
    # ``{"n": 2}`` the plugin must keep, so reading it back is visible in the next panel.
    seen = previous_state()
    n = seen.get("n") if isinstance(seen, dict) else None
    # bool is a subclass of int, and a boolean count is a bug rather than a count.
    if isinstance(n, bool) or not isinstance(n, int):
        emit({"panel": "n=1", "state": {"n": 1}})
    elif n == 1:
        emit({"panel": "n=2", "state": {"n": 2}})
    elif n == 2:
        emit({"panel": "", "state": {"n": 1002}})
    elif n == 3:
        emit({"panel": "count=3", "state": {"n": 3}})
    else:
        emit({"panel": f"count={n}", "state": {"n": n}})
elif MODE == "fail":
    emit({"panel": "never used"})
    sys.exit(3)
elif MODE == "hang":
    time.sleep(30)
elif MODE == "deaf":
    time.sleep(30)
elif MODE == "linger":
    # Replies, releases both pipes so the plugin sees EOF, then keeps running.
    # `os.close` and not `sys.stdout.close()`: CPython builds the standard streams
    # with closefd=False, so closing the wrapper flushes but never releases the
    # descriptor — which is exactly the leak this mode reproduces.
    emit({"panel": "bye", "state": {}})
    sys.stdout.flush()
    os.close(1)
    os.close(2)
    time.sleep(30)
elif MODE == "big_state":
    emit({"panel": "x", "state": {"blob": "y" * 5000}})
elif MODE == "bad_json":
    sys.stdout.write("not json at all")
elif MODE == "bad_shape":
    emit({"panel": "x", "state": "not-an-object"})
elif MODE == "nan":
    sys.stdout.write('{"panel": "x", "state": {"v": NaN}}')
elif MODE == "long":
    emit({"panel": "x" * 400, "state": {}})
elif MODE == "flood":
    # Never stops, so a cap that is only consulted after buffering everything
    # would grow until the test times out instead of failing fast.
    while True:
        sys.stdout.write("x" * 8192)
        sys.stdout.flush()
else:
    raise SystemExit(f"unknown PANEL_STUB_MODE: {MODE}")
