"""The panel's state, kept in a file the plugin owns.

The state used to ride on the turn's user message in ``display_metadata``. That broke in
the gateway: before handing history to the agent it rebuilds every replayed message from a
whitelist as ``{role, content}`` (``gateway/run.py``), so plugin-owned fields are dropped
and the state can never be read back. A file under the plugin's own data directory has no
such dependency on the message shape.

``data_dir`` is the host's ``PluginContext.state.data_dir`` — ``<HERMES_HOME>/plugin-data/
<namespace>/`` for the profile the plugin was loaded under. This module never resolves a
home itself, so a multiplexed service cannot write one profile's state into another's.

A stored state is used only while the panel that produced it is still in the model's visible
context, and the test for that is ``api_content``: the host composes the panel into the user
message the provider actually receives and stamps that text onto the message as a sidecar
(``agent/turn_context.py``), deleting the sidecar when the content is rewritten. The panel
being present in ``api_content`` therefore means the model can still see it, no matter where
it sits in the history or how long the history is — which is what lets it survive compaction,
rewind, ``/branch``, and concurrent sessions, none of which a length watermark can. What
``load_state`` returns is the script's own last reply — ``panel`` and ``state`` — because the
script is fed back exactly what it emitted, and the file on disk holds exactly that reply and
nothing else. The write
is a same-directory temp file plus ``os.replace`` so a crash never leaves half a document
behind; no lock is taken, because the worst a lost concurrent update can do is repeat one
panel, which is benign. The file name avoids the host's disk-cleanup patterns (no ``test_``/
``tmp_`` prefix, no ``*.test.*``) so it is never reclaimed as test debris.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

# Not ``state.json``: that is the host's own per-plugin key/value store
# (``hermes_cli/plugins_state.py::PluginState.path``), and sharing the name would let one
# overwrite the other.
STATE_FILENAME = "panel_state.json"

# The stamped message normally sits near the tail; a panel older than this has almost
# certainly been dropped or compressed away. Bounding the scan keeps the search cheap on a
# long history, and stopping early only ever costs one redundant full panel — the safe
# direction, unlike a false positive, which would suppress a panel the model never saw.
_PANEL_SCAN_LIMIT = 200


def load_state(data_dir: Any, *, history: list) -> dict | None:
    """Return the record to feed this turn's script, or ``None`` when there is none.

    The record is ``{"panel": ..., "state": ...}``: the script's own previous reply, handed
    back verbatim, so the script sees nothing but what it once produced. ``None`` means
    "start from nothing": the file is missing or unreadable, it does not hold a record with a
    dict ``state``, or the panel it describes is no longer in ``history``. That last case is
    the deliberate reset; the rest are plain absences.
    """
    path = _state_path(data_dir)
    if path is None:
        return None
    payload = _read_json(path)
    if payload is None:
        return None

    state = payload.get("state")
    if not isinstance(state, dict):
        return None

    panel = payload.get("panel")
    if not _panel_visible(history, panel):
        return None
    return {"panel": panel, "state": state}


def save_state(
    data_dir: Any,
    state: dict,
    *,
    panel: str,
) -> None:
    """Atomically record ``state`` and the ``panel`` that carried it.

    ``panel`` is what the host appends to the turn's user message and stamps into
    ``api_content``, so it is the fingerprint a later run looks for and is stored exactly as
    the model received it. The document is the script's own reply and nothing more: nothing
    here records when it was written, because only the two fields the script exchanges are
    ever read back.
    """
    path = _state_path(data_dir)
    if path is None:
        return
    payload = {
        "state": state,
        "panel": panel,
    }
    _atomic_write_json(path, payload)


def _panel_visible(history: Any, panel: Any) -> bool:
    """Whether ``panel`` is still present in the model-visible context.

    Scans ``api_content``, the sidecar the host stamps with the exact text the provider
    received. If a surface never stamps it, the panel can never be replayed and this returns
    ``False`` every turn, so the script restates in full every turn. That is deliberate: there
    is no length fallback, because a length would silently *suppress* the panel on exactly the
    sessions where the premise has broken, leaving the model permanently time-blind.
    """
    if not isinstance(panel, str) or not panel:
        return False
    if not isinstance(history, list):
        return False
    for message in reversed(history[-_PANEL_SCAN_LIMIT:]):
        if not isinstance(message, dict):
            continue
        content = message.get("api_content")
        if isinstance(content, str) and panel in content:
            return True
    return False


def _state_path(data_dir: Any) -> Path | None:
    if data_dir is None:
        return None
    try:
        return Path(data_dir) / STATE_FILENAME
    except TypeError:
        return None


def _read_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write through a same-directory temp file so a reader never sees a partial document."""
    # mkdir before mkstemp: the host resolves the data directory but may not create it.
    path.parent.mkdir(parents=True, exist_ok=True)
    document = json.dumps(payload, ensure_ascii=False)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(document)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
