"""Example panel script: the date and time, then the time alone every three minutes.

Implements the contract in docs/panel-script-contract.md. main() is the only clock reader.
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime


_LABEL = "[hermes-panel: status update]"

_DATE_FORMAT = "%Y-%m-%d"
_TIME_FORMAT = "%H:%M:%S"

_EMPTY_REPLY = '{"panel": "", "state": null}'


def _fetch_timestamp(previous_state: dict | None, now: datetime) -> int:
    return int(now.timestamp())


# State registry: name -> fetch(previous_state, now). This is all the script remembers.
STATE_FIELDS = (("timestamp", _fetch_timestamp),)


def _is_full_restate(
    previous_state: dict | None, current_state: dict, now: datetime
) -> bool:
    """True when the previous reading is missing, unreadable, from another day, or ahead."""

    if previous_state is None:
        return True
    previous = previous_state.get("timestamp")
    if isinstance(previous, bool) or not isinstance(previous, (int, float)):
        return True
    try:
        last = datetime.fromtimestamp(previous, tz=now.tzinfo)
    except (OverflowError, OSError, ValueError):
        return True
    if last.date() != now.date():
        return True
    return now.timestamp() - previous < 0


def _update_date(
    previous_state: dict | None, current_state: dict, now: datetime
) -> str | None:
    return (
        f"当前日期: {now.strftime(_DATE_FORMAT)}"
        if _is_full_restate(previous_state, current_state, now)
        else None
    )


def _update_time(
    previous_state: dict | None, current_state: dict, now: datetime
) -> str | None:
    # Safe to subscript only because _is_full_restate ran first and ``or`` short-circuits: to
    # reach the subtraction, the timestamp must already be known a real number.
    if (
        _is_full_restate(previous_state, current_state, now)
        or now.timestamp() - previous_state["timestamp"] >= 3 * 60  # 3 minutes
    ):
        return f"当前时间: {now.strftime(_TIME_FORMAT)}"
    return None


# Render registry: (name, update(previous_state, current_state, now)). None = say nothing.
# The name is only a handle for reading this table; each update owns the whole line it emits,
# so it is free to put its label, units, or nothing at all wherever it likes.
#
# Order is load-bearing. The plugin finds its own panel by looking for that text as a
# *substring* of what the model was shown, so no reachable panel may contain another: put the
# rarest optional line first, right behind the label, and never append an optional line after
# one that always speaks.
RENDER_FIELDS = (
    ("date", _update_date),
    ("time", _update_time),
)


def _read_previous() -> object:
    try:
        record = json.loads(sys.stdin.read())
    except (ValueError, OSError):
        return None
    return record if isinstance(record, dict) else None


def _read_state(previous: object) -> dict | None:
    if not isinstance(previous, dict):
        return None
    state = previous.get("state")
    if not isinstance(state, dict):
        return None
    return state


def _assemble(previous: object, now: datetime) -> dict:
    previous_state = _read_state(previous)
    current_state = {name: fetch(previous_state, now) for name, fetch in STATE_FIELDS}
    lines = [
        text
        for name, update in RENDER_FIELDS
        if (text := update(previous_state, current_state, now)) is not None
    ]
    # Silence hands back the previous state: the model saw nothing, so storing the fresh one
    # would claim a reading it never received.
    if not lines:
        return {"panel": "", "state": previous_state}
    return {
        "panel": "\n".join(["", "", _LABEL, "", *lines]),
        "state": current_state,
    }


def main() -> None:
    try:
        reply = _assemble(_read_previous(), datetime.now().astimezone())
    except Exception:
        # Never break the turn: a non-zero exit discards the reply instead.
        traceback.print_exc()
        sys.stdout.write(_EMPTY_REPLY)
        return

    sys.stdout.write(json.dumps(reply, ensure_ascii=False))


if __name__ == "__main__":
    main()
