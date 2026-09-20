"""The shipped example: its gate pinned by an injected clock, its bytes by the real protocol.

The example reads its clock once, in ``main``, and threads the instant into ``assemble`` as an
argument; that call is the seam tests drive. A few end-to-end runs through a real subprocess and
the real parser then prove the reply still survives the contract.

These tests deliberately assert only the gate's *behaviour* and the protocol's *invariants* —
never the wording or the number of the example's own fields. Silencing a field, adding one, or
removing one must leave every test here green; the only fixed vocabulary is the baseline date and
time lines, which are the gate's whole reason to exist.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.runner import run_script

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "time_panel.py"

_spec = importlib.util.spec_from_file_location("example_time_panel", EXAMPLE)
time_panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(time_panel)
assemble = time_panel._assemble

ZONE = timezone(timedelta(hours=8))
NOW = datetime(2026, 9, 20, 9, 35, tzinfo=ZONE)

# The two lines every first turn / restate must carry, and the only field names these tests know.
DATE_LINE = "当前日期: "
TIME_LINE = "当前时间: "
DATE_TEXT = f"{DATE_LINE}{NOW.strftime('%Y-%m-%d')}"
TIME_TEXT = f"{TIME_LINE}{NOW.strftime('%H:%M:%S')}"

# The registered state keys, read from the registry so a field change cannot stale this.
STATE_KEYS = {name for name, _ in time_panel.STATE_FIELDS}


def _record(moment: datetime, state: dict) -> dict:
    """A previous reply in the shape the plugin hands back, around a real recorded state."""
    return {"panel": "whatever the model was shown", "state": state}


def _roundtrip(previous: dict | None, now: datetime, *, seconds: int = 0) -> dict:
    """Drive a turn, then the next one ``seconds`` later fed the state it really produced.

    Building the follow-up state by hand would couple every test to the field list; round-tripping
    keeps them honest and field-agnostic.
    """
    first = assemble(previous, now)
    later = now + timedelta(seconds=seconds)
    return assemble(_record(now, first["state"]), later)


class GateTest(unittest.TestCase):
    def test_a_first_turn_speaks_the_date_and_the_time(self):
        reply = assemble(None, NOW)
        self.assertTrue(reply["panel"])
        self.assertIn(DATE_TEXT, reply["panel"])
        self.assertIn(TIME_TEXT, reply["panel"])

    def test_a_previous_reply_without_a_usable_timestamp_restates_in_full(self):
        # Without a reading to measure from there is no elapsed time, and guessing one would
        # either spam the panel or silently suppress it. None of these may raise.
        broken = (
            None,
            {},
            "not a record",
            7,
            [],
            {"panel": "x"},
            {"panel": "x", "state": None},
            {"panel": "x", "state": {}},
            {"panel": "x", "state": {"timestamp": "09:35"}},
            {"panel": "x", "state": {"timestamp": True}},
        )
        for previous in broken:
            with self.subTest(previous=previous):
                panel = assemble(previous, NOW)["panel"]
                self.assertIn(DATE_TEXT, panel)
                self.assertIn(TIME_TEXT, panel)

    def test_a_new_date_states_the_date_again(self):
        # A bare clock says nothing about which day it is, so a rollover must spell it out.
        earlier = assemble(None, NOW - timedelta(days=1))
        reply = assemble(_record(NOW - timedelta(days=1), earlier["state"]), NOW)
        self.assertIn(DATE_TEXT, reply["panel"])
        self.assertIn(TIME_TEXT, reply["panel"])

    def test_the_same_date_after_the_interval_states_the_time_only(self):
        seen = assemble(None, NOW - timedelta(minutes=10))
        reply = assemble(_record(NOW - timedelta(minutes=10), seen["state"]), NOW)
        self.assertIn(TIME_TEXT, reply["panel"])
        self.assertNotIn(DATE_LINE, reply["panel"])

    def test_the_interval_is_measured_exactly(self):
        # One second under the interval stays silent; the interval itself speaks.
        under = _roundtrip(None, NOW, seconds=179)
        self.assertEqual(under["panel"], "")
        over = _roundtrip(None, NOW, seconds=180)
        self.assertIn(TIME_LINE, over["panel"])

    def test_a_reading_inside_the_interval_says_nothing(self):
        # Round-trip the state the example really produced, so adding a field cannot make this
        # silent turn accidentally speak.
        first = assemble(None, NOW - timedelta(seconds=30))
        reply = assemble(_record(NOW - timedelta(seconds=30), first["state"]), NOW)
        self.assertEqual(reply["panel"], "")
        # Nothing was shown, so the record must keep the reading the model last received.
        self.assertEqual(reply["state"], first["state"])

    def test_a_silent_turn_returns_the_previous_state_untouched(self):
        # Silence hands back the previous snapshot verbatim, so a key this example does not set
        # survives instead of being dropped: the model saw nothing, and rebuilding the state
        # would quietly discard whatever another shape had recorded.
        seen = assemble(None, NOW - timedelta(seconds=30))
        previous = _record(
            NOW - timedelta(seconds=30), {**seen["state"], "extra": "kept"}
        )
        reply = assemble(previous, NOW)
        self.assertEqual(reply["panel"], "")
        self.assertEqual(reply["state"], previous["state"])
        self.assertEqual(reply["state"]["extra"], "kept")

    def test_a_clock_that_moved_backwards_restates_in_full(self):
        # The stored reading is in the future, so its gap cannot be trusted.
        ahead = assemble(None, NOW + timedelta(minutes=5))
        reply = assemble(_record(NOW, ahead["state"]), NOW)
        self.assertIn(DATE_TEXT, reply["panel"])
        self.assertIn(TIME_TEXT, reply["panel"])

    def test_an_out_of_range_timestamp_restates_in_full(self):
        # A number with no moment behind it must not be trusted either.
        reply = assemble(
            _record(
                NOW,
                {"timestamp": datetime.min.replace(tzinfo=timezone.utc).timestamp()},
            ),
            NOW,
        )
        self.assertIn(DATE_TEXT, reply["panel"])

    def test_a_fractional_timestamp_is_accepted(self):
        seen = assemble(None, NOW - timedelta(minutes=10))
        record = _record(NOW - timedelta(minutes=10), dict(seen["state"]))
        record["state"]["timestamp"] = record["state"]["timestamp"] - 0.5
        self.assertIn(TIME_TEXT, assemble(record, NOW)["panel"])

    def test_the_recorded_state_is_only_the_registered_fields(self):
        # The state is what the turn measured from and nothing else: whatever the registry names,
        # and no rendered date or clock, so its shape cannot drift from the panel.
        reply = assemble(None, NOW)
        self.assertIsInstance(reply["state"], dict)
        self.assertIn("timestamp", reply["state"])
        self.assertEqual(set(reply["state"]), STATE_KEYS)
        self.assertEqual(reply["state"]["timestamp"], int(NOW.timestamp()))

    def test_the_render_keeps_the_documented_format(self):
        full = assemble(None, NOW)["panel"]
        self.assertIn("[hermes-panel: status update]", full)
        self.assertIn(DATE_TEXT, full)
        self.assertIn(TIME_TEXT, full)
        seen = assemble(None, NOW - timedelta(minutes=10))
        stale = assemble(_record(NOW - timedelta(minutes=10), seen["state"]), NOW)[
            "panel"
        ]
        self.assertIn(TIME_TEXT, stale)
        self.assertNotIn(DATE_LINE, stale)

    def test_the_clock_comes_from_the_injected_now_and_not_the_process_zone(self):
        # The panel is rendered from the injected ``now`` and its zone. A zone deliberately
        # unlike this machine's must therefore govern the clock: a renderer that rebuilt an
        # instant from the stored epoch would borrow the process zone and print the wrong hour.
        # The suite runs under ``TZ=UTC`` to catch exactly that.
        elsewhere = NOW.astimezone(timezone(timedelta(hours=-5)))
        reply = assemble(None, elsewhere)
        self.assertEqual(elsewhere.strftime("%H:%M"), "20:35")
        self.assertIn(f"{DATE_LINE}{elsewhere.strftime('%Y-%m-%d')}", reply["panel"])
        self.assertIn(f"{TIME_LINE}{elsewhere.strftime('%H:%M:%S')}", reply["panel"])


class PanelIdentityTest(unittest.TestCase):
    """Reachable panels must be pairwise non-substrings, or a record reads as already shown.

    This is the prefix invariant the ``RENDER_FIELDS`` order note states: the plugin finds a record
    by looking for the panel's text inside what the model was shown, so if one reachable panel
    contains another, the shorter one can be matched by the longer one's record and be treated
    as already delivered. Since a record is only ever matched as a prefix, the invariant holds
    exactly when no optional field is appended after a mandatory one. Appending one fails this
    test — that is what it is for.

    Residual limit, honestly: the grid only drives shapes the gate reaches, so a new optional
    field the grid never triggers would slip past. This test supplements the order note; it
    does not replace it.
    """

    def _reachable_panels(self) -> dict[str, str]:
        """Every distinct non-empty panel the gate can produce, keyed by what produced it."""
        base = NOW
        readings = [
            ("none", None),
            ("yesterday", base - timedelta(days=1)),
            ("future", base + timedelta(minutes=5)),
            ("ten-min", base - timedelta(minutes=10)),
            ("three-min", base - timedelta(minutes=3)),
            ("just-under", base - timedelta(seconds=179)),
            ("thirty-sec", base - timedelta(seconds=30)),
        ]
        nows = [
            base,
            base + timedelta(minutes=5),
            datetime(2026, 9, 21, 0, 5, tzinfo=ZONE),
            datetime(2026, 12, 21, 0, 5, tzinfo=ZONE),
        ]
        panels: dict[str, str] = {}
        for label, moment in readings:
            for offset, now in enumerate(nows):
                if moment is None:
                    previous: dict | None = None
                else:
                    # Feed back the state the example produced at ``moment``, so the follow-up is
                    # a shape the gate could really meet, whatever fields it registers.
                    previous = _record(moment, assemble(None, moment)["state"])
                panel = assemble(previous, now)["panel"]
                if panel:
                    panels[f"{label}@{offset}"] = panel
        return panels

    def test_no_reachable_panel_contains_another(self):
        panels = self._reachable_panels()
        # Guard against a grid that stops producing panels and silently asserts nothing.
        self.assertIn("none@0", panels)
        self.assertIn("ten-min@0", panels)
        for short_key, short in panels.items():
            for long_key, long in panels.items():
                if short_key == long_key or len(short) >= len(long):
                    continue
                with self.subTest(short=short_key, long=long_key):
                    self.assertNotIn(short, long)


class ProtocolTest(unittest.TestCase):
    """The bytes themselves, through a real subprocess and the plugin's own parser."""

    def test_a_null_request_answers_with_the_full_panel(self):
        result = run_script(EXAMPLE, None, timeout=10.0, max_chars=300)
        self.assertIn("[hermes-panel: status update]", result.panel)
        self.assertIn("日期", result.panel)
        self.assertIn("时间", result.panel)
        self.assertIsInstance(result.state, dict)
        self.assertIn("timestamp", result.state)
        self.assertEqual(set(result.state), STATE_KEYS)
        self.assertIsInstance(result.state["timestamp"], int)

    def test_a_record_without_a_usable_state_restates_in_full(self):
        result = run_script(
            EXAMPLE, {"panel": "old", "state": {}}, timeout=10.0, max_chars=300
        )
        self.assertIn("日期", result.panel)

    def test_stdout_is_one_json_object_of_panel_and_state(self):
        process = subprocess.run(
            [sys.executable, str(EXAMPLE)],
            input="null",
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        self.assertEqual(process.returncode, 0)
        reply = json.loads(process.stdout)
        self.assertEqual(set(reply), {"panel", "state"})


if __name__ == "__main__":
    unittest.main()
