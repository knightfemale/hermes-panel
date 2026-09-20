"""Script execution: every failure has to be clean and must not leak into a panel."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from src.runner import ScriptFailure, run_script

STUB = Path(__file__).resolve().parent / "fakes" / "panel_stub.py"


class RunnerTest(unittest.TestCase):
    def run_stub(self, mode, *, timeout=5.0, max_chars=300, payload=None):
        with mock.patch.dict(os.environ, {"PANEL_STUB_MODE": mode}):
            return run_script(STUB, payload, timeout=timeout, max_chars=max_chars)

    def test_returns_the_panel_and_the_state(self):
        output = self.run_stub("echo", payload={"panel": "x", "state": {"t": "09:35"}})
        self.assertIn("09:35", output.panel)
        self.assertEqual(output.state, {"echoed": True})

    def test_a_null_payload_is_sent_as_a_json_null(self):
        # ``None`` is how a script is told it has no previous reply; the runner must pass
        # it through rather than reject it as a missing dict.
        self.assertEqual(self.run_stub("echo", payload=None).panel, "prev=null")

    def test_silence_is_a_valid_answer(self):
        output = self.run_stub("silent")
        self.assertEqual(output.panel, "")
        self.assertIsNone(output.state)

    def test_a_whitespace_panel_strips_to_empty_but_keeps_its_state(self):
        output = self.run_stub("empty")
        self.assertEqual(output.panel, "")
        self.assertEqual(output.state, {"d": "2026-09-20"})

    def test_non_ascii_survives_the_round_trip(self):
        self.assertEqual(self.run_stub("utf8").panel, "今天 09:35")

    def test_stderr_never_reaches_the_panel(self):
        output = self.run_stub("noisy", payload={"panel": "x"})
        self.assertEqual(output.panel, "clean")

    def test_a_non_zero_exit_is_a_failure(self):
        with self.assertRaises(ScriptFailure) as caught:
            self.run_stub("fail")
        self.assertIn("3", str(caught.exception))

    def test_a_timeout_is_a_failure(self):
        with self.assertRaises(ScriptFailure) as caught:
            self.run_stub("hang", timeout=0.3)
        self.assertIn("timed out", str(caught.exception))

    def test_invalid_json_is_a_failure(self):
        with self.assertRaises(ScriptFailure):
            self.run_stub("bad_json")

    def test_wrong_field_types_are_a_failure(self):
        with self.assertRaises(ScriptFailure):
            self.run_stub("bad_shape")

    def test_non_standard_json_constants_are_rejected(self):
        # NaN would otherwise reach the database column as invalid JSON.
        with self.assertRaises(ScriptFailure):
            self.run_stub("nan")

    def test_an_endless_script_fails_at_the_byte_cap(self):
        # Fails fast on the cap rather than buffering until the timeout, which is
        # the only way the cap protects the process's memory at all.
        with self.assertRaises(ScriptFailure) as caught:
            self.run_stub("flood", timeout=5.0)
        self.assertIn("bytes of panel output", str(caught.exception))

    def test_an_overlong_panel_is_discarded_rather_than_truncated(self):
        with self.assertRaises(ScriptFailure) as caught:
            self.run_stub("long", max_chars=100)
        self.assertIn("100 limit", str(caught.exception))

    def test_a_panel_within_the_limit_is_kept(self):
        self.assertEqual(len(self.run_stub("long", max_chars=400).panel), 400)

    def test_a_deaf_script_cannot_stall_the_turn(self):
        # A payload larger than the pipe buffer, sent to a script that never reads
        # stdin: writing it up front would block here for good.
        payload = {"panel": "x" * (256 * 1024)}
        with self.assertRaises(ScriptFailure) as caught:
            self.run_stub("deaf", timeout=1.0, payload=payload)
        self.assertIn("timed out", str(caught.exception))

    def test_a_lingering_script_is_reclaimed_but_its_reply_is_kept(self):
        # It closed its pipes and kept running; the reply is complete either way.
        self.assertEqual(self.run_stub("linger", timeout=5.0).panel, "bye")

    def test_an_oversized_state_is_rejected(self):
        # The state is echoed back as the next turn's input, so its size is a
        # recurring cost rather than a one-off.
        with self.assertRaises(ScriptFailure) as caught:
            self.run_stub("big_state")
        self.assertIn("state is", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
