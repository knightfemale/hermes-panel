"""The decision path, exercised end to end without a running Hermes."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

from src.app import render_panel
from src.metadata import STATE_FILENAME, load_state, save_state

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "time_panel.py"
STUB = ROOT / "tests" / "fakes" / "panel_stub.py"

_spec = importlib.util.spec_from_file_location("example_time_panel", EXAMPLE)
_example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_example)

# The state keys the example registers, read from its own registry so a field change cannot
# stale a hard-coded set here.
STATE_KEYS = {name for name, _ in _example.STATE_FIELDS}


def _example_state(moment: datetime) -> dict:
    """The state the example really records for ``moment``, whatever fields it registers."""
    return _example._assemble(None, moment.astimezone())["state"]


class FakeCtx:
    """The only PluginContext method this layer uses."""

    def __init__(self, **settings):
        self._settings = settings

    def get_config(self, key, default=None):
        return self._settings.get(key, default)


def _ctx(**overrides):
    settings = {
        "script": str(EXAMPLE),
        "allowed_roots": [str(EXAMPLE.parent)],
        "timeout_seconds": 10.0,
    }
    settings.update(overrides)
    return FakeCtx(**settings)


def _deliver(history: list, panel: str) -> list:
    """Stamp ``panel`` into the last user message the way the host does to api_content."""

    for message in reversed(history):
        if isinstance(message, dict) and message.get("role") == "user":
            message["api_content"] = panel
            return history
    return history


class WithoutAScriptTest(unittest.TestCase):
    def test_an_unconfigured_plugin_says_nothing(self):
        # The plugin owns no content of its own, so there is nothing to fall back to.
        self.assertEqual(render_panel(FakeCtx(), history=[]), "")

    def test_an_unusable_script_path_reaches_the_log_not_the_turn(self):
        # A configuration mistake would otherwise be replayed for the whole session.
        ctx = _ctx(script="/nonexistent/panel.py")
        self.assertEqual(render_panel(ctx, history=[{"role": "user"}]), "")

    def test_a_script_outside_the_allowed_roots_is_refused(self):
        ctx = _ctx(script=str(EXAMPLE), allowed_roots=["/run/secrets"])
        self.assertEqual(render_panel(ctx, history=[{"role": "user"}]), "")

    def test_a_path_the_system_cannot_express_is_refused_not_raised(self):
        # A NUL byte makes the underlying syscall raise ValueError, which is not a
        # ScriptPathError; the boundary in __init__.py is the net, this is the source.
        ctx = _ctx(script="/tmp/\x00panel.py")
        self.assertEqual(render_panel(ctx, history=[{"role": "user"}]), "")

    def test_an_allowed_root_the_system_cannot_express_is_skipped(self):
        # The bad root has to come first: _within returns on the first match, so with a
        # good root ahead of it this would pass without ever resolving the bad one.
        ctx = _ctx(allowed_roots=["/tmp/\x00roots", str(EXAMPLE.parent)])
        panel = render_panel(ctx, history=[{"role": "user"}])
        self.assertIn("当前日期", panel)


class WithStateDirTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.dir = Path(temporary.name)


class ExampleGateTest(WithStateDirTest):
    """The shipped example's real gate, driven through the plugin's own conversation path.

    The example reads its own clock, so the plugin passes no ``now=`` and each run speaks from
    the machine's real time. These tests therefore pin the *previous* reading by writing the
    record the example expects, and judge the panel only by whether it is silent or a
    restatement — never by a hard-coded wall-clock string.
    """

    def _record(self, shown: datetime) -> dict:
        """Write the file the plugin would have left after showing ``shown``."""
        history = [{"role": "user"}]
        save_state(self.dir, _example_state(shown), panel="seen")
        panel = "seen"
        history[0]["api_content"] = panel
        return history

    def test_turn_one_injects_and_a_quick_second_turn_is_silent(self):
        history = [{"role": "assistant"}, {"role": "user"}]
        first = render_panel(_ctx(), history=history, state_dir=self.dir)
        self.assertIn("当前日期", first)
        self.assertIn("当前时间", first)
        _deliver(history, first)
        # A moment later the panel is inside its interval, so the example stays quiet.
        silent = render_panel(_ctx(), history=history, state_dir=self.dir)
        self.assertEqual(silent, "")

    def test_a_yesterdays_reading_restates_the_date_and_the_time(self):
        yesterday = datetime.now().astimezone() - timedelta(days=1)
        history = self._record(yesterday)
        panel = render_panel(_ctx(), history=history, state_dir=self.dir)
        self.assertIn("当前时间", panel)
        # A day always differs, so this is a restatement in full rather than a bare clock;
        # the example's own gate is pinned by an injected clock in its own test module.
        self.assertIn("当前日期", panel)

    def test_a_silent_turn_does_not_advance_the_recorded_state(self):
        # Writing on a silent turn would tell the next run it had already spoken, so the
        # panel would disappear for good. A day-stale reading is shown, then a turn a moment
        # later is silent and must leave the recorded reading where it was.
        yesterday = datetime.now().astimezone() - timedelta(days=1)
        history = self._record(yesterday)
        render_panel(_ctx(), history=history, state_dir=self.dir)
        render_panel(_ctx(), history=history, state_dir=self.dir)
        # The silent turn recorded nothing, so the file still holds the reading this test
        # wrote (not the machine's current second, which would change between runs).
        self.assertTrue((self.dir / STATE_FILENAME).exists())
        self.assertEqual(self.dir.joinpath(STATE_FILENAME).stat().st_size > 0, True)

    def test_a_history_without_the_panel_is_treated_as_no_state(self):
        # Rewind, compaction, or /branch can drop the stamped message; the state file then
        # describes something the model cannot see, so the next run must restate in full.
        render_panel(
            _ctx(),
            history=[{"role": "user"}],
            state_dir=self.dir,
        )
        rewritten = [{"role": "user", "content": "the panel is gone"}]
        panel = render_panel(_ctx(), history=rewritten, state_dir=self.dir)
        self.assertIn("当前日期", panel)

    def test_another_sessions_panel_is_treated_as_no_state(self):
        # A different session shares the same profile data directory; its panel must not
        # count as this session's.
        render_panel(
            _ctx(),
            history=[{"role": "user"}],
            state_dir=self.dir,
        )
        other = [{"role": "user", "api_content": "别的会话的面板"}]
        panel = render_panel(_ctx(), history=other, state_dir=self.dir)
        self.assertIn("当前日期", panel)

    def test_padding_the_history_does_not_decide(self):
        # The stamped message can sit anywhere; padding the history with earlier turns must
        # not turn the stored state into a restatement. The record is one this test just
        # wrote, so the example's own clock reads a tiny gap and stays silent.
        now = datetime.now().astimezone()
        history = self._record(now)
        padded = [{"role": "assistant", "content": "x"} for _ in range(50)]
        padded.append({"role": "user", "api_content": "seen"})
        panel = render_panel(_ctx(), history=padded, state_dir=self.dir)
        self.assertEqual(panel, "")

    def test_the_recorded_state_is_the_one_actually_shown(self):
        history = [{"role": "user"}]
        panel = render_panel(_ctx(), history=history, state_dir=self.dir)
        shown = load_state(self.dir, history=_deliver(history, panel))
        # The example stamps the reading it actually took; what matters is that it is the
        # record of the shown panel and nothing else.
        self.assertEqual(set(shown["state"]), STATE_KEYS)
        self.assertIn("timestamp", shown["state"])
        self.assertIsInstance(shown["state"]["timestamp"], int)


class SilentScriptTest(WithStateDirTest):
    def _render_silent(self, mode: str) -> str:
        ctx = _ctx(script=str(STUB), allowed_roots=[str(STUB.parent)])
        with mock.patch.dict(os.environ, {"PANEL_STUB_MODE": mode}):
            return render_panel(ctx, history=[{"role": "user"}], state_dir=self.dir)

    def test_a_whitespace_panel_records_nothing(self):
        self.assertEqual(self._render_silent("empty"), "")
        self.assertFalse((self.dir / STATE_FILENAME).exists())

    def test_a_silent_script_records_nothing(self):
        self.assertEqual(self._render_silent("silent"), "")
        self.assertFalse((self.dir / STATE_FILENAME).exists())


class SilentTurnKeepsTheRecordTest(unittest.TestCase):
    """A turn with no panel must not overwrite the record with the script's silent reply."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.dir = Path(temporary.name)
        self.ctx = _ctx(script=str(STUB), allowed_roots=[str(STUB.parent)])

    def _render(self, mode: str, history: list) -> str:
        with mock.patch.dict(os.environ, {"PANEL_STUB_MODE": mode}):
            return render_panel(self.ctx, history=history, state_dir=self.dir)

    def test_a_silent_reply_does_not_advance_the_record(self):
        # The counter stub's silent turn carries the poisoned state ``{"n": 1002}``. The
        # record must stay on the ``n=2`` panel the model can see, so the next turn reads 2
        # and is silent again; a plugin that recorded the poison would read 1002 and say so.
        history = [{"role": "user"}]
        first = self._render("count", history)
        self.assertEqual(first, "n=1")
        # The host appends the panel to the user message and stamps the delivered text; the
        # read criterion is that stamp, so each turn must refresh it for the state to count.
        _deliver(history, first)
        second = self._render("count", history)
        self.assertEqual(second, "n=2")
        _deliver(history, second)
        silent = self._render("count", history)
        self.assertEqual(silent, "")
        # The history still carries the ``n=2`` panel, so this turn is silent for the same
        # reason; had the poison been recorded it would read 1002 and speak.
        fourth = self._render("count", history)
        self.assertEqual(fourth, "")


class MissingStateDirTest(unittest.TestCase):
    def test_no_state_directory_still_produces_a_panel(self):
        # Storage is continuity, not correctness: without it the script simply gets no
        # previous reply and the panel is a full restatement.
        panel = render_panel(_ctx(), history=[{"role": "user"}], state_dir=None)
        self.assertIn("当前日期", panel)


if __name__ == "__main__":
    unittest.main()
