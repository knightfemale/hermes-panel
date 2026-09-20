"""The plugin's own state file: the durable memory behind the script's input."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.metadata import (
    STATE_FILENAME,
    _PANEL_SCAN_LIMIT,
    load_state,
    save_state,
)

STATE = {"d": "2026-09-20", "t": "09:35", "ts": 1789868100}
PANEL = "2026-09-20 09:35"
OTHER_PANEL = "[hermes-panel: status update]\n\n当前时间: 08:00:00"

# What the script is handed back: its own last reply, and nothing of the plugin's file.
RECORD = {"panel": PANEL, "state": STATE}


def _carrying(panel: str = PANEL, *, before: int = 0) -> list:
    """A history whose last user message carries ``panel`` the way the host stamps it."""
    history = [{"role": "assistant", "content": "filler"} for _ in range(before)]
    history.append({"role": "user", "api_content": panel})
    return history


class StateFileTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _path(self) -> Path:
        return self.dir / STATE_FILENAME

    def _write_raw(self, document: str) -> None:
        self._path().write_text(document, encoding="utf-8")

    def _write(self, payload: dict) -> None:
        self._write_raw(json.dumps(payload))


class LoadStateTest(StateFileTest):
    def test_a_missing_file_is_no_state(self):
        self.assertIsNone(load_state(self.dir, history=_carrying()))

    def test_a_missing_directory_is_no_state(self):
        self.assertIsNone(load_state(self.dir / "absent", history=_carrying()))

    def test_an_absent_directory_argument_is_no_state(self):
        self.assertIsNone(load_state(None, history=_carrying()))

    def test_a_directory_the_system_cannot_express_is_no_state(self):
        # Path(123) raises TypeError; a state read must never take the panel down with it.
        self.assertIsNone(load_state(123, history=_carrying()))

    def test_a_stored_state_round_trips(self):
        save_state(self.dir, STATE, panel=PANEL)
        self.assertEqual(load_state(self.dir, history=_carrying()), RECORD)

    def test_the_record_carries_only_the_panel_and_the_state(self):
        # No version, no timestamp: the plugin stores the script's own reply and nothing
        # else, so the script gets back exactly what it once emitted.
        save_state(self.dir, STATE, panel=PANEL)
        record = load_state(self.dir, history=_carrying())
        self.assertEqual(set(record), {"panel", "state"})

    def test_the_panel_must_still_be_in_the_history(self):
        # The state and its panel are written together; a history that no longer shows the
        # panel (rewind, compaction, /branch, another session) must not resurface the state.
        save_state(self.dir, STATE, panel=PANEL)
        gone = [{"role": "user", "content": "the panel was rewritten away"}]
        self.assertIsNone(load_state(self.dir, history=gone))

    def test_another_sessions_panel_does_not_count(self):
        save_state(self.dir, STATE, panel=PANEL)
        self.assertIsNone(load_state(self.dir, history=_carrying(OTHER_PANEL)))

    def test_an_empty_history_is_no_state(self):
        save_state(self.dir, STATE, panel=PANEL)
        self.assertIsNone(load_state(self.dir, history=[]))

    def test_a_history_that_is_not_a_list_is_no_state(self):
        save_state(self.dir, STATE, panel=PANEL)
        for history in (None, "nope", 7):
            with self.subTest(history=history):
                self.assertIsNone(load_state(self.dir, history=history))

    def test_messages_that_are_not_dicts_are_skipped(self):
        save_state(self.dir, STATE, panel=PANEL)
        mixed = [None, 7, "x", *_carrying()]
        self.assertEqual(load_state(self.dir, history=mixed), RECORD)
        self.assertIsNone(load_state(self.dir, history=[None, 7, "x"]))

    def test_api_content_that_is_not_text_does_not_raise(self):
        save_state(self.dir, STATE, panel=PANEL)
        for content in (None, 7, [], {}, b"bytes"):
            with self.subTest(content=content):
                history = [{"role": "user", "api_content": content}]
                self.assertIsNone(load_state(self.dir, history=history))

    def test_the_history_size_does_not_decide_anything(self):
        # The same state is usable from a one-message history and from a long one; only the
        # panel's presence matters.
        save_state(self.dir, STATE, panel=PANEL)
        self.assertEqual(load_state(self.dir, history=_carrying()), RECORD)
        self.assertEqual(load_state(self.dir, history=_carrying(before=500)), RECORD)

    def test_a_panel_beyond_the_scan_horizon_is_no_state(self):
        # Bounded scanning is the safe direction: an unread panel costs one restatement.
        save_state(self.dir, STATE, panel=PANEL)
        history = [{"role": "user", "api_content": PANEL}] + [
            {"role": "assistant", "content": "later"} for _ in range(_PANEL_SCAN_LIMIT)
        ]
        self.assertIsNone(load_state(self.dir, history=history))

    # The structural guards below are the whole defence now that no version field gates the
    # read: anything that is not exactly ``{"panel": <visible str>, "state": <dict>}`` is
    # treated as absent, so the script restates in full rather than trusting a half-record.

    def test_a_non_object_state_is_not_trusted(self):
        for state in ("nope", 7, [], None):
            with self.subTest(state=state):
                self._write({"state": state, "panel": PANEL})
                self.assertIsNone(load_state(self.dir, history=_carrying()))

    def test_a_missing_state_is_not_trusted(self):
        self._write({"panel": PANEL})
        self.assertIsNone(load_state(self.dir, history=_carrying()))

    def test_a_document_without_a_panel_is_no_state(self):
        # Written before the presence check existed: cannot be confirmed, so restate in full.
        self._write({"state": STATE})
        self.assertIsNone(load_state(self.dir, history=_carrying()))

    def test_a_non_string_panel_is_no_state(self):
        for panel in (None, 7, [], {}):
            with self.subTest(panel=panel):
                self._write({"state": STATE, "panel": panel})
                self.assertIsNone(load_state(self.dir, history=_carrying()))

    def test_a_panel_not_in_the_history_is_no_state(self):
        self._write({"state": STATE, "panel": PANEL})
        self.assertIsNone(load_state(self.dir, history=_carrying(OTHER_PANEL)))

    def test_unreadable_json_is_no_state(self):
        self._write_raw("{not json")
        self.assertIsNone(load_state(self.dir, history=_carrying()))

    def test_a_non_object_document_is_no_state(self):
        self._write_raw("[1, 2, 3]")
        self.assertIsNone(load_state(self.dir, history=_carrying()))


class SaveStateTest(StateFileTest):
    def test_save_creates_the_file_and_its_directory(self):
        target = self.dir / "nested"
        save_state(target, STATE, panel=PANEL)
        self.assertEqual(load_state(target, history=_carrying()), RECORD)

    def test_save_writes_exactly_the_state_and_the_panel(self):
        save_state(self.dir, STATE, panel=PANEL)
        document = json.loads(self._path().read_text(encoding="utf-8"))
        self.assertEqual(set(document), {"state", "panel"})
        self.assertEqual(document["state"], STATE)
        self.assertEqual(document["panel"], PANEL)

    def test_save_replaces_the_previous_state(self):
        save_state(self.dir, {"t": "old"}, panel="old panel")
        save_state(self.dir, STATE, panel=PANEL)
        self.assertEqual(load_state(self.dir, history=_carrying()), RECORD)

    def test_save_leaves_no_temp_files_behind(self):
        save_state(self.dir, STATE, panel=PANEL)
        self.assertEqual([entry.name for entry in self.dir.iterdir()], [STATE_FILENAME])

    def test_save_creates_no_subdirectories(self):
        save_state(self.dir, STATE, panel=PANEL)
        self.assertTrue(all(entry.is_file() for entry in self.dir.iterdir()))

    def test_an_absent_directory_argument_is_a_no_op(self):
        save_state(None, STATE, panel=PANEL)

    def test_a_directory_the_system_cannot_express_is_a_no_op(self):
        save_state(123, STATE, panel=PANEL)

    def test_a_failed_replace_keeps_the_previous_state_and_cleans_up(self):
        # The write goes through a temp file plus os.replace; if the rename fails, the old
        # document must be intact and the temp file must not survive as debris.
        save_state(self.dir, {"t": "old"}, panel=PANEL)
        with mock.patch("src.metadata.os.replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                save_state(self.dir, STATE, panel=PANEL)
        self.assertEqual(
            load_state(self.dir, history=_carrying()),
            {"panel": PANEL, "state": {"t": "old"}},
        )
        self.assertEqual([entry.name for entry in self.dir.iterdir()], [STATE_FILENAME])


class FilenameTest(unittest.TestCase):
    def test_the_filename_avoids_the_host_cleanup_patterns(self):
        # The host's disk cleanup reclaims test_/tmp_ prefixed files and *.test.* paths.
        self.assertFalse(STATE_FILENAME.startswith(("test_", "tmp_")))
        self.assertNotIn(".test.", STATE_FILENAME)
        self.assertEqual(STATE_FILENAME, "panel_state.json")


if __name__ == "__main__":
    unittest.main()
