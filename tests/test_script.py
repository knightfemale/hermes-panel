"""Script path resolution: the root whitelist is all that stands between config
and an arbitrary program launch."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.script import ScriptPathError, resolve_script


class ResolveScriptTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.root = self.tmp / "root"
        self.root.mkdir()
        self.script = self.root / "panel.py"
        self.script.write_text("print('{}')\n", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def resolve(self, raw, roots=None):
        return resolve_script(raw, (str(self.root),) if roots is None else roots)

    def test_resolves_a_file_under_a_root(self):
        self.assertEqual(self.resolve(str(self.script)), self.script.resolve())

    def test_rejects_a_relative_path(self):
        # A relative path would silently resolve against the service's cwd.
        with self.assertRaises(ScriptPathError):
            self.resolve("panel.py")

    def test_rejects_a_missing_file(self):
        with self.assertRaises(ScriptPathError):
            self.resolve(str(self.root / "nope.py"))

    def test_rejects_a_directory(self):
        with self.assertRaises(ScriptPathError):
            self.resolve(str(self.root))

    def test_rejects_a_path_under_no_known_root(self):
        with self.assertRaises(ScriptPathError):
            self.resolve(str(self.script), roots=("/definitely/not/here",))

    def test_rejects_a_symlink_that_escapes_its_root(self):
        outside = self.tmp / "outside.py"
        outside.write_text("print('{}')\n", encoding="utf-8")
        escaping = self.root / "escape.py"
        escaping.symlink_to(outside)

        with self.assertRaises(ScriptPathError):
            self.resolve(str(escaping))

    def test_rejects_an_unreadable_file(self):
        if os.geteuid() == 0:
            self.skipTest("root bypasses file permissions")
        self.script.chmod(0o000)
        with self.assertRaises(ScriptPathError):
            self.resolve(str(self.script))


if __name__ == "__main__":
    unittest.main()
