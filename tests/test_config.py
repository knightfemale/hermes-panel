"""Settings coercion: the hook has to survive any YAML a user can write."""

from __future__ import annotations

import unittest

from src.config import (
    DEFAULT_ALLOWED_ROOTS,
    DEFAULT_MAX_CHARS,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_MAX_CHARS,
    MAX_TIMEOUT_SECONDS,
    PanelConfig,
    load_config,
)


class FakeContext:
    """Stands in for ``PluginContext``, which the config layer only ever reads from."""

    def __init__(self, **settings):
        self._settings = settings

    def get_config(self, key, default=None):
        return self._settings.get(key, default)


class DefaultsTest(unittest.TestCase):
    def test_unconfigured_plugin_is_inert(self):
        config = load_config(FakeContext())
        self.assertFalse(config.enabled)
        self.assertEqual(config.timeout_seconds, DEFAULT_TIMEOUT_SECONDS)
        self.assertEqual(config.max_chars, DEFAULT_MAX_CHARS)
        self.assertEqual(config.allowed_roots, DEFAULT_ALLOWED_ROOTS)

    def test_script_is_trimmed_and_switches_the_plugin_on(self):
        config = load_config(FakeContext(script="  /run/secrets/panel.py \n"))
        self.assertEqual(config.script, "/run/secrets/panel.py")
        self.assertTrue(config.enabled)

    def test_blank_script_leaves_the_plugin_inert(self):
        self.assertFalse(load_config(FakeContext(script="   ")).enabled)


class TimeoutTest(unittest.TestCase):
    def test_clamped_to_the_shared_hook_budget(self):
        self.assertEqual(
            load_config(FakeContext(timeout_seconds=999)).timeout_seconds,
            MAX_TIMEOUT_SECONDS,
        )

    def test_unusable_values_fall_back(self):
        for value in ("5", True, None, 0, -1, [5]):
            with self.subTest(value=value):
                self.assertEqual(
                    load_config(FakeContext(timeout_seconds=value)).timeout_seconds,
                    DEFAULT_TIMEOUT_SECONDS,
                )

    def test_a_huge_integer_is_clamped_not_overflowed(self):
        # YAML yields arbitrary-precision ints, and float(10**400) raises; the
        # value still has to be clamped like any other oversized timeout.
        self.assertEqual(
            load_config(FakeContext(timeout_seconds=10**400)).timeout_seconds,
            MAX_TIMEOUT_SECONDS,
        )


class MaxCharsTest(unittest.TestCase):
    def test_clamped(self):
        self.assertEqual(
            load_config(FakeContext(max_chars=10**6)).max_chars, MAX_MAX_CHARS
        )

    def test_unusable_values_fall_back(self):
        for value in ("300", False, 0, -3, 1.5):
            with self.subTest(value=value):
                self.assertEqual(
                    load_config(FakeContext(max_chars=value)).max_chars,
                    DEFAULT_MAX_CHARS,
                )


class AllowedRootsTest(unittest.TestCase):
    def test_entries_are_trimmed_and_filtered(self):
        config = load_config(FakeContext(allowed_roots=[" /srv/a ", "", 7, "/srv/b"]))
        self.assertEqual(config.allowed_roots, ("/srv/a", "/srv/b"))

    def test_a_lone_string_is_accepted(self):
        self.assertEqual(
            load_config(FakeContext(allowed_roots="/srv/a")).allowed_roots, ("/srv/a",)
        )

    def test_relative_roots_are_discarded(self):
        # A relative root would silently mean something else under another cwd.
        config = load_config(FakeContext(allowed_roots=["/srv/a", "relative/path"]))
        self.assertEqual(config.allowed_roots, ("/srv/a",))

    def test_unusable_values_fall_back(self):
        for value in ([], "", 7, {"a": 1}):
            with self.subTest(value=value):
                self.assertEqual(
                    load_config(FakeContext(allowed_roots=value)).allowed_roots,
                    DEFAULT_ALLOWED_ROOTS,
                )


class UnreadableContextTest(unittest.TestCase):
    def test_a_context_that_raises_falls_back_to_defaults(self):
        # The plugin must stay inert rather than half-configured when the store
        # itself is broken; this runs inside the turn's hook budget.
        class Exploding:
            def get_config(self, key, default=None):
                raise RuntimeError("config store exploded")

        config = load_config(Exploding())
        self.assertEqual(config, PanelConfig())
        self.assertFalse(config.enabled)


if __name__ == "__main__":
    unittest.main()
