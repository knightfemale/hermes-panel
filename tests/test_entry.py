"""The entry point, loaded the way Hermes loads it.

Hermes imports the plugin directory as a package rooted at ``__init__.py`` with
``submodule_search_locations`` pointing at the plugin directory, and reads a
``register`` attribute off the result. Repeating that here is the only way to catch a
relative import that would break under it.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMESPACE_PARENT = "hermes_plugins"


class RecordingState:
    def __init__(self, data_dir):
        self.data_dir = data_dir


class RecordingCtx:
    def __init__(self, data_dir, **settings):
        self._settings = settings
        self.hooks: dict[str, object] = {}
        self.state = RecordingState(data_dir)

    def get_config(self, key, default=None):
        return self._settings.get(key, default)

    def register_hook(self, name, callback):
        self.hooks[name] = callback


def _load_as_host():
    """Import the plugin exactly as ``hermes_cli/plugins_loader.py:436-457`` does.

    The namespace parent has to exist first: the plugin's relative imports resolve through
    it, so skipping this step fails with ``No module named 'hermes_plugins'``.
    """
    if NAMESPACE_PARENT not in sys.modules:
        ns_pkg = types.ModuleType(NAMESPACE_PARENT)
        ns_pkg.__path__ = []
        ns_pkg.__package__ = NAMESPACE_PARENT
        sys.modules[NAMESPACE_PARENT] = ns_pkg

    name = f"{NAMESPACE_PARENT}.hermes_panel"
    for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        del sys.modules[key]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    module.__package__ = name
    module.__path__ = [str(ROOT)]
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class EntryPointTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.data_dir = Path(temporary.name)

    def _ctx(self, **settings):
        return RecordingCtx(self.data_dir, **settings)

    def test_the_host_can_import_the_package_and_register(self):
        module = _load_as_host()
        self.assertTrue(callable(getattr(module, "register", None)))

        ctx = self._ctx(script="", allowed_roots=[])
        module.register(ctx)
        self.assertIn("pre_llm_call", ctx.hooks)

    def test_the_hook_says_nothing_without_a_script(self):
        module = _load_as_host()
        ctx = self._ctx(script="", allowed_roots=[])
        module.register(ctx)
        self.assertIsNone(ctx.hooks["pre_llm_call"](conversation_history=[]))

    def test_the_hook_injects_a_panel_through_the_real_example(self):
        example = ROOT / "examples" / "time_panel.py"
        module = _load_as_host()
        ctx = self._ctx(script=str(example), allowed_roots=[str(example.parent)])
        module.register(ctx)

        reply = ctx.hooks["pre_llm_call"](
            session_id="s", turn_id="t", conversation_history=[{"role": "user"}]
        )
        self.assertIsInstance(reply, dict)
        self.assertIn("[hermes-panel: status update]", reply["context"])

    def test_the_state_survives_a_second_turn(self):
        # The whole point of the file store: turn two reads what turn one showed, so a
        # reading inside the script's interval is silent instead of restated.
        example = ROOT / "examples" / "time_panel.py"
        module = _load_as_host()
        ctx = self._ctx(script=str(example), allowed_roots=[str(example.parent)])
        module.register(ctx)
        history = [{"role": "user"}]

        first = ctx.hooks["pre_llm_call"](conversation_history=history)
        self.assertIn("[hermes-panel: status update]", first["context"])
        # The host appends the panel to the user message and stamps api_content with the
        # delivered text; without that sidecar the panel cannot be seen as replayed.
        history[0]["api_content"] = first["context"]
        second = ctx.hooks["pre_llm_call"](conversation_history=history)
        self.assertIsNone(second)

    def test_a_missing_history_is_treated_as_empty(self):
        example = ROOT / "examples" / "time_panel.py"
        module = _load_as_host()
        ctx = self._ctx(script=str(example), allowed_roots=[str(example.parent)])
        module.register(ctx)
        # The panel still lands; there is simply no history that could show a previous one.
        reply = ctx.hooks["pre_llm_call"](conversation_history=None)
        self.assertIn("[hermes-panel: status update]", reply["context"])

    def test_an_unexpected_failure_costs_the_panel_and_not_the_turn(self):
        # The host isolates a raising hook, but the panel would vanish every turn and each
        # turn would log a traceback, so the boundary has to absorb it here.
        module = _load_as_host()
        ctx = self._ctx(script="/x.py")
        module.register(ctx)

        def boom(*_args, **_kwargs):
            raise RuntimeError("boom")

        module.render_panel = boom
        with self.assertLogs(module.__name__, level="ERROR"):
            self.assertIsNone(ctx.hooks["pre_llm_call"](conversation_history=[]))


if __name__ == "__main__":
    unittest.main()
