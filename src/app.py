"""Turning the configured script's answer into this turn's panel.

Everything here takes plain arguments and returns a string, so the whole decision path
can be exercised without a running Hermes. The only host contact is the config lookup, and
the state lives in a file the caller points at.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import load_config
from .metadata import load_state, save_state
from .runner import ScriptFailure, run_script
from .script import ScriptPathError, resolve_script

logger = logging.getLogger(__name__)


def render_panel(
    ctx: Any,
    *,
    history: list,
    session_id: str = "",
    turn_id: str = "",
    state_dir: Any = None,
) -> str:
    """Return the panel to inject into this turn, or "" when there is nothing to say.

    The script is handed its own previous reply — the ``load_state`` record, or ``None`` on
    the first turn — and answers with the next one. The plugin never reads a clock: the
    script owns whatever notion of time it needs. ``session_id`` and ``turn_id`` exist only
    to name the turn in warnings. ``state_dir`` is the host's plugin data directory, and
    ``None`` — no storage — costs only continuity: the script is handed no record and its
    answer is not remembered.
    """
    config = load_config(ctx)
    if not config.enabled:
        # No script means no content: the plugin owns no panel of its own.
        return ""

    try:
        script = resolve_script(config.script, config.allowed_roots)
    except ScriptPathError as exc:
        logger.warning("hermes-panel: %s", exc)
        return ""

    try:
        result = run_script(
            script,
            load_state(state_dir, history=history),
            timeout=config.timeout_seconds,
            max_chars=config.max_chars,
        )
    except ScriptFailure as exc:
        # A failing script has nothing to say. The reason belongs in the log, not in the
        # conversation, where it would be replayed for the rest of the session.
        logger.warning(
            "hermes-panel: %s (session=%s turn=%s)", exc, session_id, turn_id
        )
        return ""

    if not result.panel:
        # Silence is the script's answer for "nothing changed", so it must not cost a
        # state update either — writing one would tell the next run it had spoken.
        return ""

    if isinstance(result.state, dict):
        try:
            save_state(
                state_dir,
                result.state,
                panel=result.panel,
            )
        except Exception:
            # A panel the model needs beats a state file that failed to land; the next run
            # then just restates a little more than it had to.
            logger.warning(
                "hermes-panel: could not record state (session=%s turn=%s)",
                session_id,
                turn_id,
                exc_info=True,
            )

    return result.panel
