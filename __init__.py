"""hermes-panel — Hermes Agent plugin entry point.

Hermes imports this file as a package rooted at the plugin directory, so the modules
below are reached with relative imports. An absolute import would break as soon as a
second profile scope renames the package (``hermes_plugins.<slug>__home_<digest>``).
"""

from __future__ import annotations

import logging
from typing import Any

from .src.app import render_panel

logger = logging.getLogger(__name__)


def register(ctx: Any) -> None:
    """Register the panel hook with the host."""

    def on_pre_llm_call(**_kwargs: Any) -> dict[str, str] | None:
        # The host isolates a raising hook, but it would swallow the panel on every turn
        # and log a traceback each time. This is the host boundary, so it is the place to
        # be total: anything unexpected costs one panel, never the turn.
        try:
            # The host hands over the live messages list. Normalise it here so everything
            # downstream can assume a list.
            history = _kwargs.get("conversation_history")
            panel = render_panel(
                ctx,
                history=history if isinstance(history, list) else [],
                session_id=str(_kwargs.get("session_id") or ""),
                turn_id=str(_kwargs.get("turn_id") or ""),
                # Resolved per call rather than captured at register time: the host resolves
                # the profile on each access, and caching the path would pin whichever
                # profile happened to register first.
                state_dir=ctx.state.data_dir,
            )
        except Exception:
            logger.exception(
                "hermes-panel: unexpected failure; skipping this turn's panel"
            )
            return None
        return {"context": panel} if panel else None

    ctx.register_hook("pre_llm_call", on_pre_llm_call)
