"""Plugin settings.

Hermes stores plugin settings as untyped YAML, so every value is coerced and
range-checked here. This runs inside the turn's hook budget, where an exception
would cost the user their entire turn, so nothing below is allowed to raise —
not even on settings the config store itself cannot read.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

KEY_SCRIPT = "script"
KEY_TIMEOUT_SECONDS = "timeout_seconds"
KEY_MAX_CHARS = "max_chars"
KEY_ALLOWED_ROOTS = "allowed_roots"

# `pre_llm_call` hooks share one 30s budget per turn, so the default has to leave
# room for the others on that path; a missing panel is never worth a stalled turn.
DEFAULT_TIMEOUT_SECONDS = 5.0
MAX_TIMEOUT_SECONDS = 20.0

# A panel is replayed on every later request, which makes its length a recurring
# cost rather than a one-off. Oversized output is discarded, never truncated: a
# cut panel would misreport the very state the script meant to convey.
DEFAULT_MAX_CHARS = 300
MAX_MAX_CHARS = 2000

# `examples/` ships inside the plugin, so it has to work before the user edits
# any config; `/run/secrets` is where sops-nix lands decrypted scripts.
DEFAULT_ALLOWED_ROOTS = (
    str(Path(__file__).resolve().parent.parent),
    "/run/secrets",
)


@dataclass(frozen=True)
class PanelConfig:
    """Validated settings for a single turn."""

    script: str = ""
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_chars: int = DEFAULT_MAX_CHARS
    allowed_roots: tuple[str, ...] = DEFAULT_ALLOWED_ROOTS

    @property
    def enabled(self) -> bool:
        """No script configured means the plugin stays inert, by design."""
        return bool(self.script)


def load_config(ctx: Any) -> PanelConfig:
    """Build a config from the plugin context; never raises."""
    try:
        return PanelConfig(
            script=_string(ctx, KEY_SCRIPT, ""),
            timeout_seconds=_bounded_float(
                ctx, KEY_TIMEOUT_SECONDS, DEFAULT_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS
            ),
            max_chars=_bounded_int(
                ctx, KEY_MAX_CHARS, DEFAULT_MAX_CHARS, MAX_MAX_CHARS
            ),
            allowed_roots=_roots(ctx),
        )
    except Exception:
        # The default config is inert, so a broken store degrades to "do nothing"
        # rather than to a half-configured plugin.
        logger.warning("panel settings unreadable; using defaults", exc_info=True)
        return PanelConfig()


def _string(ctx: Any, key: str, default: str) -> str:
    value = ctx.get_config(key, default)
    return value.strip() if isinstance(value, str) else default


def _bounded_float(ctx: Any, key: str, default: float, upper: float) -> float:
    value = ctx.get_config(key, default)
    # bool is an int subclass, and `timeout_seconds: true` is a typo, not 1 second.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    # Clamp before converting: YAML yields arbitrary-precision ints, and
    # float(10**400) overflows where min() would simply have clamped it. The
    # comparison also rejects NaN, which is never greater than zero.
    if not value > 0:
        return default
    return float(min(value, upper))


def _bounded_int(ctx: Any, key: str, default: int, upper: int) -> int:
    value = ctx.get_config(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return min(value, upper) if value > 0 else default


def _roots(ctx: Any) -> tuple[str, ...]:
    value = ctx.get_config(KEY_ALLOWED_ROOTS, list(DEFAULT_ALLOWED_ROOTS))
    if isinstance(value, str):
        candidates: Sequence[Any] = (value,)
    elif isinstance(value, (list, tuple)):
        candidates = value
    else:
        return DEFAULT_ALLOWED_ROOTS
    # A relative root would resolve against the service's working directory, so it
    # would silently mean something else after the unit is started from elsewhere.
    roots = tuple(
        entry.strip()
        for entry in candidates
        if isinstance(entry, str) and os.path.isabs(entry.strip())
    )
    # An empty list would silently disable every configured script, so treat it as
    # "unset" and keep the examples working.
    return roots or DEFAULT_ALLOWED_ROOTS
