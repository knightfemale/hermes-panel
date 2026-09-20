"""Locating the configured panel script.

The script path comes from configuration, so it is untrusted input as far as this
plugin is concerned: a bad or hostile value must never turn into an arbitrary
program launch. Resolution is therefore a whitelist rather than a sanitiser —
the *real* path, with symlinks already followed, has to live under one of the
configured roots.

This is a trust boundary only because those roots are not writable by anyone
less trusted than the service itself. A root that a lower-privileged user can
write to would let them swap the file between this check and the launch.
"""

from __future__ import annotations

import os
from pathlib import Path

# Path.resolve() reports symlink loops as OSError from 3.13 and as RuntimeError before
# that, and an embedded NUL as ValueError. All three have to be caught: a path this
# system cannot express is a configuration mistake, not a crash.
_RESOLVE_ERRORS = (OSError, RuntimeError, ValueError)


class ScriptPathError(ValueError):
    """The configured script cannot be run. The message is safe to show the user.

    A ``ValueError`` because a bad path is a bad value: a caller guarding configuration
    broadly can catch it without importing this module.
    """


def resolve_script(raw: str, allowed_roots: tuple[str, ...]) -> Path:
    """Return the canonical, readable path of ``raw``, or raise ``ScriptPathError``."""
    try:
        candidate = Path(raw).expanduser()
    except RuntimeError as exc:
        # `~nosuchuser` is unresolvable rather than merely absent, and callers
        # only have to handle ScriptPathError.
        raise ScriptPathError(f"script path cannot be expanded: {raw}") from exc

    if not candidate.is_absolute():
        # A relative path would resolve against the service's working directory,
        # which is not what anyone means when they write a path into config.
        raise ScriptPathError(f"script path must be absolute: {raw}")

    try:
        resolved = candidate.resolve(strict=True)
    except _RESOLVE_ERRORS as exc:
        raise ScriptPathError(f"script does not exist: {candidate}") from exc

    if not resolved.is_file():
        raise ScriptPathError(f"script is not a regular file: {resolved}")

    if not _within(resolved, allowed_roots):
        raise ScriptPathError(f"script is outside the allowed roots: {resolved}")

    if not os.access(resolved, os.R_OK):
        raise ScriptPathError(f"script is not readable: {resolved}")

    return resolved


def _within(path: Path, allowed_roots: tuple[str, ...]) -> bool:
    """Compare real paths, so a symlink placed inside a root cannot escape it."""
    for root in allowed_roots:
        try:
            root_path = Path(root).expanduser().resolve(strict=True)
        except _RESOLVE_ERRORS:
            # A root that does not exist cannot contain anything; skip it rather
            # than failing the whole lookup on an unrelated missing directory.
            continue
        if path == root_path or root_path in path.parents:
            return True
    return False
