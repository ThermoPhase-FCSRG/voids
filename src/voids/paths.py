from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT_ENV = "VOIDS_PROJECT_ROOT"
NOTEBOOKS_PATH_ENV = "VOIDS_NOTEBOOKS_PATH"
EXAMPLES_PATH_ENV = "VOIDS_EXAMPLES_PATH"
DATA_PATH_ENV = "VOIDS_DATA_PATH"


def _repo_root_from_source_tree() -> Path:
    """Return the repository root when running from an editable source checkout.

    This fallback is intentionally narrow: it avoids cwd-based guessing and only
    succeeds when the installed package lives under ``<repo>/src/voids``.
    """

    root = Path(__file__).resolve().parents[2]
    if (root / "src" / "voids").exists() and (root / "pixi.toml").exists():
        return root
    raise RuntimeError(
        "Could not resolve the project paths from the installed package layout. "
        "Run inside a Pixi environment with VOIDS_* path variables set."
    )


def _resolve_path(env_name: str, relative_to_root: str) -> Path:
    value = os.getenv(env_name)
    if value:
        return Path(value).expanduser().resolve()
    return (_repo_root_from_source_tree() / relative_to_root).resolve()


def project_root() -> Path:
    return _resolve_path(PROJECT_ROOT_ENV, ".")


def notebooks_path() -> Path:
    return _resolve_path(NOTEBOOKS_PATH_ENV, "notebooks")


def examples_path() -> Path:
    return _resolve_path(EXAMPLES_PATH_ENV, "examples")


def data_path() -> Path:
    return _resolve_path(DATA_PATH_ENV, "examples/data")
