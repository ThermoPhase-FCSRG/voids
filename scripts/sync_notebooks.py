from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


def main() -> int:
    ipynb_files = sorted(str(path) for path in NOTEBOOKS_DIR.glob("*.ipynb"))
    if not ipynb_files:
        print("No notebooks found to sync.")
        return 0

    command = ["jupytext", "--sync", *ipynb_files]
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
