from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, gmres, spsolve


def solve_linear_system(A: sparse.csr_matrix, b: np.ndarray, *, method: str = "direct") -> tuple[np.ndarray, dict[str, Any]]:
    if method == "direct":
        x = spsolve(A, b)
        return np.asarray(x, dtype=float), {"method": method, "info": 0}
    if method == "cg":
        x, info = cg(A, b)
        return np.asarray(x, dtype=float), {"method": method, "info": int(info)}
    if method == "gmres":
        x, info = gmres(A, b)
        return np.asarray(x, dtype=float), {"method": method, "info": int(info)}
    raise ValueError(f"Unknown solver method '{method}'")
