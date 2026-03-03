from __future__ import annotations

import numpy as np
from scipy import sparse


def residual_norm(A: sparse.csr_matrix, x: np.ndarray, b: np.ndarray) -> float:
    r = A @ x - b
    return float(np.linalg.norm(r))
