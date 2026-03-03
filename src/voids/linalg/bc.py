from __future__ import annotations

import numpy as np
from scipy import sparse


def apply_dirichlet_rowcol(A: sparse.csr_matrix, b: np.ndarray, values: np.ndarray, mask: np.ndarray) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Apply Dirichlet BC by row/column modification and diagonal=1.

    `values` is the full-length vector of target values; only entries where `mask` is True are applied.
    """
    A = A.tolil(copy=True)
    b2 = np.asarray(b, dtype=float).copy()
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != b2.shape or mask.shape != b2.shape:
        raise ValueError("values, mask and b must have the same shape")
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return A.tocsr(), b2

    # Column elimination contribution on RHS before zeroing column
    A_csr = A.tocsr()
    b2 = b2 - A_csr[:, idx] @ values[idx]

    for k in idx:
        A[:, k] = 0.0
        A[k, :] = 0.0
        A[k, k] = 1.0
        b2[k] = values[k]
    return A.tocsr(), b2
