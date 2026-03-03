from __future__ import annotations

import numpy as np
from scipy import sparse

from ..core.network import Network


def assemble_pressure_system(net: Network, throat_conductance: np.ndarray) -> sparse.csr_matrix:
    g = np.asarray(throat_conductance, dtype=float)
    if g.shape != (net.Nt,):
        raise ValueError("throat_conductance must have shape (Nt,)")
    if (g < 0).any():
        raise ValueError("throat_conductance must be nonnegative")
    i = net.throat_conns[:, 0]
    j = net.throat_conns[:, 1]
    # Off-diagonal contributions
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    data = np.concatenate([-g, -g])
    # Diagonal contributions
    diag = np.zeros(net.Np, dtype=float)
    np.add.at(diag, i, g)
    np.add.at(diag, j, g)
    rows = np.concatenate([rows, np.arange(net.Np)])
    cols = np.concatenate([cols, np.arange(net.Np)])
    data = np.concatenate([data, diag])
    return sparse.coo_matrix((data, (rows, cols)), shape=(net.Np, net.Np)).tocsr()
