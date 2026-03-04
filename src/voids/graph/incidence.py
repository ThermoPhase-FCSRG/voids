from __future__ import annotations

import numpy as np
from scipy import sparse

from voids.core.network import Network


def incidence_matrix(net: Network) -> sparse.csr_matrix:
    rows = np.repeat(np.arange(net.Nt), 2)
    cols = net.throat_conns.reshape(-1)
    data = np.tile(np.array([1.0, -1.0]), net.Nt)
    return sparse.coo_matrix((data, (rows, cols)), shape=(net.Nt, net.Np)).tocsr()
