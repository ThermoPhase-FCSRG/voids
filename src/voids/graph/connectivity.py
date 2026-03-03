from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components as _cc

from ..core.network import Network


def adjacency_matrix(net: Network) -> sparse.csr_matrix:
    i = net.throat_conns[:, 0]
    j = net.throat_conns[:, 1]
    data = np.ones(net.Nt, dtype=float)
    A = sparse.coo_matrix((data, (i, j)), shape=(net.Np, net.Np))
    return (A + A.T).tocsr()


def connected_components(net: Network) -> tuple[int, np.ndarray]:
    A = adjacency_matrix(net)
    n, labels = _cc(A, directed=False, return_labels=True)
    return int(n), labels.astype(np.int64)


def _axis_boundary_labels(axis: str) -> tuple[str, str]:
    amap = {
        "x": ("inlet_xmin", "outlet_xmax"),
        "y": ("inlet_ymin", "outlet_ymax"),
        "z": ("inlet_zmin", "outlet_zmax"),
    }
    if axis not in amap:
        raise ValueError(f"Unsupported axis '{axis}'")
    return amap[axis]


def spanning_component_ids(net: Network, axis: str, labels: np.ndarray | None = None) -> np.ndarray:
    if labels is None:
        _, labels = connected_components(net)
    inlet_name, outlet_name = _axis_boundary_labels(axis)
    if inlet_name not in net.pore_labels or outlet_name not in net.pore_labels:
        raise KeyError(f"Missing pore labels '{inlet_name}'/'{outlet_name}'")
    inlet_mask = net.pore_labels[inlet_name]
    outlet_mask = net.pore_labels[outlet_name]
    inlet_ids = np.unique(labels[inlet_mask])
    outlet_ids = np.unique(labels[outlet_mask])
    return np.intersect1d(inlet_ids, outlet_ids)


def spanning_component_mask(net: Network, axis: str, labels: np.ndarray | None = None) -> np.ndarray:
    if labels is None:
        _, labels = connected_components(net)
    comp_ids = spanning_component_ids(net, axis=axis, labels=labels)
    return np.isin(labels, comp_ids)
