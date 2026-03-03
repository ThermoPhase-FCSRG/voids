from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.network import Network
from ..graph.connectivity import connected_components, spanning_component_mask
from ..graph.metrics import ConnectivitySummary, connectivity_metrics as _connectivity_metrics


@dataclass(slots=True)
class PorosityBreakdown:
    void_volume: float
    bulk_volume: float
    porosity: float


def _void_volume(net: Network, pore_mask: np.ndarray | None = None) -> float:
    if "volume" not in net.pore or "volume" not in net.throat:
        raise KeyError("Both pore.volume and throat.volume are required for porosity calculations")
    pv = np.asarray(net.pore["volume"], dtype=float)
    tv = np.asarray(net.throat["volume"], dtype=float)
    if pore_mask is None:
        throat_mask = np.ones(net.Nt, dtype=bool)
        pore_mask_local = np.ones(net.Np, dtype=bool)
    else:
        pore_mask_local = np.asarray(pore_mask, dtype=bool)
        c = net.throat_conns
        throat_mask = pore_mask_local[c[:, 0]] & pore_mask_local[c[:, 1]]
    return float(pv[pore_mask_local].sum() + tv[throat_mask].sum())


def absolute_porosity(net: Network) -> float:
    return _void_volume(net) / net.sample.resolved_bulk_volume()


def effective_porosity(net: Network, axis: str | None = None, mode: str | None = None) -> float:
    """Compute effective porosity.

    Modes:
      - axis provided => spanning-by-axis using inlet/outlet labels for that axis
      - axis None and mode in {"boundary_connected", None}: components connected to any boundary labels
    """
    _, comp_labels = connected_components(net)
    if axis is not None:
        pore_mask = spanning_component_mask(net, axis=axis, labels=comp_labels)
    else:
        if mode is None:
            mode = "boundary_connected"
        if mode != "boundary_connected":
            raise ValueError(f"Unsupported effective porosity mode '{mode}'")
        boundary_ids: list[int] = []
        for name, mask in net.pore_labels.items():
            lname = name.lower()
            if lname.startswith("inlet") or lname.startswith("outlet") or lname == "boundary":
                boundary_ids.extend(np.unique(comp_labels[np.asarray(mask, dtype=bool)]).tolist())
        pore_mask = np.isin(comp_labels, np.unique(boundary_ids)) if boundary_ids else np.zeros(net.Np, dtype=bool)
    return _void_volume(net, pore_mask=pore_mask) / net.sample.resolved_bulk_volume()


def connectivity_metrics(net: Network) -> ConnectivitySummary:
    return _connectivity_metrics(net)
