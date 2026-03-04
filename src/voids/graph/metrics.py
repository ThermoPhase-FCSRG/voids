from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voids.core.network import Network
from voids.graph.connectivity import connected_components, spanning_component_ids


@dataclass(slots=True)
class ConnectivitySummary:
    n_components: int
    giant_component_fraction: float
    isolated_pore_fraction: float
    dead_end_fraction: float
    mean_coordination: float
    coordination_histogram: dict[int, int]
    spans: dict[str, bool]


def coordination_numbers(net: Network) -> np.ndarray:
    deg = np.zeros(net.Np, dtype=np.int64)
    np.add.at(deg, net.throat_conns[:, 0], 1)
    np.add.at(deg, net.throat_conns[:, 1], 1)
    return deg


def connectivity_metrics(net: Network) -> ConnectivitySummary:
    n_comp, labels = connected_components(net)
    counts = np.bincount(labels, minlength=n_comp)
    deg = coordination_numbers(net)
    hist_keys, hist_counts = np.unique(deg, return_counts=True)
    spans: dict[str, bool] = {}
    for ax in ("x", "y", "z"):
        try:
            spans[ax] = spanning_component_ids(net, ax, labels=labels).size > 0
        except KeyError:
            continue
    return ConnectivitySummary(
        n_components=n_comp,
        giant_component_fraction=float(counts.max() / net.Np if net.Np else 0.0),
        isolated_pore_fraction=float(np.mean(deg == 0)),
        dead_end_fraction=float(np.mean(deg == 1)),
        mean_coordination=float(np.mean(deg) if deg.size else 0.0),
        coordination_histogram={int(k): int(v) for k, v in zip(hist_keys, hist_counts)},
        spans=spans,
    )
