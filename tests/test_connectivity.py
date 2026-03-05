from __future__ import annotations

import numpy as np
import pytest

from voids.graph.connectivity import (
    connected_components,
    induced_subnetwork,
    spanning_component_mask,
    spanning_subnetwork,
)
from voids.graph.metrics import connectivity_metrics


def test_connected_components(branched_network):
    """Test connected-component counting on a branched toy network."""

    n, labels = connected_components(branched_network)
    assert n == 2
    assert labels.shape == (branched_network.Np,)


def test_spanning_component_mask(branched_network):
    """Test spanning-component masking along the x axis."""

    mask = spanning_component_mask(branched_network, axis="x")
    assert mask.tolist() == [True, True, True, True, False]


def test_connectivity_metrics(branched_network):
    """Test graph connectivity summary statistics."""

    m = connectivity_metrics(branched_network)
    assert m.n_components == 2
    assert m.spans["x"] is True
    assert 0 in m.coordination_histogram


def test_induced_subnetwork_reindexes_and_filters_fields(branched_network):
    """Induced subnetworks should retain only selected pores and connecting throats."""

    pore_mask = np.array([True, True, True, False, False])
    sub, pore_idx, throat_mask = induced_subnetwork(branched_network, pore_mask)

    assert pore_idx.tolist() == [0, 1, 2]
    assert throat_mask.tolist() == [True, True, False]
    assert sub.Np == 3
    assert sub.Nt == 2
    assert sub.throat_conns.tolist() == [[0, 1], [1, 2]]
    assert sub.pore_labels["inlet_xmin"].tolist() == [True, False, False]
    assert sub.pore_labels["outlet_xmax"].tolist() == [False, False, True]


def test_induced_subnetwork_preserves_non_indexed_metadata(branched_network):
    """Fields not indexed by pore/throat count should be copied unchanged."""

    net = branched_network.copy()
    net.pore["global_meta"] = np.array(42.0)
    net.throat["global_meta"] = np.array(0.125)
    net.pore_labels["global_meta"] = np.array(True)
    net.throat_labels["global_meta"] = np.array(False)

    pore_mask = np.array([True, True, True, False, False])
    sub, _, _ = induced_subnetwork(net, pore_mask)

    assert float(sub.pore["global_meta"]) == pytest.approx(42.0)
    assert float(sub.throat["global_meta"]) == pytest.approx(0.125)
    assert bool(sub.pore_labels["global_meta"]) is True
    assert bool(sub.throat_labels["global_meta"]) is False


def test_spanning_subnetwork_matches_spanning_mask(branched_network):
    """Axis-spanning subnetwork should remove isolated non-spanning pores."""

    mask = spanning_component_mask(branched_network, axis="x")
    sub, pore_idx, throat_mask = spanning_subnetwork(branched_network, axis="x")

    assert mask.tolist() == [True, True, True, True, False]
    assert pore_idx.tolist() == [0, 1, 2, 3]
    assert throat_mask.tolist() == [True, True, True]
    assert sub.Np == 4
    assert sub.Nt == 3


def test_induced_subnetwork_validates_pore_mask_shape(branched_network):
    """Induced subnetworks should reject masks with the wrong pore count."""

    with pytest.raises(ValueError, match="pore_mask must have shape \\(Np,\\)"):
        induced_subnetwork(branched_network, np.array([True, False]))
