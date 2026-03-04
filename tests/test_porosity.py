from __future__ import annotations

import numpy as np

from voids.core.network import Network
from voids.core.sample import SampleGeometry
from voids.physics.petrophysics import absolute_porosity, effective_porosity


def test_absolute_porosity(line_network):
    """Test absolute porosity on the canonical line network."""

    phi = absolute_porosity(line_network)
    # void volume = 3*1 + 2*0.5 = 4, bulk = 10
    assert np.isclose(phi, 0.4)


def test_effective_porosity_axis(branched_network):
    """Test axis-based effective porosity on a branched network."""

    # spanning pores: 0,1,2,3 (vol 4) + throats 3*0.2 = 0.6 ; bulk=20 => 0.23
    phi_eff = effective_porosity(branched_network, axis="x")
    assert np.isclose(phi_eff, 0.23)


def test_effective_porosity_boundary_mode(line_network):
    """Test boundary-connected effective porosity on the line network."""

    phi_eff = effective_porosity(line_network)
    assert np.isclose(phi_eff, 0.4)


def test_porosity_prefers_region_volume_for_voxel_extracted_networks():
    """Use region-volume bookkeeping when segmented pore regions are available."""

    net = Network(
        throat_conns=np.array([[0, 1]], dtype=int),
        pore_coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        sample=SampleGeometry(bulk_volume=10.0),
        pore={
            "region_volume": np.array([2.0, 3.0]),
            "volume": np.array([2.0, 3.0]),
        },
        throat={"volume": np.array([7.0])},
        pore_labels={
            "inlet_xmin": np.array([True, False]),
            "outlet_xmax": np.array([False, True]),
        },
    )

    assert np.isclose(absolute_porosity(net), 0.5)
    assert np.isclose(effective_porosity(net, axis="x"), 0.5)
