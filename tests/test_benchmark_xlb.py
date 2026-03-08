from __future__ import annotations

import warnings

import numpy as np
import pytest

from voids.benchmarks import (
    XLBOptions,
    benchmark_segmented_volume_with_xlb,
    solve_binary_volume_with_xlb,
)
from voids.physics.singlephase import FluidSinglePhase


def test_benchmark_segmented_volume_with_xlb_rejects_nonbinary_inputs() -> None:
    """Test binary-volume validation before optional XLB imports."""

    phases = np.array([[[0, 2], [1, 0]], [[1, 0], [0, 1]]], dtype=int)

    with pytest.raises(ValueError, match="phases must be binary with void=1 and solid=0"):
        benchmark_segmented_volume_with_xlb(phases, voxel_size=1.0)


def test_benchmark_segmented_volume_with_xlb_rejects_invalid_rank() -> None:
    """Test rank validation before extraction or optional XLB imports."""

    phases = np.array([0, 1, 0, 1], dtype=int)

    with pytest.raises(ValueError, match="phases must be a 2D or 3D binary segmented volume"):
        benchmark_segmented_volume_with_xlb(phases, voxel_size=1.0)


def test_xlb_direct_solver_api_available_or_clean_import_error() -> None:
    """Test that the direct XLB solve either runs or fails with a clean import error."""

    phases = np.zeros((8, 10, 10), dtype=int)
    phases[:, 3:7, 3:7] = 1

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            solve_binary_volume_with_xlb(
                phases,
                voxel_size=1.0,
                flow_axis="x",
                options=XLBOptions(max_steps=10, min_steps=0, check_interval=5),
            )
    except ImportError:
        return


def test_benchmark_segmented_volume_with_xlb_returns_positive_permeabilities() -> None:
    """Test end-to-end extraction plus direct-image XLB comparison on a tiny segmented volume."""

    pytest.importorskip("xlb")

    phases = np.zeros((12, 16, 16), dtype=int)
    phases[:, 5:11, 5:11] = 1
    phases[2:4, 1:3, 1:3] = 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = benchmark_segmented_volume_with_xlb(
            phases,
            voxel_size=1.0,
            flow_axis="x",
            length_unit="voxel",
            fluid=FluidSinglePhase(viscosity=1.0),
            provenance_notes={"case": "tiny_xlb"},
            xlb_options=XLBOptions(
                max_steps=120,
                min_steps=60,
                check_interval=20,
                lattice_viscosity=0.1,
                rho_inlet=1.001,
                rho_outlet=1.0,
            ),
        )
    record = result.to_record()

    assert result.extract.flow_axis == "x"
    assert result.extract.provenance.user_notes["case"] == "tiny_xlb"
    assert result.image_porosity == pytest.approx(float(phases.mean()))
    assert record["Np"] == result.extract.net.Np
    assert record["Nt"] == result.extract.net.Nt
    assert record["phi_abs"] == pytest.approx(result.absolute_porosity)
    assert record["phi_eff"] == pytest.approx(result.effective_porosity)
    assert record["k_voids"] > 0.0
    assert record["k_xlb"] > 0.0
    assert record["xlb_steps"] > 0
    assert record["xlb_backend"] == "jax"


def test_xlb_direct_solver_open_duct_returns_finite_positive_permeability() -> None:
    """Test that an all-void duct does not produce NaNs or negative permeability."""

    pytest.importorskip("xlb")

    phases = np.ones((16, 8, 8), dtype=int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = solve_binary_volume_with_xlb(
            phases,
            voxel_size=1.0,
            flow_axis="x",
            options=XLBOptions(
                max_steps=160,
                min_steps=80,
                check_interval=20,
                steady_rtol=1.0e-4,
                lattice_viscosity=0.1,
                rho_inlet=1.001,
                rho_outlet=1.0,
            ),
        )

    assert np.isfinite(result.superficial_velocity_lattice)
    assert np.isfinite(result.permeability)
    assert result.superficial_velocity_lattice > 0.0
    assert result.permeability > 0.0
