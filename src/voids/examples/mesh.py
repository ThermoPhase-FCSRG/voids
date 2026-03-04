from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from voids.core.network import Network
from voids.core.provenance import Provenance
from voids.core.sample import SampleGeometry


_AXES = ("x", "y", "z")
_CIRCULAR_SHAPE_FACTOR = 1.0 / (4.0 * np.pi)


def _normalize_shape(shape: Sequence[int]) -> tuple[int, ...]:
    dims = tuple(int(n) for n in shape)
    if len(dims) not in {2, 3}:
        raise ValueError("shape must have length 2 or 3, e.g. (20, 20) or (20, 20, 20)")
    if any(n < 2 for n in dims):
        raise ValueError(
            "each entry in shape must be >= 2 to define distinct inlet/outlet boundaries"
        )
    return dims


def _build_cartesian_connectivity(shape3: tuple[int, int, int], ndim: int) -> np.ndarray:
    node_ids = np.arange(np.prod(shape3), dtype=np.int64).reshape(shape3)
    conns: list[np.ndarray] = []
    for axis in range(ndim):
        start = [slice(None)] * 3
        stop = [slice(None)] * 3
        start[axis] = slice(0, -1)
        stop[axis] = slice(1, None)
        conns.append(
            np.column_stack(
                [
                    node_ids[tuple(start)].ravel(),
                    node_ids[tuple(stop)].ravel(),
                ]
            )
        )
    return np.vstack(conns)


def _build_boundary_labels(shape3: tuple[int, int, int], ndim: int) -> dict[str, np.ndarray]:
    node_ids = np.arange(np.prod(shape3), dtype=np.int64).reshape(shape3)
    labels: dict[str, np.ndarray] = {"all": np.ones(node_ids.size, dtype=bool)}
    boundary = np.zeros(node_ids.size, dtype=bool)
    for axis_index, axis_name in enumerate(_AXES[:ndim]):
        inlet = np.zeros(node_ids.size, dtype=bool)
        outlet = np.zeros(node_ids.size, dtype=bool)
        inlet[node_ids.take(indices=0, axis=axis_index).ravel()] = True
        outlet[node_ids.take(indices=shape3[axis_index] - 1, axis=axis_index).ravel()] = True
        labels[f"{axis_name}min"] = inlet.copy()
        labels[f"{axis_name}max"] = outlet.copy()
        labels[f"inlet_{axis_name}min"] = inlet
        labels[f"outlet_{axis_name}max"] = outlet
        boundary |= inlet | outlet
    labels["boundary"] = boundary
    return labels


def make_cartesian_mesh_network(
    shape: Sequence[int],
    *,
    spacing: float = 1.0,
    pore_radius: float | None = None,
    throat_radius: float | None = None,
    thickness: float | None = None,
    units: dict[str, str] | None = None,
) -> Network:
    """Build a synthetic regular mesh network where each mesh node is a pore.

    Parameters
    ----------
    shape
        Number of pores along each axis. Supports 2D and 3D shapes such as
        ``(20, 20)`` or ``(20, 20, 20)``.
    spacing
        Center-to-center pore spacing along each active axis.
    pore_radius, throat_radius
        Geometric parameters for the synthetic pore bodies and throats. They
        must be smaller than half the pore spacing so neighboring pores remain
        distinct in the conduit decomposition.
    thickness
        Extrusion thickness for 2D meshes. Ignored for 3D meshes.

    Notes
    -----
    This is a controlled Cartesian lattice for method development and solver
    validation. It is not intended to represent a geologically realistic pore
    network. Pore volumes are modeled as cylinders in 2D (extruded through the
    thickness) and spheres in 3D. Throat volumes use the core throat length so
    the pore and throat void volumes are not double-counted excessively.
    """

    dims = _normalize_shape(shape)
    ndim = len(dims)
    if spacing <= 0:
        raise ValueError("spacing must be positive")

    pore_radius = 0.2 * spacing if pore_radius is None else float(pore_radius)
    throat_radius = 0.1 * spacing if throat_radius is None else float(throat_radius)
    if pore_radius <= 0 or throat_radius <= 0:
        raise ValueError("pore_radius and throat_radius must be positive")
    if pore_radius >= 0.5 * spacing:
        raise ValueError("pore_radius must be smaller than half the pore spacing")
    if throat_radius >= 0.5 * spacing:
        raise ValueError("throat_radius must be smaller than half the pore spacing")

    if ndim == 2:
        nz = 1
        depth = float(spacing if thickness is None else thickness)
        if depth <= 0:
            raise ValueError("thickness must be positive for 2D meshes")
        shape3 = (dims[0], dims[1], nz)
        x = (np.arange(dims[0], dtype=float) + 0.5) * spacing
        y = (np.arange(dims[1], dtype=float) + 0.5) * spacing
        z = np.array([0.5 * depth], dtype=float)
        pore_volume_scalar = np.pi * pore_radius**2 * depth
        cross_sections = {
            "x": dims[1] * spacing * depth,
            "y": dims[0] * spacing * depth,
        }
        lengths = {
            "x": dims[0] * spacing,
            "y": dims[1] * spacing,
        }
        bulk_volume = dims[0] * dims[1] * spacing**2 * depth
    else:
        shape3 = (dims[0], dims[1], dims[2])
        x = (np.arange(dims[0], dtype=float) + 0.5) * spacing
        y = (np.arange(dims[1], dtype=float) + 0.5) * spacing
        z = (np.arange(dims[2], dtype=float) + 0.5) * spacing
        pore_volume_scalar = (4.0 / 3.0) * np.pi * pore_radius**3
        cross_sections = {
            "x": dims[1] * dims[2] * spacing**2,
            "y": dims[0] * dims[2] * spacing**2,
            "z": dims[0] * dims[1] * spacing**2,
        }
        lengths = {
            "x": dims[0] * spacing,
            "y": dims[1] * spacing,
            "z": dims[2] * spacing,
        }
        bulk_volume = dims[0] * dims[1] * dims[2] * spacing**3

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    pore_coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    throat_conns = _build_cartesian_connectivity(shape3, ndim=ndim)
    pore_labels = _build_boundary_labels(shape3, ndim=ndim)

    throat_area_scalar = np.pi * throat_radius**2
    throat_perimeter_scalar = 2.0 * np.pi * throat_radius
    throat_core_length_scalar = spacing - 2.0 * pore_radius
    if throat_core_length_scalar <= 0:  # pragma: no cover - guarded by pore_radius < spacing / 2
        raise ValueError(
            "pore_radius is too large relative to spacing; throat core length must stay positive"
        )

    pore_area_scalar = np.pi * pore_radius**2
    pore_perimeter_scalar = 2.0 * np.pi * pore_radius
    n_pores = pore_coords.shape[0]
    n_throats = throat_conns.shape[0]

    pore = {
        "volume": np.full(n_pores, pore_volume_scalar, dtype=float),
        "area": np.full(n_pores, pore_area_scalar, dtype=float),
        "perimeter": np.full(n_pores, pore_perimeter_scalar, dtype=float),
        "shape_factor": np.full(n_pores, _CIRCULAR_SHAPE_FACTOR, dtype=float),
        "radius_inscribed": np.full(n_pores, pore_radius, dtype=float),
        "diameter_inscribed": np.full(n_pores, 2.0 * pore_radius, dtype=float),
    }
    throat = {
        "volume": np.full(n_throats, throat_area_scalar * throat_core_length_scalar, dtype=float),
        "area": np.full(n_throats, throat_area_scalar, dtype=float),
        "perimeter": np.full(n_throats, throat_perimeter_scalar, dtype=float),
        "shape_factor": np.full(n_throats, _CIRCULAR_SHAPE_FACTOR, dtype=float),
        "radius_inscribed": np.full(n_throats, throat_radius, dtype=float),
        "diameter_inscribed": np.full(n_throats, 2.0 * throat_radius, dtype=float),
        "length": np.full(n_throats, spacing, dtype=float),
        "direct_length": np.full(n_throats, spacing, dtype=float),
        "pore1_length": np.full(n_throats, pore_radius, dtype=float),
        "core_length": np.full(n_throats, throat_core_length_scalar, dtype=float),
        "pore2_length": np.full(n_throats, pore_radius, dtype=float),
    }

    sample = SampleGeometry(
        bulk_volume=float(bulk_volume),
        lengths={k: float(v) for k, v in lengths.items()},
        cross_sections={k: float(v) for k, v in cross_sections.items()},
        units=units or {"length": "m", "pressure": "Pa"},
    )
    provenance = Provenance(
        source_kind="synthetic_mesh",
        extraction_method="cartesian_lattice",
        voxel_size_original=float(spacing),
        user_notes={"shape": list(dims)},
    )

    return Network(
        throat_conns=throat_conns,
        pore_coords=pore_coords,
        sample=sample,
        provenance=provenance,
        pore=pore,
        throat=throat,
        pore_labels=pore_labels,
        extra={
            "mesh_shape": tuple(dims),
            "mesh_spacing": float(spacing),
            "mesh_ndim": ndim,
        },
    )
