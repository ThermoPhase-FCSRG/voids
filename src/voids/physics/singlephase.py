from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from voids.core.network import Network
from voids.geom.hydraulic import throat_conductance as _throat_conductance
from voids.graph.connectivity import connected_components, induced_subnetwork
from voids.linalg.assemble import assemble_pressure_system
from voids.linalg.bc import apply_dirichlet_rowcol
from voids.linalg.diagnostics import residual_norm
from voids.linalg.solve import solve_linear_system


@dataclass(slots=True)
class FluidSinglePhase:
    """Single-phase fluid properties used by the flow solver.

    Attributes
    ----------
    viscosity :
        Dynamic viscosity of the fluid.
    density :
        Optional fluid density. It is stored for bookkeeping but is not used by the
        incompressible Darcy-scale solver in ``v0.1``.
    """

    viscosity: float
    density: float | None = None


@dataclass(slots=True)
class PressureBC:
    """Dirichlet pressure boundary conditions.

    Attributes
    ----------
    inlet_label, outlet_label :
        Names of pore labels identifying fixed-pressure pores.
    pin, pout :
        Pressure values imposed on inlet and outlet pores.
    """

    inlet_label: str
    outlet_label: str
    pin: float
    pout: float


@dataclass(slots=True)
class SinglePhaseOptions:
    """Numerical and constitutive options for the single-phase solver.

    Attributes
    ----------
    conductance_model :
        Name of the hydraulic conductance model passed to
        :func:`voids.geom.hydraulic.throat_conductance`.
    solver :
        Linear solver backend name.
    check_mass_balance :
        If ``True``, compute a normalized divergence residual on free pores.
    regularization :
        Optional diagonal shift added to the matrix before Dirichlet elimination.
    """

    conductance_model: str = "generic_poiseuille"
    solver: str = "direct"
    check_mass_balance: bool = True
    regularization: float | None = None


@dataclass(slots=True)
class SinglePhaseResult:
    """Results returned by :func:`solve`.

    Attributes
    ----------
    pore_pressure :
        Pressure solution at pores.
    throat_flux :
        Volumetric flux on each throat, positive when flowing from
        ``throat_conns[:, 0]`` to ``throat_conns[:, 1]``.
    throat_conductance :
        Throat conductance values used during assembly.
    total_flow_rate :
        Net inlet flow rate associated with the imposed pressure drop.
    permeability :
        Dictionary containing the apparent permeability for the simulated axis.
    residual_norm :
        Algebraic residual norm of the solved linear system.
    mass_balance_error :
        Normalized divergence residual on free pores.
    solver_info :
        Backend-specific diagnostic information.
    """

    pore_pressure: np.ndarray
    throat_flux: np.ndarray
    throat_conductance: np.ndarray
    total_flow_rate: float
    permeability: dict[str, float] | None
    residual_norm: float
    mass_balance_error: float
    solver_info: dict[str, Any] = field(default_factory=dict)


def _make_dirichlet_vector(net: Network, bc: PressureBC) -> tuple[np.ndarray, np.ndarray]:
    """Construct Dirichlet values and mask from labeled pores.

    Parameters
    ----------
    net :
        Network carrying pore labels.
    bc :
        Pressure boundary-condition specification.

    Returns
    -------
    tuple of numpy.ndarray
        Pair ``(values, mask)`` where ``values`` contains prescribed pressures and
        ``mask`` selects constrained pores.

    Raises
    ------
    KeyError
        If the requested labels are missing.
    ValueError
        If one label is empty or if inlet and outlet labels overlap.
    """

    if bc.inlet_label not in net.pore_labels:
        raise KeyError(f"Missing pore label '{bc.inlet_label}'")
    if bc.outlet_label not in net.pore_labels:
        raise KeyError(f"Missing pore label '{bc.outlet_label}'")
    inlet = np.asarray(net.pore_labels[bc.inlet_label], dtype=bool)
    outlet = np.asarray(net.pore_labels[bc.outlet_label], dtype=bool)
    if inlet.sum() == 0 or outlet.sum() == 0:
        raise ValueError("BC labels must contain at least one pore each")
    if np.any(inlet & outlet):
        raise ValueError("Inlet and outlet labels overlap")
    mask = inlet | outlet
    values = np.zeros(net.Np, dtype=float)
    values[inlet] = float(bc.pin)
    values[outlet] = float(bc.pout)
    return values, mask


def _inlet_total_flow(net: Network, q: np.ndarray, inlet_mask: np.ndarray) -> float:
    """Compute net volumetric flow entering through inlet pores.

    Parameters
    ----------
    net :
        Network topology.
    q :
        Throat flux array with sign convention ``q_t > 0`` for flow from pore ``i`` to pore ``j``.
    inlet_mask :
        Boolean pore mask identifying inlet pores.

    Returns
    -------
    float
        Net inlet flow rate.

    Notes
    -----
    The implementation sums:

    - ``+q_t`` for throats leaving an inlet pore toward a non-inlet pore
    - ``-q_t`` for throats entering an inlet pore from a non-inlet pore

    Internal inlet-inlet throats are ignored because their contributions cancel.
    """

    i = net.throat_conns[:, 0]
    j = net.throat_conns[:, 1]
    total = 0.0
    total += float(q[inlet_mask[i] & ~inlet_mask[j]].sum())
    total += float((-q[~inlet_mask[i] & inlet_mask[j]]).sum())
    return total


def _mass_balance_error(net: Network, q: np.ndarray, fixed_mask: np.ndarray) -> float:
    """Compute a normalized mass-balance residual on unconstrained pores.

    Parameters
    ----------
    net :
        Network topology.
    q :
        Throat flux array.
    fixed_mask :
        Boolean mask selecting Dirichlet pores.

    Returns
    -------
    float
        Quantity
        ``||div(q)||_2 / max(||q||_2, 1)``
        evaluated only on free pores.
    """

    i = net.throat_conns[:, 0]
    j = net.throat_conns[:, 1]
    div = np.zeros(net.Np, dtype=float)
    np.add.at(div, i, q)
    np.add.at(div, j, -q)
    free = ~fixed_mask
    denom = max(float(np.linalg.norm(q)), 1.0)
    return float(np.linalg.norm(div[free]) / denom)


def _active_bc_component_mask(net: Network, fixed_mask: np.ndarray) -> np.ndarray:
    """Select pores in components touched by at least one Dirichlet pore."""

    _, comp_labels = connected_components(net)
    active_ids = np.unique(comp_labels[np.asarray(fixed_mask, dtype=bool)])
    return np.isin(comp_labels, active_ids)


def solve(
    net: Network,
    fluid: FluidSinglePhase,
    bc: PressureBC,
    *,
    axis: str,
    options: SinglePhaseOptions | None = None,
) -> SinglePhaseResult:
    """Solve steady incompressible single-phase flow on a pore network.

    Parameters
    ----------
    net :
        Network containing topology, geometry, and sample metadata.
    fluid :
        Fluid properties. Only viscosity enters the current formulation.
    bc :
        Pressure boundary conditions.
    axis :
        Macroscopic flow axis used when converting total flow to apparent permeability.
    options :
        Optional solver and conductance settings.

    Returns
    -------
    SinglePhaseResult
        Pressure, flux, conductance, and derived transport metrics.

    Raises
    ------
    ValueError
        If viscosity is not positive or if the imposed pressure drop is zero.

    Notes
    -----
    The solver assembles a graph-Laplacian system

    ``A p = b``

    with throat fluxes

    ``q_t = g_t * (p_i - p_j)``

    where ``g_t`` is the hydraulic conductance of throat ``t``. After solving for
    pore pressure, the apparent permeability is computed from Darcy's law:

    ``K = |Q| * mu * L / (A * |delta_p|)``

    where ``Q`` is total inlet flow rate, ``mu`` is viscosity, ``L`` is the sample
    length along ``axis``, and ``A`` is the corresponding cross-sectional area.

    Connected components that do not touch any Dirichlet pore are excluded from the
    linear solve because they form floating pressure blocks. Returned pressures and
    fluxes on those excluded components are reported as ``nan``.
    """

    if fluid.viscosity <= 0:
        raise ValueError("Fluid viscosity must be positive")
    options = options or SinglePhaseOptions()

    values, fixed_mask = _make_dirichlet_vector(net, bc)
    active_pores = _active_bc_component_mask(net, fixed_mask)
    active_net, active_idx, active_throats = induced_subnetwork(net, active_pores)
    active_values = values[active_idx]
    active_fixed_mask = fixed_mask[active_idx]

    g_active = _throat_conductance(
        active_net, viscosity=fluid.viscosity, model=options.conductance_model
    )
    A = assemble_pressure_system(active_net, g_active)
    b = np.zeros(active_net.Np, dtype=float)
    if options.regularization is not None:
        A = A.copy().tocsr()
        A.setdiag(A.diagonal() + float(options.regularization))

    A_bc, b_bc = apply_dirichlet_rowcol(A, b, values=active_values, mask=active_fixed_mask)
    p_active, solver_info = solve_linear_system(A_bc, b_bc, method=options.solver)

    p = np.full(net.Np, np.nan, dtype=float)
    p[active_idx] = p_active

    g = np.full(net.Nt, np.nan, dtype=float)
    g[active_throats] = g_active

    q = np.full(net.Nt, np.nan, dtype=float)
    i_active = active_net.throat_conns[:, 0]
    j_active = active_net.throat_conns[:, 1]
    q_active = g_active * (p_active[i_active] - p_active[j_active])
    q[active_throats] = q_active

    inlet_mask = np.asarray(active_net.pore_labels[bc.inlet_label], dtype=bool)
    Q = _inlet_total_flow(active_net, q_active, inlet_mask)
    dP = float(bc.pin - bc.pout)
    if abs(dP) == 0.0:
        raise ValueError("Pressure drop pin-pout must be nonzero")
    L = net.sample.length_for_axis(axis)
    Axs = net.sample.area_for_axis(axis)
    K = abs(Q) * fluid.viscosity * L / (Axs * abs(dP))

    res = residual_norm(A_bc, p_active, b_bc)
    mbe = (
        _mass_balance_error(active_net, q_active, active_fixed_mask)
        if options.check_mass_balance
        else float("nan")
    )

    return SinglePhaseResult(
        pore_pressure=p,
        throat_flux=q,
        throat_conductance=g,
        total_flow_rate=Q,
        permeability={axis: float(K)},
        residual_norm=res,
        mass_balance_error=mbe,
        solver_info=solver_info,
    )
