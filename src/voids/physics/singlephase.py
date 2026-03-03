from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.network import Network
from ..geom.hydraulic import throat_conductance as _throat_conductance
from ..linalg.assemble import assemble_pressure_system
from ..linalg.bc import apply_dirichlet_rowcol
from ..linalg.diagnostics import residual_norm
from ..linalg.solve import solve_linear_system


@dataclass(slots=True)
class FluidSinglePhase:
    viscosity: float
    density: float | None = None


@dataclass(slots=True)
class PressureBC:
    inlet_label: str
    outlet_label: str
    pin: float
    pout: float


@dataclass(slots=True)
class SinglePhaseOptions:
    conductance_model: str = "generic_poiseuille"
    solver: str = "direct"
    check_mass_balance: bool = True
    regularization: float | None = None


@dataclass(slots=True)
class SinglePhaseResult:
    pore_pressure: np.ndarray
    throat_flux: np.ndarray
    throat_conductance: np.ndarray
    total_flow_rate: float
    permeability: dict[str, float] | None
    residual_norm: float
    mass_balance_error: float
    solver_info: dict[str, Any] = field(default_factory=dict)


def _make_dirichlet_vector(net: Network, bc: PressureBC) -> tuple[np.ndarray, np.ndarray]:
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
    i = net.throat_conns[:, 0]
    j = net.throat_conns[:, 1]
    total = 0.0
    # q positive when flowing from i -> j
    total += float(q[inlet_mask[i] & ~inlet_mask[j]].sum())
    total += float((-q[~inlet_mask[i] & inlet_mask[j]]).sum())
    # Internal inlet-inlet throats cancel physically and are ignored by construction above.
    return total


def _mass_balance_error(net: Network, q: np.ndarray, fixed_mask: np.ndarray) -> float:
    i = net.throat_conns[:, 0]
    j = net.throat_conns[:, 1]
    div = np.zeros(net.Np, dtype=float)
    # Net outflow from pore k
    np.add.at(div, i, q)
    np.add.at(div, j, -q)
    free = ~fixed_mask
    denom = max(float(np.linalg.norm(q)), 1.0)
    return float(np.linalg.norm(div[free]) / denom)


def solve(
    net: Network,
    fluid: FluidSinglePhase,
    bc: PressureBC,
    *,
    axis: str,
    options: SinglePhaseOptions | None = None,
) -> SinglePhaseResult:
    if fluid.viscosity <= 0:
        raise ValueError("Fluid viscosity must be positive")
    options = options or SinglePhaseOptions()

    g = _throat_conductance(net, viscosity=fluid.viscosity, model=options.conductance_model)
    A = assemble_pressure_system(net, g)
    b = np.zeros(net.Np, dtype=float)
    if options.regularization is not None:
        A = A.copy().tocsr()
        A.setdiag(A.diagonal() + float(options.regularization))

    values, fixed_mask = _make_dirichlet_vector(net, bc)
    A_bc, b_bc = apply_dirichlet_rowcol(A, b, values=values, mask=fixed_mask)
    p, solver_info = solve_linear_system(A_bc, b_bc, method=options.solver)

    i = net.throat_conns[:, 0]
    j = net.throat_conns[:, 1]
    q = g * (p[i] - p[j])

    inlet_mask = np.asarray(net.pore_labels[bc.inlet_label], dtype=bool)
    Q = _inlet_total_flow(net, q, inlet_mask)
    dP = float(bc.pin - bc.pout)
    if abs(dP) == 0.0:
        raise ValueError("Pressure drop pin-pout must be nonzero")
    L = net.sample.length_for_axis(axis)
    Axs = net.sample.area_for_axis(axis)
    K = abs(Q) * fluid.viscosity * L / (Axs * abs(dP))

    res = residual_norm(A_bc, p, b_bc)
    mbe = _mass_balance_error(net, q, fixed_mask) if options.check_mass_balance else float("nan")

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
