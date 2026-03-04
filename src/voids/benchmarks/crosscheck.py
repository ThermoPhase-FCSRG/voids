from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from voids.core.network import Network
from voids.io.openpnm import to_openpnm_dict, to_openpnm_network
from voids.io.porespy import from_porespy
from voids.physics.singlephase import (
    FluidSinglePhase,
    PressureBC,
    SinglePhaseOptions,
    SinglePhaseResult,
    solve,
)


@dataclass(slots=True)
class SinglePhaseCrosscheckSummary:
    reference: str
    axis: str
    permeability_abs_diff: float
    permeability_rel_diff: float
    total_flow_abs_diff: float
    total_flow_rel_diff: float
    details: dict[str, Any]


def _rel_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom


def _summary_from_values(
    *,
    reference: str,
    axis: str,
    k_voids: float,
    k_ref: float,
    q_voids: float,
    q_ref: float,
    details: dict[str, Any],
) -> SinglePhaseCrosscheckSummary:
    return SinglePhaseCrosscheckSummary(
        reference=reference,
        axis=axis,
        permeability_abs_diff=abs(k_voids - k_ref),
        permeability_rel_diff=_rel_diff(k_voids, k_ref),
        total_flow_abs_diff=abs(q_voids - q_ref),
        total_flow_rel_diff=_rel_diff(q_voids, q_ref),
        details={"k_voids": k_voids, "k_ref": k_ref, "Q_voids": q_voids, "Q_ref": q_ref, **details},
    )


def _summary_from_results(
    reference: str, axis: str, r0: SinglePhaseResult, r1: SinglePhaseResult
) -> SinglePhaseCrosscheckSummary:
    k0 = float((r0.permeability or {}).get(axis, np.nan))
    k1 = float((r1.permeability or {}).get(axis, np.nan))
    q0 = float(r0.total_flow_rate)
    q1 = float(r1.total_flow_rate)
    return _summary_from_values(
        reference=reference, axis=axis, k_voids=k0, k_ref=k1, q_voids=q0, q_ref=q1, details={}
    )


def crosscheck_singlephase_roundtrip_openpnm_dict(
    net: Network,
    fluid: FluidSinglePhase,
    bc: PressureBC,
    *,
    axis: str,
    options: SinglePhaseOptions | None = None,
) -> SinglePhaseCrosscheckSummary:
    """Cross-check `voids` solver against a dict roundtrip through OpenPNM/PoreSpy keys.

    This validates interoperability and importer/exporter consistency even when
    OpenPNM is not installed.
    """
    options = options or SinglePhaseOptions()
    r0 = solve(net, fluid=fluid, bc=bc, axis=axis, options=options)
    op_dict = to_openpnm_dict(net)
    net_rt = from_porespy(op_dict, sample=net.sample, provenance=net.provenance)
    r1 = solve(net_rt, fluid=fluid, bc=bc, axis=axis, options=options)
    return _summary_from_results("openpnm_dict_roundtrip", axis, r0, r1)


def _openpnm_phase_factory(op, pn):
    # OpenPNM 3.x: op.phase.Phase; older variants may expose op.phases.GenericPhase
    for factory in (
        lambda: op.phase.Phase(network=pn),
        lambda: op.phases.GenericPhase(network=pn),
    ):
        try:
            return factory()
        except Exception:
            continue
    raise RuntimeError("Unable to construct OpenPNM phase object")


def _get_openpnm_pressure(sf):
    for getter in (
        lambda: sf["pore.pressure"],
        lambda: sf.soln["pore.pressure"],
    ):
        try:
            arr = np.asarray(getter(), dtype=float)
            if arr.ndim == 1:
                return arr
        except Exception:
            continue
    raise RuntimeError("Unable to extract pore pressures from OpenPNM StokesFlow result")


def crosscheck_singlephase_with_openpnm(
    net: Network,
    fluid: FluidSinglePhase,
    bc: PressureBC,
    *,
    axis: str,
    options: SinglePhaseOptions | None = None,
) -> SinglePhaseCrosscheckSummary:
    """Run `voids` and OpenPNM StokesFlow on the same conductance field and compare Q/K.

    Notes
    -----
    This adapter injects *voids-computed* ``throat.hydraulic_conductance`` into OpenPNM so the
    cross-check isolates assembly/BC/solver consistency rather than geometry-model differences.
    """
    try:
        import openpnm as op
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise ImportError(
            "OpenPNM is not installed. Use the 'test' pixi environment or install openpnm."
        ) from exc

    options = options or SinglePhaseOptions()
    r_voids = solve(net, fluid=fluid, bc=bc, axis=axis, options=options)
    g = np.asarray(r_voids.throat_conductance, dtype=float)

    pn = to_openpnm_network(net, copy_properties=False, copy_labels=True)
    phase = _openpnm_phase_factory(op, pn)
    phase["throat.hydraulic_conductance"] = g

    sf = op.algorithms.StokesFlow(network=pn, phase=phase)
    inlet_mask = np.asarray(net.pore_labels[bc.inlet_label], dtype=bool)
    outlet_mask = np.asarray(net.pore_labels[bc.outlet_label], dtype=bool)
    inlet = np.where(inlet_mask)[0]
    outlet = np.where(outlet_mask)[0]

    # BC setter compatibility (OpenPNM versions vary slightly)
    if hasattr(sf, "set_value_BC"):
        sf.set_value_BC(pores=inlet, values=float(bc.pin))
        sf.set_value_BC(pores=outlet, values=float(bc.pout))
    elif hasattr(sf, "set_BC"):
        sf.set_BC(pores=inlet, bctype="value", bcvalues=float(bc.pin))
        sf.set_BC(pores=outlet, bctype="value", bcvalues=float(bc.pout))
    else:  # pragma: no cover
        raise RuntimeError("OpenPNM StokesFlow object does not expose a recognizable BC API")

    sf.run()
    p_ref = _get_openpnm_pressure(sf)

    q_rate = np.asarray(sf.rate(pores=inlet), dtype=float)
    q_ref_raw = float(q_rate.sum())
    q_ref = q_ref_raw
    # Align sign convention if only sign differs (common across APIs) while preserving raw value in details.
    if np.isfinite(q_ref) and np.isfinite(r_voids.total_flow_rate):
        if np.isclose(abs(q_ref), abs(r_voids.total_flow_rate), rtol=1e-8, atol=1e-14):
            q_ref = float(np.copysign(abs(q_ref), r_voids.total_flow_rate))

    dP = float(bc.pin - bc.pout)
    if abs(dP) == 0.0:
        raise ValueError("Pressure drop pin-pout must be nonzero")
    L = net.sample.length_for_axis(axis)
    Axs = net.sample.area_for_axis(axis)
    k_ref = abs(q_ref_raw) * fluid.viscosity * L / (Axs * abs(dP))
    k_voids = float((r_voids.permeability or {}).get(axis, np.nan))

    return _summary_from_values(
        reference="openpnm_stokesflow",
        axis=axis,
        k_voids=k_voids,
        k_ref=float(k_ref),
        q_voids=float(r_voids.total_flow_rate),
        q_ref=float(q_ref),
        details={
            "openpnm_version": getattr(op, "__version__", "unknown"),
            "q_ref_raw": q_ref_raw,
            "n_inlet_pores": int(inlet.size),
            "n_outlet_pores": int(outlet.size),
            "conductance_model": options.conductance_model,
            "solver_voids": options.solver,
            "p_ref_min": float(np.min(p_ref)) if p_ref.size else np.nan,
            "p_ref_max": float(np.max(p_ref)) if p_ref.size else np.nan,
        },
    )
