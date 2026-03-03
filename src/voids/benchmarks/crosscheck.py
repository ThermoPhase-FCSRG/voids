from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.network import Network
from ..io.openpnm import to_openpnm_dict
from ..io.porespy import from_porespy
from ..physics.singlephase import (
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


def _summary_from_results(
    reference: str, axis: str, r0: SinglePhaseResult, r1: SinglePhaseResult
) -> SinglePhaseCrosscheckSummary:
    k0 = float((r0.permeability or {}).get(axis, np.nan))
    k1 = float((r1.permeability or {}).get(axis, np.nan))
    q0 = float(r0.total_flow_rate)
    q1 = float(r1.total_flow_rate)
    return SinglePhaseCrosscheckSummary(
        reference=reference,
        axis=axis,
        permeability_abs_diff=abs(k0 - k1),
        permeability_rel_diff=_rel_diff(k0, k1),
        total_flow_abs_diff=abs(q0 - q1),
        total_flow_rel_diff=_rel_diff(q0, q1),
        details={"k_voids": k0, "k_ref": k1, "Q_voids": q0, "Q_ref": q1},
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


def crosscheck_singlephase_with_openpnm(
    net: Network,
    fluid: FluidSinglePhase,
    bc: PressureBC,
    *,
    axis: str,
    options: SinglePhaseOptions | None = None,
) -> SinglePhaseCrosscheckSummary:
    """Best-effort solver cross-check against OpenPNM (optional dependency).

    Notes
    -----
    OpenPNM API variants differ across versions. This helper currently provides
    a strict ImportError if OpenPNM is unavailable and a NotImplementedError for
    unsupported versions. It is included to stabilize the public cross-check API
    before wiring a version-specific adapter.
    """
    try:
        import openpnm as op  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise ImportError(
            "OpenPNM is not installed. Use the 'interop' pixi environment or install openpnm."
        ) from exc

    raise NotImplementedError(
        "OpenPNM solver cross-check adapter is version-sensitive and not yet wired in v0.1.1. "
        "Use crosscheck_singlephase_roundtrip_openpnm_dict for interoperability checks."
    )
