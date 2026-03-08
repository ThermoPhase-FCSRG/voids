"""Shared helpers for high-level segmented-volume benchmarks."""

from __future__ import annotations

import math

from voids.physics.singlephase import PressureBC

DEFAULT_BENCHMARK_PRESSURE_DROP = 1.0
DEFAULT_BENCHMARK_REFERENCE_PRESSURE = 0.0


def resolve_benchmark_pressures(
    *,
    delta_p: float | None = None,
    pin: float | None = None,
    pout: float | None = None,
    default_delta_p: float = DEFAULT_BENCHMARK_PRESSURE_DROP,
    default_reference_pressure: float = DEFAULT_BENCHMARK_REFERENCE_PRESSURE,
) -> tuple[float, float, float]:
    """Resolve physical pressure inputs for a high-level segmented-volume benchmark.

    Notes
    -----
    The current high-level segmented-volume benchmarks are formulated for
    incompressible permeability estimation. In that setting, the physically
    relevant input is the imposed pressure drop ``delta_p = pin - pout``.
    Absolute pressure offsets are therefore treated as a gauge choice: they can
    be accepted for clarity or provenance, but they do not alter the current
    permeability estimate as long as ``delta_p`` is unchanged.
    """

    delta_p_input = None if delta_p is None else float(delta_p)
    pin_input = None if pin is None else float(pin)
    pout_input = None if pout is None else float(pout)

    if delta_p_input is None and pin_input is None and pout_input is None:
        delta_p_input = float(default_delta_p)
        pout_input = float(default_reference_pressure)
        pin_input = pout_input + delta_p_input
    elif delta_p_input is None:
        if pin_input is None or pout_input is None:
            raise ValueError(
                "Provide either `delta_p`, or both `pin` and `pout`, for a "
                "high-level segmented-volume benchmark"
            )
        delta_p_input = pin_input - pout_input
    else:
        if pin_input is None and pout_input is None:
            pout_input = float(default_reference_pressure)
            pin_input = pout_input + delta_p_input
        elif pin_input is None:
            assert pout_input is not None
            pin_input = pout_input + delta_p_input
        elif pout_input is None:
            pout_input = pin_input - delta_p_input
        else:
            if not math.isclose(
                pin_input - pout_input,
                delta_p_input,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "Inconsistent pressure inputs: `delta_p` must equal `pin - pout` "
                    "when all three are provided"
                )

    assert delta_p_input is not None
    assert pin_input is not None
    assert pout_input is not None

    if not all(math.isfinite(value) for value in (delta_p_input, pin_input, pout_input)):
        raise ValueError("Benchmark pressure inputs must be finite")
    if delta_p_input <= 0.0 or pin_input <= pout_input:
        raise ValueError(
            "High-level segmented-volume benchmarks require a positive physical "
            "pressure drop (`delta_p > 0`, `pin > pout`)"
        )
    return pin_input, pout_input, delta_p_input


def make_benchmark_pressure_bc(axis: str, *, pin: float, pout: float) -> PressureBC:
    """Return the standard inlet/outlet pressure BC for a benchmark axis."""

    pin_float, pout_float, _ = resolve_benchmark_pressures(pin=pin, pout=pout)
    return PressureBC(
        f"inlet_{axis}min",
        f"outlet_{axis}max",
        pin=pin_float,
        pout=pout_float,
    )
