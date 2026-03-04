from __future__ import annotations

import warnings
from typing import Final

import numpy as np

from voids.core.network import Network


DEFAULT_G_REF: Final[float] = 1.0 / (4.0 * np.pi)  # circular duct shape factor A/P^2
_CIRCLE_COEFF_AG2: Final[float] = 0.5  # gives Hagen-Poiseuille when G=1/(4π)


def _require(net: Network, kind: str, names: tuple[str, ...]) -> None:
    store = net.throat if kind == "throat" else net.pore
    missing = [n for n in names if n not in store]
    if missing:
        raise KeyError(f"Missing required {kind} fields: {missing}")


def _diameter_from_area(area: np.ndarray) -> np.ndarray:
    return 2.0 * np.sqrt(area / np.pi)


def _area_from_diameter(d: np.ndarray) -> np.ndarray:
    r = 0.5 * d
    return np.pi * r**2


def _shape_factor_from_area_perimeter(area: np.ndarray, perimeter: np.ndarray) -> np.ndarray:
    return area / np.maximum(perimeter, 1e-30) ** 2


def _get_entity_area(net: Network, kind: str) -> np.ndarray:
    store = net.throat if kind == "throat" else net.pore
    if "area" in store:
        return np.asarray(store["area"], dtype=float)
    if "diameter_inscribed" in store:
        d = np.asarray(store["diameter_inscribed"], dtype=float)
        return _area_from_diameter(d)
    if "radius_inscribed" in store:
        r = np.asarray(store["radius_inscribed"], dtype=float)
        return np.pi * r**2
    raise KeyError(f"Need {kind}.area or {kind}.diameter_inscribed (or radius_inscribed)")


def _get_entity_shape_factor(net: Network, kind: str, area: np.ndarray | None = None) -> np.ndarray:
    store = net.throat if kind == "throat" else net.pore
    if "shape_factor" in store:
        return np.asarray(store["shape_factor"], dtype=float)
    if "perimeter" in store:
        a = _get_entity_area(net, kind) if area is None else np.asarray(area, dtype=float)
        p = np.asarray(store["perimeter"], dtype=float)
        return _shape_factor_from_area_perimeter(a, p)
    raise KeyError(f"Need {kind}.shape_factor or {kind}.perimeter (with area/diameter)")


def _segment_conductance_from_agl(
    area: np.ndarray,
    shape_factor: np.ndarray,
    length: np.ndarray,
    viscosity: float,
    *,
    clip_shape_factor: bool = True,
) -> np.ndarray:
    if viscosity <= 0:
        raise ValueError("viscosity must be positive")
    a = np.asarray(area, dtype=float)
    gsf = np.asarray(shape_factor, dtype=float)
    L = np.asarray(length, dtype=float)
    if np.any(a < 0):
        raise ValueError("area contains negative values")
    if np.any(L < 0):
        raise ValueError("length contains negative values")
    if np.any(gsf < 0):
        raise ValueError("shape_factor contains negative values")
    if clip_shape_factor:
        # protect against extraction noise and outliers without silently exploding conductance
        gsf = np.clip(gsf, 1e-12, 1.0)
    out = np.full_like(a, np.inf, dtype=float)
    nz = L > 0
    out[nz] = (_CIRCLE_COEFF_AG2 * gsf[nz] * a[nz] ** 2) / (viscosity * L[nz])
    return out


def generic_poiseuille_conductance(net: Network, viscosity: float) -> np.ndarray:
    if viscosity <= 0:
        raise ValueError("viscosity must be positive")
    if "hydraulic_conductance" in net.throat:
        g = np.asarray(net.throat["hydraulic_conductance"], dtype=float)
        if (g < 0).any():
            raise ValueError("throat.hydraulic_conductance contains negative values")
        return g.copy()

    _require(net, "throat", ("length",))
    L = np.asarray(net.throat["length"], dtype=float)
    if "diameter_inscribed" in net.throat:
        d = np.asarray(net.throat["diameter_inscribed"], dtype=float)
    elif "area" in net.throat:
        d = _diameter_from_area(np.asarray(net.throat["area"], dtype=float))
    else:
        raise KeyError(
            "Need throat.diameter_inscribed or throat.area (or precomputed hydraulic_conductance)"
        )
    r = 0.5 * d
    return (np.pi * r**4) / (8.0 * viscosity * L)


def _conduit_lengths_available(net: Network) -> bool:
    keys = ("pore1_length", "core_length", "pore2_length")
    return all(k in net.throat for k in keys)


def _harmonic_combine_segments(*segments: np.ndarray) -> np.ndarray:
    recip = np.zeros_like(np.asarray(segments[0], dtype=float))
    for s in segments:
        arr = np.asarray(s, dtype=float)
        positive = arr > 0
        recip[positive] += 1.0 / arr[positive]
    out = np.zeros_like(recip)
    positive = recip > 0
    out[positive] = 1.0 / recip[positive]
    return out


def _throat_only_shape_factor_conductance(net: Network, viscosity: float) -> np.ndarray:
    _require(net, "throat", ("length",))
    L = np.asarray(net.throat["length"], dtype=float)
    A = _get_entity_area(net, "throat")
    G = _get_entity_shape_factor(net, "throat", area=A)
    return _segment_conductance_from_agl(A, G, L, viscosity)


def _valvatne_conduit_baseline(net: Network, viscosity: float) -> np.ndarray:
    if not _conduit_lengths_available(net):
        raise KeyError("Missing conduit lengths (pore1_length, core_length, pore2_length)")

    # Core throat segment
    At = _get_entity_area(net, "throat")
    Gt = _get_entity_shape_factor(net, "throat", area=At)
    Lt = np.asarray(net.throat["core_length"], dtype=float)
    gt = _segment_conductance_from_agl(At, Gt, Lt, viscosity)

    # End pore segments use connected pore geometry and per-throat conduit lengths
    conns = net.throat_conns
    p1_idx = conns[:, 0]
    p2_idx = conns[:, 1]
    Ap = _get_entity_area(net, "pore")
    Gp = _get_entity_shape_factor(net, "pore", area=Ap)
    g1 = _segment_conductance_from_agl(
        Ap[p1_idx], Gp[p1_idx], np.asarray(net.throat["pore1_length"], dtype=float), viscosity
    )
    g2 = _segment_conductance_from_agl(
        Ap[p2_idx], Gp[p2_idx], np.asarray(net.throat["pore2_length"], dtype=float), viscosity
    )
    return _harmonic_combine_segments(g1, gt, g2)


def valvatne_blunt_baseline_conductance(net: Network, viscosity: float) -> np.ndarray:
    """Return a shape-factor-informed conduit conductance baseline.

    Notes
    -----
    This is a *single-phase baseline* inspired by conduit decomposition used in
    PNM formulations. It is not a full reproduction of the Valvatne & Blunt
    implementation (e.g., no wetting layers, no multiphase occupancy updates).
    When conduit lengths and pore/throat shape data are available, it computes a
    harmonic pore1-core-pore2 conductance using an A^2 G / (mu L) scaling that
    recovers Hagen–Poiseuille for circular sections. Otherwise it falls back to a
    throat-only shape-factor conductance, and finally to generic Poiseuille.
    """

    if viscosity <= 0:
        raise ValueError("viscosity must be positive")

    # If user provided a trusted hydraulic conductance explicitly, honor it.
    if "hydraulic_conductance" in net.throat:
        return generic_poiseuille_conductance(net, viscosity)

    try:
        return _valvatne_conduit_baseline(net, viscosity)
    except KeyError:
        pass

    try:
        return _throat_only_shape_factor_conductance(net, viscosity)
    except KeyError:
        warnings.warn(
            "Insufficient geometry for shape-factor model; falling back to generic_poiseuille",
            RuntimeWarning,
            stacklevel=2,
        )
        return generic_poiseuille_conductance(net, viscosity)


def available_conductance_models() -> tuple[str, ...]:
    return ("generic_poiseuille", "valvatne_blunt_baseline")


def throat_conductance(
    net: Network, viscosity: float, model: str = "generic_poiseuille"
) -> np.ndarray:
    if model == "generic_poiseuille":
        return generic_poiseuille_conductance(net, viscosity)
    if model == "valvatne_blunt_baseline":
        return valvatne_blunt_baseline_conductance(net, viscosity)
    raise ValueError(f"Unknown conductance model '{model}'")
