from __future__ import annotations

import warnings

import numpy as np

from ..core.network import Network


DEFAULT_G_REF = 1.0 / (4.0 * np.pi)


def _require(net: Network, kind: str, names: tuple[str, ...]) -> None:
    store = net.throat if kind == "throat" else net.pore
    missing = [n for n in names if n not in store]
    if missing:
        raise KeyError(f"Missing required {kind} fields: {missing}")


def _diameter_from_area(area: np.ndarray) -> np.ndarray:
    return 2.0 * np.sqrt(area / np.pi)


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
        raise KeyError("Need throat.diameter_inscribed or throat.area (or precomputed hydraulic_conductance)")
    r = 0.5 * d
    return (np.pi * r**4) / (8.0 * viscosity * L)


def valvatne_blunt_baseline_conductance(net: Network, viscosity: float) -> np.ndarray:
    # Conservative MVP: use shape-factor-informed correction when available, otherwise fallback.
    g = generic_poiseuille_conductance(net, viscosity)
    if "shape_factor" not in net.throat:
        warnings.warn(
            "throat.shape_factor unavailable; falling back to generic_poiseuille",
            RuntimeWarning,
            stacklevel=2,
        )
        return g
    G = np.asarray(net.throat["shape_factor"], dtype=float)
    # Mild correction around a circular-equivalent reference. Clamp to avoid nonphysical blow-up.
    corr = np.clip(np.sqrt(np.maximum(G, 1e-30) / DEFAULT_G_REF), 0.25, 4.0)
    return g * corr


def throat_conductance(net: Network, viscosity: float, model: str = "generic_poiseuille") -> np.ndarray:
    if model == "generic_poiseuille":
        return generic_poiseuille_conductance(net, viscosity)
    if model == "valvatne_blunt_baseline":
        return valvatne_blunt_baseline_conductance(net, viscosity)
    raise ValueError(f"Unknown conductance model '{model}'")
