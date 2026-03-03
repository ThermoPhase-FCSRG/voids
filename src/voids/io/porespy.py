from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..core.network import Network
from ..core.provenance import Provenance
from ..core.sample import SampleGeometry
from ..core.validation import validate_network

_PORESPY_KEYMAP = {
    "throat.conns": ("throat", None, "conns"),
    "pore.coords": ("pore", None, "coords"),
    "pore.volume": ("pore", "volume", None),
    "throat.volume": ("throat", "volume", None),
    "throat.length": ("throat", "length", None),
    "throat.area": ("throat", "area", None),
    "pore.area": ("pore", "area", None),
    "throat.shape_factor": ("throat", "shape_factor", None),
    "pore.shape_factor": ("pore", "shape_factor", None),
    "throat.diameter": ("throat", "diameter_inscribed", None),
    "pore.diameter": ("pore", "diameter_inscribed", None),
    "throat.diameter_inscribed": ("throat", "diameter_inscribed", None),
    "pore.diameter_inscribed": ("pore", "diameter_inscribed", None),
    "throat.hydraulic_conductance": ("throat", "hydraulic_conductance", None),
}


def from_porespy(
    network_dict: Mapping[str, object],
    *,
    sample: SampleGeometry | None = None,
    provenance: Provenance | None = None,
    strict: bool = True,
) -> Network:
    if "throat.conns" not in network_dict or "pore.coords" not in network_dict:
        if strict:
            raise KeyError("PoreSpy/OpenPNM-style dict must include 'throat.conns' and 'pore.coords'")

    pore_data: dict[str, np.ndarray] = {}
    throat_data: dict[str, np.ndarray] = {}
    pore_labels: dict[str, np.ndarray] = {}
    throat_labels: dict[str, np.ndarray] = {}
    extra: dict[str, object] = {}

    for key, value in network_dict.items():
        arr = np.asarray(value)
        if key in _PORESPY_KEYMAP:
            family, canonical, reserved = _PORESPY_KEYMAP[key]
            if reserved == "conns":
                throat_conns = arr
                continue
            if reserved == "coords":
                pore_coords = arr
                continue
            if family == "pore":
                pore_data[canonical] = arr
            else:
                throat_data[canonical] = arr
            continue

        if key.startswith("pore."):
            sub = key[5:]
            if arr.dtype == bool:
                pore_labels[sub] = arr
            else:
                pore_data[sub.replace('.', '_')] = arr
        elif key.startswith("throat."):
            sub = key[7:]
            if arr.dtype == bool:
                throat_labels[sub] = arr
            else:
                throat_data[sub.replace('.', '_')] = arr
        else:
            extra[key] = value

    # Required variables; if strict=False and absent, fall back to sensible empty arrays and let validation fail later if used.
    throat_conns = np.asarray(network_dict.get("throat.conns"))
    pore_coords = np.asarray(network_dict.get("pore.coords"), dtype=float)
    if pore_coords.ndim == 2 and pore_coords.shape[1] == 2:
        pore_coords = np.column_stack([pore_coords, np.zeros(pore_coords.shape[0])])

    # Auto-generate axis labels if standard min/max labels are available in source dict form
    for alias, canonical in [
        ("left", "inlet_xmin"),
        ("right", "outlet_xmax"),
        ("front", "inlet_ymin"),
        ("back", "outlet_ymax"),
        ("bottom", "inlet_zmin"),
        ("top", "outlet_zmax"),
    ]:
        if alias in pore_labels and canonical not in pore_labels:
            pore_labels[canonical] = pore_labels[alias]

    net = Network(
        throat_conns=throat_conns,
        pore_coords=pore_coords,
        sample=sample or SampleGeometry(),
        provenance=provenance or Provenance(source_kind="porespy"),
        pore=pore_data,
        throat=throat_data,
        pore_labels=pore_labels,
        throat_labels=throat_labels,
        extra=extra,
    )
    validate_network(net)
    return net
