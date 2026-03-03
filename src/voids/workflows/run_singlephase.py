from __future__ import annotations

import json

import numpy as np

from ..core.network import Network
from ..core.sample import SampleGeometry
from ..physics.singlephase import FluidSinglePhase, PressureBC, solve


def _toy_network() -> Network:
    pore_coords = np.array([[0,0,0],[0.5,0,0],[1,0,0]], dtype=float)
    throat_conns = np.array([[0,1],[1,2]], dtype=int)
    net = Network(
        throat_conns=throat_conns,
        pore_coords=pore_coords,
        sample=SampleGeometry(bulk_volume=10.0, lengths={"x":1.0}, cross_sections={"x":1.0}),
        pore={"volume": np.array([1.0, 1.0, 1.0])},
        throat={
            "volume": np.array([0.5, 0.5]),
            "hydraulic_conductance": np.array([1.0, 1.0]),
            "length": np.array([1.0, 1.0]),
        },
        pore_labels={
            "inlet_xmin": np.array([True, False, False]),
            "outlet_xmax": np.array([False, False, True]),
        },
    )
    return net


def main() -> None:
    net = _toy_network()
    result = solve(
        net,
        fluid=FluidSinglePhase(viscosity=1.0),
        bc=PressureBC("inlet_xmin", "outlet_xmax", pin=1.0, pout=0.0),
        axis="x",
    )
    summary = {
        "Q": result.total_flow_rate,
        "Kx": result.permeability["x"] if result.permeability else None,
        "residual_norm": result.residual_norm,
        "mass_balance_error": result.mass_balance_error,
        "p": result.pore_pressure.tolist(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
