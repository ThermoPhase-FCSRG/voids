from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from voids.core.provenance import Provenance
from voids.core.sample import SampleGeometry


@dataclass(slots=True)
class Network:
    throat_conns: np.ndarray
    pore_coords: np.ndarray
    sample: SampleGeometry
    provenance: Provenance = field(default_factory=Provenance)
    schema_version: str = "0.1.0"
    pore: dict[str, np.ndarray] = field(default_factory=dict)
    throat: dict[str, np.ndarray] = field(default_factory=dict)
    pore_labels: dict[str, np.ndarray] = field(default_factory=dict)
    throat_labels: dict[str, np.ndarray] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.throat_conns = np.asarray(self.throat_conns, dtype=np.int64)
        self.pore_coords = np.asarray(self.pore_coords, dtype=float)
        for d in (self.pore, self.throat):
            for k, v in list(d.items()):
                d[k] = np.asarray(v)
        for d in (self.pore_labels, self.throat_labels):
            for k, v in list(d.items()):
                d[k] = np.asarray(v, dtype=bool)

    @property
    def Np(self) -> int:
        return int(self.pore_coords.shape[0])

    @property
    def Nt(self) -> int:
        return int(self.throat_conns.shape[0])

    def get_pore_array(self, name: str) -> np.ndarray:
        if name not in self.pore:
            raise KeyError(f"Missing pore field '{name}'")
        return self.pore[name]

    def get_throat_array(self, name: str) -> np.ndarray:
        if name not in self.throat:
            raise KeyError(f"Missing throat field '{name}'")
        return self.throat[name]

    def copy(self) -> "Network":
        return Network(
            throat_conns=self.throat_conns.copy(),
            pore_coords=self.pore_coords.copy(),
            sample=self.sample,
            provenance=self.provenance,
            schema_version=self.schema_version,
            pore={k: v.copy() for k, v in self.pore.items()},
            throat={k: v.copy() for k, v in self.throat.items()},
            pore_labels={k: v.copy() for k, v in self.pore_labels.items()},
            throat_labels={k: v.copy() for k, v in self.throat_labels.items()},
            extra={**self.extra},
        )
