from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SampleGeometry:
    voxel_size: float | tuple[float, float, float] | None = None
    bulk_shape_voxels: tuple[int, int, int] | None = None
    bulk_volume: float | None = None
    lengths: dict[str, float] = field(default_factory=dict)
    cross_sections: dict[str, float] = field(default_factory=dict)
    axis_map: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=lambda: {"length": "m", "pressure": "Pa"})

    def resolved_bulk_volume(self) -> float:
        if self.bulk_volume is not None:
            return float(self.bulk_volume)
        if self.bulk_shape_voxels is None or self.voxel_size is None:
            raise ValueError("bulk_volume is unavailable and cannot be derived")
        if isinstance(self.voxel_size, tuple):
            vx, vy, vz = self.voxel_size
        else:
            vx = vy = vz = float(self.voxel_size)
        nx, ny, nz = self.bulk_shape_voxels
        return float(nx * ny * nz * vx * vy * vz)

    def length_for_axis(self, axis: str) -> float:
        if axis not in self.lengths:
            raise KeyError(f"Missing sample length for axis '{axis}'")
        return float(self.lengths[axis])

    def area_for_axis(self, axis: str) -> float:
        if axis not in self.cross_sections:
            raise KeyError(f"Missing sample cross-section for axis '{axis}'")
        return float(self.cross_sections[axis])

    def to_metadata(self) -> dict[str, Any]:
        return {
            "voxel_size": self.voxel_size,
            "bulk_shape_voxels": self.bulk_shape_voxels,
            "bulk_volume": self.bulk_volume,
            "lengths": self.lengths,
            "cross_sections": self.cross_sections,
            "axis_map": self.axis_map,
            "units": self.units,
        }

    @classmethod
    def from_metadata(cls, data: dict[str, Any]) -> "SampleGeometry":
        return cls(
            voxel_size=data.get("voxel_size"),
            bulk_shape_voxels=tuple(data["bulk_shape_voxels"]) if data.get("bulk_shape_voxels") is not None else None,
            bulk_volume=data.get("bulk_volume"),
            lengths={str(k): float(v) for k, v in (data.get("lengths") or {}).items()},
            cross_sections={str(k): float(v) for k, v in (data.get("cross_sections") or {}).items()},
            axis_map={str(k): str(v) for k, v in (data.get("axis_map") or {}).items()},
            units={str(k): str(v) for k, v in (data.get("units") or {}).items()},
        )
