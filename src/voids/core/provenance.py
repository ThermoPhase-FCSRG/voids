from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class Provenance:
    source_kind: str = "custom"
    source_version: str | None = None
    extraction_method: str | None = None
    segmentation_notes: str | None = None
    voxel_size_original: float | tuple[float, float, float] | None = None
    image_hash: str | None = None
    preprocessing_hash: str | None = None
    random_seed: int | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    user_notes: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "source_version": self.source_version,
            "extraction_method": self.extraction_method,
            "segmentation_notes": self.segmentation_notes,
            "voxel_size_original": self.voxel_size_original,
            "image_hash": self.image_hash,
            "preprocessing_hash": self.preprocessing_hash,
            "random_seed": self.random_seed,
            "created_at": self.created_at,
            "user_notes": self.user_notes,
        }

    @classmethod
    def from_metadata(cls, data: dict[str, Any]) -> "Provenance":
        return cls(**data)
