from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class Evidence:
    """Traceable reason why a value exists."""

    source: str
    method: str
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None


@dataclass
class Component:
    """A single element inside a reconstructed scene."""

    id: str
    name: str
    kind: str  # e.g. WALL, FLOOR, CEILING, OPENING, FURNITURE

    # Abstract reference into exported geometry (e.g. glTF node name).
    geometry_ref: Optional[str] = None

    # Spatial properties (meters, scene-local)
    center: Optional[Vec3] = None
    dims: Optional[Vec3] = None
    bbox_min: Optional[Vec3] = None
    bbox_max: Optional[Vec3] = None

    # Free-form semantic/engineering attributes
    attrs: Dict[str, Any] = field(default_factory=dict)

    # Provenance
    evidence: List[Evidence] = field(default_factory=list)

    def add_evidence(self, ev: Evidence) -> None:
        self.evidence.append(ev)


@dataclass
class Relationship:
    """Typed edge between two components."""

    src_id: str
    rel_type: str  # e.g. PART_OF, HAS_OPENING, ADJACENT_TO
    dst_id: str
    confidence: float = 1.0
    evidence: Optional[Evidence] = None


@dataclass
class SceneModel:
    """Source-agnostic scene graph representation."""

    version: str = "scene_v1"
    units: str = "m"
    up_axis: str = "Y"
    components: Dict[str, Component] = field(default_factory=dict)
    relationships: List[Relationship] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def add_component(self, c: Component) -> None:
        self.components[c.id] = c

    def add_relationship(self, r: Relationship) -> None:
        self.relationships.append(r)

    def component_list(self) -> List[Component]:
        return list(self.components.values())

    def bounds(self) -> Optional[Tuple[Vec3, Vec3]]:
        mins: List[Vec3] = []
        maxs: List[Vec3] = []

        for c in self.components.values():
            if c.bbox_min and c.bbox_max:
                mins.append(c.bbox_min)
                maxs.append(c.bbox_max)
            elif c.center and c.dims:
                cx, cy, cz = c.center
                dx, dy, dz = c.dims
                mins.append((cx - dx / 2.0, cy - dy / 2.0, cz - dz / 2.0))
                maxs.append((cx + dx / 2.0, cy + dy / 2.0, cz + dz / 2.0))

        if not mins:
            return None

        minx = min(p[0] for p in mins)
        miny = min(p[1] for p in mins)
        minz = min(p[2] for p in mins)
        maxx = max(p[0] for p in maxs)
        maxy = max(p[1] for p in maxs)
        maxz = max(p[2] for p in maxs)
        return (minx, miny, minz), (maxx, maxy, maxz)

    def scene_diag(self) -> float:
        b = self.bounds()
        if not b:
            return 1.0
        (minx, miny, minz), (maxx, maxy, maxz) = b
        return math.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2 + (maxz - minz) ** 2)

    def to_dict(self) -> Dict[str, Any]:
        def ev_to_dict(ev: Evidence) -> Dict[str, Any]:
            return {
                "source": ev.source,
                "method": ev.method,
                "confidence": float(ev.confidence),
                "details": ev.details,
                "timestamp": ev.timestamp,
            }

        def comp_to_dict(c: Component) -> Dict[str, Any]:
            return {
                "id": c.id,
                "name": c.name,
                "kind": c.kind,
                "geometry_ref": c.geometry_ref,
                "center": c.center,
                "dims": c.dims,
                "bbox_min": c.bbox_min,
                "bbox_max": c.bbox_max,
                "attrs": c.attrs,
                "evidence": [ev_to_dict(e) for e in c.evidence],
            }

        def rel_to_dict(r: Relationship) -> Dict[str, Any]:
            return {
                "src_id": r.src_id,
                "rel_type": r.rel_type,
                "dst_id": r.dst_id,
                "confidence": float(r.confidence),
                "evidence": ev_to_dict(r.evidence) if r.evidence else None,
            }

        return {
            "version": self.version,
            "units": self.units,
            "up_axis": self.up_axis,
            "components": {k: comp_to_dict(v) for k, v in self.components.items()},
            "relationships": [rel_to_dict(r) for r in self.relationships],
            "meta": self.meta,
        }

    def to_json(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SceneModel":
        scene = SceneModel(
            version=str(d.get("version", "scene_v1")),
            units=str(d.get("units", "m")),
            up_axis=str(d.get("up_axis", "Y")),
            meta=d.get("meta") or {},
        )

        for cid, cd in (d.get("components") or {}).items():
            evs = []
            for e in cd.get("evidence") or []:
                evs.append(
                    Evidence(
                        source=str(e.get("source", "")),
                        method=str(e.get("method", "")),
                        confidence=float(e.get("confidence", 1.0)),
                        details=e.get("details") or {},
                        timestamp=e.get("timestamp"),
                    )
                )

            c = Component(
                id=str(cd.get("id", cid)),
                name=str(cd.get("name", cid)),
                kind=str(cd.get("kind", "UNKNOWN")),
                geometry_ref=cd.get("geometry_ref"),
                center=tuple(cd.get("center")) if cd.get("center") else None,
                dims=tuple(cd.get("dims")) if cd.get("dims") else None,
                bbox_min=tuple(cd.get("bbox_min")) if cd.get("bbox_min") else None,
                bbox_max=tuple(cd.get("bbox_max")) if cd.get("bbox_max") else None,
                attrs=cd.get("attrs") or {},
                evidence=evs,
            )
            scene.add_component(c)

        for rd in d.get("relationships") or []:
            ev = None
            if rd.get("evidence"):
                e = rd["evidence"]
                ev = Evidence(
                    source=str(e.get("source", "")),
                    method=str(e.get("method", "")),
                    confidence=float(e.get("confidence", 1.0)),
                    details=e.get("details") or {},
                    timestamp=e.get("timestamp"),
                )

            scene.add_relationship(
                Relationship(
                    src_id=str(rd.get("src_id")),
                    rel_type=str(rd.get("rel_type")),
                    dst_id=str(rd.get("dst_id")),
                    confidence=float(rd.get("confidence", 1.0)),
                    evidence=ev,
                )
            )

        return scene

    @staticmethod
    def from_json(path: str) -> "SceneModel":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return SceneModel.from_dict(d)
