from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple
import json


@dataclass
class PlannedObject:
    """
    Ein geplantes Objekt aus dem Plan (as-designed), als Prior für Matching.
    Koordinaten sind erstmal RELATIV (0..1) in der Plan-Raum-Boundingbox gedacht.
    (Alignment Plan<->Scan kommt später.)
    """
    id: str
    coarse: str
    fine: Optional[str]
    center_rel: Tuple[float, float, float]   # (cx,cy,cz) in [0..1]
    dims_rel: Tuple[float, float, float]     # (dx,dy,dz) in [0..1] grob
    mount: Optional[str] = None              # "wall"|"ceiling"|"floor"|None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class PlanPrior:
    version: str
    source_drawing: str
    source_plan_model: Optional[str]
    objects: List[PlannedObject]
    meta: Optional[Dict[str, Any]] = None


def save_prior(prior: PlanPrior, path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(prior)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(p)


def load_prior(path: str) -> PlanPrior:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    objs = []
    for o in data.get("objects", []):
        objs.append(
            PlannedObject(
                id=str(o.get("id")),
                coarse=str(o.get("coarse")),
                fine=o.get("fine"),
                center_rel=tuple(o.get("center_rel", (0.5, 0.5, 0.5))),
                dims_rel=tuple(o.get("dims_rel", (0.2, 0.2, 0.2))),
                mount=o.get("mount"),
                meta=o.get("meta"),
            )
        )
    return PlanPrior(
        version=str(data.get("version", "prior_v0")),
        source_drawing=str(data.get("source_drawing", "")),
        source_plan_model=data.get("source_plan_model"),
        objects=objs,
        meta=data.get("meta"),
    )
