from __future__ import annotations

from typing import Tuple

from domain.scene_model import SceneModel, Evidence, Relationship
from ai.plan_prior import PlanPrior


def _rel_to_abs(scene: SceneModel, center_rel: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Map prior-relative coordinates [0..1] to scene-local coordinates using scene bounds."""
    b = scene.bounds()
    if not b:
        return (center_rel[0], center_rel[1], center_rel[2])

    (minx, miny, minz), (maxx, maxy, maxz) = b
    cx = minx + center_rel[0] * (maxx - minx)
    cy = miny + center_rel[1] * (maxy - miny)
    cz = minz + center_rel[2] * (maxz - minz)
    return (cx, cy, cz)


def _dist3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) ** 0.5


def apply_plan_prior(scene: SceneModel, prior: PlanPrior) -> SceneModel:
    """Apply plan priors to the SceneModel as evidence + constraints (no hard overwrite)."""

    comps = [c for c in scene.component_list() if c.center is not None]

    for po in prior.objects:
        target_center = _rel_to_abs(scene, po.center_rel)

        best = None
        best_d = 1e18
        for c in comps:
            d = _dist3(target_center, c.center)  # type: ignore[arg-type]
            if d < best_d:
                best_d = d
                best = c

        if best is None:
            continue

        ev = Evidence(
            source="plan_prior",
            method="nearest_center_match",
            confidence=0.70,
            details={
                "planned_object_id": po.id,
                "planned_coarse": po.coarse,
                "planned_fine": po.fine,
                "dist": float(best_d),
            },
        )

        best.attrs.setdefault("priors", [])
        best.attrs["priors"].append(
            {
                "id": po.id,
                "coarse": po.coarse,
                "fine": po.fine,
                "mount": po.mount,
                "confidence": ev.confidence,
            }
        )
        best.add_evidence(ev)

        scene.add_relationship(
            Relationship(
                src_id=po.id,
                rel_type="PRIOR_MATCHED_TO",
                dst_id=best.id,
                confidence=ev.confidence,
                evidence=ev,
            )
        )

    scene.meta.setdefault("priors", [])
    scene.meta["priors"].append(
        {
            "version": prior.version,
            "source_drawing": prior.source_drawing,
            "source_plan_model": prior.source_plan_model,
            "n_objects": len(prior.objects),
        }
    )

    return scene
