from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

from .adapters import load_image
from .inference import overlay
from .manifest import load_manifest
from .schema import MASK_CLASSES, write_json
from .spatial import mask_polygons
from .storage import ensure_artifact_directory


def inspect_data(manifest: Path, output: Path, per_source=8):
    if per_source < 1:
        raise ValueError("per_source must be positive")
    ensure_artifact_directory(output)
    records = load_manifest(manifest, verify_files=True)
    counts = {}
    reports = []
    output.mkdir(parents=True, exist_ok=True)
    for record in sorted(records, key=lambda r: r["id"]):
        source = record["source"]
        if counts.get(source, 0) >= per_source:
            continue
        image = load_image(manifest.parent / record["image"])
        with np.load(manifest.parent / record["targets"], allow_pickle=False) as z:
            targets = {k: z[k] for k in z.files}
        instances = []
        for channel, name in enumerate(("wall", "room")):
            if name == "wall":
                from scipy import ndimage

                label_image = ndimage.label(targets["masks"][channel])[0]
            else:
                label_image = targets["instances"][0]
            for label in np.unique(label_image):
                if not label:
                    continue
                mask = label_image == label
                yy, xx = np.where(mask)
                instances.append(
                    {
                        "id": f"GT-{name}-{label}",
                        "class": name,
                        "confidence": 1.0,
                        "bbox": [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)],
                        "polygons": mask_polygons(mask),
                    }
                )
        for idx, (label, x1, y1, x2, y2) in enumerate(targets["objects"]):
            name = ("door", "window", "stairs")[int(label)]
            instances.append(
                {
                    "id": f"GT-{name}-{idx}",
                    "class": name,
                    "confidence": 1.0,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                }
            )
        name = record["id"].replace(":", "_")
        Image.fromarray(overlay(image, {"instances": instances})).save(output / (name + ".png"))
        coverage = {
            c: {
                "supervised_pixels": int(targets["valid"][i].sum()),
                "positive_pixels": int(targets["masks"][i].sum()),
            }
            for i, c in enumerate(MASK_CLASSES)
        }
        report = {
            "id": record["id"],
            "source": source,
            "split": record["split"],
            "ground_truth_only": True,
            "coverage": coverage,
            "object_complete": targets["object_complete"].tolist(),
            "object_instances": len(targets["objects"]),
            "weak_category_boxes": len(targets["weak_types"]),
        }
        reports.append(report)
        counts[source] = counts.get(source, 0) + 1
    write_json(
        output / "inspection.json",
        {
            "sources": counts,
            "samples": reports,
            "note": "Adapter inspection only; these are annotations, not model predictions.",
        },
    )
    return {"out": str(output), "rendered": len(reports), "sources": counts}
