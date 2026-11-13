"""Generated illustrative fixtures, never publisher data or accuracy evidence."""

from __future__ import annotations
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import yaml

from .schema import write_json
from .targets import encode, derived, normalize_image
from .manifest import write_manifest


def check_example(size=64):
    features = [
        {
            "class": "wall",
            "outer": [[3, 3], [size - 3, 3], [size - 3, size - 3], [3, size - 3]],
            "holes": [[[7, 7], [size - 7, 7], [size - 7, size - 7], [7, size - 7]]],
        },
        {
            "class": "room",
            "room_type": "kitchen",
            "outer": [[9, 9], [size - 9, 9], [size - 9, size - 9], [9, size - 9]],
        },
        {"class": "door", "outer": [[25, 2], [35, 2], [35, 8], [25, 8]]},
    ]
    targets = encode((size, size), features, ["wall", "room", "door", "window", "stairs"])
    centers, valid, boundary = derived(targets["instances"], targets["valid"])
    return {
        "image": normalize_image(np.full((size, size, 3), 255, np.uint8)),
        "masks": targets["masks"].astype(np.float32),
        "valid": targets["valid"].astype(np.float32),
        "types": targets["types"].astype(np.int64),
        "centers": centers,
        "center_valid": valid,
        "boundary": boundary,
        "boundary_valid": targets["valid"][1:2].astype(np.float32),
        "boxes": np.zeros((12, size, size), np.float32),
        "box_valid": np.zeros((3, size, size), np.float32),
        "weak_types": np.empty((0, 5), np.float32),
    }


def check_corpus(output: Path):
    from .schema import sha256

    output.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, split in enumerate(("train", "train", "val", "test")):
        size = 64
        image = np.full((size, size, 3), 255, np.uint8)
        image[3:7, 3:61] = 20 + idx
        image[7:61, 3:7] = 20 + idx
        image[58:62, 3:61] = 20 + idx
        image[3:61, 58:62] = 20 + idx
        image[2:8, 25:35] = 80 + idx
        image_path = output / f"fixture-{idx}.png"
        labels_path = output / f"fixture-{idx}.npz"
        Image.fromarray(image).save(image_path)
        data = encode(
            (size, size),
            [
                {
                    "class": "wall",
                    "outer": [[3, 3], [61, 3], [61, 61], [3, 61]],
                    "holes": [[[7, 7], [57, 7], [57, 57], [7, 57]]],
                },
                {
                    "class": "room",
                    "room_type": "kitchen",
                    "outer": [[9, 9], [55 - idx, 9], [55 - idx, 55], [9, 55]],
                },
                {"class": "door", "outer": [[25, 2], [35, 2], [35, 8], [25, 8]]},
            ],
            ["wall", "room", "door", "window", "stairs"],
        )
        np.savez_compressed(labels_path, **data)
        records.append(
            {
                "id": f"fixture:{idx}",
                "source": "fixture",
                "group_id": f"fixture:{idx}",
                "family": "synthetic",
                "split": split,
                "publisher_split": split,
                "image": image_path.name,
                "targets": labels_path.name,
                "image_sha256": sha256(image_path),
                "simulated": True,
            }
        )
    write_manifest(output / "manifest.jsonl", records)
    return output / "manifest.jsonl"


def smoke(output: Path | None = None, device="cpu", with_export=True):
    from .train import train
    from .evaluate import evaluate
    from .model import load_detector

    temporary = tempfile.TemporaryDirectory(prefix="planproof-detector-check-") if output is None else None
    root = Path(temporary.name) if temporary else output
    root.mkdir(parents=True, exist_ok=True)
    try:
        manifest = check_corpus(root / "fixtures")
        cfg = {
            "name": "generated-fixture-check",
            "seed": 42,
            "image_size": 64,
            "effective_batch": 2,
            "micro_batch": 2,
            "encoder_weights": None,
            "validation_every": 1,
            "checkpoint_every": 1,
            "validation_limit_per_source": 2,
            "allow_simulated": True,
            "phases": [{"name": "check", "steps": 3, "sources": {"fixture": 1.0}}],
        }
        config = root / "check.yaml"
        config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        first = train(config, manifest, root / "run", device=device, max_steps=1)
        resumed = train(
            config, manifest, root / "run", device=device, resume=root / "run/last.pt", max_steps=2
        )
        model, data = load_detector(root / "run/last.pt", device)
        report = evaluate(model, manifest, "test", device, 64, output=root / "evaluation", overlays=1)
        result = {
            "generated_fixture_only": True,
            "real_data_accuracy_claim": False,
            "device": device,
            "train_steps": first["steps"],
            "resumed_steps": resumed["steps"],
            "evaluation_completed": report["sources"]["fixture"]["samples"] == 1,
            "onnx_export_checked": False,
        }
        if with_export:
            from .export import export

            export(root / "run/last.pt", root / "export", 64)
            result["onnx_export_checked"] = True
        write_json(root / "readiness.json", result)
        return result
    finally:
        if temporary:
            temporary.cleanup()
