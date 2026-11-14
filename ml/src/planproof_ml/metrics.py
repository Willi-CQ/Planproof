"""Detection metrics; absent ground-truth classes are reported as null."""

from __future__ import annotations
from collections import defaultdict
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from .inference import box_iou
from .schema import ROOM_TYPES
from .spatial import polygon_mask


def assignment(iou, threshold=0.5):
    if not iou.size:
        return []
    costs = np.where(iou >= threshold, 1 - iou, 100.0)
    rows, cols = linear_sum_assignment(costs)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if iou[r, c] >= threshold]


def average_precision(flags, scores, total):
    if total == 0:
        return None
    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="stable")
    tp = np.cumsum(np.asarray(flags)[order])
    fp = np.cumsum(1 - np.asarray(flags)[order])
    recall = tp / total
    precision = tp / np.maximum(tp + fp, 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(
        np.mean([precision[recall >= r].max() if np.any(recall >= r) else 0 for r in np.linspace(0, 1, 101)])
    )


def item_mask(item, shape):
    out = np.zeros(shape, np.uint8)
    for polygon in item.get("polygons", []):
        out |= polygon_mask(shape, polygon["outer"], polygon.get("holes", []))
    return out.astype(bool)


def item_roi_mask(item, shape):
    polygons = item.get("polygons", [])
    if not polygons:
        return (slice(0, 0), slice(0, 0)), np.zeros((0, 0), bool)
    points = np.rint(np.vstack([p["outer"] for p in polygons])).astype(np.int64)
    low = np.maximum(points.min(0), 0)
    high = np.minimum(points.max(0) + 1, [shape[1], shape[0]])
    if np.any(high <= low):
        return (slice(0, 0), slice(0, 0)), np.zeros((0, 0), bool)
    bounds = (slice(int(low[1]), int(high[1])), slice(int(low[0]), int(high[0])))
    mask = np.zeros((high[1] - low[1], high[0] - low[0]), np.uint8)
    for polygon in polygons:
        outer = np.rint(np.asarray(polygon["outer"])).astype(np.int64) - low
        holes = [np.rint(np.asarray(r)).astype(np.int64) - low for r in polygon.get("holes", [])]
        mask |= polygon_mask(mask.shape, outer, holes)
    return bounds, mask.astype(bool)


class SourceMetrics:
    def __init__(self):
        self.samples = 0
        self.wall_intersection = 0
        self.wall_union = 0
        self.wall_pixels = 0
        self.wall_boundary_counts = np.zeros(4, np.int64)
        self.raw_wall_counts = np.zeros(2, np.int64)
        self.raw_wall_evaluated = False
        self.raw_room_counts = np.zeros(2, np.int64)
        self.raw_room_evaluated = False
        self.room_tp = self.room_fp = self.room_fn = 0
        self.room_evaluated = 0
        self.type_confusion = np.zeros((len(ROOM_TYPES), len(ROOM_TYPES) + 1), np.int64)
        self.types_matched = 0
        self.ap = defaultdict(lambda: {"flags": [], "scores": [], "total": 0, "covered": 0})

    def add(self, result, targets, raw_maps=None):
        self.samples += 1
        masks, valid = targets["masks"], targets["valid"].astype(bool)
        h, w = masks.shape[1:]
        if valid[0].any():
            prediction = np.zeros((h, w), bool)
            for item in result["instances"]:
                if item["class"] == "wall":
                    bounds, region = item_roi_mask(item, (h, w))
                    prediction[bounds] |= region
            truth = masks[0].astype(bool)
            self.wall_intersection += int(np.sum(prediction & truth & valid[0]))
            self.wall_union += int(np.sum((prediction | truth) & valid[0]))
            self.wall_pixels += int(valid[0].sum())
            pred_edge = (prediction ^ ndimage.binary_erosion(prediction)) & valid[0]
            true_edge = (truth ^ ndimage.binary_erosion(truth)) & valid[0]
            tolerance = max(1, round(0.002 * max(h, w)))
            pred_match = pred_edge & ndimage.binary_dilation(true_edge, iterations=tolerance)
            true_match = true_edge & ndimage.binary_dilation(pred_edge, iterations=tolerance)
            self.wall_boundary_counts += np.array(
                [pred_match.sum(), pred_edge.sum(), true_match.sum(), true_edge.sum()]
            )
            if raw_maps is not None:
                self.raw_wall_evaluated = True
                import cv2

                raw = cv2.resize(
                    raw_maps["mask_probabilities"][0], (w, h), interpolation=cv2.INTER_LINEAR
                ) >= raw_maps.get("wall_threshold", 0.5)
                self.raw_wall_counts += np.array(
                    [np.sum(raw & truth & valid[0]), np.sum((raw | truth) & valid[0])]
                )
        if valid[1].any():
            if raw_maps is not None:
                import cv2

                raw_room = cv2.resize(
                    raw_maps["mask_probabilities"][1], (w, h), interpolation=cv2.INTER_LINEAR
                ) >= raw_maps.get("room_threshold", 0.5)
                room_truth = masks[1].astype(bool)
                self.raw_room_counts += np.array(
                    [np.sum(raw_room & room_truth & valid[1]), np.sum((raw_room | room_truth) & valid[1])]
                )
                self.raw_room_evaluated = True
            self.room_evaluated += 1
            ids = [i for i in np.unique(targets["instances"][0]) if i > 0]
            truth = [(targets["instances"][0] == i) & valid[1] for i in ids]
            predictions = [p for p in result["instances"] if p["class"] == "room"]
            pred_masks = []
            for prediction in predictions:
                bounds, region = item_roi_mask(prediction, (h, w))
                pred_masks.append((bounds, region & valid[1][bounds]))
            overlaps = np.zeros((len(pred_masks), len(truth)))
            truth_areas = [int(t.sum()) for t in truth]
            for i, (bounds, p) in enumerate(pred_masks):
                area = int(p.sum())
                for j, t in enumerate(truth):
                    intersection = int(np.sum(p & t[bounds]))
                    union = area + truth_areas[j] - intersection
                    overlaps[i, j] = intersection / union if union else 0
            matches = assignment(overlaps)
            self.room_tp += len(matches)
            self.room_fp += len(pred_masks) - len(matches)
            self.room_fn += len(truth) - len(matches)
            for pi, gi in matches:
                values = targets["types"][truth[gi]]
                values = values[values != 255]
                name = predictions[pi].get("room_type", "unknown")
                if len(values):
                    category = int(np.bincount(values, minlength=len(ROOM_TYPES)).argmax())
                    predicted_type = ROOM_TYPES.index(name) if name in ROOM_TYPES else len(ROOM_TYPES)
                    self.type_confusion[category, predicted_type] += 1
                    self.types_matched += 1
        for cls, name in enumerate(("door", "window", "stairs")):
            if not targets["object_complete"][cls]:
                continue
            gt = [row[1:].tolist() for row in targets["objects"] if int(row[0]) == cls]
            pred = sorted(
                [p for p in result["instances"] if p["class"] == name], key=lambda p: -p["confidence"]
            )
            if "pixel_valid" in targets:
                pred = [
                    p
                    for p in pred
                    if targets["pixel_valid"][
                        min(h - 1, int((p["bbox"][1] + p["bbox"][3]) / 2)),
                        min(w - 1, int((p["bbox"][0] + p["bbox"][2]) / 2)),
                    ]
                    or any(
                        g[0] <= (p["bbox"][0] + p["bbox"][2]) / 2 < g[2]
                        and g[1] <= (p["bbox"][1] + p["bbox"][3]) / 2 < g[3]
                        for g in gt
                    )
                ]
            for threshold in np.arange(0.5, 0.96, 0.05):
                entry = self.ap[(name, round(float(threshold), 2))]
                entry["total"] += len(gt)
                entry["covered"] += 1
                used = set()
                for item in pred:
                    options = [(box_iou(item["bbox"], g), idx) for idx, g in enumerate(gt) if idx not in used]
                    best = max(options, default=(0.0, -1))
                    matched = best[0] >= threshold
                    if matched:
                        used.add(best[1])
                    entry["flags"].append(int(matched))
                    entry["scores"].append(item["confidence"])

    def report(self):
        classes = {}
        for name in ("door", "window", "stairs"):
            results = {}
            for threshold in np.arange(0.5, 0.96, 0.05):
                e = self.ap.get((name, round(float(threshold), 2)))
                results[round(float(threshold), 2)] = (
                    average_precision(e["flags"], e["scores"], e["total"]) if e else None
                )
            known = [v for v in results.values() if v is not None]
            totals = self.ap.get((name, 0.5), {"total": 0, "covered": 0})
            classes[name] = {
                "ap50": results[0.5],
                "ap50_95": float(np.mean(known)) if known else None,
                "ground_truth_instances": totals["total"],
                "annotated_pages": totals["covered"],
            }
        ap50 = [x["ap50"] for x in classes.values() if x["ap50"] is not None]
        apall = [x["ap50_95"] for x in classes.values() if x["ap50_95"] is not None]
        wall = (
            self.wall_intersection / self.wall_union
            if self.wall_union
            else (1.0 if self.wall_pixels else None)
        )
        denominator = 2 * self.room_tp + self.room_fp + self.room_fn
        rooms = 2 * self.room_tp / denominator if denominator else (1.0 if self.room_evaluated else None)
        matched_pred, pred_count, matched_true, true_count = self.wall_boundary_counts
        precision = matched_pred / pred_count if pred_count else 0.0
        recall = matched_true / true_count if true_count else 0.0
        boundary_f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else ((1.0 if not pred_count and not true_count else 0.0) if self.wall_pixels else None)
        )
        conf = self.type_confusion
        type_f1 = []
        for idx in range(len(ROOM_TYPES)):
            total = conf[idx].sum() + conf[:, idx].sum()
            if conf[idx].sum() > 0:
                type_f1.append(2 * conf[idx, idx] / total)
        return {
            "samples": self.samples,
            "wall_iou": wall,
            "wall_boundary_f1_tolerance_0002": boundary_f1,
            "raw_wall_iou": float(self.raw_wall_counts[0] / self.raw_wall_counts[1])
            if self.raw_wall_counts[1]
            else (1.0 if self.raw_wall_evaluated else None),
            "raw_room_iou": float(self.raw_room_counts[0] / self.raw_room_counts[1])
            if self.raw_room_counts[1]
            else (1.0 if self.raw_room_evaluated else None),
            "room_instance_f1_iou50": rooms,
            "room_tp_fp_fn": [self.room_tp, self.room_fp, self.room_fn],
            "objects": classes,
            "object_map50": float(np.mean(ap50)) if ap50 else None,
            "object_map50_95": float(np.mean(apall)) if apall else None,
            "room_type_macro_f1_matched": float(np.mean(type_f1)) if type_f1 else None,
            "room_type_matched_count": self.types_matched,
            "room_type_unknown_count": int(conf[:, -1].sum()),
        }


class Evaluator:
    def __init__(self):
        self.sources = defaultdict(SourceMetrics)

    def add(self, source, prediction, targets, raw_maps=None):
        self.sources[source].add(prediction, targets, raw_maps)

    def report(self):
        reports = {k: v.report() for k, v in sorted(self.sources.items())}

        def macro(key):
            values = [r[key] for r in reports.values() if r[key] is not None]
            return float(np.mean(values)) if values else None

        wall, objects, rooms = macro("wall_iou"), macro("object_map50"), macro("room_instance_f1_iou50")
        score = (
            0.4 * wall + 0.3 * objects + 0.3 * rooms
            if all(v is not None for v in (wall, objects, rooms))
            else None
        )
        gates = {
            "all_object_classes_supported": all(
                any(r["objects"][name]["ground_truth_instances"] > 0 for r in reports.values())
                for name in ("door", "window", "stairs")
            ),
            "wall_iou_080": wall is not None and wall >= 0.8,
            "room_f1_080": rooms is not None and rooms >= 0.8,
            "object_map50_070": objects is not None and objects >= 0.7,
        }
        return {
            "sources": reports,
            "macro": {"wall_iou": wall, "object_map50": objects, "room_instance_f1_iou50": rooms},
            "selection_score": score,
            "acceptance_gates": gates,
            "meets_numeric_targets": all(gates.values()),
            "notes": "Project pixel/mask and 101-point box-AP protocol; not a claim of COCO or publisher benchmark reproduction. Type F1 is conditional on matched rooms.",
        }
