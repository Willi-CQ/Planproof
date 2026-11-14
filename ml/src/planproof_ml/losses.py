from __future__ import annotations
import torch
from torch.nn import functional as F


def masked_mean(value, valid):
    return (value * valid).sum() / valid.sum().clamp_min(1)


def focal_centers(logits, target, valid):
    p = logits.float().sigmoid().clamp(1e-5, 1 - 1e-5)
    positive = (target == 1).float() * valid
    negative = (target < 1).float() * valid
    loss = -(torch.log(p) * (1 - p) ** 2 * positive + torch.log(1 - p) * p**2 * (1 - target) ** 4 * negative)
    return loss.sum() / positive.sum().clamp_min(1)


def giou_boxes(encoded, target, valid):
    """Paired GIoU at supervised centers, preserving the existing box encoding.

    Width/height are image-normalized; offsets are half-pixel coordinates.
    Common center-grid coordinates cancel, but offsets must use image units.
    """
    selected = valid.bool()
    if not selected.any():
        return encoded.sum() * 0
    prediction = encoded.permute(0, 1, 3, 4, 2)[selected]
    truth = target.permute(0, 1, 3, 4, 2)[selected]
    size = encoded.new_tensor([encoded.shape[-1], encoded.shape[-2]])

    def corners(values):
        center = values[:, 2:] / size
        return center - values[:, :2] / 2, center + values[:, :2] / 2

    plow, phigh = corners(prediction)
    tlow, thigh = corners(truth)
    intersection = (torch.minimum(phigh, thigh) - torch.maximum(plow, tlow)).clamp_min(0).prod(1)
    union = (phigh - plow).prod(1) + (thigh - tlow).prod(1) - intersection
    enclosing = (torch.maximum(phigh, thigh) - torch.minimum(plow, tlow)).clamp_min(0).prod(1)
    eps = torch.finfo(encoded.dtype).eps
    giou = intersection / union.clamp_min(eps) - (enclosing - union) / enclosing.clamp_min(eps)
    return (1 - giou).mean()


def detection_loss(pred, batch, box_loss="smooth_l1"):
    logits = pred["masks"].float()
    gt, valid = batch["masks"], batch["valid"]
    bce = masked_mean(F.binary_cross_entropy_with_logits(logits, gt, reduction="none"), valid)
    p = logits.sigmoid() * valid
    sums = (p + gt * valid).sum((-2, -1))
    dice = 1 - (2 * (p * gt).sum((-2, -1)) + 1) / (sums + 1)
    covered = (valid.sum((-2, -1)) > 0).float()
    mask_loss = bce + (dice * covered).sum() / covered.sum().clamp_min(1)
    boundary = masked_mean(
        F.binary_cross_entropy_with_logits(
            pred["boundary"].float(),
            batch["boundary"],
            reduction="none",
            pos_weight=torch.tensor(5.0, device=logits.device),
        ),
        batch["boundary_valid"],
    )
    centers = focal_centers(pred["centers"], batch["centers"], batch["center_valid"])
    box_parts = pred["boxes"].float().reshape(-1, 3, 4, *logits.shape[-2:])
    encoded = torch.cat((box_parts[:, :, :2].sigmoid(), box_parts[:, :, 2:].tanh() * 0.5), dim=2)
    target = batch["boxes"].reshape_as(encoded)
    box_valid = batch["box_valid"][:, :, None].expand_as(encoded)
    if box_loss == "giou":
        boxes = giou_boxes(encoded, target, batch["box_valid"])
    elif box_loss == "smooth_l1":
        boxes = masked_mean(F.smooth_l1_loss(encoded, target, reduction="none"), box_valid)
    else:
        raise ValueError("box_loss must be smooth_l1 or giou")
    known = batch["types"] != 255
    types = (
        F.cross_entropy(pred["types"].float().permute(0, 2, 3, 1)[known], batch["types"][known])
        if known.any()
        else pred["types"].sum() * 0
    )
    weak = pred["types"].sum() * 0
    weak_count = 0
    for idx, weak_annotations in enumerate(batch["weak_types"]):
        for category, x1, y1, x2, y2 in weak_annotations:
            h, w = pred["types"].shape[-2:]
            xa, ya = max(0, int(x1)), max(0, int(y1))
            xb, yb = min(w, int(x2)), min(h, int(y2))
            if xb > xa and yb > ya:
                pooled = pred["types"][idx, :, ya:yb, xa:xb].mean((-2, -1))[None]
                weak = weak + F.cross_entropy(
                    pooled.float(), torch.tensor([int(category)], device=pooled.device)
                )
                weak_count += 1
    weak = weak / max(1, weak_count)
    total = mask_loss + types + 0.5 * boundary + 0.5 * centers + 2 * boxes + 0.25 * weak
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite detection loss")
    return total, {
        "mask": float(mask_loss.detach()),
        "type": float(types.detach()),
        "boundary": float(boundary.detach()),
        "center": float(centers.detach()),
        "box": float(boxes.detach()),
        "weak_type": float(weak.detach()),
    }
