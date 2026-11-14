import pytest
import torch
from torchvision.ops import generalized_box_iou_loss

from planproof_ml.losses import giou_boxes, detection_loss
from planproof_ml.checks import check_example
from planproof_ml.dataset import tensor_batch
from planproof_ml.model import Detector


def test_paired_giou_matches_reference_with_pixel_offsets_and_small_boxes():
    rng = torch.Generator().manual_seed(3)
    predicted = torch.rand(1, 3, 4, 64, 64, generator=rng)
    target = torch.rand(1, 3, 4, 64, 64, generator=rng)
    predicted[:, :, :2] *= 0.1
    target[:, :, :2] *= 0.1
    predicted[:, :, 2:] -= 0.5
    target[:, :, 2:] -= 0.5
    valid = torch.zeros(1, 3, 64, 64)
    valid[:, :, 12, 30] = 1
    actual = giou_boxes(predicted, target, valid)

    def corners(values):
        v = values.permute(0, 1, 3, 4, 2)[valid.bool()]
        return torch.cat((v[:, 2:] / 64 - v[:, :2] / 2, v[:, 2:] / 64 + v[:, :2] / 2), 1)

    expected = generalized_box_iou_loss(corners(predicted), corners(target), reduction="mean", eps=1e-12)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert giou_boxes(target, target, valid) == pytest.approx(0.0, abs=1e-6)


def test_box_overlap_loss_is_scale_invariant_and_empty_coverage_has_no_gradient():
    target = torch.tensor([0.01, 0.02, 0.0, 0.0]).reshape(1, 1, 4, 1, 1)
    predicted = (target * 2).requires_grad_()
    valid = torch.ones(1, 1, 1, 1)
    small = giou_boxes(predicted, target, valid)
    large = giou_boxes(predicted * 10, target * 10, valid)
    torch.testing.assert_close(small, large)
    small.backward()
    assert predicted.grad[0, 0, 0].abs().sum() > 0
    predicted.grad.zero_()
    empty = giou_boxes(predicted, target, valid * 0)
    empty.backward()
    assert empty == 0 and predicted.grad.abs().sum() == 0


def test_giou_detection_update_is_finite_for_real_encoding():
    example = check_example()
    example["boxes"][:4, 32, 32] = [0.02, 0.05, 0.1, -0.2]
    example["box_valid"][0, 32, 32] = 1
    batch = tensor_batch([example, example], "cpu")
    model = Detector(None)
    loss, parts = detection_loss(model(batch["image"]), batch, "giou")
    loss.backward()
    assert torch.isfinite(loss)
    assert 0 <= parts["box"] <= 2
    assert model.box_head.weight.grad.abs().sum() > 0


def test_box_weight_changes_only_its_task_contribution():
    example = check_example()
    example["boxes"][:4, 32, 32] = [0.02, 0.05, 0.1, -0.2]
    example["box_valid"][0, 32, 32] = 1
    batch = tensor_batch([example, example], "cpu")
    prediction = Detector(None)(batch["image"])
    original, parts = detection_loss(prediction, batch, "giou", 2.0)
    balanced, _ = detection_loss(prediction, batch, "giou", 0.5)
    torch.testing.assert_close(original - balanced, original.new_tensor(1.5 * parts["box"]))


def test_room_foreground_metric_is_separate_from_instance_splitting():
    import numpy as np
    from planproof_ml.targets import encode
    from planproof_ml.metrics import SourceMetrics

    targets = encode((32, 32), [{"class": "room", "outer": [[4, 4], [20, 4], [20, 20], [4, 20]]}], ["room"])
    metrics = SourceMetrics()
    metrics.add({"instances": []}, targets, {"mask_probabilities": targets["masks"].astype(np.float32)})
    report = metrics.report()
    assert report["raw_room_iou"] == 1.0
    assert report["room_instance_f1_iou50"] == 0.0
    assert report["raw_wall_iou"] is None
