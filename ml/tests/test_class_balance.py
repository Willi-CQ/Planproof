import torch

from planproof_ml.losses import focal_centers, giou_boxes


def test_rare_center_gradient_is_not_diluted_by_other_class_counts():
    def gradient(common_count, balanced):
        logits = torch.full((1, 4, 2, 3), -2.0, requires_grad=True)
        target = torch.zeros_like(logits)
        target[0, 3, 0, 0] = 1
        target[0, 1].view(-1)[:common_count] = 1
        loss = focal_centers(logits, target, torch.ones_like(logits), balanced)
        return torch.autograd.grad(loss, logits)[0][0, 3, 0, 0]

    torch.testing.assert_close(gradient(1, True), gradient(6, True), atol=0, rtol=0)
    assert gradient(6, False).abs() < gradient(1, False).abs()


def test_rare_box_gradient_is_not_diluted_by_common_class_duplicates():
    def gradient(duplicate, balanced):
        encoded = torch.zeros(1, 3, 4, 1, 2)
        encoded[:, :, :2] = 0.1
        encoded.requires_grad_()
        target = encoded.detach().clone()
        target[:, :, :2] = 0.2
        valid = torch.zeros(1, 3, 1, 2)
        valid[0, 0, 0, 0] = valid[0, 2, 0, 0] = 1
        valid[0, 0, 0, 1] = int(duplicate)
        loss = giou_boxes(encoded, target, valid, balanced)
        return torch.autograd.grad(loss, encoded)[0][0, 2, 0, 0, 0]

    torch.testing.assert_close(gradient(False, True), gradient(True, True), atol=0, rtol=0)
    assert gradient(True, False).abs() < gradient(False, False).abs()


def test_uncovered_balanced_heads_have_finite_zero_loss_and_gradient():
    logits = torch.zeros(2, 4, 8, 8, requires_grad=True)
    loss = focal_centers(logits, torch.zeros_like(logits), torch.zeros_like(logits), True)
    loss.backward()
    assert loss == 0 and logits.grad.abs().sum() == 0
    encoded = torch.zeros(2, 3, 4, 8, 8, requires_grad=True)
    boxes = giou_boxes(encoded, encoded.detach(), torch.zeros(2, 3, 8, 8), True)
    boxes.backward()
    assert boxes == 0 and encoded.grad.abs().sum() == 0
