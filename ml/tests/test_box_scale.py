import pytest
import torch

from planproof_ml.losses import logarithmic_box_size


def test_log_size_loss_and_logit_gradient_are_scale_invariant():
    valid = torch.ones(1, 1, 1, 1)
    gradients = []
    losses = []
    for factor in (0.01, 0.1):
        values = torch.tensor([factor, factor * 2, 0.0, 0.0]).reshape(1, 1, 4, 1, 1).requires_grad_()
        target = values.detach() * 5
        loss = logarithmic_box_size(values, target, valid)
        gradient = torch.autograd.grad(loss, values)[0]
        gradients.append(gradient * values.detach())
        losses.append(loss.detach())
    torch.testing.assert_close(losses[0], losses[1])
    torch.testing.assert_close(gradients[0], gradients[1])
    assert losses[0] == pytest.approx(torch.log(torch.tensor(5.0)).item() - 0.5)


def test_identical_sizes_ignore_offsets_and_missing_coverage():
    values = torch.tensor([0.1, 0.2, 0.1, -0.1]).reshape(1, 1, 4, 1, 1).requires_grad_()
    target = values.detach().clone()
    target[:, :, 2:] *= -1
    assert logarithmic_box_size(values, target, torch.ones(1, 1, 1, 1)) == 0
    loss = logarithmic_box_size(values, target, torch.zeros(1, 1, 1, 1), True)
    loss.backward()
    assert loss == 0 and values.grad.abs().sum() == 0
