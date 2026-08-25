import pytest

torch = pytest.importorskip("torch")

from dribblebot.world_model.losses import reward_prediction_loss


def test_reward_mean_gradient_cannot_be_hidden_by_large_variance():
    target = torch.tensor([3.0])
    small_variance_mean = torch.tensor([0.0], requires_grad=True)
    large_variance_mean = torch.tensor([0.0], requires_grad=True)

    small_loss, _, _ = reward_prediction_loss(
        small_variance_mean, torch.tensor([-10.0]), target
    )
    large_loss, _, _ = reward_prediction_loss(
        large_variance_mean, torch.tensor([10.0]), target
    )
    small_loss.backward()
    large_loss.backward()

    assert small_variance_mean.grad.item() == pytest.approx(
        large_variance_mean.grad.item()
    )
    assert small_variance_mean.grad.item() == pytest.approx(-6.0)


def test_reward_variance_calibration_does_not_change_mean_gradient():
    mean = torch.tensor([1.0], requires_grad=True)
    log_variance = torch.tensor([0.0], requires_grad=True)
    loss, mse, calibration = reward_prediction_loss(
        mean, log_variance, torch.tensor([3.0]), variance_weight=1.0
    )
    loss.backward()

    assert mse.item() == pytest.approx(4.0)
    assert calibration.item() > 0.0
    assert mean.grad.item() == pytest.approx(-4.0)
    assert log_variance.grad is not None
