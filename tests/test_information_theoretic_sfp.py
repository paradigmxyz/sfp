import torch
from unittest.mock import MagicMock
from methods import METHODS


def test_information_theoretic_sfp_registered():
    """Our method must be in the METHODS registry."""
    assert "information_theoretic_sfp" in METHODS


def test_information_theoretic_sfp_has_setup():
    """Our loss function must declare SETUP attribute."""
    loss_fn = METHODS["information_theoretic_sfp"]
    assert hasattr(loss_fn, "SETUP")
    assert loss_fn.SETUP == "information_theoretic_sfp"


def test_information_theoretic_sfp_loss_returns_scalar():
    """Loss function must return a scalar tensor."""
    loss_fn = METHODS["information_theoretic_sfp"]

    # Create minimal mock inputs
    model = MagicMock()
    batch = {
        "input_ids": torch.randint(0, 100, (2, 32)),
        "attention_mask": torch.ones(2, 32, dtype=torch.long),
        "labels": torch.randint(0, 100, (2, 32)),
    }

    # Mock model forward to return a loss
    mock_output = MagicMock()
    mock_output.loss = torch.tensor(1.0, requires_grad=True)
    model.return_value = mock_output

    # Provide empty basis/anchor (no preservation when missing)
    result = loss_fn(model, batch, basis={}, anchor_acts={}, lam=0.1)

    assert isinstance(result, torch.Tensor)
    assert result.dim() == 0, "Loss must be scalar"
