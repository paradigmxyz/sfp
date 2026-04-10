import torch
import pytest
from features import estimate_forgetting_constant, davis_kahan_buffer_size


def test_estimate_C_known_slope():
    """When forgetting = 2 * drift + noise, C should be close to 2."""
    torch.manual_seed(42)
    n = 500
    drift = torch.rand(n)  # D_ℓ values
    noise = torch.randn(n) * 0.1
    forgetting = 2.0 * drift + noise  # C = 2.0

    C_hat, std_err = estimate_forgetting_constant(drift, forgetting)

    assert abs(C_hat - 2.0) < 0.1, f"Expected C ≈ 2.0, got {C_hat:.3f}"
    assert std_err > 0, "Standard error must be positive"
    assert std_err < 0.5, f"Standard error too large: {std_err:.3f}"


def test_estimate_C_zero_drift():
    """When drift is constant (zero variance), should return inf or raise."""
    drift = torch.ones(100)
    forgetting = torch.randn(100)

    C_hat, std_err = estimate_forgetting_constant(drift, forgetting)

    assert C_hat == float("inf") or torch.isnan(torch.tensor(C_hat))


def test_estimate_C_negative_slope():
    """Negative C means more drift = less forgetting (good layer)."""
    torch.manual_seed(42)
    n = 500
    drift = torch.rand(n)
    forgetting = -1.5 * drift + torch.randn(n) * 0.1

    C_hat, _ = estimate_forgetting_constant(drift, forgetting)

    assert C_hat < 0, f"Expected negative C, got {C_hat:.3f}"


def test_davis_kahan_basic():
    """Buffer size increases when eigenvalue gap shrinks."""
    n_large_gap = davis_kahan_buffer_size(delta_k=1.0, sigma=1.0, confidence=0.95, target_sin_theta=0.1)
    n_small_gap = davis_kahan_buffer_size(delta_k=0.1, sigma=1.0, confidence=0.95, target_sin_theta=0.1)

    assert n_small_gap > n_large_gap, "Smaller gap should require more samples"
    assert n_large_gap >= 1, "Must require at least 1 sample"


def test_davis_kahan_clamp():
    """Buffer size is clamped to [128, 1024]."""
    n = davis_kahan_buffer_size(delta_k=100.0, sigma=1.0, confidence=0.95, target_sin_theta=0.1)
    assert n >= 128, f"Should be clamped to at least 128, got {n}"

    n = davis_kahan_buffer_size(delta_k=0.001, sigma=1.0, confidence=0.95, target_sin_theta=0.1)
    assert n <= 1024, f"Should be clamped to at most 1024, got {n}"
