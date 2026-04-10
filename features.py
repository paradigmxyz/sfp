"""
features.py — Activation hooks, PCA/gradient subspace construction, importance ranking.

~250 lines. Uses torch + numpy. No imports from other project files.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def get_layer_names(
    model,
    positions: list[float] | None = None,
) -> list[str]:
    """Return layer module names at given relative depth positions.

    Args:
        model: A transformer model.
        positions: Relative depth positions, e.g. [0.25, 0.5, 0.75].
            Defaults to [0.25, 0.5, 0.75].

    Returns:
        List of module name strings (e.g. "model.layers.8").
    """
    if positions is None:
        positions = [0.25, 0.5, 0.75]

    # Find transformer layers by scanning named_modules for ".layers.0" or ".h.0"
    all_names = dict(model.named_modules())
    prefix = None
    n_layers = 0

    for name in all_names:
        if name.endswith(".layers.0"):
            prefix = name.rsplit(".0", 1)[0]
            n_layers = sum(
                1 for n in all_names
                if n.startswith(prefix + ".") and n[len(prefix)+1:].isdigit()
            )
            break
        if name.endswith(".h.0"):
            prefix = name.rsplit(".0", 1)[0]
            n_layers = sum(
                1 for n in all_names
                if n.startswith(prefix + ".") and n[len(prefix)+1:].isdigit()
            )
            break

    if prefix is None or n_layers == 0:
        raise ValueError(
            "Cannot detect transformer layers in model modules."
        )

    indices = [int(p * n_layers) for p in positions]
    indices = [min(i, n_layers - 1) for i in indices]
    return [f"{prefix}.{i}" for i in indices]


def collect_activations(
    model,
    dataset: list[dict],
    tokenizer,
    layer_names: list[str],
    pool: str = "mean",
    max_samples: int = 512,
    device: str = "auto",
) -> dict[str, Tensor]:
    """Forward pass over dataset, collect activations at specified layers.

    Args:
        pool: "mean" for token-mean pooling, "last" for last token.

    Returns:
        {layer_name: tensor of shape [n_samples, hidden_dim]}
    """
    if device == "auto":
        device = str(next(model.parameters()).device)

    samples = dataset[:max_samples]
    buffers: dict[str, list[Tensor]] = {name: [] for name in layer_names}
    hooks = []

    def _make_hook(name: str):
        def hook_fn(_module, _input, output):
            if isinstance(output, Tensor):
                h = output
            elif isinstance(output, tuple):
                h = output[0]
            elif hasattr(output, "last_hidden_state"):
                h = output.last_hidden_state
            else:
                h = output[0]
            if pool == "last":
                buffers[name].append(h[:, -1, :].detach().cpu())
            else:
                buffers[name].append(h.mean(dim=1).detach().cpu())

        return hook_fn

    # Register hooks
    for name in layer_names:
        module = dict(model.named_modules())[name]
        hooks.append(module.register_forward_hook(_make_hook(name)))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    with torch.no_grad():
        for sample in samples:
            text = sample.get("input", sample.get("text", ""))
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False,
            ).to(device)
            model(**enc)

    for h in hooks:
        h.remove()

    return {name: torch.cat(tensors, dim=0) for name, tensors in buffers.items()}


def build_pca_basis(activations: Tensor, k: int = 128) -> tuple[Tensor, Tensor]:
    """Compute PCA basis via SVD.

    Args:
        activations: [n_samples, hidden_dim] tensor.
        k: Number of principal components to keep.

    Returns:
        (U, explained_variance) where U is [hidden_dim, k].
    """
    X = activations.float()
    X = X - X.mean(dim=0, keepdim=True)

    # SVD: X = U_full @ diag(S) @ Vh  →  columns of Vh^T are PCs in hidden-dim space
    _U_full, S, Vh = torch.linalg.svd(X, full_matrices=False)

    k = min(k, S.shape[0], Vh.shape[0])
    U = Vh[:k].T  # [hidden_dim, k]
    explained_variance = (S[:k] ** 2) / (X.shape[0] - 1)
    return U, explained_variance


def build_gradient_basis(
    model,
    tokenizer,
    memory_set: list[dict],
    layer_names: list[str],
    k: int = 128,
    batch_size: int = 4,
    device: str = "auto",
) -> dict[str, Tensor]:
    """Compute gradient-informed basis via SVD of activation gradients.

    For each layer, collects ∂L_old/∂a over the memory set, then takes the
    top-k SVD directions of the gradient matrix. These are the directions
    along which old-task loss is most sensitive to activation changes.

    Args:
        memory_set: List of {"input": str, "output": str, ...} examples.
        layer_names: Which layers to probe.
        k: Number of top gradient directions to keep.
        batch_size: Process this many examples at a time.

    Returns:
        {layer_name: U_grad tensor [hidden_dim, k]}
    """
    from data import get_batch

    if device == "auto":
        device = str(next(model.parameters()).device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    grad_buffers: dict[str, list[Tensor]] = {name: [] for name in layer_names}
    hooks = []
    hidden_states: dict[str, Tensor] = {}

    def _make_hook(name: str):
        def hook_fn(_module, _input, output):
            h = output[0] if isinstance(output, tuple) else output
            h.retain_grad()
            hidden_states[name] = h
        return hook_fn

    modules = dict(model.named_modules())
    for name in layer_names:
        hooks.append(modules[name].register_forward_hook(_make_hook(name)))

    try:
        model.eval()
        for i in range(0, len(memory_set), batch_size):
            chunk = memory_set[i : i + batch_size]
            batch = get_batch(
                chunk, batch_size=len(chunk), tokenizer=tokenizer, device=device,
            )
            hidden_states.clear()
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            out.loss.backward()
            for name in layer_names:
                # ∂L/∂h has shape [batch, seq, hidden]; mean-pool over seq
                g = hidden_states[name].grad.float().mean(dim=1)  # [batch, hidden]
                grad_buffers[name].append(g.detach().cpu())
            model.zero_grad()
    finally:
        for h in hooks:
            h.remove()

    result: dict[str, Tensor] = {}
    for name in layer_names:
        G = torch.cat(grad_buffers[name], dim=0).float()  # [n, hidden]
        _U_full, _S, Vh = torch.linalg.svd(G, full_matrices=False)
        k_actual = min(k, Vh.shape[0])
        result[name] = Vh[:k_actual].T  # [hidden, k]

    return result


def rank_importance(
    model,
    tokenizer,
    basis: Tensor,
    memory_set: list[dict],
    eval_fn,
    layer_name: str,
) -> Tensor:
    """For each PC dimension j, zero it and measure metric drop.

    Args:
        eval_fn: callable(model, tokenizer, memory_set) -> float (higher is better).

    Returns:
        importance scores tensor of shape [k].
    """
    k = basis.shape[1]
    baseline = eval_fn(model, tokenizer, memory_set)
    importance = torch.zeros(k)

    module = dict(model.named_modules())[layer_name]

    for j in range(k):
        direction = basis[:, j].clone()

        def _make_ablate_hook(col: Tensor):
            col_dev = None

            def hook_fn(_module, _input, output):
                nonlocal col_dev
                is_tensor = isinstance(output, Tensor)
                is_tuple = isinstance(output, tuple)
                if is_tensor:
                    h = output
                elif is_tuple:
                    h = output[0]
                elif hasattr(output, "last_hidden_state"):
                    h = output.last_hidden_state
                else:
                    return output
                orig_dtype = h.dtype
                if col_dev is None:
                    col_dev = col.to(h.device).float()
                # h may be [batch, seq, hidden] or [seq, hidden]
                h_flat = h.float().reshape(-1, h.shape[-1])
                proj = h_flat @ col_dev.unsqueeze(-1)  # [..., 1]
                h_new = (h.float() - (proj.reshape(*h.shape[:-1], 1) * col_dev)).to(orig_dtype)
                if is_tensor:
                    return h_new
                if is_tuple:
                    return (h_new, *output[1:])
                output.last_hidden_state = h_new
                return output

            return hook_fn

        handle = module.register_forward_hook(_make_ablate_hook(direction))
        score = eval_fn(model, tokenizer, memory_set)
        handle.remove()

        importance[j] = max(baseline - score, 0.0)

    return importance


def estimate_forgetting_constant(
    drift: Tensor, forgetting: Tensor
) -> tuple[float, float]:
    """Estimate the forgetting constant C_ℓ = Cov(F, D) / Var(D) via OLS.

    Args:
        drift: [n] tensor of subspace drift magnitudes ‖P_W(Δa)‖ per sample
        forgetting: [n] tensor of forgetting values (Δloss) per sample

    Returns:
        (C_hat, std_err): OLS slope estimate and its standard error
    """
    assert drift.shape == forgetting.shape and drift.dim() == 1
    n = drift.shape[0]

    var_d = drift.var().item()
    if var_d < 1e-12:
        return float("inf"), float("inf")

    cov_fd = ((drift - drift.mean()) * (forgetting - forgetting.mean())).mean().item()
    C_hat = cov_fd / var_d

    # Residual standard error
    residuals = forgetting - (C_hat * drift + (forgetting.mean() - C_hat * drift.mean()))
    residual_var = (residuals**2).sum().item() / max(n - 2, 1)
    std_err = math.sqrt(residual_var / (n * var_d))

    return C_hat, std_err


def davis_kahan_buffer_size(
    delta_k: float,
    sigma: float = 1.0,
    confidence: float = 0.95,
    target_sin_theta: float = 0.1,
    clamp_min: int = 128,
    clamp_max: int = 1024,
) -> int:
    """Compute minimum buffer size for PCA subspace estimation via Davis-Kahan.

    n >= (sigma^4 / (c * delta_k^2 * t^2)) * log(2/delta)

    Args:
        delta_k: eigenvalue gap lambda_k - lambda_{k+1}
        sigma: sub-Gaussian parameter of activations
        confidence: 1 - delta (probability of success)
        target_sin_theta: target principal angle bound t
        clamp_min: minimum buffer size
        clamp_max: maximum buffer size

    Returns:
        Required buffer size, clamped to [clamp_min, clamp_max]
    """
    delta = 1.0 - confidence
    c = 0.5  # universal constant (conservative estimate)
    log_term = math.log(2.0 / delta)
    n = (sigma**4 * log_term) / (c * delta_k**2 * target_sin_theta**2)
    return max(clamp_min, min(clamp_max, math.ceil(n)))


def select_top_r(basis: Tensor, importance: Tensor, r: int = 32) -> Tensor:
    """Select top-r most important dimensions from basis.

    Returns:
        U_r of shape [hidden_dim, r].
    """
    r = min(r, basis.shape[1])
    indices = torch.argsort(importance, descending=True)[:r]
    return basis[:, indices]
