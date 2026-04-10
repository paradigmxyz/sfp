"""Information-theoretic SFP: adaptive per-layer preservation weighting.

Uses standard SFP setup (PCA basis + anchor activations) but applies
theory-driven per-layer lambda weighting instead of uniform regularization.

From Theorem T10 (per-layer optimal weighting): layers with larger
preservation signal should receive proportionally more regularization.
We approximate this by weighting each layer's MSE by the anchor norm
(proxy for signal magnitude s_ℓ), normalized across layers.
"""
import torch
import torch.nn.functional as F
from torch import Tensor

CONTRIBUTOR = "0xQuinto"
SETUP = "sfp"


def information_theoretic_sfp_loss(
    model,
    batch: dict,
    memory_batch: dict | None = None,
    basis: dict | None = None,
    anchor_acts: dict | None = None,
    lam: float = 0.1,
    **kw,
) -> Tensor:
    """SFP loss with adaptive per-layer weighting (T10).

    Instead of L = L_new + lam * sum_l MSE_l, uses:
        L = L_new + sum_l (lam_l * MSE_l)
    where lam_l is proportional to the anchor signal magnitude at layer l.
    """
    out_new = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    ce_loss = out_new.loss

    if basis is None or anchor_acts is None or memory_batch is None:
        return ce_loss

    layer_names = list(basis.keys())
    current_acts: dict[str, Tensor] = {}

    def _make_hook(name: str):
        def hook(_module, _input, output):
            out = output[0] if isinstance(output, tuple) else output
            current_acts[name] = out.float().mean(dim=1)
        return hook

    handles = []
    modules = dict(model.named_modules())
    for name in layer_names:
        handles.append(modules[name].register_forward_hook(_make_hook(name)))

    try:
        model(
            input_ids=memory_batch["input_ids"],
            attention_mask=memory_batch["attention_mask"],
        )
    finally:
        for h in handles:
            h.remove()

    # Compute per-layer weights from anchor signal magnitude (T10 proxy)
    # lam_l proportional to ||anchor_l||_F (layers with larger preserved
    # signal get more regularization budget)
    weights = {}
    total_w = 0.0
    for name in layer_names:
        if name in current_acts:
            w = anchor_acts[name].norm().item() + 1e-8
            weights[name] = w
            total_w += w

    # Normalize so weights sum to len(layers) * lam (same total budget)
    n_layers = len(weights)
    if total_w > 0 and n_layers > 0:
        scale = n_layers * lam / total_w
        for name in weights:
            weights[name] *= scale

    preserve = torch.tensor(0.0, device=ce_loss.device)
    for name in layer_names:
        if name not in current_acts:
            continue
        u_r = basis[name].to(current_acts[name].device)
        projected = current_acts[name] @ u_r
        anchor = anchor_acts[name].to(projected.device)
        n = min(projected.shape[0], anchor.shape[0])
        layer_loss = F.mse_loss(projected[:n], anchor[:n])
        preserve = preserve + weights.get(name, lam) * layer_loss

    return ce_loss + preserve
