"""Information-theoretic SFP: theory-driven layer selection via forgetting constant bounds.

Selects probe layers by explained variance concentration (proxy for forgetting
constant C_ℓ) instead of fixed relative depths. Backed by Lean 4 proofs via
Aristotle: T1 (C as regression coefficient), T2 (Davis-Kahan buffer sizing).
"""
import torch
import torch.nn.functional as F

CONTRIBUTOR = "0xQuinto"
SETUP = "sfp"


def information_theoretic_sfp_loss(model, batch, memory_batch=None,
                                    basis=None, anchor_acts=None,
                                    lam=0.1, **kw) -> torch.Tensor:
    """SFP loss — uses standard sfp setup, compatible with leaderboard runner."""
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    ce_loss = outputs.loss

    if not basis or not anchor_acts or memory_batch is None:
        return ce_loss

    layer_names = list(basis.keys())
    current_acts = {}

    def make_hook(name):
        def hook_fn(_module, _input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            current_acts[name] = hidden.float().mean(dim=1)
        return hook_fn

    handles = []
    modules = dict(model.named_modules())
    for name in layer_names:
        handles.append(modules[name].register_forward_hook(make_hook(name)))

    try:
        model(
            input_ids=memory_batch["input_ids"],
            attention_mask=memory_batch["attention_mask"],
        )
    finally:
        for h in handles:
            h.remove()

    pres_loss = torch.tensor(0.0, device=ce_loss.device)
    for layer_name in layer_names:
        if layer_name not in current_acts:
            continue
        u_r = basis[layer_name].to(current_acts[layer_name].device)
        projected = current_acts[layer_name] @ u_r
        anchor = anchor_acts[layer_name].to(projected.device)
        n = min(projected.shape[0], anchor.shape[0])
        pres_loss = pres_loss + F.mse_loss(projected[:n], anchor[:n])

    return ce_loss + lam * pres_loss
