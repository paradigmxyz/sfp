"""SFP + DER++ + EWC-LoRA: five complementary retention signals.

Directly addresses the paper's own stated threats to validity:

  Threat 1 (LoRA confound): EWC-LoRA penalises LoRA weight drift toward
    the previous-task optimum — retention in parameter space, not just
    activation space, so forgetting is constrained regardless of whether
    it is a LoRA-subspace artifact.
    (Zheng et al., ICLR 2026, arXiv:2602.17559: +8.9% over vanilla LoRA)

  Threat 2 (layer sensitivity): SFP weight reduced to 0.05; EWC + logit
    + CE_mem provide retention independent of which three layer depths
    were chosen by sfp_setup.

  Threat 3 (task-order): Additive terms are symmetric across orderings.

Loss (fully additive — new-task gradient weight always 1.0):

  L = L_new                                  (plasticity, full weight)
    + 0.50 * CE(model(x_mem), y_mem)         (output retention ~1:1)
    + 0.20 * MSE(logits_now, logits_ref)      (DER++: logit distillation)
    + 0.05 * SFP_preserve                    (geometric retention, 3 layers)
    + 0.10 * EWC_LoRA                        (LoRA weight anchor)

DER++ reference logits: captured once at task boundary (on the same
memory batch used for the first forward pass that step), then held
constant for the entire task. This is the correct DER++ semantics —
"dark experience" = logits frozen at the moment of task transition.

EWC-LoRA: snapshot of LoRA A/B weights at task boundary, L2 penalty
throughout the task. Uniform Fisher (L2) — robust for small memory (128).

Both are reset automatically when the basis fingerprint changes (i.e.
at each new task boundary detected by sfp_setup returning a new basis).

References:
  DER++:      Buzzega et al. NeurIPS 2020 (arXiv:2004.07211)
  EWC-LoRA:   Zheng et al. ICLR 2026 (arXiv:2602.17559)
  Replay 1:1: Scalable Strategies (arXiv:2505.12512)
"""

import torch
import torch.nn.functional as F
from torch import Tensor

CONTRIBUTOR = "mzoubkoff"
SETUP = "sfp"

# Module-level state — persists across steps, reset at each task boundary.
# Only torch, torch.nn.functional, copy, math are allowed per leaderboard rules.
_state = {
    "fp": None,        # basis fingerprint (int) — sentinel for task boundary
    "theta_star": {},  # {name: Tensor on CPU} — LoRA weights snapshot
    "logit_ref": None, # Tensor on CPU — DER++ reference logits from boundary
}


def _fp(basis: dict) -> int:
    """Fingerprint of current basis: changes exactly when sfp_setup re-runs."""
    if not basis:
        return 0
    v = next(iter(basis.values()))
    return id(v) + int(v[0, 0].item() * 1e6) % (2 ** 31)


def _snapshot(model, memory_batch: dict, device) -> None:
    """Capture theta* and reference logits at task boundary."""
    # LoRA weight snapshot (CPU to save GPU memory)
    snap = {}
    for name, p in model.named_parameters():
        if p.requires_grad and ("lora_A" in name or "lora_B" in name):
            snap[name] = p.detach().clone().cpu()
    _state["theta_star"] = snap

    # DER++ reference logits: one forward pass on memory, no grad
    with torch.no_grad():
        out = model(
            input_ids=memory_batch["input_ids"],
            attention_mask=memory_batch["attention_mask"],
        )
    # Mean-pool over sequence → [B, V]; store on CPU
    _state["logit_ref"] = out.logits.float().mean(dim=1).cpu()


def sfp_ewc_der_loss(
    model,
    batch: dict,
    memory_batch: dict | None = None,
    basis: dict | None = None,
    anchor_acts: dict | None = None,
    lam: float = 0.05,
    alpha: float = 0.5,
    beta: float = 0.20,
    lam_ewc: float = 0.10,
    **kw,
) -> Tensor:
    """SFP + DER++ + EWC-LoRA: additive five-signal continual learning loss."""

    # ── New-task CE (always full weight) ──────────────────────────────────
    out_new = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    loss = out_new.loss

    if memory_batch is None or basis is None or anchor_acts is None:
        return loss

    # ── Task boundary: snapshot weights + reference logits once per task ──
    fp = _fp(basis)
    if _state["fp"] != fp:
        _state["fp"] = fp
        _snapshot(model, memory_batch, loss.device)

    # ── One memory forward pass: CE + activations + current logits ────────
    layer_names = list(basis.keys())
    current_acts: dict[str, Tensor] = {}

    def _hook(name: str):
        def h(_m, _i, output):
            x = output[0] if isinstance(output, tuple) else output
            current_acts[name] = x.float().mean(dim=1)  # [B, hidden]
        return h

    handles = []
    mods = dict(model.named_modules())
    for name in layer_names:
        handles.append(mods[name].register_forward_hook(_hook(name)))

    try:
        out_mem = model(
            input_ids=memory_batch["input_ids"],
            attention_mask=memory_batch["attention_mask"],
            labels=memory_batch["labels"],
        )
    finally:
        for h in handles:
            h.remove()

    # ── alpha: Memory CE (~1:1 replay ratio) ──────────────────────────────
    loss = loss + alpha * out_mem.loss

    # ── beta: DER++ logit distillation against task-boundary reference ─────
    if _state["logit_ref"] is not None:
        ref = _state["logit_ref"].to(loss.device)          # [n_ref, V]
        cur = out_mem.logits.float().mean(dim=1)            # [B, V]
        n = min(cur.shape[0], ref.shape[0])
        loss = loss + beta * F.mse_loss(cur[:n], ref[:n])

    # ── lam: SFP geometric preservation (3 layers from sfp_setup) ─────────
    preserve = torch.tensor(0.0, device=loss.device)
    for name in layer_names:
        if name not in current_acts:
            continue
        u_r = basis[name].to(current_acts[name].device)
        projected = current_acts[name] @ u_r                # [B, r]
        anchor = anchor_acts[name].to(projected.device)     # [n, r]
        n = min(projected.shape[0], anchor.shape[0])
        preserve = preserve + F.mse_loss(projected[:n], anchor[:n])
    loss = loss + lam * preserve

    # ── lam_ewc: EWC-LoRA (LoRA weight anchor, uniform Fisher) ───────────
    if _state["theta_star"]:
        ewc = torch.tensor(0.0, device=loss.device)
        n_params = 0
        for name, p in model.named_parameters():
            if not p.requires_grad or name not in _state["theta_star"]:
                continue
            star = _state["theta_star"][name].to(p.device)
            ewc = ewc + F.mse_loss(p, star)
            n_params += 1
        if n_params > 0:
            loss = loss + lam_ewc * ewc

    return loss
