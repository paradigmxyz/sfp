"""
methods.py — All continual learning methods as plain functions.

~250 lines. Each method is a function that returns a loss tensor.
No classes, no inheritance. Just functions.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def naive_loss(model, batch: dict, **kw) -> Tensor:
    """Standard cross-entropy loss. No forgetting mitigation. ~5 lines."""
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return out.loss


def replay_loss(
    model,
    batch: dict,
    memory_batch: dict | None = None,
    replay_ratio: float = 0.1,
    **kw,
) -> Tensor:
    """Mix new-task CE with memory replay CE at given ratio. ~15 lines."""
    out_new = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    if memory_batch is None:
        return out_new.loss

    out_mem = model(
        input_ids=memory_batch["input_ids"],
        attention_mask=memory_batch["attention_mask"],
        labels=memory_batch["labels"],
    )
    return (1 - replay_ratio) * out_new.loss + replay_ratio * out_mem.loss


def logit_distill_loss(
    model,
    batch: dict,
    teacher=None,
    memory_batch: dict | None = None,
    temperature: float = 2.0,
    alpha: float = 0.5,
    **kw,
) -> Tensor:
    """LwF-style: CE on new task + KL(teacher || student) on memory. ~25 lines."""
    out_new = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    ce_loss = out_new.loss

    if teacher is None or memory_batch is None:
        return ce_loss

    with torch.no_grad():
        teacher_out = teacher(
            input_ids=memory_batch["input_ids"],
            attention_mask=memory_batch["attention_mask"],
        )
    student_out = model(
        input_ids=memory_batch["input_ids"],
        attention_mask=memory_batch["attention_mask"],
    )

    t_logits = teacher_out.logits / temperature
    s_logits = student_out.logits / temperature

    kl = F.kl_div(
        F.log_softmax(s_logits, dim=-1),
        F.softmax(t_logits, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)

    return ce_loss + alpha * kl

logit_distill_loss.SETUP = "distill"


def hidden_distill_loss(
    model,
    batch: dict,
    teacher=None,
    memory_batch: dict | None = None,
    layers: list[str] | None = None,
    **kw,
) -> Tensor:
    """POD-style: CE + L2 match of normalized hidden states at selected layers. ~30 lines."""
    out_new = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    ce_loss = out_new.loss

    if teacher is None or memory_batch is None or not layers:
        return ce_loss

    student_acts: dict[str, Tensor] = {}
    teacher_acts: dict[str, Tensor] = {}

    def _make_hook(store: dict, name: str):
        def hook(_module, _input, output):
            out = output[0] if isinstance(output, tuple) else output
            store[name] = out
        return hook

    s_handles = []
    t_handles = []
    for name in layers:
        s_mod = dict(model.named_modules())[name]
        t_mod = dict(teacher.named_modules())[name]
        s_handles.append(s_mod.register_forward_hook(_make_hook(student_acts, name)))
        t_handles.append(t_mod.register_forward_hook(_make_hook(teacher_acts, name)))

    try:
        model(
            input_ids=memory_batch["input_ids"],
            attention_mask=memory_batch["attention_mask"],
        )
        with torch.no_grad():
            teacher(
                input_ids=memory_batch["input_ids"],
                attention_mask=memory_batch["attention_mask"],
            )
    finally:
        for h in s_handles + t_handles:
            h.remove()

    distill = torch.tensor(0.0, device=ce_loss.device)
    for name in layers:
        s_norm = F.normalize(student_acts[name].float(), dim=-1)
        t_norm = F.normalize(teacher_acts[name].float(), dim=-1)
        distill = distill + F.mse_loss(s_norm, t_norm)

    return ce_loss + distill

hidden_distill_loss.SETUP = "hidden_distill"


def orthogonal_loss(
    model,
    batch: dict,
    prev_lora_weights: dict | None = None,
    ortho_lambda: float = 1.0,
    **kw,
) -> Tensor:
    """Constrain new LoRA to be orthogonal to previous adapters. ~30 lines."""
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    ce_loss = out.loss

    if not prev_lora_weights:
        return ce_loss

    ortho_reg = torch.tensor(0.0, device=ce_loss.device)
    for name, param in model.named_parameters():
        if "lora_B" not in name or not param.requires_grad:
            continue
        prev_key = name.replace("lora_B", "lora_B_prev")
        if prev_key not in prev_lora_weights:
            base_key = name
            if base_key not in prev_lora_weights:
                continue
            prev_key = base_key
        b_prev = prev_lora_weights[prev_key].to(param.device)
        # ||B_new^T @ B_prev||^2_F
        ortho_reg = ortho_reg + torch.norm(param.T @ b_prev, p="fro") ** 2

    return ce_loss + ortho_lambda * ortho_reg

orthogonal_loss.SETUP = "orthogonal"


def forgetting_curve_loss(
    model,
    batch: dict,
    memory_batch: dict | None = None,
    teacher=None,
    temperature: float = 2.0,
    alpha: float = 1.0,
    beta: float = 0.5,
    focus: float = 1.0,
    **kw,
) -> Tensor:
    """DER++ + per-example forgetting-curve weighting on the distill term.

    Replay CE on memory + dark-knowledge KL from a frozen task-boundary teacher,
    with softmax-over-drift weights that concentrate preservation on memory
    examples most at risk of being forgotten (spaced repetition / Ebbinghaus).
    Submitted by lilaclilac09. ~35 lines.
    """
    out_new = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    loss = out_new.loss

    if memory_batch is None:
        return loss

    mem_ids = memory_batch["input_ids"]
    mem_mask = memory_batch["attention_mask"]

    out_mem = model(input_ids=mem_ids, attention_mask=mem_mask, labels=memory_batch["labels"])
    loss = loss + beta * out_mem.loss

    if teacher is None:
        return loss

    with torch.no_grad():
        t_logits = teacher(input_ids=mem_ids, attention_mask=mem_mask).logits

    # Per-example KL, computed one batch index at a time to avoid holding
    # multiple [B, S, V] tensors simultaneously. With Qwen2.5-1.5B (vocab
    # ~151k) the naive expression t_p * (log t_p - s_logp) materialises
    # ~5 [B,S,V] bf16 tensors at once (~3 GiB) which OOMs even on A10G with
    # a teacher copy resident. Looping over batch costs 1/B the peak memory.
    s_logits = out_mem.logits
    B = s_logits.shape[0]
    m = mem_mask.to(s_logits.dtype)
    kl_ex_list = []
    for i in range(B):
        t_pi = F.softmax(t_logits[i] / temperature, dim=-1)
        s_lpi = F.log_softmax(s_logits[i] / temperature, dim=-1)
        kl_tok_i = (t_pi * (t_pi.clamp_min(1e-9).log() - s_lpi)).sum(-1)  # [S]
        mi = m[i]
        kl_ex_list.append((kl_tok_i * mi).sum() / mi.sum().clamp_min(1.0))
        del t_pi, s_lpi, kl_tok_i
    kl_ex = torch.stack(kl_ex_list)  # [B]

    w = F.softmax(focus * kl_ex.detach(), dim=0)
    distill = (w * kl_ex).sum() * (temperature ** 2)

    return loss + alpha * distill

forgetting_curve_loss.SETUP = "distill"


def forgetting_curve_dist_loss(model, batch, memory_batch=None, teacher=None, **kw):
    """forgetting_curve variant — alpha=2.0, beta=0.3. Doubles down on the
    strongest single-mechanism baseline on the board (distill=0.6350 > replay=
    0.5892), keeping the replay term light so plasticity isn't taxed.
    """
    return forgetting_curve_loss(
        model, batch, memory_batch=memory_batch, teacher=teacher,
        alpha=2.0, beta=0.3, **kw,
    )

forgetting_curve_dist_loss.SETUP = "distill"


def forgetting_curve_replay_loss(model, batch, memory_batch=None, teacher=None, **kw):
    """forgetting_curve variant — alpha=0.5, beta=1.0. Bet that hard-label
    rehearsal carries retention more than softer distill on a 1.5B model.
    """
    return forgetting_curve_loss(
        model, batch, memory_batch=memory_batch, teacher=teacher,
        alpha=0.5, beta=1.0, **kw,
    )

forgetting_curve_replay_loss.SETUP = "distill"


def forgetting_curve_sharp_loss(model, batch, memory_batch=None, teacher=None, **kw):
    """forgetting_curve variant — focus=3.0. Sharpens the per-example forgetting
    softmax so review concentrates harder on the most-drifted memory examples.
    Leans into the unique mechanism nothing else on the board uses.
    """
    return forgetting_curve_loss(
        model, batch, memory_batch=memory_batch, teacher=teacher,
        focus=3.0, **kw,
    )

forgetting_curve_sharp_loss.SETUP = "distill"


def forgetting_curve_soft_loss(model, batch, memory_batch=None, teacher=None, **kw):
    """forgetting_curve variant — temperature=4.0. Softer distill target spreads
    teacher knowledge across more classes per token; the T^2 scaling keeps the
    distill term contribution roughly stable in magnitude.
    """
    return forgetting_curve_loss(
        model, batch, memory_batch=memory_batch, teacher=teacher,
        temperature=4.0, **kw,
    )

forgetting_curve_soft_loss.SETUP = "distill"


# ---------------------------------------------------------------------------
# SFP (our method)
# ---------------------------------------------------------------------------


def _sfp_loss_impl(
    model,
    batch: dict,
    memory_batch: dict | None = None,
    basis: dict | None = None,
    anchor_acts: dict | None = None,
    lam: float = 0.1,
    **kw,
) -> Tensor:
    """Shared SFP loss body for all SFP variants.

    L = L_new + lam * sum_l || U_r^T a_l(x;t) - anchor_acts_l ||^2
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
            # mean-pool over sequence dimension → [batch, hidden]
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

    preserve = torch.tensor(0.0, device=ce_loss.device)
    for name in layer_names:
        u_r = basis[name].to(current_acts[name].device)  # [hidden, r]
        projected = current_acts[name] @ u_r  # [batch, r]
        anchor = anchor_acts[name].to(projected.device)  # [n, r]
        # align batch sizes (take min)
        n = min(projected.shape[0], anchor.shape[0])
        preserve = preserve + F.mse_loss(projected[:n], anchor[:n])

    return ce_loss + lam * preserve


def sfp_loss(
    model,
    batch: dict,
    memory_batch: dict | None = None,
    basis: dict | None = None,
    anchor_acts: dict | None = None,
    lam: float = 0.1,
    **kw,
) -> Tensor:
    """Sparse Feature Preservation loss.

    L = L_new + lam * sum_l || U_r^T a_l(x;t) - anchor_acts_l ||^2

    Args:
        basis: {layer_name: U_r tensor [hidden_dim, r]}
        anchor_acts: {layer_name: projected activations [n, r] at anchor checkpoint}
        lam: Preservation weight.
    """
    return _sfp_loss_impl(model, batch, memory_batch, basis, anchor_acts, lam, **kw)

sfp_loss.SETUP = "sfp"


def sfp_grad_loss(
    model,
    batch: dict,
    memory_batch: dict | None = None,
    basis: dict | None = None,
    anchor_acts: dict | None = None,
    lam: float = 0.1,
    **kw,
) -> Tensor:
    """SFP with gradient-informed basis instead of PCA.

    Uses top SVD directions of ∂L_old/∂a to define the preserved subspace.
    """
    return _sfp_loss_impl(model, batch, memory_batch, basis, anchor_acts, lam, **kw)

sfp_grad_loss.SETUP = "sfp_grad"


def sfp_random_basis_loss(
    model,
    batch: dict,
    memory_batch: dict | None = None,
    basis: dict | None = None,
    anchor_acts: dict | None = None,
    lam: float = 0.1,
    **kw,
) -> Tensor:
    """Same as SFP but with random orthonormal basis. Control for PCA."""
    return _sfp_loss_impl(model, batch, memory_batch, basis, anchor_acts, lam, **kw)

sfp_random_basis_loss.SETUP = "sfp"


def sfp_random_selection_loss(
    model,
    batch: dict,
    memory_batch: dict | None = None,
    basis: dict | None = None,
    anchor_acts: dict | None = None,
    lam: float = 0.1,
    **kw,
) -> Tensor:
    """Same as SFP with PCA basis but random dim selection. Control for importance."""
    return _sfp_loss_impl(model, batch, memory_batch, basis, anchor_acts, lam, **kw)

sfp_random_selection_loss.SETUP = "sfp"


# ---------------------------------------------------------------------------
# Setup functions (called at task boundaries)
# ---------------------------------------------------------------------------


SETUP_BUDGET_SECONDS = 120  # max wall-clock time for method setup (2 min)


def method_setup(
    method_name: str, model, tokenizer, memory_buffers: dict, **kw
) -> dict:
    """Method-specific setup at each task boundary.

    Returns a state dict passed to the loss function each step.
    Dispatches on the loss function's .SETUP attribute.

    Setup is budgeted: if it takes longer than SETUP_BUDGET_SECONDS the run
    is aborted.  This prevents submissions from smuggling unbounded
    computation (extra training, expensive searches) into the setup phase.
    """
    import time as _time

    loss_fn = METHODS.get(method_name)
    setup_key = getattr(loss_fn, "SETUP", "none") if loss_fn else "none"

    if setup_key == "none":
        return {}

    t0 = _time.monotonic()

    if setup_key == "distill":
        teacher = copy.deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        result = {"teacher": teacher}
    elif setup_key == "hidden_distill":
        teacher = copy.deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        from features import get_layer_names
        layers = kw.get("layers") or get_layer_names(model)
        result = {"teacher": teacher, "layers": layers}
    elif setup_key == "orthogonal":
        prev = {}
        for name, param in model.named_parameters():
            if "lora_B" in name:
                prev[name] = param.detach().clone()
        result = {"prev_lora_weights": prev}
    elif setup_key == "sfp":
        result = sfp_setup(model, tokenizer, memory_buffers, **kw)
    elif setup_key == "sfp_grad":
        result = sfp_grad_setup(model, tokenizer, memory_buffers, **kw)
    else:
        result = {}

    elapsed = _time.monotonic() - t0
    print(f"  method_setup({setup_key}) took {elapsed:.1f}s (budget: {SETUP_BUDGET_SECONDS}s)")
    if elapsed > SETUP_BUDGET_SECONDS:
        raise RuntimeError(
            f"method_setup exceeded budget: {elapsed:.1f}s > {SETUP_BUDGET_SECONDS}s. "
            f"Setup must complete within {SETUP_BUDGET_SECONDS}s to prevent "
            f"smuggling unbounded computation into the setup phase."
        )
    return result


def sfp_setup(
    model,
    tokenizer,
    memory_buffers: dict,
    layers: list[str] | None = None,
    k: int = 128,
    r: int = 32,
    **kw,
) -> dict:
    """Build PCA basis, rank importance, select top-r, cache anchor activations.

    Returns: {"basis": {...}, "anchor_acts": {...}}
    """
    from features import (
        build_pca_basis,
        collect_activations,
        get_layer_names,
        select_top_r,
    )

    if layers is None:
        layers = get_layer_names(model)

    # Flatten memory buffers into a list of samples
    mem_samples: list[dict] = []
    for samples in memory_buffers.values():
        if isinstance(samples, list):
            mem_samples.extend(samples)

    # Collect activations at each layer
    acts = collect_activations(model, mem_samples, tokenizer, layers)

    basis: dict[str, Tensor] = {}
    anchor_acts: dict[str, Tensor] = {}

    for layer_name in layers:
        a = acts[layer_name].float()  # [n, hidden] — ensure float32 for PCA
        u, _var = build_pca_basis(a, k=k)  # [hidden, k]
        u_r = select_top_r(u, _var, r=r)  # [hidden, r]
        basis[layer_name] = u_r
        anchor_acts[layer_name] = (a @ u_r).detach()  # [n, r]

    return {"basis": basis, "anchor_acts": anchor_acts}


def sfp_grad_setup(
    model,
    tokenizer,
    memory_buffers: dict,
    layers: list[str] | None = None,
    k: int = 128,
    r: int = 32,
    **kw,
) -> dict:
    """Build gradient-informed basis, cache anchor activations.

    Returns: {"basis": {...}, "anchor_acts": {...}}
    """
    from features import (
        build_gradient_basis,
        collect_activations,
        get_layer_names,
    )

    if layers is None:
        layers = get_layer_names(model)

    # Flatten memory buffers into a list of samples
    mem_samples: list[dict] = []
    for samples in memory_buffers.values():
        if isinstance(samples, list):
            mem_samples.extend(samples)

    # Build gradient basis (already returns top-k directions per layer)
    grad_basis = build_gradient_basis(
        model, tokenizer, mem_samples, layers, k=k,
    )

    # Collect activations to compute anchor projections
    acts = collect_activations(model, mem_samples, tokenizer, layers)

    basis: dict[str, Tensor] = {}
    anchor_acts: dict[str, Tensor] = {}

    for layer_name in layers:
        u = grad_basis[layer_name]  # [hidden, k]
        # Select top-r (they're already ranked by singular value)
        u_r = u[:, :min(r, u.shape[1])]  # [hidden, r]
        basis[layer_name] = u_r
        a = acts[layer_name].float()  # [n, hidden]
        anchor_acts[layer_name] = (a @ u_r).detach()  # [n, r]

    return {"basis": basis, "anchor_acts": anchor_acts}


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHODS: dict[str, Callable] = {
    "naive": naive_loss,
    "replay": replay_loss,
    "distill": logit_distill_loss,
    "hidden_distill": hidden_distill_loss,
    "orthogonal": orthogonal_loss,
    # Reordered after local PK data (iter 8 in ITERATION.md):
    # forgetting_curve_replay (alpha=0.5, beta=1.0) wins the local proxy
    # -0.6*math_loss - 0.4*code_loss on three independent runs (leaky-eval,
    # held-out math-only, held-out both axes). Placed first so
    # leaderboard.yml's `head -1` detector picks it as the primary auto-run.
    # Other variants registered as alternatives for follow-up PRs.
    "forgetting_curve_replay": forgetting_curve_replay_loss,
    "forgetting_curve": forgetting_curve_loss,
    "forgetting_curve_dist": forgetting_curve_dist_loss,
    "forgetting_curve_sharp": forgetting_curve_sharp_loss,
    "forgetting_curve_soft": forgetting_curve_soft_loss,
    "sfp": sfp_loss,
    "sfp_grad": sfp_grad_loss,
    "sfp_random_basis": sfp_random_basis_loss,
    "sfp_random_selection": sfp_random_selection_loss,
}
