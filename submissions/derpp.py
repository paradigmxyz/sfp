"""DER++: experience replay + dark-knowledge distillation on the memory buffer.

Stacks two complementary retention mechanisms on top of full-weight new-task
learning:
  - experience replay   — cross-entropy on the memory batch (rehearsal), and
  - dark-knowledge distill — temperature-scaled KL from a frozen task-boundary
                             teacher on that same memory batch.

New-task CE stays at full weight to protect plasticity; the two memory terms
protect retention. Distillation is the dominant retention signal here (on this
benchmark logit-distill > replay among the single-mechanism baselines), so it
carries full weight while replay adds half-weight data-level rehearsal. The
leaderboard ranks by 0.6*retention + 0.4*plasticity, so retention is the more
valuable axis — but the soft distillation target preserves the old function
without hard-fitting old labels, so plasticity stays high rather than being
traded away.

The student forward on the memory batch is reused for both the replay CE and
the distillation logits, so the step costs one student forward + one teacher
forward on memory — same as the distill baseline, but with replay for free.

Ref: Buzzega et al., "Dark Experience for General Continual Learning",
NeurIPS 2020.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

CONTRIBUTOR = "lilaclilac09"
SETUP = "distill"  # harness injects a frozen `teacher` snapshot at each task boundary


def derpp_loss(
    model,
    batch: dict,
    memory_batch: dict | None = None,
    teacher=None,
    temperature: float = 2.0,
    alpha: float = 1.0,
    beta: float = 0.5,
    **kw,
) -> Tensor:
    """L = CE_new + beta * CE_mem + alpha * T^2 * KL(teacher || student) on mem."""
    # New-task cross-entropy — the plasticity signal, full weight.
    out_new = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    loss = out_new.loss

    # First task: no memory yet, nothing to preserve.
    if memory_batch is None:
        return loss

    # One student forward on memory serves both the replay CE and the
    # distillation logits.
    out_mem = model(
        input_ids=memory_batch["input_ids"],
        attention_mask=memory_batch["attention_mask"],
        labels=memory_batch["labels"],
    )
    # Experience replay: rehearse old-task labels.
    loss = loss + beta * out_mem.loss

    # Dark-knowledge distillation against the frozen teacher snapshot.
    if teacher is not None:
        with torch.no_grad():
            teacher_out = teacher(
                input_ids=memory_batch["input_ids"],
                attention_mask=memory_batch["attention_mask"],
            )
        s_logits = out_mem.logits / temperature
        t_logits = teacher_out.logits / temperature
        kl = F.kl_div(
            F.log_softmax(s_logits, dim=-1),
            F.softmax(t_logits, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)
        loss = loss + alpha * kl

    return loss
