"""Forgetting-curve replay: review hardest what the model is forgetting fastest.

Three mechanisms, each mirroring how brains fight forgetting:

  1. Replay (rehearsal) — re-train on a small memory buffer of old-task
     examples. Mirrors the hippocampus reinstating experiences to the
     neocortex so new learning is interleaved with old instead of overwriting
     it (complementary learning systems; McClelland, McNaughton & O'Reilly,
     1995).

  2. Distillation (function preservation) — keep the model's outputs on old
     examples close to a frozen pre-task "teacher" snapshot, not just close to
     the old hard labels (LwF; Li & Hoiem, 2017).

  3. Forgetting-curve weighting (the part nothing on the board does) — spend
     the preservation budget where forgetting is most imminent. For each
     memory example the gap between the frozen teacher and the current student
     measures how far that knowledge has already drifted; the examples
     drifting most get the largest review weight. This is spaced repetition /
     the Ebbinghaus forgetting curve ("review just before you forget") and
     prioritised retrieval of the most-interfered memories (Aljundi et al.,
     2019), implemented with the teacher we already have.

Why this is built to beat the leader: sfp wins on retention by being
*selective* — it hard-protects only the ~32 most important activation
directions (out of ~2048) and lets the rest move, so plasticity survives.
This method is selective in a different currency: it concentrates preservation
on the *examples* most at risk rather than all of them uniformly, which
likewise buys retention without taxing plasticity. The board ranks by
0.6*retention + 0.4*plasticity, so retention is the axis worth fighting for —
but because the weighting frees up effort on already-safe knowledge, the
new-task signal (plasticity) is not crushed.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

CONTRIBUTOR = "lilaclilac09"
SETUP = "distill"  # harness injects a frozen `teacher` snapshot at each task boundary


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
    """L = CE_new + beta*CE_mem + alpha*T^2 * sum_i w_i * KL(teacher||student)_i

    where w_i = softmax(focus * forgetting_i) puts more review weight on the
    memory examples that have drifted furthest from the teacher.
    """
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

    mem_ids = memory_batch["input_ids"]
    mem_mask = memory_batch["attention_mask"]

    # One student forward on memory feeds both replay CE and distillation.
    out_mem = model(input_ids=mem_ids, attention_mask=mem_mask, labels=memory_batch["labels"])
    loss = loss + beta * out_mem.loss  # broad rehearsal (replay)

    # No teacher snapshot -> fall back to replay only.
    if teacher is None:
        return loss

    with torch.no_grad():
        t_logits = teacher(input_ids=mem_ids, attention_mask=mem_mask).logits

    # Qwen runs in bfloat16; upcast to float32 so the softmax / log / KL over a
    # ~150k vocab stays numerically stable (bf16 underflows the small probs).
    t_logits = t_logits.float()
    s_logits = out_mem.logits.float()

    # Per-example forgetting signal = token-averaged KL(teacher || student).
    t_p = F.softmax(t_logits / temperature, dim=-1)
    s_logp = F.log_softmax(s_logits / temperature, dim=-1)
    kl_tok = (t_p * (t_p.clamp_min(1e-9).log() - s_logp)).sum(-1)  # [B, S]
    m = mem_mask.to(kl_tok.dtype)
    kl_ex = (kl_tok * m).sum(1) / m.sum(1).clamp_min(1.0)          # [B]

    # Spaced-repetition weights: bigger drift -> more review. Detached so we
    # weight *by* forgetting rather than optimising the weights themselves.
    w = F.softmax(focus * kl_ex.detach(), dim=0)                   # [B], sums to 1
    distill = (w * kl_ex).sum() * (temperature ** 2)

    return loss + alpha * distill
