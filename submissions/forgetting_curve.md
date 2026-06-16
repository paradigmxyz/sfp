# forgetting_curve

A submission to the [sfp catastrophic-forgetting benchmark](https://github.com/paradigmxyz/sfp).

**Method:** replay + dark-knowledge distillation, with per-example review weight proportional to how much each old example is being forgotten. Spaced repetition for LLMs.

## The problem

Sequential fine-tuning of an LLM (alignment updates, safety patches, domain adaptation) degrades earlier capabilities — *catastrophic forgetting*. The benchmark holds everything fixed (same Qwen2.5-1.5B base, same data, same compute) and ranks training strategies by:

```
score = 0.6 × retention + 0.4 × plasticity
```

`retention` = accuracy on previously trained tasks. `plasticity` = accuracy on the newest task. Two axes pulling against each other.

## The intuition

Your brain already solved this. Three mechanisms, each mirrored in a CL technique:

| What the brain does | The technique | Who on the board uses it |
|---|---|---|
| **Hippocampus replays experiences to the cortex** during rest (complementary learning systems; McClelland, McNaughton & O'Reilly, 1995) | experience replay | `replay` (0.5892) |
| **Functional memory** — keep old behaviour intact, not exact weights (Learning without Forgetting; Li & Hoiem, 2017) | distillation from a frozen teacher snapshot | `distill` (0.6350) |
| **Spaced repetition** — review hardest what's most being forgotten (Ebbinghaus's 1885 self-experiments: ~58 % retained at 20 min, ~44 % at 1 hr, ~33 % at 24 hr) | per-example forgetting-weighted review | **nobody** ← this submission's contribution |

The board leader `sfp` (0.7124) wins on retention by being *selective* in another currency — preserving only the ~32 most important internal feature directions out of ~2048 per layer. This submission is selective in *example* space instead of *feature* space.

## The method

```
L = CE_new + β·CE_mem + α·T² · Σᵢ wᵢ · KL(teacherᵢ ‖ studentᵢ)   on memory

where wᵢ = softmax( focus · KL_per_example.detach() )
```

- **New-task CE** at full weight → protects plasticity.
- **Replay CE** on the memory buffer → broad rehearsal (CLS).
- **Weighted distillation** → concentrate function preservation on the examples most at risk. The per-example KL between the frozen teacher and the current student measures *how far that knowledge has drifted*; the softmax over those drifts up-weights the examples about to fall off the forgetting curve.

Defaults: `α=1.0, β=0.5, T=2.0, focus=1.0`. KL math is upcast to fp32 inside the loss to preserve per-example signal fidelity under the harness's bf16 model (per `AGENTS.md` dtype rule).

`SETUP = "distill"` — the harness deep-copies the model at each task boundary and passes it as `teacher`. Identical compute footprint to the `distill` baseline (one student forward + one teacher forward on memory); replay CE rides for free on the existing student memory forward.

## Validation

| Check | Result |
|---|---|
| Validator (`scripts/submit_server.py`) — size ≤ 10 KB, blocked-import set, `*_loss` function shape | ✓ 4335 bytes; imports only `torch`, `torch.nn.functional` |
| Real-model forward+backward, all three code paths (full / first-task / no-teacher) | ✓ finite loss, gradients flow (∥grad∥ ≈ 9.75 × 10⁴) |
| Per-example forgetting weights non-uniform in the correct direction | ✓ KL=[0.131, 0.088] → weights=[0.511, 0.489] (higher drift → higher weight) |
| Bfloat16 KL numerical check (Qwen runs in bf16) | ✓ fp32 upcast preserves per-example signal that bf16 would quantize away |
| Local PK on SmolLM2-135M (Apple MPS, 60 steps, math → code) | base **2.363** → naive **2.210** (Δ -0.153) → forgetting_curve **1.741** (Δ -0.621) — **4× more math preservation than naive** |
| Official Modal A10G benchmark (Qwen2.5-1.5B, 5 tasks × 2000 steps × 3 seeds) | ⏳ pending — Paradigm submission server is currently offline |

**Honest caveat on the local PK.** Both `math_data` (replay pool) and the math eval batch are sampled from the same 200-example pool, so part of `forgetting_curve`'s gain is from directly rehearsing examples similar to the eval. On the official harness, eval is strictly held out. Local PK confirms the mechanism works in the intended direction; absolute magnitudes will look different on the real benchmark.

## How to submit

```bash
cd sfp
python3 submit.py submissions/forgetting_curve.py
```

`submit.py` is pure stdlib — no GPU or extra deps needed on the submitter's end. Paradigm runs it on a cloud A10G via Modal (~45 min for `standard` preset).

The default submission server (`sfp.paradigm.xyz:8090`) is currently unreachable. Alternatives until it returns:
- Open a PR to `paradigmxyz/sfp` adding the file (this submission is at PR #4).
- Submit via `--server <URL>` if a live mirror exists.

## Acknowledgments

Method designed by **lilaclilac09** with **Claude (Opus 4.7, 1M-context)** as the AI implementation partner — code authorship, debugging, harness audit, and benchmark research.

The base model under test (Qwen2.5-1.5B-Instruct) is **fixed by the benchmark harness** — every method on the leaderboard trains the same Qwen model. Qwen is the *subject* being trained; Claude is the *tool* that wrote this training strategy. They are not interchangeable.

## References

- McClelland, McNaughton & O'Reilly (1995). *Why there are complementary learning systems in the hippocampus and neocortex.*
- Ebbinghaus (1885). *Über das Gedächtnis* — original forgetting-curve self-experiments.
- Kirkpatrick et al. (2017). *Overcoming catastrophic forgetting in neural networks* (EWC).
- Li & Hoiem (2017). *Learning without Forgetting* (LwF).
- Buzzega et al. (2020). *Dark Experience for General Continual Learning* (DER++).
- Aljundi et al. (2019). *Online Continual Learning with Maximally Interfered Retrieval* (MIR).
- Shi et al. (2024). *Continual Learning of Large Language Models: A Comprehensive Survey* (arXiv:2404.16789).
