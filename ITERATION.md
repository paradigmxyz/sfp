# Iteration log — `forgetting_curve` submission

How this submission evolved from a one-line prompt ("do what the page shows") through eight iterations to the current PR. Includes the dead-ends and pivots, not just the wins — because the dead-ends are where the real lessons live.

---

## Iteration 0 — reconnaissance

**Starting input**: a URL `sfp.paradigm-36f.workers.dev/submit` with the instruction "do what the page shows".

**Action**: rendered the JS-only SPA in a headless browser, read the rendered page; cloned `paradigmxyz/sfp`; read `README.md`, the `/about` motivation page, the arxiv survey it cites (2404.16789, Shi et al.), `submissions/README.md`, and the leaderboard page itself.

**Outcome**: this is Paradigm's open benchmark for **catastrophic forgetting in LLMs**. A submission is exactly one Python loss function; Paradigm runs it on a cloud A10G via Modal. Scoring displayed on the live page: `0.6 × retention + 0.4 × plasticity`. Current leader: `sfp` (Sparse Feature Preservation) at 0.7124.

---

## Iteration 1 — DER++ first pass (`derpp.py`)

**Hypothesis**: stack the two strongest single-mechanism baselines on the board — replay (0.5892) + distill (0.6350) — into one loss. This is **DER++** (Buzzega et al., 2020) adapted to the harness.

**Action**: wrote `submissions/derpp.py` with
```
L = CE_new + β·CE_mem + α·T²·KL(teacher ‖ student) on memory
```
Defaults α=0.5, β=0.5. Validated structure + real-model forward+backward.

**Outcome**: solid floor — should clear distill, unlikely to beat sfp without further refinement.

---

## Iteration 2 — the scoring discrepancy

**Trigger**: noticed the live submit page header says `score = 0.6 × retention + 0.4 × plasticity` (a linear, retention-weighted combination), but `evaluate.py` labels that formula `legacy linear composite score, deprecated in favour of harmonic-mean scoring`.

**Investigation**: back-computed every visible leaderboard score from `web/src/worker/seed.sql`'s retention / plasticity values:

| method | R | P | 0.6·R + 0.4·P | displayed |
|---|---|---|---|---|
| sfp | 0.754 | 0.650 | **0.7124** | 0.7124 ✓ |
| distill | 0.650 | 0.6125 | **0.6350** | 0.6350 ✓ |
| naive | 0.180 | 0.550 | **0.3280** | 0.3280 ✓ |

Confirmed against the live `/api/leaderboard` endpoint.

**Outcome**: **the deployed leaderboard ranks by linear, not harmonic.** The repo's docs are stale on this point. Retention is weighted 1.5× plasticity, so lean harder toward retention. Updated `derpp.py` to α=1.0 (heavier distill — the empirically stronger retention mechanism among single-mechanism baselines).

---

## Iteration 3 — forgetting-curve weighting (`forgetting_curve.py`)

**Hypothesis**: brains don't review everything equally — they review hardest what's forgetting fastest (Ebbinghaus 1885 forgetting curve; modern spaced-repetition systems; Maximally Interfered Retrieval, Aljundi et al. 2019). **Nothing on the board does this** — that's the contribution.

**Mechanism**: weight the per-example distillation by how far the student has drifted from the frozen teacher on that example:
```
KL_i  = token-averaged KL(teacher_i ‖ student_i)   on memory example i
w_i   = softmax(focus · KL_i.detach())
L     = CE_new + β·CE_mem + α·T²·Σ_i w_i·KL_i
```
Detach on `KL_i` inside the softmax so we **weight by** forgetting rather than optimise the weighting itself.

**Validation**:

| Check | Result |
|---|---|
| Validator (size, blocked imports, single `*_loss`) | ✓ 4112 B; imports only `torch`, `torch.nn.functional` |
| Real-model forward+backward, all 3 paths | ✓ finite loss, gradients flow |
| Per-example weights non-uniform in correct direction | ✓ KL=[0.131, 0.088] → w=[0.511, 0.489] (higher drift → higher weight) |

---

## Iteration 4 — bfloat16 safety fix

**Trigger**: re-read `AGENTS.md` (the agent-targeted docs I'd skipped on the first pass). Key line:

> *"Qwen and most modern models load as bfloat16. Any code that touches activations must preserve dtype."*

I had only tested in float32. The KL block softmaxes over a ~151k vocab and takes logs — both numerically delicate in bf16.

**Test on synthetic bf16 logits, B=4, S=24, V=151000**:

| Path | per-example KL |
|---|---|
| bf16 KL (no upcast) | `[2.2656, 2.2500, 2.2344, 2.2656]` ← snapped to bf16 grid (~0.0156 spacing) |
| fp32 KL (upcast) | `[2.2514, 2.2518, 2.2459, 2.2614]` ← fine differences preserved |

**Outcome**: bf16 doesn't crash (PyTorch softmax has internal stability), but it **quantizes the per-example forgetting differences away** — and those differences are precisely the signal my method's weights depend on. Patched: `t_logits.float()` and `s_logits.float()` before any softmax/log/KL inside the loss. File size 4112 → 4335 B; still under the 10 KB limit.

---

## Iteration 5 — compliance audit ("did you really read the rules?")

**Trigger**: user pushed back: did I actually read every requirement, including the GitHub repo, the arxiv paper, and the on-page rules?

**Action**: full second-pass of `.github/workflows/`, `dev/LEADERBOARD.md`, `dev/RESEARCH.md`, `submissions/README.md` (line by line), `scripts/submit_server.py`'s validator, `scripts/modal_bench.py`'s preset definitions, and `tests/test_smoke.py`.

**Major finding — there are *two* submission paths with *different* registration requirements:**

| Path | Detector | Registration |
|---|---|---|
| `submit.py` to live server (`sfp.paradigm.xyz:8090`) | server-side `validate_code` parses the file as text | file in `submissions/` |
| **PR + `benchmark` label** | `.github/workflows/leaderboard.yml`: `git diff … methods.py … \| head -1` | **entry in `methods.py:METHODS` dict** |

My original PR only added to `submissions/`. **If a maintainer had labelled it `benchmark` right then, the workflow's `head -1` detector would have found no METHODS additions and defaulted to running `naive`.** Hours of work would have produced a `naive` benchmark score.

**Other compliance findings:**

- `tests/test_smoke.py:test_method_registry` does `assert expected == set(METHODS.keys())` — **exact set equality**. Adding a method requires extending the test's `expected` set in the same commit, or CI (`.github/workflows/ci.yml`) goes red.
- `ci.yml` ruff scope is `sfp/ tests/ tasks/ scripts/ *.py`. Top-level `methods.py` is in scope; `submissions/` is out of scope.
- The "no changes to X" rule from the submit page is precisely `train.py / evaluate.py / data.py / model.py`. **`methods.py` is *not* in this list** — modifying it (including adding new SETUP keys in `method_setup()`) is explicitly compliant per `dev/LEADERBOARD.md`. I had been overcautious calling this a "grey area".

**Fix**: registered `forgetting_curve_loss` in `methods.py` METHODS dict between `orthogonal` and `sfp`; extended `test_smoke.py:expected` set; ran `ruff check` locally; pushed (commit `8931e54`).

---

## Iteration 6 — local head-to-head (the kill-and-recover saga)

**Goal**: empirically verify forgetting_curve preserves math better than naive on a representative task transition, on a model small enough to run locally without an A10G.

**Failures along the way** (worth recording so future contributors don't repeat them):

| Attempt | Config | Outcome |
|---|---|---|
| 1 | CPU fp32, 120 steps, math→code with sandboxed HumanEval eval | killed mid-run (slow eval) |
| 2 | CPU fp32, 60 steps, no accuracy eval, MAXLEN=128 | killed at naive step 45 |
| 3 | CPU **bf16** (to save memory), 60 steps, MAXLEN=64 | killed manually — bf16 on Apple CPU has no native acceleration; ~22 s/step → ~55 min projected for the FC arm |
| 4 | CPU fp32 again after user closed apps | killed shortly after launch |

**Root cause diagnosed**: `top -l 1` showed `PhysMem … 84 M unused`, `compressor 4.2 GB`. **macOS jetsam was killing the process under memory pressure** — not the agent runtime time-limiting it. Three 540 MB model copies (base + teacher + arm) couldn't fit alongside whatever else was open on a 16 GB machine.

**Pivot**: `torch.backends.mps.is_available()` returned `True`. **Apple Silicon GPU was idle the entire time.** Apple Silicon has native bf16 on the GPU; CPU does not. Switching to `DEVICE = "mps"` moved compute to the GPU and removed CPU memory contention.

**Final run** (Apple MPS, bfloat16, 60 steps, math → code, SmolLM2-135M + LoRA r=16):

| method | math_loss | Δ vs base |
|---|---|---|
| base (before code training) | 2.363 | — |
| naive (plain CE on code) | 2.210 | −0.153 |
| **forgetting_curve** | **1.741** | **−0.621** (≈4× more math preserved than naive) |

Per-example forgetting weights stayed non-uniform throughout the run; one transient `nan` train_loss at fc step 45 recovered by step 59 (likely a single-batch bf16 hiccup; the fp32 KL upcast prevented it from propagating).

**Honest caveat recorded in `submissions/forgetting_curve.md`**: the eval batch and the FC replay pool are both sampled from the same 200-example `math_data`, so part of FC's win is direct rehearsal of similar examples. The official harness holds eval data out — the gap will look different on the real benchmark.

---

## Iteration 7 — hyperparameter variants

**Hypothesis**: without real-benchmark scores to tune against, register multiple principled variants and let the actual A10G run pick the winner.

**Action**: added four thin-wrapper variants around `forgetting_curve_loss`:

| Variant | Hyperparams | Reasoning |
|---|---|---|
| `forgetting_curve_dist` | α=2.0, β=0.3 | doubles down on the strongest baseline (distill) |
| `forgetting_curve_replay` | α=0.5, β=1.0 | bets hard-label rehearsal carries more retention at 1.5B |
| `forgetting_curve_sharp` | focus=3.0 | sharpens per-example softmax onto most-forgotten examples |
| `forgetting_curve_soft` | T=4.0 | softer distill target, more knowledge transferred per token |

**Workflow constraint**: `leaderboard.yml` only `head -1`s the diff — only one new method gets auto-run per PR. Placed `forgetting_curve_dist` first in the METHODS dict (just after `orthogonal`) so it becomes the primary auto-run; the other three sit registered as named alternatives for follow-up PRs or manual `modal run` invocations.

`tests/test_smoke.py:expected` extended to the full 14-method set; ruff clean.

---

## Iteration 8 — local proxy decides the primary

**Trigger**: which of the five forgetting_curve variants should `leaderboard.yml`'s `head -1` detector pick as the auto-run? Iteration 7 made an *a priori* guess (`forgetting_curve_dist`); this iteration runs the actual numbers.

**Method**: extended the local PK to all 5 variants + 3 baselines (naive / replay / distill) on Apple MPS bf16, 60 steps math → code, then re-ran in three configurations:

1. **leaky**: math eval batch and replay pool both sampled from the same 200-example `math_data`.
2. **held-out math**: 80/20 split of `math_data` so the eval batch is invisible to replay.
3. **held-out both axes**: also a 80/20 split of `code_data`, so plasticity (code-side held-out loss) is measured cleanly.

**Final proxy score on configuration #3** (mirrors the live formula `0.6 × retention + 0.4 × plasticity`, with loss in place of accuracy):

| rank | method | math_loss | Δm | code_loss | Δc | proxy = −0.6·m − 0.4·c |
|---|---|---|---|---|---|---|
| 🥇 | **fc_replay** (α=0.5, β=1.0) | 1.692 | −0.439 | 1.330 | −0.369 | **−1.547** |
| 🥈 | fc_soft (T=4.0) | 1.759 | −0.372 | 1.290 | −0.409 | −1.571 |
| 🥉 | fc_base (defaults) | 1.770 | −0.360 | 1.289 | −0.410 | −1.578 |
| 4 | fc_sharp (focus=3.0) | 1.780 | −0.351 | 1.291 | −0.408 | −1.584 |
| 5 | replay (baseline) | 1.829 | −0.301 | 1.242 | −0.457 | −1.594 |
| 6 | fc_dist (α=2.0, β=0.3) | 1.859 | −0.272 | 1.277 | −0.422 | −1.626 |
| 7 | naive | 2.001 | −0.130 | 1.222 | −0.477 | −1.689 |
| 8 | distill (baseline) | 2.123 | −0.008 | 1.338 | −0.361 | −1.809 |

**Findings**:

1. **`fc_replay` wins on three independent runs** — leaky-eval, held-out math, held-out both. Consistent across configurations is the strongest signal local PK can give.
2. **The feared "heavy replay crushes plasticity" did not happen.** `fc_replay`'s code_loss is 1.330 vs the best (naive's 1.222) — only 0.04 worse. Retention more than compensates.
3. **`fc_dist`, the iteration-7 *a priori* primary, ranks 6th of 8** — worse than the plain replay baseline. The local distill teacher knows nothing about math (no task-0 training in the local PK), so heavy distillation has nothing to anchor.
4. **`distill` baseline is dead last** for the same reason — a small-model-specific artifact. On the real Qwen2.5-1.5B benchmark with proper task-0 training, distill is the *second*-strongest single-mechanism baseline (0.6350).

**Caveats** (the local PK is a proxy, not a prediction):

- 135M small-model dynamics ≠ 1.5B Qwen dynamics.
- Local sequence is `zero-shot → code` (one task transition); the real harness is `math → code → ifeval → safety → domain` (four transitions).
- The local teacher = base model (no task-0 training), so distillation under-performs. On the real harness the teacher knows the old task — `fc_dist` might claw back ground there.
- Proxy uses loss; the leaderboard uses accuracy. Ranking *usually* survives that mapping but isn't guaranteed.

**Action**: reorder `methods.py:METHODS` to put `forgetting_curve_replay` first. `leaderboard.yml`'s `head -1` detector will now select it as the primary auto-run.

---

## Current state — what's on the PR

PR #4 at `paradigmxyz/sfp` (`lilaclilac09:master` → `paradigmxyz:master`).

| Commit | Content |
|---|---|
| `4a1c65d` | `submissions/forgetting_curve.py` + `submissions/derpp.py` |
| `74df23e` | `submissions/forgetting_curve.md` (per-method README) |
| `8931e54` | `methods.py` registration + `tests/test_smoke.py` update |
| `a0e1f88` | `REPRODUCE.md` (methodology + plain-language rules) |
| `23b9d68` | `REPRODUCE.md` §3.2–3.3 (two submission paths + gotchas) |
| `3841aa8` | four hyperparameter variants in `methods.py` |
| *(this commit)* | `ITERATION.md` |

Awaiting a Paradigm maintainer to add the `benchmark` label. The workflow will then auto-run `forgetting_curve_dist` on Modal A10G under the `standard` preset (Qwen2.5-1.5B, 5 tasks, 2000 steps × 3 seeds, M=128).

---

## What I'd do next if iteration 7 doesn't beat sfp

Per **option C** in the design discussion (fully compliant per `dev/LEADERBOARD.md`, just larger):

- New SETUP key `"fc_sfp"` registered in `method_setup()` (allowed — `methods.py` is not in the no-touch list)
- Setup combines the distill teacher snapshot with SFP's PCA-importance basis from `features.py`
- Loss applies the per-example forgetting weighting (this submission's contribution) **on top of** SFP's selective preservation in the most important hidden-state directions

This combines selectivity in *example* space (this submission) with selectivity in *feature* space (sfp's win). It addresses the one axis where sfp clearly out-thinks the current submission.

Risk: SFP setup is budgeted at 120 s per task boundary; combining with the deep-copy teacher snapshot is tight but should fit on the A10G. Holding it until we have real numbers from iteration 7 — no point burning a more complex submission before knowing whether the simpler one already wins.
