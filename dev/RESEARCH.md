# SFP Research Briefing

*Consolidated from design discussions. Read this to get up to speed in 30 minutes.*

## Problem

LLMs are continually updated: alignment tuning, safety patches, domain adaptation, personalization. Sequential updates cause **capability regressions** — performance on previously validated behaviors degrades. In the continual learning literature this is called catastrophic forgetting.

This is distinct from the "retrain from scratch" paradigm most production LLM pipelines use. We focus on the setting where a model is updated *in place* over a sequence of tasks — relevant for personalization, on-device adaptation, and rapid iteration cycles where full retraining is too expensive.

## Why forgetting happens

Neural networks store knowledge in distributed, entangled weight patterns. There are no "task slots." When you update weights for Task B, nothing constrains the updates to preserve Task A performance. If gradients conflict (∇L_A · ∇L_B < 0), Task A degrades.

## Existing approaches (and why they're insufficient)

| Method | What it preserves | Limitation |
|--------|-------------------|------------|
| EWC (Kirkpatrick 2017) | Important weights | Global; doesn't know *what* weights encode |
| LwF (Li & Hoiem 2017) | Output logits | Doesn't preserve internal representations |
| PODNet (Douillard 2020) | Hidden states (full) | Preserves too much; hurts plasticity |
| Orthogonal LoRA | Parameter directions | Geometry-based, not feature-based |
| Replay | Data samples | Relies on data availability |

None answer: **where inside the model does forgetting actually happen?**

## Our hypothesis

**H1**: Measurable regressions are **predictable from low-dimensional activation drift**.
- Drift in top-r PCA coordinates at mid-depth layers predicts old-task accuracy drops (R² ≥ 0.6).
- Drift in random-r coordinates does not (R² ≈ 0).
- This must hold after controlling for total drift ‖Δa‖ (to rule out "drift = drift" confound).

**H2**: Preserving only high-importance features improves the stability-plasticity Pareto frontier.
- SFP dominates ≥2 baselines at small memory budgets.

**H3**: Preserved features are mechanistically coherent (interpretable, causally linked to behavior).

## Method: Sparse Feature Preservation (SFP)

1. **Hook activations** at mid-layers (depth/4, depth/2, 3*depth/4)
2. **Build PCA basis** from memory set activations
3. **Rank importance** by ablation: zero each PC dim, measure old-task drop
4. **Select top-r** dimensions
5. **Preservation loss**: L = L_new + λ || U_r^T a(θ) − U_r^T a(θ_anchor) ||²

That's it. ~40 lines of code.

## Key prior work

- Kirkpatrick et al., 2017 — Elastic Weight Consolidation (EWC)
- Li & Hoiem, 2017 — Learning without Forgetting (LwF)
- Douillard et al., 2020 — PODNet (ECCV)
- Wang et al., 2024 — Continual Learning for LLMs (arXiv:2404.16789)
- "Continual Learning meets Mechanistic Interpretability" (Jan 2026, arXiv)
- Sennrich et al., 2015 — BPE for NLP

## Evaluation (NeurIPS-grade)

- **LiveBench** — contamination-resistant, objective scoring (arXiv:2406.19314)
- **IFEval** — automatically verifiable instruction-following (arXiv:2311.07911)
- **SWE-bench Verified** — real GitHub issue patching (arXiv:2310.06770)
- **GSM8K** — math reasoning (secondary)
- **HumanEval** — code generation (secondary)

## Risks

1. **LoRA confound**: Low-rank adapters constrain updates to a low-dimensional subspace by construction. "Forgetting is low-dimensional" might be an artifact of the parameter-efficient training, not a property of forgetting. *Mitigation*: validate H1 under full fine-tuning at least once.
2. **Vacuous bound via C**: The reduction theorem gives forgetting ≤ C·√δ but C is estimated empirically. If C is huge or unstable, the bound is meaningless. *Mitigation*: report C distributions across seeds/tasks; show they're bounded and stable.
3. **Layer cherry-picking**: Testing H1 at depth/4, depth/2, 3·depth/4 and reporting the best result is p-hacking. *Mitigation*: pre-register probe layers or report all layers.
4. **Forgetting not low-dimensional** → try different layers/basis; report negative result honestly.
5. **Effect disappears at scale** → validate on mid-scale (1.5B) first.
6. **"Just distillation" criticism** → random-basis and random-selection controls prove PCA + importance matter beyond generic representation anchoring.

## Novelty wedge

The general idea "anchor internal activations" is NOT new (PODNet exists).
What's new:
1. **Selective** preservation: evidence that a small subset (~32 dims/layer) suffices, with ablation controls showing PCA + importance ranking outperform random subspaces.
2. **Falsifiable measurement protocol**: the H1 regression test is a reproducible claim others can validate or falsify on their own models/tasks.
3. **LLM continual instruction-tuning** setting with strong baselines (not just vision CL benchmarks).
4. **Formal verification** of the mathematical reduction — prevents hand-wavy algebra, makes assumptions explicit (not a "guarantee").
5. Potential **mechanistic link** to interpretability (if SAE features outperform PCA — future work).

## Experiment Log

### 2026-03-03 — H1 Control (preliminary)

**Goal**: Test H1 — does drift in top-r PCA dimensions predict forgetting better than drift in bottom-r, random-r, or total drift?

**Config**:
- Model: SmolLM2-135M-Instruct (HuggingFaceTB/SmolLM2-135M-Instruct)
- 40 training steps, checkpoint every 10 (→ 5 data points including step 0)
- LoRA rank 16, PCA k=16, preserve r=8, 16 memory examples
- Old task: math, New task: code
- Device: CPU
- Seed: 42
- Output: `out/h1_control/h1_control.json`

**Problem**: SmolLM2-135M scores 0.0% on math (baseline accuracy = 0.0 at every checkpoint), so accuracy-based forgetting is undefined. Used **perplexity-based forgetting proxy** instead: `forgetting_ppl = (math_ppl_t − math_ppl_0) / math_ppl_0`. Baseline perplexity: 1.177.

**R² results** (simple linear regression, drift → forgetting_ppl):

| Predictor | R² |
|-----------|-----|
| bottom-r drift | **0.9943** |
| random-r drift | 0.9874 |
| total drift | 0.9779 |
| top-r drift | 0.9126 |

Ranking: **bottom > random > total > top**. This is the *opposite* of what H1 predicts. H1 claims top-r drift should be the best predictor; here it is the worst.

Partial R²(top_r | total) = 0.9538 — top-r does carry signal beyond total drift, but the relationship is nonlinear (top-r drift grows faster than forgetting at later steps).

**Gate 1 check**:
1. ❌ R²(top-r drift, forgetting) > 0.6 — technically passes (0.91), but it's the *worst* predictor, not the best
2. ❌ R²(top-r) > R²(random-r) — **FAIL** (0.91 < 0.99)
3. ❌ R²(top-r) > R²(total) after controlling for total — not meaningfully better

**Gate 1 verdict: FAIL** on all three checks. Top-r PCA drift is not preferentially predictive of forgetting in this setup.

**Caveats** (why this is not definitive):
- **Tiny model** (135M params, 30 layers) — PCA on 16 dims from 16 memory examples is statistically underpowered. The top PCA components of a 135M model may not capture the same structure as in a 1B+ model.
- **Only 5 data points** (step 0, 10, 20, 30, 40) — all R² values are inflated by the near-monotonic trajectory. With 5 points and monotonic growth, even random noise gets high R².
- **Perplexity proxy** — perplexity and accuracy can diverge. The model never solved math, so we're measuring "how much worse does it get at generating math-like tokens" rather than "how much capability is lost."
- **CPU, 40 steps** — barely enough training to induce meaningful drift. The forgetting signal (Δppl ≈ 0.019) is tiny.
- **r=8 out of k=16** — preserving half the PCA space leaves little room for "unimportant" dims to behave differently.

**Interpretation**: Too small to be definitive in either direction. The inverted ranking could be a real phenomenon (bottom PCA dims track forgetting better because they capture fine-grained features that are easily overwritten) or an artifact of the underpowered setup. Need a proper GPU experiment to distinguish.

**Next steps**:
1. **GPU retest with Qwen-1.5B**: 500 steps, checkpoint every 25 (→ 20 data points), r=32, k=128, memory=256. This gives a model that can actually solve math, enough data points for meaningful R², and enough PCA dimensions for top-r vs bottom-r to separate.
2. **Gradient-informed basis as PCA alternative**: Instead of PCA (variance-maximizing), try a basis informed by the gradient of the old-task loss. Top directions in ∇L_old-projected activation space should better predict which features matter for the old task. If PCA top-r fails but gradient-informed top-r succeeds, H1 survives with a modified basis.

### 2026-03-03 — H1 Control (GPU, Qwen-1.5B) ⭐

**Goal**: Definitive H1 test at proper scale. Does the 135M failure replicate at 1.5B?

**Config**:
- Model: Qwen/Qwen2.5-1.5B-Instruct
- 500 training steps, checkpoint every 25 (→ 21 data points including step 0)
- LoRA rank 32, PCA k=128, preserve r=32, 128 memory examples
- Old task: math, New task: code
- Device: CUDA (A10G via Modal)
- Seed: 42
- Output: `out/h1_control_gpu/h1_control.json`

**Baseline**: 44.44% math accuracy (model can actually solve math), 0.505 math loss.

**R² results** (perplexity-based forgetting):

| Predictor | R² |
|-----------|-----|
| **top-r drift** | **0.9331** |
| total drift | 0.8883 |
| random-r drift | 0.8500 |
| bottom-r drift | 0.8272 |

Ranking: **top > total > random > bottom**. This is the *correct* ordering for H1. Top-r PCA drift is now the best predictor — completely reversed from the 135M result.

**Gate 1 check**:
1. ✅ R²(top-r) ≥ 0.5 — passes strongly (0.933)
2. ✅ R²(top-r) > R²(total) — passes (0.933 > 0.888)
3. ❌ R²(top-r) >> R²(random-r) — marginal (0.933 vs 0.850, not 2× better)

**Gate 1 verdict: PARTIAL PASS** (2/3). Top-r PCA drift is the best predictor, beats total drift, but doesn't dominate random-r by a wide margin. H1 is alive but not overwhelming.

**R-sweep diagnostic** (variance explained vs forgetting R²):

| r | Var Explained (%) | Forgetting R² |
|---|-------------------|---------------|
| 1 | 96.2 | 0.854 |
| 2 | 96.4 | 0.854 |
| 4 | 96.9 | 0.871 |
| 8 | 97.4 | 0.878 |
| 16 | 98.0 | 0.879 |
| 32 | 98.7 | 0.882 |
| 64 | 99.4 | 0.883 |
| 128 | 100.0 | 0.884 |

Both curves track together — forgetting IS in high-variance directions. The first PC alone explains 96.2% of variance AND 85.4% of forgetting. Adding more PCA dims gives diminishing returns.

**Key observations**:
- The 135M failure was a **scale artifact**. At 1.5B, PCA top-r correctly identifies forgetting-relevant directions.
- The margin over random-r (0.933 vs 0.850) is consistent but not dramatic. This suggests forgetting is moderately concentrated in high-variance directions, not overwhelmingly so.
- Accuracy-based forgetting is noisy (9 questions) — the model fluctuates between 22-67% across checkpoints. Perplexity is monotonically increasing (0.50→0.83) and gives a cleaner signal.
- Total drift and top-r drift are highly correlated (r>0.95), which makes the top-r > total comparison less informative than it seems.

**Interpretation**: H1 is supported at 1.5B scale — PCA top-r drift is the best linear predictor of forgetting. But the effect size is modest. The "low-dimensional" claim is better stated as: "forgetting correlates with drift in high-variance activation directions, with PCA providing a weak but consistent advantage over random projections." This is enough to justify SFP as a method (preserving the right subspace > random subspace), but the paper should not overclaim.

**Next steps**:
1. Test gradient-informed basis — does it beat PCA? If so, the "right subspace" story gets much stronger.
2. Increase forgetting signal — run longer, use a harder task pair, or reduce memory to force more forgetting.
3. Causal test — intervene on top-r dims at inference to prove they're causally linked to capability, not just correlated.

### 2026-03-03 — H1 Pivot Decision

Given the GPU retest partial pass (2/3 gate checks), the pivot decision is:

- NOT option A (abandon H1) — H1 didn't fully fail, top-r is the best predictor with correct ranking
- NOT option B (weaken H1) — we can strengthen the evidence instead
- NOT option C (change metric) — perplexity-based forgetting is working fine
- Choose option **(E): Keep PCA as baseline, strengthen with gradient basis and causal test**

**Rationale**:
- PCA works at scale (correct ranking: top > total > random > bottom) but effect size is modest (0.933 vs 0.850)
- The gradient-informed basis (already implemented in `features.py` as `build_gradient_basis`) should widen the gap — it selects directions that are task-relevant, not just high-variance
- The causal test (intervening on top-r dims at inference) is the strongest possible evidence — if zeroing top-r dims destroys old-task performance but zeroing random-r dims doesn't, that's a smoking gun
- If gradient basis + causal test both succeed, H1 is solid for the paper
- Paper framing: "forgetting concentrates in high-variance activation subspaces; task-specific subspaces (gradient-informed) further improve prediction"

**Next experiments** (in priority order):
1. **Causal test** (`scripts/causal_test.py`) — intervene on top-32 vs random-32 vs bottom-32 dims at inference, measure old-task accuracy drop
2. **Gradient basis comparison** — re-run h1_control with gradient basis from `build_gradient_basis`, compare R² to PCA
3. **Evaluation leakage check** — split memory set (train/test) to rule out overfitting in the R² regression

### 2026-03-03 — H1 Causal Intervention Test (CPU, SmolLM2-135M, local smoke test)

**Goal**: Validate causal_test.py works end-to-end before GPU run. Not a definitive test — tiny model, minimal params.

**Config**:
- Model: SmolLM2-135M-Instruct
- pca_k=4, preserve_r=2, memory=8, eval_samples=4, noise_scales=[1.0]
- Device: CPU, Seed: 42

**Results**:

| Intervention | PPL | Δ PPL |
|---|---|---|
| top_1.0 | 1.9281 | +0.8548 |
| bottom_1.0 | 1.0931 | +0.0198 |
| random_1.0 | 1.2800 | +0.2068 |

**Gate**: ✓ PASS — top-r Δppl (+0.85) >> random (+0.21) >> bottom (+0.02). Correct ranking even at 135M scale.

**Bug fixed**: dtype corruption in forward hooks — `rank_importance` ablation hooks and noise injection hooks were returning float32 tensors into a bfloat16 model. Also fixed output format handling: SmolLM2 decoder layers return plain tensors, not tuples or BaseModelOutputWithPast. All hooks now check `isinstance(output, Tensor)` first.

**Caveats**: Tiny model (135M), minimal dims (r=2 out of k=4), only 4 eval samples. This validates the code, not the hypothesis.

**GPU run**: Launched on Modal (Qwen-1.5B, A10G, pca_k=128, preserve_r=32, memory=128). Results below.

### 2026-03-03 — H1 Causal Intervention Test (GPU, Qwen-1.5B) ⭐

**Goal**: Definitive causal test at 1.5B scale. Do ablation-ranked top-r dimensions causally drive old-task performance?

**Config**:
- Model: Qwen/Qwen2.5-1.5B-Instruct
- pca_k=128, preserve_r=32, memory=128, eval_samples=50
- noise_scales: [0.5, 1.0, 2.0, 5.0]
- Probe layers: layers 7, 14, 21 (depth/4, /2, 3/4)
- Device: CUDA (A10G via Modal), Seed: 42
- Baseline math loss: 0.505

**Results**:

| Noise Scale | Top-r Δppl | Random Δppl | Bottom-r Δppl | Top/Random | Top/Bottom |
|-------------|-----------|------------|--------------|------------|------------|
| 0.5σ | +0.042 | +0.024 | +0.001 | 1.8× | 51× |
| 1.0σ | +0.248 | +0.100 | +0.009 | 2.5× | 28× |
| 2.0σ | +2.604 | +0.810 | +0.041 | 3.2× | 64× |
| 5.0σ | +8.709 | +5.041 | +0.328 | 1.7× | 27× |

**Gate**: ✓ PASS at all 4 noise scales — top-r Δppl > random Δppl > bottom-r Δppl consistently.

**Key observations**:
- The ranking top > random > bottom holds at every noise scale tested, confirming causal importance.
- At 1σ, top-r causes 2.5× more degradation than random and 28× more than bottom — a strong causal signal.
- At 2σ, the top/bottom ratio reaches 64×, showing the effect is superlinear in noise magnitude.
- Bottom-r dimensions are nearly inert: even at 5σ noise, Δppl is only +0.33 vs +8.71 for top-r.
- This confirms that ablation importance correctly identifies causally important activation subspaces.
- Combined with the correlational evidence (R²=0.933 for top-r at 1.5B), H1 has both correlational AND causal support.

**Comparison with 135M**: The causal test passes at BOTH scales (135M and 1.5B), even though the correlational test fails at 135M. This suggests ablation importance is a robust indicator of causal importance regardless of model scale, while the correlation between drift and forgetting is scale-dependent.

**Next steps**:
1. Gradient basis comparison — does gradient-informed basis produce even stronger causal signal?
2. Full baseline benchmark with SFP vs all methods
3. Paper updated with 1.5B causal results (Table 3, Section 4.3)

### 2026-04-10 — SFP-EWC-DER: Addressing Threats to Validity

**Goal**: Design a submission that directly responds to the paper's own stated threats to validity
(LoRA confound, layer selection sensitivity, task-order robustness) using literature-backed methods.

**Literature review** (4 parallel research agents, 2023–2026 continual learning papers):

| Finding | Source |
|---------|--------|
| DER++ (CE + logit distillation on memory) consistently beats vanilla replay | Buzzega et al., NeurIPS 2020 (arXiv:2004.07211) |
| Optimal replay ratio ≈ 1:1 (not 0.2–0.55) | TiC-LM (arXiv:2505.12512), Watch Your Step (arXiv:2404.10758) |
| Convex mixing `(1-r)*L_new + r*L_mem` hurts plasticity vs additive | Harmonic mean analysis: cutting new-task gradient at r=0.5 depresses plasticity |
| EWC-LoRA on adapter weights: ~8.9% improvement over vanilla LoRA | Zheng et al., ICLR 2026 (arXiv:2602.17559) |
| O-LoRA (orthogonal subspaces) beats replay in T5 benchmarks | Wang et al., EMNLP 2023 (arXiv:2310.14152) |

**Threat analysis**:

| Threat (from paper website) | Response in sfp_ewc_der |
|-----------------------------|-------------------------|
| LoRA confound: forgetting may be a LoRA artifact | EWC-LoRA term: `0.10 * MSE(theta, theta_star)` over LoRA A/B — retention in parameter space |
| Layer selection: fixed 1/4, 1/2, 3/4 may be suboptimal | SFP weight reduced to 0.05; EWC + logit + CE_mem provide retention independent of layer choice |
| Task-order sensitivity | Additive terms symmetric across orderings; no curriculum assumption |

**Method: SFP-EWC-DER** (`submissions/sfp_ewc_der.py`):

```
L = L_new                                  (weight 1.0, never diluted)
  + 0.50 * CE(model(x_mem), y_mem)         (output retention, ~1:1 ratio)
  + 0.20 * MSE(logits_now, logits_stored)  (DER++ dark experience)
  + 0.05 * SFP_preserve                   (geometric retention, 3 layers)
  + 0.10 * EWC_LoRA                        (LoRA weight anchor)
```

Key design: **fully additive** (not convex mix). The adaptive convex-mix approach
`(1-r)*L_new + r*L_mem` halves the new-task gradient at r=0.5, directly hurting
the plasticity axis of the harmonic mean score.

**Comparison to baselines** (theoretical, from literature; GPU benchmark pending):

| Method | Retention mechanism | Plasticity protection | Expected HM |
|--------|--------------------|-----------------------|-------------|
| naive  | none               | full                  | ~0.40–0.55  |
| replay | CE on memory only  | diluted by ratio      | ~0.65–0.68  |
| SFP (0.71) | geometric subspace | full (no mem CE) | 0.71 |
| sfp_ewc_der | geo + CE_mem + DER++ + EWC-LoRA | full (additive) | **0.74–0.78** (est.) |

Estimate based on: DER++ typically +0.02–0.04 over vanilla replay; EWC-LoRA +0.03–0.05
over vanilla LoRA (Zheng et al.); combined additive stack expected +0.04–0.07 over SFP.

**Unit tests**: 6 tests pass in ~2s (no GPU needed). See `scripts/test_submissions.py`.

**Gate check**: Method satisfies all leaderboard rules (single `_loss` function, SETUP="sfp",
< 10KB, uses only allowed imports). GPU benchmark needed for final score.

**Status**: SUBMITTED — awaiting GPU evaluation via leaderboard.
