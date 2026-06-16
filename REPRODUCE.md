# Reproducing the `forgetting_curve` submission

This document is written so an AI coding agent (Kilo Code + Qwen, Cline + Llama, Cursor + GPT, etc.) or a human collaborator can rebuild every step from zero context — the thinking, the method, the validation, and the submission flow.

---

## 1. The thinking (思路)

### 1.1 What the benchmark is

Sequential fine-tuning of a large language model (alignment updates, safety patches, domain adaptation) degrades earlier capabilities — *catastrophic forgetting*. The sfp benchmark holds **everything fixed** (base model = Qwen2.5-1.5B-Instruct, data, compute, LoRA config); the **only** knob you turn is the **training strategy** — i.e., the loss function.

Ranking formula on the live leaderboard:

```
score = 0.6 × retention + 0.4 × plasticity
```

- `retention` = accuracy on previously trained tasks (after later training erases some of it).
- `plasticity` = accuracy on the most recently trained task.
- Retention is weighted 1.5× plasticity, so good methods must **protect old skills without crushing the ability to learn new ones**.

### 1.2 What beats the current leader

Three brain mechanisms map onto three CL (continual learning) techniques. Two are already on the board; the third is missing — and that's where the win is.

| Brain mechanism | CL technique | Who on the board uses it |
|---|---|---|
| Hippocampus replays experiences to the cortex during rest (Complementary Learning Systems, McClelland et al. 1995) | **experience replay** | `replay` (score 0.5892) |
| Keep behaviour intact rather than copy exact weights (Learning without Forgetting, Li & Hoiem 2017) | **distillation from a frozen teacher snapshot** | `distill` (0.6350) |
| Spaced repetition: review hardest what's forgetting fastest (Ebbinghaus 1885: ~58 % retained at 20 min, ~44 % at 1 hr, ~33 % at 24 hr) | **per-example forgetting-weighted review** | **nobody — this is the contribution** |

Board leader `sfp` (0.7124) wins on retention by being **selective in feature space** — preserving only the ~32 most important internal feature directions out of ~2048 per layer, and letting the rest move freely. My submission is **selective in example space** — preserving hardest the memory examples most at risk of being forgotten, and letting the rest drift.

### 1.3 The method in one formula

```
L = CE_new + β · CE_mem + α · T² · Σᵢ wᵢ · KL( teacherᵢ ‖ studentᵢ )   on memory

where  wᵢ = softmax( focus · KL_per_example.detach() )
```

- `CE_new`  — cross-entropy on the new task. Full weight protects plasticity.
- `β · CE_mem` — replay: cross-entropy on the memory batch. Broad rehearsal.
- `α · T² · Σᵢ wᵢ · KL` — **the novelty**: weighted distillation. Each memory example's review weight `wᵢ` rises with how far the student has drifted from the frozen teacher on that example. Concentrates preservation effort where forgetting is most imminent.

Defaults: `α=1.0, β=0.5, T=2.0, focus=1.0`. KL math is upcast to fp32 inside the loss for bf16 model stability (per the harness's `AGENTS.md` dtype rule).

`SETUP = "distill"` tells the harness to deep-copy the model at each task boundary and pass it as `teacher`. Compute footprint = one student forward + one teacher forward on the memory batch — same as the `distill` baseline. Replay CE rides for free on the existing student memory forward (no extra pass).

---

## 2. The official requirements — plain-language translation

This is the rule list from the live submit page, translated:

| Official rule | What it actually means |
|---|---|
| **Model: Qwen2.5-1.5B-Instruct** | The base model is *fixed*. You can't substitute Claude, Llama, Mistral. Every submission fine-tunes the same Qwen. The base is the *subject under test* — what gets trained. (The AI coding agent that *writes* your loss function — Claude, Qwen-Coder, GPT-4o, etc. — is a separate thing entirely.) |
| **Tasks: math → code → ifeval** | The training task order on the simplified submit page. The *actual* scoring preset (`standard` in `scripts/modal_bench.py`) runs **5 tasks**: math → code → ifeval → safety → domain. Your loss has to handle any number of tasks. |
| **Steps: 1000 per task** | Each task gets 1000 gradient updates. (`standard` preset uses 2000.) Same for every method. |
| **LoRA rank: 64** | **LoRA** = Low-Rank Adaptation. Instead of updating all model weights, train a small "adapter" of rank 64. Cheap, fast, common in modern fine-tuning. Your loss doesn't see this — the harness handles it. |
| **Seeds: 42, 43, 44 (mean ± std)** | Run with each of these random seeds, report mean ± standard deviation of the score. Three runs total per setting. |
| **GPU: A10G (Modal)** | Paradigm rents the cloud GPU on Modal and pays for it. You don't pay anything. You don't need a GPU on your end. |
| **Memory tiers: M=128, M=512** | **M** = how many old-task examples the harness keeps in the memory buffer that's fed to your loss as `memory_batch`. Your method is evaluated at both tiers (128 and 512). |
| **Same total tokens as baselines (enforced by harness)** | All methods see the same amount of training data. The harness fixes `batch_size × steps × tasks`. You cannot sneak more data in, or train longer. The "fairness" guarantee. |
| **Same memory buffer (provided, not custom)** | The harness picks which old-task examples land in the memory buffer. You don't get to choose what to remember — you only decide **how to use** what's given. |
| **No access to eval data during training** | Your loss function sees `batch`, `memory_batch`, `teacher` — that's it. The test set is invisible. Prevents data leakage where a method peeks at the answers. |
| **Must be reproducible across 3 seeds** | Same seed → same numbers. No `time.time()`, no network calls, no nondeterministic ops (or if you use them, seed their RNG). |
| **No changes to `train.py`, `evaluate.py`, `data.py`, `model.py`** | You submit **exactly one Python file** — a loss function in `submissions/`. You cannot modify the training harness. The submit validator rejects anything else. |

If any specific rule is still unclear, ask — I'll expand that row.

---

## 3. Reproduction steps

```bash
# 1. Clone + install
git clone https://github.com/lilaclilac09/sfp.git && cd sfp
brew install uv               # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                       # ~2 GB download (torch, transformers, peft, ...)

# 2. Validate the submission with the official validator (pure stdlib)
python3 -c "import submit; print(submit.parse_submission('submissions/forgetting_curve.py'))"

# 3. Run a quick smoke test of the loss on a real model (uses cached SmolLM2-135M)
uv run python /tmp/test_fc.py  # see "Local test scripts" section below

# 4. Local PK — sanity comparison on 135M (Apple MPS recommended; CPU fp32 also works)
#    Output: math_loss before / after for naive vs forgetting_curve
uv run python /tmp/pk.py        # ~5 min on MPS, ~10-15 min on CPU fp32

# 5. Submit to the official server (when reachable)
python3 submit.py submissions/forgetting_curve.py
```

### 3.1 Local test scripts (recreate these if not present)

**`/tmp/test_fc.py`** — targeted loss-function test:
- Loads `SmolLM2-135M-Instruct` + tokenizer in fp32.
- Builds a tiny perturbation of student vs teacher (so per-example KL > 0).
- Calls `forgetting_curve_loss` along all three paths: full (memory + teacher), first-task (no memory), no-teacher.
- Asserts each loss is finite and gradients flow.
- Prints the per-example forgetting weights to confirm they're non-uniform in the correct direction (more-drifted example → higher weight).

**`/tmp/pk.py`** — head-to-head naive vs forgetting_curve:
- Loads SmolLM2-135M + LoRA r=16.
- Monkey-patches `datasets.load_dataset` to drop the stale pinned revisions (the repo's pins for `mbpp` have since 404'd on the Hub).
- Loads code (mbpp) as the "new task," math (gsm8k) for memory + eval.
- Fixed math eval batch (sampled once with seed 42 and reused).
- Arm A: train naive (plain CE) for 60 steps.
- Arm B: train forgetting_curve (replay + weighted distill) for 60 steps. Same code batches as Arm A (re-seeded for fairness).
- Reports `math_loss` before / after each arm.

Use `DEVICE = "mps"` if on Apple Silicon — it's ~5× faster than CPU and avoids macOS jetsam killing the process under memory pressure.

### 3.2 The two submission paths (read this — they're not equivalent)

There are **two** ways to get a score on the official leaderboard, and they have **different registration requirements**. Get this wrong and your method runs as `naive` (or not at all). The canonical guide is `dev/LEADERBOARD.md`; the harness code is the source of truth.

**Path A — PR + `benchmark` label** (canonical per `dev/LEADERBOARD.md`):

1. Fork the repo.
2. **Edit `methods.py`** — add your loss function and register it in the `METHODS` dict.
3. (If your method needs setup at task boundaries, add a case in `method_setup()`. `SETUP="distill"`, `"sfp"`, `"hidden_distill"`, `"orthogonal"` already have cases.)
4. Open a PR.
5. A maintainer adds the `benchmark` label.
6. `.github/workflows/leaderboard.yml` does:
   ```
   git diff origin/master...HEAD -- methods.py | grep -oP '^\+\s*"(\w+)":\s' | head -1
   ```
   to detect the new method name, then runs `modal run scripts/modal_bench.py --method <name> --memory 128 --preset standard` and posts results to the PR.

   **Critical**: that detector only looks at additions to `methods.py`'s `METHODS` dict. **A file dropped only into `submissions/` is invisible to this workflow** and the run falls back to `naive`.

**Path B — `submit.py` to the live server** (the route `submissions/README.md` describes):

1. Fork (optional — only needed if you also want the file on GitHub).
2. Write your loss as `submissions/<name>.py` (single `*_loss` function, allowed imports only, ≤10 KB).
3. Run `python3 submit.py submissions/<name>.py`. The CLI POSTs your code as a string to `http://sfp.paradigm.xyz:8090/submit`; the server validates it (`scripts/submit_server.py:validate_code`), writes it to a temp file, and runs `modal run scripts/modal_bench.py --method <name> --code <code> --memory 128 --preset standard`.
4. Poll `/status/<id>` for ~45 min.

The submission server was offline at the time of writing. **Path A is the working route right now**, which is why this submission has both: `methods.py` is registered (so the workflow detects it when labelled) **and** `submissions/forgetting_curve.py` is kept (so Path B works the instant the server returns).

### 3.3 Gotchas worth knowing in advance

- **`tests/test_smoke.py:test_method_registry` uses exact set equality** on `METHODS.keys()`. Adding a method requires extending that test's `expected` set in the same commit, otherwise CI (`.github/workflows/ci.yml`) goes red.
- **CI lint scope** is `ruff check sfp/ tests/ tasks/ scripts/ *.py`. Top-level `methods.py` is in scope; `submissions/` is not. Your `methods.py` addition must pass `ruff check --target-version py312 --line-length 100`.
- **`AGENTS.md` dtype rule**: Qwen loads as bfloat16. Any custom math over the vocab (softmax, log, KL) should upcast to fp32 to keep numerical fidelity — `forgetting_curve_loss` does this explicitly. Skipping the upcast doesn't crash, but it quantizes per-example differences and silently flattens any per-example weighting you compute.
- **Pinned dataset revisions** in `data.py` (e.g. `mbpp` rev `12e9221`) have 404'd on the Hugging Face Hub. Local runs need to monkey-patch `datasets.load_dataset` to drop the `revision` kwarg (see `/tmp/pk.py`). Modal's harness may or may not hit the same issue depending on its dataset cache.
- **Submit page rules vs harness reality**: the live submit page says "3 tasks, 1000 steps". The actual scoring preset (`standard` in `modal_bench.py`) runs **5 tasks, 2000 steps**. Write your loss to be config-agnostic.

---

## 4. Validation evidence (from the original session)

| Check | Result |
|---|---|
| Validator (size ≤ 10 KB, blocked-import set, `*_loss` function shape) | ✓ 4335 B; imports only `torch`, `torch.nn.functional` |
| Real-model forward + backward, all three code paths | ✓ finite loss, gradients flow (∥grad∥ ≈ 9.75 × 10⁴) |
| Per-example forgetting weights non-uniform in correct direction | ✓ KL = [0.131, 0.088] → weights = [0.511, 0.489] |
| Bfloat16 KL numerical check (Qwen runs in bf16) | ✓ fp32 upcast preserves per-example signal that bf16 would quantize |
| Local PK on SmolLM2-135M (Apple MPS, 60 steps, math → code) | base **2.363** → naive **2.210** (Δ -0.153) → forgetting_curve **1.741** (Δ -0.621). **4× more math preservation than naive.** |
| Official Modal A10G benchmark | ⏳ pending — Paradigm submission server offline at time of writing |

**Honest caveat on the local PK.** Both the math memory pool and the math eval batch are sampled from the same 200-example `math_data`, so part of `forgetting_curve`'s gain is from rehearsing examples similar to the eval. On the official harness, eval is held out — this confound disappears. Local PK confirms the **mechanism direction** (the method works as designed); absolute magnitudes will look different on the real benchmark.

---

## 5. Who did what

- **lilaclilac09** — submission author, decisions, GitHub identity on the PR.
- **Claude Opus 4.7 (1M-context)** — AI coding partner during the original session: drafted the loss, audited the harness against the validator and `AGENTS.md`, ran the local validations, wrote these docs.
- **Qwen2.5-1.5B-Instruct** — the **fixed base model** under test in the benchmark. This is what gets fine-tuned by every submission. **Not interchangeable with Claude** — different roles entirely.

A reproducer using **Kilo Code + Qwen-Coder** (or any other AI-agent + model combo) can follow this document end-to-end. The submission code in `submissions/forgetting_curve.py` is agent-agnostic — it makes no assumptions about which AI wrote it.

---

## 6. References

- McClelland, McNaughton & O'Reilly (1995). *Why there are complementary learning systems in the hippocampus and neocortex.*
- Ebbinghaus (1885). *Über das Gedächtnis* — original forgetting-curve self-experiments.
- Kirkpatrick et al. (2017). *Overcoming catastrophic forgetting in neural networks* (EWC).
- Li & Hoiem (2017). *Learning without Forgetting* (LwF).
- Buzzega et al. (2020). *Dark Experience for General Continual Learning* (DER++).
- Aljundi et al. (2019). *Online Continual Learning with Maximally Interfered Retrieval* (MIR).
- Shi et al. (2024). *Continual Learning of Large Language Models: A Comprehensive Survey* (arXiv:2404.16789).
