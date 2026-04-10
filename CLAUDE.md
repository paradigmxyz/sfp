# SFP Formal Improvements

## Overview

Fork of [paradigmxyz/sfp](https://github.com/paradigmxyz/sfp) — Sparse Feature Preservation for continual LLM fine-tuning. We formalize tighter mathematical foundations via Aristotle (Lean 4) and implement theory-derived algorithmic improvements.

**Spec:** `../autoresearch-trading/docs/superpowers/specs/2026-04-09-sfp-formal-improvements-design.md`
**Plan:** `../autoresearch-trading/docs/superpowers/plans/2026-04-09-sfp-formal-improvements.md`

## Stack

Python 3.12+, PyTorch, transformers, peft, Aristotle CLI (Lean 4), uv

## Key Files

- `methods.py` — all loss functions + setup functions + METHODS registry
- `features.py` — PCA basis, gradient basis, importance ranking, select_top_r
- `model.py` — model loading, LoRA config, checkpointing
- `train.py` — main training loop, reads METHODS dict
- `evaluate.py` — leaderboard scoring (harmonic mean of retention x plasticity)
- `lean-proofs/` — our Aristotle theorem inputs and Lean 4 results
- `submissions/` — leaderboard submission files
- `scripts/estimate_C.py` — empirical forgetting constant estimation

## Upstream SFP Architecture

Loss: `L = L_new + lambda * sum_l ||U_r^T a_l(theta;x) - U_r^T a_l(theta*;x)||^2`

Setup at task boundary (`sfp_setup`):
1. Collect activations from memory buffer at fixed layers [0.25, 0.5, 0.75] relative depth
2. PCA via SVD -> top-k basis U [hidden_dim, k]
3. select_top_r by explained variance -> U_r [hidden_dim, r]
4. Cache anchor projections: anchor_acts[layer] = activations @ U_r

Our improvement: replace fixed layer selection with C_l-based selection (smallest forgetting constant = tightest bound).

## Aristotle Workflow

1. Write `lean-proofs/inputs/TN-name.txt`
2. `aristotle formalize lean-proofs/inputs/TN-name.txt` -> UUID
3. `aristotle list` -> wait for COMPLETE (~30-60 min)
4. `aristotle result <UUID> --destination lean-proofs/results/TN-name.tar.gz`
5. Extract, verify 0 sorry, update ROADMAP.md

## Submission Format

- Single Python file in `submissions/`
- Exactly one function ending in `_loss`
- `SETUP` attribute: one of none/distill/hidden_distill/orthogonal/sfp
- Allowed imports: torch, torch.nn.functional, copy, math
- Max 10KB, 120s setup budget
- Submit: `python submit.py submissions/information_theoretic_sfp.py`

## Running

```bash
uv run python forget.py                              # ~5 min CPU demo
uv run python train.py --method sfp --memory 128      # GPU training
uv run python scripts/estimate_C.py                   # empirical C analysis
```

## Conventions

- Commit style: `feat:`, `fix:`, `chore:`, `theorem:`, `experiment:`
- Branch: `formal-improvements` (from master which tracks upstream)
- Only stage specific files, never `git add -A`

## Gotchas

1. `sfp_setup` uses PCA explained variance as importance proxy — NOT the ablation-based `rank_importance` function
2. All source files at repo root — no `sfp/` subdirectory
3. Setup budget is 120s wall-clock — `estimate_forgetting_constant()` must be fast
4. Leaderboard uses harmonic mean score but live site may show old linear formula (0.6R + 0.4P)
5. `memory` config default is 0 (disabled) — must set `--memory 128` for SFP methods
6. Run all commands with `uv run` to use the project venv
