# Summary of changes for run 73ad895e-4182-4f98-aad7-0b5b56a17d0e
Successfully formalized all 6 claims from T12-lora-confound-conditions.txt in `RequestProject/T12LoRAConfound.lean`. The file builds cleanly with no warnings, no sorries, and all theorems use only standard axioms.

## Formalized Theorems (12 total)

**Claim 1 — Drift Dimension Bound:**
- `drift_mem_range_B`: The LoRA activation drift Δa(x) = B(Ax) lies in im(B) for all inputs x.
- `drift_subspace_finrank_le`: dim(im(B)) ≤ r_lora.

**Claim 2 — H1 Trivially Holds When im(B) ⊆ W_SFP:**
- `drift_in_SFP_of_range_le`: If im(B) ⊆ W_SFP, drift lies in W_SFP.
- `orthogonal_drift_eq_zero_of_range_le`: If im(B) ⊆ W_SFP, the projection of drift onto W_SFP⊥ is zero.
- `proj_SFP_eq_self_of_range_le`: If im(B) ⊆ W_SFP, projection onto W_SFP fixes the drift.

**Claim 3 — Confound Condition:**
- `confound_orthogonal_bound`: Near-containment hypothesis transfers to pointwise bound.
- `confound_drift_bound`: Confound condition applied to LoRA drift: ‖P_{W_SFP⊥}(Δa(x))‖ ≤ ε·‖Δa(x)‖.
- `confound_from_proj_norm`: Operator norm characterization connects ‖P_{SFP⊥} ∘ P_{lora}‖ ≤ ε to pointwise bounds.

**Claim 4 — Confound Severity / Absence Conditions:**
- `no_confound_when_orthogonal`: When im(B) ⊥ W_SFP, the projection P_{W_SFP}(Δa) = 0.

**Claim 5 — Full Fine-Tuning as Gold Standard:**
- `full_finetune_range_top`: If B is surjective, im(B) = ℝ^{d_out} (unconstrained drift).
- `full_finetune_unconstrained`: Definitional unfolding of loRA_drift as B ∘ A.

**Claim 6 — Quantitative Threshold:**
- `out_of_SFP_finrank_bound`: dim(im(B) ∩ W_SFP⊥) ≥ dim(im(B)) − dim(W_SFP).
- `out_of_SFP_sufficient_when_double_rank`: When dim(im(B)) ≥ 2·r_sfp and dim(W_SFP) = r_sfp, then dim(im(B) ∩ W_SFP⊥) ≥ r_sfp.

## Approach
The formalization uses `EuclideanSpace ℝ (Fin n)` for finite-dimensional inner product spaces, `LinearMap` for the LoRA matrices A and B, `Submodule` for subspaces, and `orthogonalProjection` for orthogonal projections. All dimension arguments use `Module.finrank`.