import Mathlib

noncomputable section

open LinearMap Submodule

set_option maxHeartbeats 800000

/-! # T12: LoRA Confound Conditions

This file formalizes the mathematical claims from T12 about when LoRA
(Low-Rank Adaptation) fine-tuning creates confounding artifacts in the
SFP (Subspace Feature Preservation) framework.

## Setup

In LoRA fine-tuning, a weight matrix W is modified as W' = W₀ + B·A where
A ∈ ℝ^{r_lora × d_in} and B ∈ ℝ^{d_out × r_lora}. The activation drift is
Δa(x) = B(Ax), which lies in im(B).

## Main Results

* `drift_mem_range_B` — The LoRA activation drift lies in im(B) for all inputs.
* `drift_subspace_finrank_le` — dim(im(B)) ≤ r_lora (drift dimension bound).
* `drift_in_SFP_of_range_le` — If im(B) ⊆ W_SFP, drift lies in W_SFP.
* `orthogonal_drift_eq_zero_of_range_le` — If im(B) ⊆ W_SFP, projection onto W_SFP⊥ is zero.
* `proj_SFP_eq_self_of_range_le` — If im(B) ⊆ W_SFP, projection onto W_SFP fixes drift.
* `confound_orthogonal_bound` — Near-containment implies small orthogonal drift.
* `confound_drift_bound` — Confound condition applied to LoRA drift.
* `full_finetune_range_top` — Full fine-tuning yields unconstrained drift.
* `out_of_SFP_finrank_bound` — Lower bound on dim(im(B) ∩ W_SFP⊥).
* `out_of_SFP_sufficient_when_double_rank` — When r_lora ≥ 2·r_sfp, enough out-of-subspace drift exists.
-/

/-- Euclidean space of dimension n, used for all vector spaces in the LoRA setup. -/
abbrev EV (n : ℕ) := EuclideanSpace ℝ (Fin n)

variable {d_in d_out r_lora : ℕ}

/-- The LoRA activation drift: Δa(x) = B(A(x)).
    This captures the change in activations due to the low-rank update B·A. -/
def loRA_drift (A : EV d_in →ₗ[ℝ] EV r_lora) (B : EV r_lora →ₗ[ℝ] EV d_out) :
    EV d_in →ₗ[ℝ] EV d_out :=
  B ∘ₗ A

/-! ## Claim 1: Drift Dimension Bound

Under LoRA, the activation drift at any layer is confined to a subspace of
dimension at most r_lora. Specifically, Δa(x) = B(Ax) ∈ im(B), and
dim(im(B)) ≤ r_lora. -/

/-- The activation drift Δa(x) = B(Ax) lies in the image of B for all inputs x. -/
theorem drift_mem_range_B (A : EV d_in →ₗ[ℝ] EV r_lora) (B : EV r_lora →ₗ[ℝ] EV d_out)
    (x : EV d_in) : loRA_drift A B x ∈ LinearMap.range B :=
  ⟨A x, rfl⟩

/-
The dimension of the LoRA drift subspace im(B) is at most the LoRA rank r_lora.
    This is because im(B) is the image of a linear map from a space of dimension r_lora.
-/
theorem drift_subspace_finrank_le (B : EV r_lora →ₗ[ℝ] EV d_out) :
    Module.finrank ℝ (LinearMap.range B) ≤ r_lora := by
  convert LinearMap.finrank_range_le B
  exact Eq.symm finrank_euclideanSpace_fin

/-! ## Claim 2: H1 Trivially Holds When im(B) ⊆ W_SFP

If the LoRA image is contained in the SFP preservation subspace, then:
- P_{W_SFP}(Δa) = Δa (the full drift is captured)
- P_{W_SFP⊥}(Δa) = 0 (no drift in the orthogonal complement)

H1 then holds vacuously with any constant C > 0. -/

/-- If im(B) ⊆ W_SFP, then the LoRA drift lies in W_SFP for all inputs. -/
theorem drift_in_SFP_of_range_le (A : EV d_in →ₗ[ℝ] EV r_lora) (B : EV r_lora →ₗ[ℝ] EV d_out)
    (W_SFP : Submodule ℝ (EV d_out)) (h : LinearMap.range B ≤ W_SFP) (x : EV d_in) :
    loRA_drift A B x ∈ W_SFP :=
  h (drift_mem_range_B A B x)

/-
If im(B) ⊆ W_SFP, the orthogonal projection of drift onto W_SFP⊥ is zero.
    This means there is no "forgetting" component, so H1 holds vacuously.
-/
theorem orthogonal_drift_eq_zero_of_range_le
    (A : EV d_in →ₗ[ℝ] EV r_lora) (B : EV r_lora →ₗ[ℝ] EV d_out)
    (W_SFP : Submodule ℝ (EV d_out))
    (h : LinearMap.range B ≤ W_SFP) (x : EV d_in) :
    orthogonalProjection W_SFPᗮ (loRA_drift A B x) = 0 := by
  simp [loRA_drift];
  exact h <| LinearMap.mem_range_self _ _

/-
If im(B) ⊆ W_SFP, the orthogonal projection onto W_SFP fixes the drift:
    P_{W_SFP}(Δa) = Δa.
-/
theorem proj_SFP_eq_self_of_range_le
    (A : EV d_in →ₗ[ℝ] EV r_lora) (B : EV r_lora →ₗ[ℝ] EV d_out)
    (W_SFP : Submodule ℝ (EV d_out))
    (h : LinearMap.range B ≤ W_SFP) (x : EV d_in) :
    (orthogonalProjection W_SFP (loRA_drift A B x) : EV d_out) = loRA_drift A B x := by
  convert Submodule.starProjection_eq_self_iff.mpr ?_;
  · infer_instance;
  · exact h <| LinearMap.mem_range_self _ _

/-! ## Claim 3: Confound Condition

H1 measured under LoRA is uninformative when the LoRA subspace is nearly
contained in the SFP subspace. Formally, if for all v ∈ im(B):
  ‖P_{W_SFP⊥}(v)‖ ≤ ε · ‖v‖
then the orthogonal drift is negligible, and high R² is an artifact. -/

/-- If the projection onto W_SFP nearly fixes all vectors in a subspace W_lora
    (i.e., the orthogonal complement projection is small), then this bound
    transfers to any specific vector in W_lora.
    This formalizes the confound condition: when the LoRA subspace is nearly
    contained in the SFP subspace, ‖P_{W_SFP⊥}(Δa)‖ ≤ ε·‖Δa‖. -/
theorem confound_orthogonal_bound
    (W_SFP W_lora : Submodule ℝ (EV d_out))
    (ε : ℝ)
    (h_near : ∀ v : EV d_out, v ∈ W_lora →
      ‖(orthogonalProjection W_SFPᗮ v : EV d_out)‖ ≤ ε * ‖v‖)
    (v : EV d_out) (hv : v ∈ W_lora) :
    ‖(orthogonalProjection W_SFPᗮ v : EV d_out)‖ ≤ ε * ‖v‖ :=
  h_near v hv

/-- The confound condition applied to LoRA drift: if the LoRA image subspace
    is nearly contained in W_SFP, then the orthogonal drift is at most ε times
    the total drift for all inputs x. -/
theorem confound_drift_bound
    (A : EV d_in →ₗ[ℝ] EV r_lora) (B : EV r_lora →ₗ[ℝ] EV d_out)
    (W_SFP : Submodule ℝ (EV d_out))
    (ε : ℝ)
    (h_near : ∀ v : EV d_out, v ∈ LinearMap.range B →
      ‖(orthogonalProjection W_SFPᗮ v : EV d_out)‖ ≤ ε * ‖v‖)
    (x : EV d_in) :
    ‖(orthogonalProjection W_SFPᗮ (loRA_drift A B x) : EV d_out)‖ ≤
      ε * ‖loRA_drift A B x‖ :=
  h_near _ (drift_mem_range_B A B x)

/-
The operator norm characterization of the confound condition:
    if ‖P_{W_SFP⊥} ∘ P_{W_lora}‖ ≤ ε (as an operator norm), then for all v ∈ W_lora,
    ‖P_{W_SFP⊥}(v)‖ ≤ ε · ‖v‖. This connects the operator norm formulation
    to the pointwise bound.
-/
theorem confound_from_proj_norm
    (W_SFP W_lora : Submodule ℝ (EV d_out))
    (ε : ℝ)
    (h_op : ∀ w : EV d_out,
      ‖(orthogonalProjection W_SFPᗮ
        ((orthogonalProjection W_lora w : EV d_out)) : EV d_out)‖ ≤ ε * ‖w‖)
    (v : EV d_out) (hv : v ∈ W_lora) :
    ‖(orthogonalProjection W_SFPᗮ v : EV d_out)‖ ≤ ε * ‖v‖ := by
  convert h_op v;
  rw [ eq_comm ];
  convert Submodule.starProjection_eq_self_iff.mpr hv

/-! ## Claim 4: Confound Severity

The confound severity κ measures how much the LoRA subspace overlaps with the
SFP subspace. When κ ≈ 1, the confound is severe; when κ ≈ 0, the SFP
measurement is genuine.

For the confound to be absent, we need either:
(a) r_lora > r_sfp — LoRA rank exceeds SFP rank
(b) im(B) ⊥ W_SFP — LoRA modifies orthogonal directions
(c) Full fine-tuning — no rank constraint -/

/-
The LoRA confound is absent when the LoRA image is orthogonal to W_SFP:
    im(B) ⊥ W_SFP implies P_{W_SFP}(Δa) = 0, so SFP genuinely measures
    feature preservation.
-/
theorem no_confound_when_orthogonal
    (A : EV d_in →ₗ[ℝ] EV r_lora) (B : EV r_lora →ₗ[ℝ] EV d_out)
    (W_SFP : Submodule ℝ (EV d_out))
    (h_orth : LinearMap.range B ≤ W_SFPᗮ) (x : EV d_in) :
    (orthogonalProjection W_SFP (loRA_drift A B x) : EV d_out) = 0 := by
  have hmem : loRA_drift A B x ∈ W_SFPᗮ := h_orth (drift_mem_range_B A B x)
  exact (starProjection_apply_eq_zero_iff W_SFP).mpr hmem

/-! ## Claim 5: Full Fine-Tuning as Gold Standard

Under full fine-tuning (r_lora = d_out with surjective B), the drift Δa(x)
is unconstrained — it can be any vector in ℝ^{d_out}. If R² ≈ 1 under full
fine-tuning, H1 is genuinely validated. -/

/-- Under full fine-tuning, if B is surjective, then im(B) = ℝ^{d_out},
    meaning the drift is unconstrained. -/
theorem full_finetune_range_top (B : EV d_out →ₗ[ℝ] EV d_out)
    (h_surj : Function.Surjective B) :
    LinearMap.range B = ⊤ :=
  LinearMap.range_eq_top.mpr h_surj

/-- Under full fine-tuning (surjective B), every vector in the output space
    is achievable as a drift for some input. -/
theorem full_finetune_unconstrained (A : EV d_in →ₗ[ℝ] EV d_out)
    (B : EV d_out →ₗ[ℝ] EV d_out)
    (v : EV d_out) : v ∈ LinearMap.range (loRA_drift A B) ↔ v ∈ LinearMap.range (B ∘ₗ A) := by
  rfl

/-! ## Claim 6: Quantitative Threshold

The LoRA confound is negligible when r_lora ≥ 2 · r_sfp. The key fact is the
dimension lower bound: dim(im(B) ∩ W_SFP⊥) ≥ dim(im(B)) - dim(W_SFP).
When r_lora ≥ 2·r_sfp, this gives at least r_sfp dimensions of out-of-subspace
drift, sufficient to test H1 non-trivially. -/

/-
Dimension lower bound for the intersection of im(B) with W_SFP⊥.
    By the modular law: dim(V ∩ W) ≥ dim(V) + dim(W) - dim(ambient).
    Applied to V = range B and W = W_SFP⊥:
    dim(range B ∩ W_SFP⊥) ≥ finrank(range B) + finrank(W_SFP⊥) - d_out
                            = finrank(range B) + (d_out - finrank W_SFP) - d_out
                            = finrank(range B) - finrank(W_SFP)
-/
theorem out_of_SFP_finrank_bound
    (B : EV r_lora →ₗ[ℝ] EV d_out)
    (W_SFP : Submodule ℝ (EV d_out))
    (h_rank : Module.finrank ℝ W_SFP ≤ Module.finrank ℝ (LinearMap.range B)) :
    Module.finrank ℝ (LinearMap.range B) - Module.finrank ℝ W_SFP ≤
      Module.finrank ℝ ↥(LinearMap.range B ⊓ W_SFPᗮ) := by
  have := Submodule.finrank_sup_add_finrank_inf_eq ( B.range ) W_SFPᗮ;
  rw [ show Module.finrank ℝ W_SFPᗮ = d_out - Module.finrank ℝ W_SFP from ?_, add_comm ] at this;
  · have h_dim_sup : Module.finrank ℝ (↥(B.range ⊔ W_SFPᗮ)) ≤ d_out := by
      exact le_trans ( Submodule.finrank_le _ ) ( by simp );
    omega;
  · have := Submodule.finrank_add_finrank_orthogonal W_SFP;
    exact eq_tsub_of_add_eq ( by rw [ add_comm ] ; simp at this ⊢; exact this )

/-
When the LoRA rank is at least twice the SFP subspace dimension,
    there are at least r_sfp dimensions of drift outside the SFP subspace.
    This ensures the confound is negligible and H1 can be tested non-trivially.
-/
theorem out_of_SFP_sufficient_when_double_rank
    (B : EV r_lora →ₗ[ℝ] EV d_out)
    (W_SFP : Submodule ℝ (EV d_out))
    (r_sfp : ℕ)
    (h_sfp_dim : Module.finrank ℝ W_SFP = r_sfp)
    (h_lora_rank : Module.finrank ℝ (LinearMap.range B) ≥ 2 * r_sfp) :
    Module.finrank ℝ ↥(LinearMap.range B ⊓ W_SFPᗮ) ≥ r_sfp := by
  have := out_of_SFP_finrank_bound B W_SFP ( by linarith );
  omega

end