/-
# Theorem T2: Davis-Kahan Perturbation Bound for PCA Subspace Estimation

This file formalizes the chain of reasoning from the Davis-Kahan sin(θ) theorem
and sub-Gaussian concentration to minimum buffer size requirements for PCA-based
subspace estimation in neural network activations.

The two deep results — Davis-Kahan and sub-Gaussian covariance concentration —
are taken as hypotheses. All downstream consequences (probability bounds,
minimum sample sizes, preservation loss error) are proved formally.
-/

import Mathlib

open Real

set_option maxHeartbeats 800000

/-! ## Abstract framework

We work with real-valued parameters representing:
- `sin_theta`: the sine of the principal angle sin(θ(Ŵ, W))
- `op_err`: the operator norm error ‖Σ̂ - Σ‖_op
- `delta_k`: the eigenvalue gap Δ_k = λ_k - λ_{k+1}
- `sigma`: the sub-Gaussian parameter
- `n`: the sample size
- `c`: the universal concentration constant
-/

section DavisKahan

/-! ### Claim 1: Davis-Kahan sin(θ) theorem (abstract version)

The Davis-Kahan theorem states: if Δ_k > 0, then
  sin(θ(Ŵ, W)) ≤ ‖Σ̂ - Σ‖_op / Δ_k

We take this as a hypothesis and derive consequences.
-/

/-
From Davis-Kahan and a bound on operator norm error, derive a bound on sin(θ).
-/
theorem davis_kahan_applied
    (sin_theta op_err delta_k : ℝ)
    (h_dk : sin_theta ≤ op_err / delta_k)
    (h_op : op_err ≤ ε)
    (h_delta_pos : 0 < delta_k)
    (h_eps_pos : 0 ≤ ε) :
    sin_theta ≤ ε / delta_k := by
  exact h_dk.trans ( by gcongr )

end DavisKahan

section Concentration

/-! ### Claim 2: Combined probability bound

From sub-Gaussian concentration:
  P(‖Σ̂ - Σ‖_op > ε) ≤ 2 exp(-c · n · ε² / σ⁴)

Combined with Davis-Kahan (setting ε = t · Δ_k):
  P(sin(θ) > t) ≤ 2 exp(-c · n · (t · Δ_k)² / σ⁴)
-/

/-- The concentration exponent: c · n · (t · Δ_k)² / σ⁴ -/
noncomputable def concentration_exponent (c_const : ℝ) (n : ℕ) (t delta_k sigma : ℝ) : ℝ :=
  c_const * n * (t * delta_k) ^ 2 / sigma ^ 4

/-
Setting ε = t · Δ_k in the concentration bound transforms the exponent correctly.
-/
theorem concentration_exponent_eq (c_const : ℝ) (n : ℕ) (t delta_k sigma : ℝ) :
    concentration_exponent c_const n t delta_k sigma =
    c_const * n * t ^ 2 * delta_k ^ 2 / sigma ^ 4 := by
  unfold concentration_exponent; ring;

end Concentration

section BufferSize

/-! ### Claim 3: Minimum buffer size

For sin(θ) ≤ t with probability ≥ 1 - δ, we need:
  2 exp(-c · n · (t · Δ_k)² / σ⁴) ≤ δ

Equivalently:
  n ≥ σ⁴ · log(2/δ) / (c · Δ_k² · t²)
-/

/-- The minimum sample size formula: n_min = σ⁴ · log(2/δ) / (c · Δ_k² · t²) -/
noncomputable def n_min (sigma : ℝ) (delta_prob : ℝ) (c_const : ℝ) (delta_k : ℝ) (t : ℝ) : ℝ :=
  sigma ^ 4 * Real.log (2 / delta_prob) / (c_const * delta_k ^ 2 * t ^ 2)

/-
If n ≥ n_min, then the concentration exponent is at least log(2/δ).
-/
theorem buffer_size_sufficient
    (c_const : ℝ) (n : ℕ) (t delta_k sigma delta_prob : ℝ)
    (hc : 0 < c_const) (ht : 0 < t) (hdk : 0 < delta_k) (hsig : 0 < sigma)
    (hdelta : 0 < delta_prob)
    (hn : n_min sigma delta_prob c_const delta_k t ≤ (n : ℝ)) :
    Real.log (2 / delta_prob) ≤ concentration_exponent c_const n t delta_k sigma := by
  unfold n_min concentration_exponent at *;
  rw [ le_div_iff₀ ] at * <;> first | positivity | rw [ div_le_iff₀ ] at * <;> first | positivity | linarith;

/-
Specialization: when t = 0.1, the minimum sample size becomes
    100 · σ⁴ · log(2/δ) / (c · Δ_k²).
-/
theorem n_min_at_t_point_one (sigma delta_prob c_const delta_k : ℝ) :
    n_min sigma delta_prob c_const delta_k 0.1 =
    sigma ^ 4 * Real.log (2 / delta_prob) / (c_const * delta_k ^ 2 * 0.01) := by
  unfold n_min; norm_num

/-
Rewriting the t=0.1 case to show the factor of 100.
-/
theorem n_min_at_t_point_one' (sigma delta_prob c_const delta_k : ℝ)
    (hc : c_const ≠ 0) (hdk : delta_k ≠ 0) :
    n_min sigma delta_prob c_const delta_k 0.1 =
    100 * (sigma ^ 4 * Real.log (2 / delta_prob)) / (c_const * delta_k ^ 2) := by
  rw [ n_min ] ; ring

end BufferSize

section PreservationLoss

/-! ### Claim 4: Preservation loss error from subspace misestimation

If the true preservation loss is L_pres = ‖P_W(Δa)‖² and we use
L̂_pres = ‖P_Ŵ(Δa)‖², then:
  |L̂_pres - L_pres| ≤ 2 · sin(θ) · ‖Δa‖²

This follows from the projection perturbation bound:
  ‖P_Ŵ(x) - P_W(x)‖ ≤ sin(θ) · ‖x‖
-/

/-
From the projection perturbation bound, derive the loss error bound.

Given:
- projection_perturb: ‖P_Ŵ(Δa) - P_W(Δa)‖ ≤ sin(θ) · ‖Δa‖
- norms are non-negative

We prove: |‖P_Ŵ(Δa)‖² - ‖P_W(Δa)‖²| ≤ 2 · sin(θ) · ‖Δa‖²

Key identity: |a² - b²| = |a+b| · |a-b| ≤ (a+b) · |a-b|
Since a,b ≤ ‖Δa‖ (projections don't increase norm), a+b ≤ 2‖Δa‖
And |a-b| ≤ ‖P_Ŵ(Δa) - P_W(Δa)‖ (reverse triangle ineq) ≤ sin(θ)·‖Δa‖
-/
theorem preservation_loss_error
    (norm_proj_hat norm_proj norm_delta sin_theta : ℝ)
    (h_hat_nn : 0 ≤ norm_proj_hat) (h_proj_nn : 0 ≤ norm_proj)
    (h_delta_nn : 0 ≤ norm_delta)
    (h_sin_nn : 0 ≤ sin_theta)
    -- Projections don't increase norm
    (h_proj_hat_le : norm_proj_hat ≤ norm_delta)
    (h_proj_le : norm_proj ≤ norm_delta)
    -- Reverse triangle inequality for norms
    (h_norm_diff : |norm_proj_hat - norm_proj| ≤ sin_theta * norm_delta) :
    |norm_proj_hat ^ 2 - norm_proj ^ 2| ≤ 2 * sin_theta * norm_delta ^ 2 := by
  exact abs_le.mpr ⟨ by nlinarith [ abs_le.mp h_norm_diff ], by nlinarith [ abs_le.mp h_norm_diff ] ⟩

end PreservationLoss

section NumericalVerification

/-! ### Claim 5: Numerical verification

With parameters:
- d = 1536, k = 128, n = 128, σ = 1.0, δ = 0.05

We verify several concrete computational claims.
-/

/-
Claim 5a: For sin(θ) ≤ 0.1, need Δ_k ≥ 10 · σ² · √(log(40) / (c · 128)).

This comes from setting n = 128, t = 0.1 in n ≥ n_min and solving for Δ_k.
From n_min = σ⁴ · log(2/δ) / (c · Δ_k² · t²) ≤ 128:
  Δ_k² ≥ σ⁴ · log(40) / (c · 128 · 0.01) = 100 · σ⁴ · log(40) / (c · 128)
  Δ_k ≥ 10 · σ² · √(log(40) / (c · 128))
-/
theorem claim_5a_gap_requirement
    (c_const sigma delta_k : ℝ)
    (hc : 0 < c_const) (hsig : 0 < sigma)
    (hdk : 0 < delta_k)
    (hn : n_min sigma 0.05 c_const delta_k 0.1 ≤ 128) :
    delta_k ^ 2 ≥ sigma ^ 4 * Real.log (2 / 0.05) / (c_const * 128 * 0.01) := by
  unfold n_min at hn;
  rw [ ge_iff_le, div_le_iff₀ ] at * <;> first | positivity | nlinarith;

/-
Claim 5c: If the sin(θ) bound exceeds 0.3, the default buffer is insufficient.

This is a simple conditional statement: if the bound gives sin(θ) > 0.3,
then we cannot guarantee sin(θ) ≤ 0.1 (i.e., the buffer is too small).
-/
theorem claim_5c_insufficient_buffer
    (bound : ℝ) (h_bound : 0.3 < bound) : 0.1 < bound := by
  exact lt_trans ( by norm_num ) h_bound

/-
Claim 5d: Computing n_required for sin(θ) ≤ 0.1.

n_required = σ⁴ · log(2/δ) / (c · Δ_k² · 0.01)
            = 100 · σ⁴ · log(2/δ) / (c · Δ_k²)

With σ=1, δ=0.05: n_required = 100 · log(40) / (c · Δ_k²)
-/
theorem claim_5d_n_required_formula
    (c_const delta_k : ℝ) (hc : c_const ≠ 0) (hdk : delta_k ≠ 0) :
    n_min 1 0.05 c_const delta_k 0.1 =
    100 * Real.log (2 / 0.05) / (c_const * delta_k ^ 2) := by
  convert n_min_at_t_point_one' 1 ( 5e-2 ) c_const delta_k hc hdk using 1 ; ring

end NumericalVerification