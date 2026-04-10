import Mathlib

open Finset Real BigOperators

noncomputable section

namespace MultiLayerWeighting

/-! # T10: Multi-Layer Optimal Weighting

We formalize the optimal per-layer weighting for the SFP preservation loss.

## Main definitions

* `forgettingBound C s w` — the total forgetting bound `F(w) = ∑ Cᵢ √(sᵢ/(2wᵢ))`
* `optWeight C s Λ` — the Lagrange-optimal per-layer weight
* `unifWeight Λ` — the uniform weight `Λ/L` at each layer

## Main results

* `optWeight_sum` — optimal weights satisfy the budget constraint `∑ wᵢ* = Λ`
* `optWeight_minimizes` — optimal weights minimize the forgetting bound
* `forgettingBound_opt_le_unif` — `F(w*) ≤ F(w_uniform)`, optimal is at least as good as uniform
* `optWeight_uniform_signal` — when gradient signal is uniform, `wᵢ* ∝ Cᵢ^{2/3}`
* `optWeight_uniform_C` — when forgetting constants are uniform, `wᵢ* ∝ sᵢ^{1/3}`
-/

variable {L : ℕ}

/-- The forgetting bound at equilibrium:
  `F(w) = ∑ᵢ Cᵢ · √(sᵢ / (2 · wᵢ))`
where `C` is the per-layer forgetting constant, `s` is the gradient signal,
and `w` is the per-layer regularization weight. -/
def forgettingBound (C s w : Fin L → ℝ) : ℝ :=
  ∑ i, C i * Real.sqrt (s i / (2 * w i))

/-- The Lagrange-optimal per-layer weight:
  `wᵢ* = Λ · (Cᵢ · √sᵢ)^{2/3} / ∑ⱼ (Cⱼ · √sⱼ)^{2/3}` -/
def optWeight (C s : Fin L → ℝ) (Λ : ℝ) : Fin L → ℝ := fun i =>
  Λ * (C i * Real.sqrt (s i)) ^ ((2 : ℝ) / 3) /
    ∑ j : Fin L, (C j * Real.sqrt (s j)) ^ ((2 : ℝ) / 3)

/-- The uniform weight: `wᵢ = Λ / L` for all layers. -/
def unifWeight (Λ : ℝ) : Fin L → ℝ := fun _ => Λ / (L : ℝ)

/-! ## Budget constraint -/

/-
The sum of optimal weights equals the total budget `Λ`.
-/
theorem optWeight_sum (C s : Fin L → ℝ) {Λ : ℝ}
    (hpos : 0 < ∑ j : Fin L, (C j * Real.sqrt (s j)) ^ ((2 : ℝ) / 3)) :
    ∑ i, optWeight C s Λ i = Λ := by
  unfold optWeight;
  rw [ ← Finset.sum_div _ _ _, ← Finset.mul_sum _ _ _, mul_div_cancel_right₀ _ hpos.ne' ]

/-
The sum of uniform weights equals the total budget `Λ`.
-/
theorem unifWeight_sum (hL : (L : ℝ) ≠ 0) {Λ : ℝ} :
    ∑ i : Fin L, unifWeight Λ i = Λ := by
  -- The sum of L terms each equal to Λ / L is indeed Λ.
  simp [unifWeight, Finset.sum_const, nsmul_eq_mul];
  rw [ mul_div_cancel₀ _ hL ]

/-! ## Key auxiliary inequality: AM-GM for three terms

The inequality `u² + 2/u ≥ 3` for `u > 0` follows from the AM-GM inequality
applied to `u², 1/u, 1/u`. Equivalently, `1/√t + t/2 ≥ 3/2` for `t > 0`.
This is the foundation for the tangent-line proof of optimality. -/

theorem sq_add_two_div_ge_three (u : ℝ) (hu : 0 < u) :
    u ^ 2 + 2 / u ≥ 3 := by
  field_simp;
  nlinarith [ sq_nonneg ( u - 1 ) ]

/-! ## Tangent line bound

For a convex function `f(x) = a/√x`, the tangent line at `x₀` provides a global lower bound:
  `a/√x ≥ 3a/(2√x₀) - ax/(2·x₀·√x₀)`
This follows from `sq_add_two_div_ge_three`. -/

theorem tangent_sqrt_inv_bound {a x x₀ : ℝ} (ha : 0 < a) (hx : 0 < x) (hx₀ : 0 < x₀) :
    a / Real.sqrt x ≥ 3 * a / (2 * Real.sqrt x₀) -
      a * x / (2 * x₀ * Real.sqrt x₀) := by
  -- Set $t = x/x₀ > 0$. Then we need to show $1/√t ≥ (3-t)/2$.
  set t := x / x₀ with ht
  have ht_pos : 0 < t := by
    positivity
  have h_ineq : 1 / Real.sqrt t ≥ (3 - t) / 2 := by
    rw [ ge_iff_le, div_le_div_iff₀ ] <;> nlinarith [ sq_nonneg ( Real.sqrt t - 1 ), Real.sqrt_nonneg t, Real.sq_sqrt ht_pos.le ];
  simp_all +decide [ division_def, mul_assoc, mul_left_comm, hx.le, hx₀.le ];
  field_simp at *;
  linarith

/-! ## Optimality of the weights

Using the tangent line bound and the Lagrange condition that
`aᵢ / (wᵢ*)^{3/2} = const` for all `i`, we prove that the optimal weights
minimize the forgetting bound among all feasible weights. -/

/-
The optimal weights minimize the forgetting bound among all positive weight vectors
summing to `Λ`.
-/
theorem optWeight_minimizes (C s : Fin L → ℝ) (Λ : ℝ)
    (hC : ∀ i, 0 < C i) (hs : ∀ i, 0 < s i) (hΛ : 0 < Λ) (hL : 0 < L)
    (w : Fin L → ℝ) (hw : ∀ i, 0 < w i) (hsum : ∑ i, w i = Λ) :
    forgettingBound C s w ≥ forgettingBound C s (optWeight C s Λ) := by
  -- Let $a_i = C_i \sqrt{s_i / 2}$, then $F(w) = \sum_i a_i / \sqrt{w_i}$.
  set a : Fin L → ℝ := fun i => C i * Real.sqrt (s i / 2);
  -- By tangent_sqrt_inv_bound applied with $a = a_i$, $x = w_i$, and $x₀ = w^*_i$:
  have h_tangent : ∀ i, a i / Real.sqrt (w i) ≥ 3 * a i / (2 * Real.sqrt (optWeight C s Λ i)) - a i * w i / (2 * optWeight C s Λ i * Real.sqrt (optWeight C s Λ i)) := by
    intro i;
    convert tangent_sqrt_inv_bound ( show 0 < a i from mul_pos ( hC i ) ( Real.sqrt_pos.mpr ( div_pos ( hs i ) zero_lt_two ) ) ) ( show 0 < w i from hw i ) ( show 0 < optWeight C s Λ i from div_pos ( mul_pos hΛ ( Real.rpow_pos_of_pos ( mul_pos ( hC i ) ( Real.sqrt_pos.mpr ( hs i ) ) ) _ ) ) ( Finset.sum_pos ( fun _ _ => Real.rpow_pos_of_pos ( mul_pos ( hC _ ) ( Real.sqrt_pos.mpr ( hs _ ) ) ) _ ) ⟨ i, Finset.mem_univ _ ⟩ ) ) using 1;
  -- The key Lagrange property: $a_i / (w^*_i \sqrt{w^*_i})$ is the SAME constant $\nu$ for all $i$.
  have h_lagrange : ∃ ν : ℝ, ∀ i, a i / (optWeight C s Λ i * Real.sqrt (optWeight C s Λ i)) = ν := by
    -- Substitute $w_i^* = \Lambda \cdot (C_i \sqrt{s_i})^{2/3} / S$ into the expression.
    have h_subst : ∀ i, a i / (optWeight C s Λ i * Real.sqrt (optWeight C s Λ i)) = (a i * (∑ j, (C j * Real.sqrt (s j)) ^ ((2 : ℝ) / 3)) ^ (3 / 2 : ℝ)) / (Λ ^ (3 / 2 : ℝ) * (C i * Real.sqrt (s i))) := by
      intro i
      simp [optWeight];
      rw [ div_mul_eq_div_div ] ; rw [ Real.sqrt_div ( by exact mul_nonneg hΛ.le <| Real.rpow_nonneg ( mul_nonneg ( le_of_lt <| hC i ) <| Real.sqrt_nonneg _ ) _ ), Real.sqrt_mul <| by positivity ] ; ring;
      rw [ show ( 3 / 2 : ℝ ) = 1 + 1 / 2 by norm_num, Real.rpow_add ( Finset.sum_pos ( fun _ _ => Real.rpow_pos_of_pos ( mul_pos ( hC _ ) ( Real.sqrt_pos.mpr ( hs _ ) ) ) _ ) ⟨ i, Finset.mem_univ _ ⟩ ), Real.rpow_one ] ; norm_num ; ring;
      rw [ show ( 3 / 2 : ℝ ) = 1 + 1 / 2 by norm_num, Real.rpow_add hΛ, Real.rpow_one ] ; norm_num [ Real.sqrt_eq_rpow, ← Real.rpow_mul ( le_of_lt ( mul_pos ( hC i ) ( Real.sqrt_pos.mpr ( hs i ) ) ) ) ] ; ring;
      rw [ ← Real.rpow_mul ( mul_nonneg ( le_of_lt ( hC i ) ) ( Real.rpow_nonneg ( le_of_lt ( hs i ) ) _ ) ) ] ; norm_num ; ring;
      rw [ show ( 2 / 3 : ℝ ) = 1 / 3 + 1 / 3 by norm_num, Real.rpow_add ( mul_pos ( hC i ) ( Real.rpow_pos_of_pos ( hs i ) _ ) ) ] ; norm_num ; ring;
      rw [ inv_pow, ← Real.rpow_natCast, ← Real.rpow_mul ( mul_nonneg ( le_of_lt ( hC i ) ) ( Real.rpow_nonneg ( le_of_lt ( hs i ) ) _ ) ) ] ; norm_num ; ring;
    use (∑ j, (C j * Real.sqrt (s j)) ^ ((2 : ℝ) / 3)) ^ (3 / 2 : ℝ) / (Λ ^ (3 / 2 : ℝ) * Real.sqrt 2);
    intro i; rw [ h_subst i ] ; ring;
    simp +zetaDelta at *;
    simp +decide [ div_eq_mul_inv, mul_assoc, mul_comm, mul_left_comm, ne_of_gt ( hC i ), ne_of_gt ( Real.sqrt_pos.mpr ( hs i ) ) ];
  -- Summing the inequalities from the tangent line bound:
  have h_sum_tangent : ∑ i, a i / Real.sqrt (w i) ≥ (3 / 2) * ∑ i, a i / Real.sqrt (optWeight C s Λ i) - (h_lagrange.choose / 2) * ∑ i, w i := by
    have h_sum_tangent : ∑ i, a i / Real.sqrt (w i) ≥ (3 / 2) * ∑ i, a i / Real.sqrt (optWeight C s Λ i) - ∑ i, (a i * w i) / (2 * optWeight C s Λ i * Real.sqrt (optWeight C s Λ i)) := by
      simpa only [ Finset.mul_sum _ _ _, Finset.sum_sub_distrib, mul_div_mul_comm ] using Finset.sum_le_sum fun i _ => h_tangent i;
    convert h_sum_tangent using 2;
    rw [ Finset.mul_sum _ _ _ ] ; refine' Finset.sum_congr rfl fun i _ => _ ; rw [ ← h_lagrange.choose_spec i ] ; ring;
  -- And $F(w^*) = \sum_i a_i / \sqrt{w^*_i} = \sum_i \nu w^*_i = \nu \Lambda$.
  have h_forgetting_bound_opt : ∑ i, a i / Real.sqrt (optWeight C s Λ i) = h_lagrange.choose * Λ := by
    have h_forgetting_bound_opt : ∑ i, a i / Real.sqrt (optWeight C s Λ i) = ∑ i, h_lagrange.choose * optWeight C s Λ i := by
      refine Finset.sum_congr rfl fun i _ => ?_;
      grind +splitIndPred;
    rw [ h_forgetting_bound_opt, ← Finset.mul_sum _ _ _, optWeight_sum ];
    exact Finset.sum_pos ( fun i _ => Real.rpow_pos_of_pos ( mul_pos ( hC i ) ( Real.sqrt_pos.mpr ( hs i ) ) ) _ ) ⟨ ⟨ 0, hL ⟩, Finset.mem_univ _ ⟩;
  convert h_sum_tangent using 1 ; ring;
  · exact Finset.sum_congr rfl fun _ _ => by rw [ ← div_eq_mul_inv ] ; rw [ show s _ / ( 2 * w _ ) = ( s _ / 2 ) / w _ by ring ] ; rw [ Real.sqrt_div ( by linarith [ hs ‹_› ] ) ] ; ring;
  · convert congr_arg ( fun x : ℝ => x ) h_forgetting_bound_opt using 1;
    · unfold forgettingBound a; norm_num [ div_eq_mul_inv, mul_assoc, mul_comm, mul_left_comm, Real.sqrt_mul ( le_of_lt ( hs _ ) ) ] ;
    · rw [ h_forgetting_bound_opt, hsum ] ; ring

/-! ## Power mean inequality

The power mean inequality states `(∑ aᵢ^{2/3})^{3/2} ≤ √L · ∑ aᵢ` for positive `aᵢ`.
This is equivalent to `M_{2/3} ≤ M₁` (power mean with exponent 2/3 ≤ arithmetic mean)
and follows from Jensen's inequality for the concave function `t ↦ t^{2/3}`. -/

theorem power_mean_two_thirds (a : Fin L → ℝ) (ha : ∀ i, 0 ≤ a i) (hL : 0 < L) :
    (∑ i, a i ^ ((2 : ℝ) / 3)) ^ ((3 : ℝ) / 2) ≤
      (L : ℝ) ^ ((1 : ℝ) / 2) * ∑ i, a i := by
  revert hL a ha;
  intros a ha hL_pos
  have h_holder : (∑ i, a i^(2 / 3 : ℝ)) ≤ (∑ i : Fin L, a i)^(2 / 3 : ℝ) * (∑ i : Fin L, 1)^(1 / 3 : ℝ) := by
    have := @Real.inner_le_Lp_mul_Lq; norm_num at *;
    convert @this ( Fin L ) Finset.univ ( fun i => a i ^ ( 2 / 3 : ℝ ) ) ( fun _ => 1 ) ( 3 / 2 ) ( 3 ) ( by constructor <;> norm_num ) using 1 <;> norm_num [ abs_of_nonneg, ha ];
    exact Or.inl ( by congr; ext i; rw [ abs_of_nonneg ( Real.rpow_nonneg ( ha i ) _ ) ] ; rw [ ← Real.rpow_mul ( ha i ) ] ; norm_num );
  convert Real.rpow_le_rpow ( Finset.sum_nonneg fun _ _ => Real.rpow_nonneg ( ha _ ) _ ) h_holder ( show ( 3 : ℝ ) / 2 ≥ 0 by norm_num ) using 1 ; norm_num [ mul_pow, ← Real.rpow_mul ( Finset.sum_nonneg fun _ _ => ha _ ) ] ; ring;
  rw [ Real.mul_rpow ( Real.rpow_nonneg ( Finset.sum_nonneg fun _ _ => ha _ ) _ ) ( by positivity ), ← Real.rpow_mul ( Finset.sum_nonneg fun _ _ => ha _ ), ← Real.rpow_mul ( by positivity ) ] ; norm_num ; ring

/-! ## Optimal ≤ Uniform comparison

Combining the power mean inequality with the closed-form expressions for `F(w*)`
and `F(w_uniform)`, we show that optimal weighting always improves upon uniform. -/

/-
Optimal weighting gives a forgetting bound no worse than uniform weighting.
Equality holds iff `Cᵢ √sᵢ` is constant across all layers.
-/
theorem forgettingBound_opt_le_unif (C s : Fin L → ℝ) (Λ : ℝ)
    (hC : ∀ i, 0 < C i) (hs : ∀ i, 0 < s i) (hΛ : 0 < Λ) (hL : 0 < L) :
    forgettingBound C s (optWeight C s Λ) ≤ forgettingBound C s (unifWeight Λ) := by
  convert optWeight_minimizes C s Λ hC hs hΛ hL _ _ _ using 1;
  · exact fun i => div_pos hΛ ( Nat.cast_pos.mpr hL );
  · exact unifWeight_sum ( by positivity )

/-! ## Simplifications -/

/-
**Claim 2**: When the gradient signal is uniform (`sᵢ = s` for all `i`),
the optimal weight is proportional to `Cᵢ^{2/3}`:
  `wᵢ* = Λ · Cᵢ^{2/3} / ∑ⱼ Cⱼ^{2/3}`
-/
theorem optWeight_uniform_signal (C : Fin L → ℝ) (s_val Λ : ℝ)
    (hC : ∀ i, 0 < C i) (hs : 0 < s_val) :
    optWeight C (fun _ => s_val) Λ = fun i =>
      Λ * C i ^ ((2 : ℝ) / 3) / ∑ j : Fin L, C j ^ ((2 : ℝ) / 3) := by
  unfold optWeight;
  norm_num [ Real.mul_rpow ( le_of_lt ( hC _ ) ) ( Real.sqrt_nonneg _ ), Finset.mul_sum _ _ _, mul_div_mul_right, ne_of_gt ( Real.sqrt_pos.mpr hs ) ];
  norm_num [ ← mul_assoc, ← Finset.sum_mul, mul_div_mul_right, ne_of_gt ( Real.rpow_pos_of_pos ( Real.sqrt_pos.mpr hs ) _ ) ]

/-
**Claim 3**: When the forgetting constants are uniform (`Cᵢ = C` for all `i`),
the optimal weight is proportional to `sᵢ^{1/3}`:
  `wᵢ* = Λ · sᵢ^{1/3} / ∑ⱼ sⱼ^{1/3}`
-/
theorem optWeight_uniform_C (s : Fin L → ℝ) (C_val Λ : ℝ)
    (hC : 0 < C_val) (hs : ∀ i, 0 < s i) :
    optWeight (fun _ => C_val) s Λ = fun i =>
      Λ * s i ^ ((1 : ℝ) / 3) / ∑ j : Fin L, s j ^ ((1 : ℝ) / 3) := by
  unfold optWeight;
  norm_num [ Real.mul_rpow ( le_of_lt hC ) ( Real.sqrt_nonneg _ ), Real.sqrt_eq_rpow, ← Real.rpow_mul ( le_of_lt ( hs _ ) ) ];
  norm_num [ Real.mul_rpow ( le_of_lt hC ) ( Real.rpow_nonneg ( le_of_lt ( hs _ ) ) _ ), ← Real.rpow_mul ( le_of_lt ( hs _ ) ) ];
  norm_num [ ← Finset.mul_sum _ _ _, mul_div_mul_left _ _ ( ne_of_gt ( Real.rpow_pos_of_pos hC _ ) ) ];
  exact funext fun i => by rw [ mul_left_comm, mul_div_mul_left _ _ ( by positivity ) ] ;

end MultiLayerWeighting