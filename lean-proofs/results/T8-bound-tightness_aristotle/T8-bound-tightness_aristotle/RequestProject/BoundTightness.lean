import Mathlib

/-!
# T8: Bound Tightness (Lower Bound on Forgetting)

We formalize the tightness of the √δ scaling in the SFP forgetting bound.

The upstream SFP bound says: forgetting ≤ C · √δ.
We show this is tight by constructing cases achieving forgetting = Ω(√δ),
and prove that no bound with exponent α > 1/2 can hold universally.
-/

open Real

set_option maxHeartbeats 800000

noncomputable section

/-! ## Claim 1: 1D Lower Bound Construction

In 1D, with L_old(a) = |a - a₀|, we have forgetting = √δ exactly.
We formalize: for x ≥ 0, if δ = x² then |x| = √δ.
More precisely: for any drift x : ℝ, forgetting = |x| and δ = x²,
so forgetting = √(δ).
-/

/-- The 1D forgetting function: L_old(a) = |a - a₀|. -/
def forgetting_1d (a a₀ : ℝ) : ℝ := |a - a₀|

/-- The 1D subspace drift: δ = (a - a₀)². -/
def drift_1d (a a₀ : ℝ) : ℝ := (a - a₀) ^ 2

/-
In the 1D construction, forgetting equals √δ exactly.
-/
theorem forgetting_eq_sqrt_drift_1d (a a₀ : ℝ) :
    forgetting_1d a a₀ = √(drift_1d a a₀) := by
  unfold forgetting_1d drift_1d; rw [ Real.sqrt_sq_eq_abs ] ;

/-
The 1D construction achieves the lower bound with constant C = 1:
    forgetting ≥ 1 · √δ.
-/
theorem lower_bound_1d (a a₀ : ℝ) :
    forgetting_1d a a₀ ≥ 1 * √(drift_1d a a₀) := by
  rw [ one_mul, forgetting_eq_sqrt_drift_1d ]

/-! ## Claim 2: General Lower Bound Construction

For any C > 0 and δ > 0, there exists a value of forgetting
that equals C · √δ, showing the √δ rate is achievable.

We construct L_old(Δa) = C · |Δa| in 1D, which is C-Lipschitz,
and set |Δa| = √δ, giving forgetting = C · √δ.
-/

/-
For any C > 0 and δ > 0, the value C · √δ is achievable as a forgetting value
    by a C-Lipschitz loss function.
-/
theorem general_lower_bound_achievable (C : ℝ) (hC : C > 0) (δ : ℝ) (hδ : δ > 0) :
    ∃ (f : ℝ → ℝ), LipschitzWith ⟨C, le_of_lt hC⟩ f ∧
      ∃ (x : ℝ), x ^ 2 = δ ∧ |f x - f 0| = C * √δ := by
  refine' ⟨ fun x => C * x, _, Real.sqrt δ, _, _ ⟩ <;> norm_num [ hC.le, hδ.le ];
  · rw [ lipschitzWith_iff_norm_sub_le ];
    norm_num [ ← mul_sub, abs_mul, abs_of_pos hC ];
  · rw [ abs_of_pos hC, abs_of_pos ( Real.sqrt_pos.mpr hδ ) ]

/-! ## Claim 3: Tightness of the √δ Rate

The exponent 1/2 is optimal: no bound of the form
  forgetting ≤ C · δ^α  with α > 1/2
can hold universally.

We prove: for any α > 1/2 and any M > 0, there exists δ > 0
such that √δ > M · δ^α. This means the 1D construction
(which achieves forgetting = √δ) violates any proposed δ^α bound
for small enough δ.
-/

/-
For α > 1/2, the function δ^(1/2) eventually dominates δ^α as δ → 0⁺.
    Specifically, for any M > 0, there exists δ > 0 with √δ > M · δ^α.
-/
theorem sqrt_dominates_higher_power (α : ℝ) (hα : α > 1 / 2) (M : ℝ) (hM : M > 0) :
    ∃ δ : ℝ, 0 < δ ∧ √δ > M * δ ^ α := by
  -- Choose δ such that δ^(α - 1/2) < 1/M.
  obtain ⟨δ, hδ_pos, hδ⟩ : ∃ δ > 0, δ ^ (α - 1 / 2) < 1 / M := by
    field_simp;
    exact ⟨ ( 1 / 2 / M ) ^ ( 2 / ( α * 2 - 1 ) ), by positivity, by rw [ ← Real.rpow_mul ( by positivity ), div_mul_div_cancel₀ ( by linarith ), div_self ( by linarith ), Real.rpow_one ] ; nlinarith [ mul_div_cancel₀ ( 1 / 2 ) hM.ne' ] ⟩;
  refine' ⟨ δ, hδ_pos, _ ⟩;
  rw [ lt_div_iff₀' hM ] at hδ;
  rw [ show δ ^ α = δ ^ ( α - 1 / 2 ) * δ ^ ( 1 / 2 : ℝ ) by rw [ ← Real.rpow_add hδ_pos ] ; ring ];
  rw [ ← Real.sqrt_eq_rpow ] ; nlinarith [ Real.sqrt_nonneg δ, Real.sq_sqrt hδ_pos.le ]

/-
No universal bound of the form `forgetting ≤ K · δ^α` with α > 1/2
    can hold for all C-Lipschitz functions.
    Formally: for any α > 1/2 and K > 0, there exists a 1-Lipschitz function f
    and a point x with drift δ = x² > 0 such that |f(x) - f(0)| > K · δ^α.
-/
theorem no_universal_higher_power_bound (α : ℝ) (hα : α > 1 / 2) (K : ℝ) (hK : K > 0) :
    ∃ (f : ℝ → ℝ) (x : ℝ), LipschitzWith 1 f ∧ 0 < x ^ 2 ∧
      |f x - f 0| > K * (x ^ 2) ^ α := by
  -- Use the lower bound construction with C = 1 and δ = x².
  obtain ⟨δ, hδ_pos, hδ⟩ : ∃ δ : ℝ, 0 < δ ∧ √δ > K * δ ^ α := by
    exact sqrt_dominates_higher_power α hα K hK;
  use fun x => x, Real.sqrt δ;
  exact ⟨ by exact LipschitzWith.id, by positivity, by simpa [ abs_of_pos ( Real.sqrt_pos.mpr hδ_pos ), Real.sq_sqrt hδ_pos.le ] using hδ ⟩

/-! ## Claim 4: Quadratic Regime (Critical Point)

When L_old is differentiable and θ₀ is a critical point (∇L_old(θ₀) = 0),
forgetting is O(δ) rather than O(√δ).

We formalize a version: if f is differentiable at 0 with f(0) = 0 and f'(0) = 0,
then |f(x)| = o(|x|) as x → 0, which means |f(x)| = o(√δ) since δ = x².
More precisely, forgetting / √δ → 0, i.e., the √δ bound is not tight.

For a concrete quadratic model: if f(x) = c · x², then forgetting = c · δ
where δ = x², showing linear (not square-root) scaling.
-/

/-
In the quadratic model f(x) = c · x², forgetting scales linearly with δ = x².
-/
theorem quadratic_model_linear_scaling (c : ℝ) (x : ℝ) :
    |c * x ^ 2| = |c| * (x ^ 2) := by
  rw [ abs_mul, abs_sq ]

/-
At a critical point with quadratic growth, forgetting/√δ → 0 as δ → 0.
    Concretely: for the quadratic f(x) = c·x², we have |f(x)|/√(x²) = |c|·|x| → 0.
-/
theorem quadratic_forgetting_over_sqrt_drift (c x : ℝ) (hx : x ≠ 0) :
    |c * x ^ 2| / √(x ^ 2) = |c| * |x| := by
  norm_num [ abs_mul, hx, Real.sqrt_sq_eq_abs ];
  rw [ div_eq_iff ( ne_of_gt ( abs_pos.mpr hx ) ) ] ; cases abs_cases x <;> simp +decide [ * ] <;> ring

/-! ## Claim 5: √δ vs δ Dichotomy

Combining Claims 1-4:
- If the derivative is nonzero (‖∇L_old(θ₀)‖ > 0): forgetting = Θ(√δ)
- If the derivative is zero (∇L_old(θ₀) = 0): forgetting = O(δ)

We formalize the key contrast:
- Linear model f(x) = c·x with c ≠ 0: forgetting/√δ → |c| ≠ 0 (tight)
- Quadratic model f(x) = c·x²: forgetting/√δ → 0 (loose)
-/

/-
In the linear model f(x) = c·x (nonzero gradient), forgetting/√δ = |c| > 0,
    showing the √δ bound is tight.
-/
theorem linear_model_tight (c x : ℝ) (_hc : c ≠ 0) (hx : x ≠ 0) :
    |c * x| / √(x ^ 2) = |c| := by
  rw [ Real.sqrt_sq_eq_abs, abs_mul, mul_div_cancel_right₀ _ ( ne_of_gt ( abs_pos.mpr hx ) ) ]

/-
The dichotomy: linear model gives |c| > 0, quadratic gives |c|·|x| which → 0.
    For any ε > 0, the quadratic model has forgetting/√δ < ε for small enough x,
    while the linear model always has forgetting/√δ = |c|.
-/
theorem dichotomy_quadratic_small (c : ℝ) (ε : ℝ) (hε : ε > 0) :
    ∃ η > 0, ∀ x : ℝ, |x| < η → |x| ≠ 0 →
      |c * x ^ 2| / √(x ^ 2) < ε := by
  use ε / ( |c| + 1 );
  norm_num [ Real.sqrt_sq_eq_abs ];
  exact ⟨ by positivity, fun x hx hx' => by rw [ div_lt_iff₀ ( abs_pos.mpr hx' ) ] ; nlinarith [ abs_pos.mpr hx', abs_mul_abs_self x, mul_div_cancel₀ ε ( by positivity : ( |c| + 1 ) ≠ 0 ), mul_lt_mul_of_pos_left hx ( abs_pos.mpr hx' ) ] ⟩

end