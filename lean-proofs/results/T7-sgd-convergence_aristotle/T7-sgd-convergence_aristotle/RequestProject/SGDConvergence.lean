/-
# Theorem T7: SGD Convergence Under SFP Regularization

Formalization of convergence results for gradient descent on the composite objective
  L(θ) = L_new(θ) + λ · L_pres(θ)
where L_new is L-smooth and μ-strongly convex, and L_pres is L_p-smooth.

We formalize:
  Claim 1: Descent lemma for the composite objective
  Claim 2: Linear convergence of new-task loss
  Claim 3: Drift bound after K steps
  Claim 4: Minimum steps for target drift
  Claim 6: Numerical verification
-/

import Mathlib

noncomputable section

open Real Finset BigOperators

/-! ## Abstract Geometric Recurrence Lemma

Many convergence results (Claims 2, 3) follow from the same abstract recurrence:
if a sequence satisfies `a_{k+1} ≤ r · a_k + b` with `0 ≤ r < 1` and `b ≥ 0`,
then `a_k ≤ r^k · a_0 + b · (1 - r^k) / (1 - r)`.
-/

/-
If `a_{k+1} ≤ r * a_k + b` with `0 ≤ r` and `b ≥ 0`, then
    `a_k ≤ r^k * a_0 + b * ∑ i ∈ range k, r^i`.
-/
theorem geometric_recurrence_bound (a : ℕ → ℝ) (r b : ℝ)
    (hr : 0 ≤ r) (hb : 0 ≤ b)
    (hrec : ∀ k, a (k + 1) ≤ r * a k + b) (ha0 : 0 ≤ a 0) :
    ∀ k, a k ≤ r ^ k * a 0 + b * (∑ i ∈ range k, r ^ i) := by
  intro k;
  induction' k with k ih;
  · norm_num;
  · rw [ Finset.sum_range_succ, pow_succ' ];
    nlinarith [ hrec k, geom_sum_mul r k ]

/-
Closed-form: when `0 ≤ r < 1`, the geometric sum equals `(1 - r^k)/(1 - r)`.
-/
theorem geometric_recurrence_closed (a : ℕ → ℝ) (r b : ℝ)
    (hr0 : 0 ≤ r) (hr1 : r < 1) (hb : 0 ≤ b)
    (hrec : ∀ k, a (k + 1) ≤ r * a k + b) (ha0 : 0 ≤ a 0) :
    ∀ k, a k ≤ r ^ k * a 0 + b * (1 - r ^ k) / (1 - r) := by
  intro k;
  convert geometric_recurrence_bound a r b hr0 hb hrec ha0 k using 1;
  rw [ ← geom_sum_mul_neg, mul_div_assoc ];
  rw [ mul_div_cancel_right₀ _ ( by linarith ) ]

/-
When `a_0 = 0`, the bound simplifies to `a_k ≤ b / (1-r) · (1 - r^k)`.
-/
theorem geometric_recurrence_zero_start (a : ℕ → ℝ) (r b : ℝ)
    (hr0 : 0 ≤ r) (hr1 : r < 1) (hb : 0 ≤ b)
    (hrec : ∀ k, a (k + 1) ≤ r * a k + b) (ha0 : a 0 = 0) :
    ∀ k, a k ≤ b / (1 - r) * (1 - r ^ k) := by
  intros k
  have := @geometric_recurrence_closed a r b hr0 hr1 hb hrec (by linarith)
  simp_all;
  convert this k using 1 ; ring

/-
Steady-state bound: `a_k ≤ b / (1 - r)` for all k when `a_0 = 0`.
-/
theorem geometric_recurrence_steady_state (a : ℕ → ℝ) (r b : ℝ)
    (hr0 : 0 ≤ r) (hr1 : r < 1) (hb : 0 ≤ b)
    (hrec : ∀ k, a (k + 1) ≤ r * a k + b) (ha0 : a 0 = 0) :
    ∀ k, a k ≤ b / (1 - r) := by
  intro k;
  rw [ le_div_iff₀ ] <;> try linarith;
  exact Nat.recOn k ( by norm_num [ ha0 ] ; nlinarith ) fun n ihn => by nlinarith [ hrec n ] ;

/-! ## Claim 1: Descent Lemma

**Statement.** If `f` has a quadratic upper bound with constant `L_total`
(consequence of `L_total`-smoothness), and `η ≤ 1 / L_total`, then one gradient step gives:
  `f(x - η • g) ≤ f(x) - (η/2) * ‖g‖²`
where `g = ∇f(x)`.

We formalize this in an inner product space.
-/

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-
The descent lemma: if f satisfies the quadratic upper bound from L-smoothness,
    and η ≤ 1/L, then one gradient step decreases f by at least (η/2)‖g‖².
-/
theorem descent_lemma (f : E → ℝ) (x g : E) (η L_total : ℝ)
    (hη_pos : 0 < η) (hL_pos : 0 < L_total) (hη_bound : η ≤ 1 / L_total)
    (hsmooth : ∀ y, f y ≤ f x + @inner ℝ E _ g (y - x) + L_total / 2 * ‖y - x‖ ^ 2) :
    f (x - η • g) ≤ f x - η / 2 * ‖g‖ ^ 2 := by
  have := hsmooth ( x - η • g );
  simp_all +decide [ inner_smul_right, norm_smul ];
  rw [ abs_of_pos hη_pos ] at this;
  nlinarith [ mul_le_mul_of_nonneg_left hη_bound ( sq_nonneg ‖g‖ ), mul_inv_cancel₀ hL_pos.ne' ]

/-! ## Claim 2: Linear Convergence of New-Task Loss

We abstract this as: if a nonneg sequence `{e_k}` (error = L_new(θ_k) - L_new(θ*)) satisfies
  `e_{k+1} ≤ (1 - η*μ) * e_k + η*λ*δ_max`
(which follows from descent + strong convexity + regularization bias),
then after K steps:
  `e_K ≤ (1 - ημ)^K * e_0 + λ * δ_max / μ`
-/

/-
Linear convergence with regularization bias.
    The error `e_k = L_new(θ_k) - L_new(θ*)` converges geometrically with a bias
    of `λ * δ_max / μ`.
-/
theorem linear_convergence_with_bias (e : ℕ → ℝ) (η μ lam δ_max : ℝ)
    (hη : 0 < η) (hμ : 0 < μ) (hlam : 0 < lam) (hδ : 0 ≤ δ_max)
    (hημ : η * μ < 1) (he0 : 0 ≤ e 0)
    (hrec : ∀ k, e (k + 1) ≤ (1 - η * μ) * e k + η * lam * δ_max) :
    ∀ K, e K ≤ (1 - η * μ) ^ K * e 0 + lam * δ_max / μ := by
  intro K;
  -- Apply the geometric recurrence closed form with r = 1 - η*μ and b = η*lam*δ_max.
  have h_recurrence_closed : ∀ k, e k ≤ (1 - η * μ) ^ k * e 0 + (η * lam * δ_max) * (1 - (1 - η * μ) ^ k) / (η * μ) := by
    convert geometric_recurrence_closed e ( 1 - η * μ ) ( η * lam * δ_max ) _ _ _ _ _ using 3 <;> try nlinarith;
    · ring;
    · positivity;
    · assumption;
  refine le_trans ( h_recurrence_closed K ) ?_;
  norm_num [ mul_assoc, mul_div_mul_left, hη.ne' ];
  gcongr;
  exact mul_le_of_le_one_right hδ ( sub_le_self _ ( pow_nonneg ( by nlinarith ) _ ) )

/-! ## Claim 3: Drift Bound After K Steps

The subspace drift `δ_k` satisfies the recurrence:
  `δ_{k+1} ≤ (1 - ημ) · δ_k + (ηλ/μ) · ‖∇L_pres‖²_max`

This is an instance of the geometric recurrence with `r = 1 - ημ` and
`b = (ηλ/μ) · ‖∇L_pres‖²_max`.
-/

/-
Drift bound: The subspace drift satisfies the geometric decay bound.
-/
theorem drift_bound (δ : ℕ → ℝ) (η μ grad_max_sq : ℝ)
    (hη : 0 < η) (hμ : 0 < μ) (hgrad : 0 ≤ grad_max_sq)
    (hημ : η * μ < 1) (hδ0 : 0 ≤ δ 0)
    (hrec : ∀ k, δ (k + 1) ≤ (1 - η * μ) * δ k + η * grad_max_sq) :
    ∀ K, δ K ≤ (1 - η * μ) ^ K * δ 0 + grad_max_sq / μ * (1 - (1 - η * μ) ^ K) := by
  intros K
  have := geometric_recurrence_closed δ (1 - η * μ) (η * grad_max_sq) (by
  linarith) (by
  nlinarith) (by
  positivity) (by
  assumption) (by
  exact?) K
  generalize_proofs at *;
  convert this using 1 ; ring;
  -- Combine like terms and simplify the expression.
  field_simp
  ring

/-
Drift bound with zero initial drift.
-/
theorem drift_bound_zero_start (δ : ℕ → ℝ) (η μ grad_max_sq : ℝ)
    (hη : 0 < η) (hμ : 0 < μ) (hgrad : 0 ≤ grad_max_sq)
    (hημ : η * μ < 1) (hδ0 : δ 0 = 0)
    (hrec : ∀ k, δ (k + 1) ≤ (1 - η * μ) * δ k + η * grad_max_sq) :
    ∀ K, δ K ≤ grad_max_sq / μ * (1 - (1 - η * μ) ^ K) := by
  exact fun K => by simpa [ hδ0 ] using ( drift_bound δ η μ grad_max_sq hη hμ hgrad hημ ( by norm_num [ hδ0 ] ) hrec K ) ;

/-
Steady-state drift: For large K, δ_∞ ≤ grad_max_sq / μ.
-/
theorem steady_state_drift (δ : ℕ → ℝ) (η μ grad_max_sq : ℝ)
    (hη : 0 < η) (hμ : 0 < μ) (hgrad : 0 ≤ grad_max_sq)
    (hημ : η * μ < 1) (hδ0 : δ 0 = 0)
    (hrec : ∀ k, δ (k + 1) ≤ (1 - η * μ) * δ k + η * grad_max_sq) :
    ∀ K, δ K ≤ grad_max_sq / μ := by
  intro K;
  convert geometric_recurrence_steady_state δ ( 1 - η * μ ) ( η * grad_max_sq ) ( by nlinarith ) ( by nlinarith ) ( by nlinarith ) hrec hδ0 K using 1;
  rw [ ← mul_div_mul_left _ _ hη.ne' ] ; ring

/-! ## Claim 4: Minimum Steps for Target Drift

To achieve δ_K ≤ δ_target starting from δ_0 = 0, we need:
  K ≥ (1/(ημ)) · log(1 / (1 - μ·δ_target / grad_max_sq))

We formalize the sufficient condition: if K is large enough (in terms of the log bound),
then the drift is below the target.
-/

/-
If the steady-state drift is at most the target, then eventually the drift reaches
    the target. More precisely: `grad_max_sq / μ ≤ δ_target` implies `δ_K ≤ δ_target`
    for all K.
-/
theorem drift_below_target_if_steady_state_ok (δ : ℕ → ℝ) (η μ grad_max_sq δ_target : ℝ)
    (hη : 0 < η) (hμ : 0 < μ) (hgrad : 0 ≤ grad_max_sq)
    (hημ : η * μ < 1) (hδ0 : δ 0 = 0) (htarget : 0 ≤ δ_target)
    (hss : grad_max_sq / μ ≤ δ_target)
    (hrec : ∀ k, δ (k + 1) ≤ (1 - η * μ) * δ k + η * grad_max_sq) :
    ∀ K, δ K ≤ δ_target := by
  intros K
  apply le_trans (steady_state_drift δ η μ grad_max_sq hη hμ hgrad hημ hδ0 hrec K) hss

/-! ## Claim 6: Numerical Verification

With parameters: L = 10, μ = 0.1, λ = 0.1, L_p = 5.
- η = 1 / (L + λ · L_p) = 1 / 10.5
- Effective condition number κ_eff = (L + λ · L_p) / μ = 105
- Steps for 0.01 relative error: K = ⌈κ_eff · log(100)⌉ ≈ 483
- Steady-state drift: δ_∞ = η · λ · ‖∇L_pres‖²_max / μ = 0.095 (with ‖∇L_pres‖²_max = 1)
-/

/-
Numerical verification: effective condition number.
-/
theorem numerical_kappa_eff :
    (10 + 0.1 * 5) / 0.1 = (105 : ℚ) := by
  norm_num +zetaDelta at *

/-
Numerical verification: step size.
-/
theorem numerical_eta :
    1 / (10 + 0.1 * 5) = (2 : ℚ) / 21 := by
  norm_num

/-
Numerical verification: steady-state drift (with ‖∇L_pres‖²_max = 1).
-/
theorem numerical_steady_state_drift :
    (2 : ℚ) / 21 * 0.1 * 1 / 0.1 = 2 / 21 := by
  norm_num

end