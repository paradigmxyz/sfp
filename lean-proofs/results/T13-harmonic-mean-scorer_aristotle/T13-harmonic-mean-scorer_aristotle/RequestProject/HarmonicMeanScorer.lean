import Mathlib

/-!
# Harmonic Mean Scorer Properties (T13)

We formalize properties of the harmonic mean scorer H(R, P) = 2RP/(R+P),
used as an F1-like score for retention (R) and plasticity (P) in [0, 1].

## Main results

- `harmonicMean_zero_iff`: H(R,P) = 0 ↔ R = 0 ∨ P = 0  (Property P1: zero forcing)
- `harmonicMean_symm`: H(R,P) = H(P,R)  (Property P2: symmetry)
- `harmonicMean_mono_left`, `harmonicMean_mono_right`: monotonicity  (Property P3)
- `harmonicMean_one_one`: H(1,1) = 1  (Property P4: scale)
- `harmonicMean_partials_eq_at_diag`: ∂H/∂R = ∂H/∂P at R = P  (Property P5)
- `arithmeticMean_violates_P1`: A(0,1) = 1/2 ≠ 0
- `harmonicMean_le_geometricMean`: H ≤ G (for nonneg inputs)
- `geometricMean_le_arithmeticMean`: G ≤ A
- `nash_linear_frontier`: Nash bargaining on R + P = c gives R = P = c/2
- `weightedLinear_violates_P1`: W(0,1) = 0.4 ≠ 0
-/

open Real

noncomputable section

/-- The harmonic mean scorer: H(R, P) = 2RP/(R+P). Returns 0 when R+P = 0. -/
def harmonicMean (R P : ℝ) : ℝ :=
  if R + P = 0 then 0 else 2 * R * P / (R + P)

/-- The arithmetic mean: A(R, P) = (R + P) / 2. -/
def arithmeticMean (R P : ℝ) : ℝ := (R + P) / 2

/-- The geometric mean: G(R, P) = √(RP). -/
def geometricMean (R P : ℝ) : ℝ := Real.sqrt (R * P)

/-- The weighted linear scorer: W(R, P) = 0.6R + 0.4P. -/
def weightedLinear (R P : ℝ) : ℝ := 0.6 * R + 0.4 * P

/-
============================================================================
Property P1: Zero forcing
============================================================================

H(R, P) = 0 iff R = 0 or P = 0, for nonneg R, P.
-/
theorem harmonicMean_zero_iff {R P : ℝ} (hR : 0 ≤ R) (hP : 0 ≤ P) :
    harmonicMean R P = 0 ↔ R = 0 ∨ P = 0 := by
  unfold harmonicMean;
  grind

/-
============================================================================
Property P2: Symmetry
============================================================================

H(R, P) = H(P, R).
-/
theorem harmonicMean_symm (R P : ℝ) : harmonicMean R P = harmonicMean P R := by
  unfold harmonicMean; ring_nf;

/-
============================================================================
Property P3: Monotonicity
============================================================================

H is monotone nondecreasing in the first argument (for nonneg inputs with positive sum).
-/
theorem harmonicMean_mono_left {R₁ R₂ P : ℝ} (hR₁ : 0 ≤ R₁) (hR₂ : 0 ≤ R₂)
    (hP : 0 < P) (h : R₁ ≤ R₂) :
    harmonicMean R₁ P ≤ harmonicMean R₂ P := by
  unfold harmonicMean;
  split_ifs <;> try linarith;
  rw [ div_le_div_iff₀ ] <;> nlinarith [ mul_le_mul_of_nonneg_right h hP.le ]

/-
H is monotone nondecreasing in the second argument (for nonneg inputs with positive sum).
-/
theorem harmonicMean_mono_right {R P₁ P₂ : ℝ} (hR : 0 < R) (hP₁ : 0 ≤ P₁)
    (hP₂ : 0 ≤ P₂) (h : P₁ ≤ P₂) :
    harmonicMean R P₁ ≤ harmonicMean R P₂ := by
  grind +suggestions

/-
============================================================================
Property P4: Scale
============================================================================

H(1, 1) = 1.
-/
theorem harmonicMean_one_one : harmonicMean 1 1 = 1 := by
  unfold harmonicMean; norm_num;

-- ============================================================================
-- Property P5: Equal marginal rates at R = P
-- ============================================================================

/-- The partial derivative ∂H/∂R = 2P²/(R+P)², for R + P > 0. -/
def partialH_R (R P : ℝ) : ℝ := 2 * P ^ 2 / (R + P) ^ 2

/-- The partial derivative ∂H/∂P = 2R²/(R+P)², for R + P > 0. -/
def partialH_P (R P : ℝ) : ℝ := 2 * R ^ 2 / (R + P) ^ 2

/-
At R = P, the partial derivatives are equal (follows from symmetry).
-/
theorem harmonicMean_partials_eq_at_diag (t : ℝ) :
    partialH_R t t = partialH_P t t := by
  unfold partialH_R partialH_P; ring;

/-
============================================================================
Claim 2: Arithmetic mean violates P1
============================================================================

A(0, 1) = 1/2, which is nonzero: the arithmetic mean violates zero forcing.
-/
theorem arithmeticMean_violates_P1 : arithmeticMean 0 1 = 1 / 2 := by
  norm_num [ arithmeticMean ]

/-
============================================================================
Claim 3: Mean inequality chain H ≤ G ≤ A
============================================================================

H(R, P) ≤ G(R, P) for nonneg R, P.
-/
theorem harmonicMean_le_geometricMean {R P : ℝ} (hR : 0 ≤ R) (hP : 0 ≤ P) :
    harmonicMean R P ≤ geometricMean R P := by
  by_cases hR_path_P : R + P = 0 <;> simp_all +contextual [ harmonicMean, geometricMean ];
  · positivity;
  · rw [ ← Real.sqrt_mul hR ];
    exact Real.le_sqrt_of_sq_le ( by rw [ div_pow, div_le_iff₀ ( by positivity ) ] ; nlinarith [ sq_nonneg ( R - P ), mul_nonneg hR hP ] )

/-
G(R, P) ≤ A(R, P) for nonneg R, P.
-/
theorem geometricMean_le_arithmeticMean {R P : ℝ} (hR : 0 ≤ R) (hP : 0 ≤ P) :
    geometricMean R P ≤ arithmeticMean R P := by
  unfold geometricMean arithmeticMean;
  exact Real.sqrt_le_iff.mpr ⟨ by positivity, by linarith [ sq_nonneg ( R - P ) ] ⟩

/-- The full chain: H ≤ G ≤ A. -/
theorem harmonicMean_le_arithmeticMean {R P : ℝ} (hR : 0 ≤ R) (hP : 0 ≤ P) :
    harmonicMean R P ≤ arithmeticMean R P :=
  le_trans (harmonicMean_le_geometricMean hR hP) (geometricMean_le_arithmeticMean hR hP)

/-
============================================================================
Claim 3 (cont.): Equality conditions
============================================================================

H = G = A iff R = P (for positive R, P).
-/
theorem harmonicMean_eq_geometricMean_iff {R P : ℝ} (hR : 0 < R) (hP : 0 < P) :
    harmonicMean R P = geometricMean R P ↔ R = P := by
  unfold harmonicMean geometricMean;
  split_ifs <;> simp_all +decide [ ne_of_gt, le_of_lt ];
  · linarith;
  · rw [ ← Real.sqrt_mul hR.le, eq_comm, Real.sqrt_eq_iff_mul_self_eq ];
    · field_simp;
      exact ⟨ fun h => by nlinarith, fun h => by rw [ h ] ; ring ⟩;
    · positivity;
    · positivity

/-
============================================================================
Claim 4: Nash bargaining on a linear Pareto frontier
============================================================================

On R + P = c with R, P ≥ 0, the product RP is maximized at R = P = c/2.
-/
theorem nash_linear_frontier {c : ℝ} (hc : 0 < c) {R P : ℝ}
    (hR : 0 ≤ R) (hP : 0 ≤ P) (hsum : R + P = c) :
    R * P ≤ (c / 2) * (c / 2) := by
  nlinarith [ sq_nonneg ( R - P ) ]

/-
============================================================================
Claim 5: Sensitivity — H is more sensitive to the weaker score
============================================================================

When R < P and R + P > 0, the elasticity w.r.t. R exceeds that w.r.t. P.
-/
theorem harmonicMean_elasticity_weaker {R P : ℝ} (hR : 0 < R) (hP : 0 < P)
    (h : R < P) : partialH_P R P < partialH_R R P := by
  unfold partialH_P partialH_R;
  gcongr

/-
============================================================================
Claim 6: Weighted linear scorer violates P1
============================================================================

W(0, 1) = 0.4, which is nonzero.
-/
theorem weightedLinear_violates_P1 : weightedLinear 0 1 = 0.4 := by
  unfold weightedLinear; norm_num;

/-
============================================================================
Claim 1 (partial): Among power means M_p, only p = -1 satisfies P1
============================================================================

The power mean M_p(R, P) = ((R^p + P^p)/2)^(1/p) for p > 0 violates P1:
    M_p(0, 1) > 0 when p > 0.
-/
theorem powerMean_pos_violates_P1 {p : ℝ} (_hp : 0 < p) :
    (((0 : ℝ) ^ p + (1 : ℝ) ^ p) / 2) ^ (1 / p) > 0 := by
  positivity

/-
============================================================================
Additional: harmonicMean is nonneg for nonneg inputs
============================================================================

H(R, P) ≥ 0 when R ≥ 0 and P ≥ 0.
-/
theorem harmonicMean_nonneg {R P : ℝ} (hR : 0 ≤ R) (hP : 0 ≤ P) :
    0 ≤ harmonicMean R P := by
  unfold harmonicMean;
  positivity

/-
============================================================================
Additional: harmonicMean is at most 1 for inputs in [0, 1]
============================================================================

H(R, P) ≤ 1 when R, P ∈ [0, 1].
-/
theorem harmonicMean_le_one {R P : ℝ} (hR0 : 0 ≤ R) (hR1 : R ≤ 1)
    (hP0 : 0 ≤ P) (hP1 : P ≤ 1) :
    harmonicMean R P ≤ 1 := by
  exact ( if h : R + P = 0 then by norm_num [ h, harmonicMean ] else by rw [ harmonicMean ] ; rw [ if_neg h ] ; rw [ div_le_iff₀ ( by contrapose! h; linarith ) ] ; nlinarith ) ;

end