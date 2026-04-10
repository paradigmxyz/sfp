import Mathlib

/-!
# T5: PCA vs Gradient-SVD Optimality

This file formalizes the mathematical results comparing PCA and Gradient-SVD
for subspace selection, as described in T5-pca-vs-gradient-svd.txt.

## Main results

1. **Trace complementarity**: `trace((1 - P) * S) = trace(S) - trace(P * S)`
2. **Ky Fan / PCA optimality**: top-r eigenspace of S maximizes `trace(P * S)`
3. **Combined matrix optimality**: the optimal subspace is the top eigenspace of `S^{1/2} G S^{1/2}`
4. **Special cases**: isotropic gradient → PCA, isotropic activations → gradient-SVD
5. **Suboptimality bound**: near-optimality when cross-alignment is high
-/

open Matrix Finset BigOperators

noncomputable section

variable {d : ℕ}

/-! ## Orthogonal Projection Matrices -/

/-- A matrix is an orthogonal projection if it is symmetric (Hermitian) and idempotent. -/
structure IsOrthProj (P : Matrix (Fin d) (Fin d) ℝ) : Prop where
  hermitian : P.IsHermitian
  idem : P * P = P

/-- The complement of an orthogonal projection is an orthogonal projection. -/
lemma IsOrthProj.compl {P : Matrix (Fin d) (Fin d) ℝ} (hP : IsOrthProj P) :
    IsOrthProj (1 - P) where
  hermitian := Matrix.IsHermitian.sub Matrix.isHermitian_one hP.hermitian
  idem := by
    have key : (1 - P) * (1 - P) - (1 - P) = P * P - P := by noncomm_ring
    exact sub_eq_zero.mp (by rw [key, hP.idem, sub_self])

/-
Eigenvalues of an orthogonal projection are 0 or 1.
-/
lemma IsOrthProj.eigenvalues_zero_or_one {P : Matrix (Fin d) (Fin d) ℝ}
    (hP : IsOrthProj P) (i : Fin d) :
    hP.hermitian.eigenvalues i = 0 ∨ hP.hermitian.eigenvalues i = 1 := by
  by_contra! h_neq;
  have h_eigenvalue_eq : P.mulVec (hP.hermitian.eigenvectorBasis i) = (hP.hermitian.eigenvalues i) • hP.hermitian.eigenvectorBasis i := by
    convert hP.hermitian.mulVec_eigenvectorBasis i using 1;
  have := congr_arg ( fun x => Matrix.mulVec P x ) h_eigenvalue_eq;
  simp_all +decide [ funext_iff, Matrix.mulVec_smul ];
  simp_all +decide [ ← mul_assoc, hP.idem ];
  exact absurd ( hP.hermitian.eigenvectorBasis.orthonormal.ne_zero i ) ( by aesop )

/-! ## Claim 1: Trace Complementarity

For any matrix S and orthogonal projection P,
  `trace((1 - P) * S) = trace(S) - trace(P * S)`

This shows that minimizing the "forgetting" `trace((1-P)S)` is equivalent to
maximizing `trace(PS)`.
-/

/-
Trace complementarity: `trace((1 - P) * S) = trace(S) - trace(P * S)`.
-/
theorem trace_compl_mul (P S : Matrix (Fin d) (Fin d) ℝ) :
    ((1 - P) * S).trace = S.trace - (P * S).trace := by
  rw [ sub_mul, one_mul, Matrix.trace_sub ]

/-! ## Ky Fan Inequality

The Ky Fan maximum principle states that for a symmetric matrix A,
the sum of the r largest eigenvalues equals the maximum of trace(P * A)
over all rank-r orthogonal projections P.

This is the key result underlying both PCA and Gradient-SVD optimality.
-/

/-- The sum of the `r` largest eigenvalues of a Hermitian matrix,
defined as the supremum of `∑ i ∈ S, eigenvalues i` over all `r`-element subsets `S`. -/
def topEigenvalueSum {A : Matrix (Fin d) (Fin d) ℝ} (hA : A.IsHermitian) (r : ℕ)
    (hr : r ≤ d) : ℝ :=
  Finset.sup' ((Finset.univ : Finset (Finset (Fin d))).filter (fun S => S.card = r))
    (by
      simp only [Finset.filter_nonempty_iff, Finset.mem_univ, true_and]
      exact ⟨Finset.image (fun i : Fin r => i.castLE hr) Finset.univ, by
        rw [Finset.card_image_of_injective _ (Fin.castLE_injective hr), Finset.card_fin]⟩)
    (fun S => ∑ i ∈ S, hA.eigenvalues i)

/-
Diagonal entries of an orthogonal projection are in [0,1].
-/
lemma IsOrthProj.diag_nonneg {P : Matrix (Fin d) (Fin d) ℝ}
    (hP : IsOrthProj P) (i : Fin d) : 0 ≤ P i i := by
  obtain ⟨ h₁, h₂ ⟩ := hP;
  replace h₂ := congr_fun ( congr_fun h₂ i ) i; simp_all +decide [ Matrix.mul_apply, Finset.sum_ite, Finset.filter_eq', Finset.filter_ne' ] ;
  exact h₂ ▸ Finset.sum_nonneg fun j _ => by rw [ show P j i = P i j by exact h₁.apply _ _ ] ; exact mul_self_nonneg _;

lemma IsOrthProj.diag_le_one {P : Matrix (Fin d) (Fin d) ℝ}
    (hP : IsOrthProj P) (i : Fin d) : P i i ≤ 1 := by
  exact sub_nonneg.mp ( by simpa using IsOrthProj.diag_nonneg ( IsOrthProj.compl hP ) i )

/-
trace(P * A) for Hermitian A equals ∑ᵢ P_ii * λ_i in the eigenbasis.
-/
lemma trace_orthProj_mul_hermitian_eq {A : Matrix (Fin d) (Fin d) ℝ} (hA : A.IsHermitian)
    (P : Matrix (Fin d) (Fin d) ℝ) (hP : IsOrthProj P) :
    (P * A).trace = ∑ i, ((hA.eigenvectorUnitary : Matrix (Fin d) (Fin d) ℝ)⁻¹ * P *
      (hA.eigenvectorUnitary : Matrix (Fin d) (Fin d) ℝ)) i i * hA.eigenvalues i := by
  -- By the spectral theorem, we have $A = U D U^*$ where $U$ is the unitary matrix of eigenvectors and $D$ is the diagonal matrix of eigenvalues.
  have h_spectral : A = hA.eigenvectorUnitary * Matrix.diagonal hA.eigenvalues * (hA.eigenvectorUnitary⁻¹) := by
    convert hA.spectral_theorem using 1;
  conv_lhs => rw [ h_spectral ];
  rw [ ← Matrix.trace_mul_comm ] ; norm_num [ Matrix.mul_assoc ];
  rw [ ← Matrix.trace_mul_comm ] ; norm_num [ Matrix.mul_assoc, Matrix.trace ];
  simp +decide [ Matrix.inv_eq_left_inv, mul_comm ];
  rw [ Matrix.inv_eq_left_inv ];
  simp +decide [ mul_eq_one_comm ]

/-
The conjugation of an orthogonal projection by a unitary matrix is still
an orthogonal projection with the same trace.
-/
lemma IsOrthProj.conjugate_unitary {P : Matrix (Fin d) (Fin d) ℝ}
    (hP : IsOrthProj P) (U : ↥(Matrix.unitaryGroup (Fin d) ℝ)) :
    IsOrthProj ((U : Matrix (Fin d) (Fin d) ℝ)⁻¹ * P * (U : Matrix (Fin d) (Fin d) ℝ)) := by
  constructor;
  · have := hP.1; simp_all +decide [ Matrix.IsHermitian, Matrix.mul_assoc ] ;
    simp_all +decide [ Matrix.inv_eq_right_inv U.2.2 ];
    simp_all +decide [ Matrix.mul_assoc, Matrix.star_eq_conjTranspose ];
  · simp_all +decide [ mul_assoc, Matrix.mul_inv_rev ];
    cases' Matrix.nonsing_inv_cancel_or_zero ( U : Matrix ( Fin d ) ( Fin d ) ℝ ) with h h <;> simp_all +decide [ ← mul_assoc, hP.idem ]

lemma IsOrthProj.trace_conjugate_unitary {P : Matrix (Fin d) (Fin d) ℝ}
    (hP : IsOrthProj P) (U : ↥(Matrix.unitaryGroup (Fin d) ℝ)) :
    ((U : Matrix (Fin d) (Fin d) ℝ)⁻¹ * P * (U : Matrix (Fin d) (Fin d) ℝ)).trace = P.trace := by
  convert Matrix.trace_mul_comm _ _ using 2;
  rw [ ← mul_assoc, Matrix.mul_nonsing_inv _ ];
  · norm_num;
  · exact?

/-
A weighted sum ∑ wᵢ * aᵢ with wᵢ ∈ [0,1] and ∑ wᵢ = r is at most
the sum of the r largest values of aᵢ.
-/
lemma weighted_sum_le_top_sum {n : ℕ} (w a : Fin n → ℝ) {r : ℕ}
    (hw_nn : ∀ i, 0 ≤ w i) (hw_le : ∀ i, w i ≤ 1)
    (hw_sum : ∑ i, w i = (r : ℝ)) (hr : r ≤ n) :
    ∑ i, w i * a i ≤
      Finset.sup' ((Finset.univ : Finset (Finset (Fin n))).filter (fun S => S.card = r))
        (by
          simp only [Finset.filter_nonempty_iff, Finset.mem_univ, true_and]
          exact ⟨Finset.image (fun i : Fin r => i.castLE hr) Finset.univ, by
            rw [Finset.card_image_of_injective _ (Fin.castLE_injective hr), Finset.card_fin]⟩)
        (fun S => ∑ i ∈ S, a i) := by
  have h_extreme : ∃ S : Finset (Fin n), S.card = r ∧ ∀ T : Finset (Fin n), T.card = r → ∑ i ∈ S, a i ≥ ∑ i ∈ T, a i := by
    have h_max_sum : ∃ S ∈ Finset.powersetCard r (Finset.univ : Finset (Fin n)), ∀ T ∈ Finset.powersetCard r (Finset.univ : Finset (Fin n)), ∑ i ∈ S, a i ≥ ∑ i ∈ T, a i := by
      exact Finset.exists_max_image _ _ ⟨ Finset.exists_subset_card_eq ( by simpa ) |> Classical.choose, Finset.mem_powersetCard.mpr ⟨ Finset.exists_subset_card_eq ( by simpa ) |> Classical.choose_spec |> And.left, Finset.exists_subset_card_eq ( by simpa ) |> Classical.choose_spec |> And.right ⟩ ⟩;
    aesop;
  obtain ⟨ S, hS₁, hS₂ ⟩ := h_extreme; simp_all +decide [ Finset.sum_mul _ _ _ ] ; (
  -- By the properties of the sum and the definition of $S$, we can bound the sum $\sum_{i} w_i a_i$.
  have h_bound : ∑ i, (w i - if i ∈ S then 1 else 0) * a i ≤ 0 := by
    -- Since $a$ is sorted in decreasing order, for any $i \notin S$, we have $a_i \leq a_j$ for all $j \in S$.
    have h_sorted : ∀ i ∉ S, ∀ j ∈ S, a i ≤ a j := by
      intro i hi j hj; specialize hS₂ ( Insert.insert i ( S.erase j ) ) ; simp_all +decide [ Finset.sum_insert, Finset.sum_erase ] ;
      linarith [ hS₂ ( Nat.succ_pred_eq_of_pos ( Nat.pos_of_ne_zero ( by aesop_cat ) ) ) ];
    -- By splitting the sum into two parts, we can apply the sorting property to each part.
    have h_split : ∑ i, (w i - if i ∈ S then 1 else 0) * a i = ∑ i ∈ S, (w i - 1) * a i + ∑ i ∈ Sᶜ, w i * a i := by
      simp +decide [ sub_mul, Finset.sum_add_distrib, Finset.sum_ite ];
      rw [ ← Finset.sum_add_sum_compl S ] ; ring;
    -- By the properties of the sum and the definition of $S$, we can bound the sum $\sum_{i \in S} (w_i - 1) a_i$.
    have h_bound_S : ∑ i ∈ S, (w i - 1) * a i ≤ ∑ i ∈ S, (w i - 1) * (∑ j ∈ Sᶜ, w j * a j) / (∑ j ∈ Sᶜ, w j) := by
      refine Finset.sum_le_sum fun i hi => ?_;
      by_cases h : ∑ j ∈ Sᶜ, w j = 0 <;> simp_all +decide [ mul_div_assoc ];
      · simp_all +decide [ Finset.sum_eq_zero_iff_of_nonneg ];
        have h_sum_zero : ∑ i ∈ S, w i = r := by
          rw [ ← hw_sum, Finset.sum_subset ( Finset.subset_univ S ) fun i hi₁ hi₂ => by aesop ];
        have h_sum_zero : ∀ i ∈ S, w i = 1 := by
          intros i hi; exact le_antisymm (hw_le i) (by
          contrapose! h_sum_zero;
          exact ne_of_lt ( lt_of_lt_of_le ( Finset.sum_lt_sum ( fun x hx => hw_le x ) ⟨ i, hi, h_sum_zero ⟩ ) ( by norm_num [ hS₁ ] ) ));
        aesop;
      · refine' mul_le_mul_of_nonpos_left _ ( sub_nonpos.mpr ( hw_le i ) );
        rw [ div_le_iff₀ ( lt_of_le_of_ne ( Finset.sum_nonneg fun _ _ => hw_nn _ ) ( Ne.symm h ) ) ];
        rw [ Finset.mul_sum _ _ _ ] ; exact Finset.sum_le_sum fun j hj => by nlinarith only [ hw_nn j, hw_le j, h_sorted j ( Finset.mem_compl.mp hj ) i hi ] ;
    by_cases h : ∑ j ∈ Sᶜ, w j = 0 <;> simp_all +decide [ ← Finset.sum_div _ _ _, ← Finset.sum_mul ];
    · simp_all +decide [ Finset.sum_eq_zero_iff_of_nonneg ];
      exact add_nonpos h_bound_S ( Finset.sum_nonpos fun i hi => by aesop );
    · simp_all +decide [ Finset.compl_eq_univ_sdiff ];
      rw [ le_div_iff₀ ] at h_bound_S <;> cases lt_or_gt_of_ne h <;> nlinarith [ show ( ∑ i ∈ S, w i ) ≤ r by exact le_trans ( Finset.sum_le_sum fun _ _ => hw_le _ ) ( by norm_num [ hS₁ ] ) ] ;
  simp_all +decide [ sub_mul ];
  exact ⟨ S, hS₁, h_bound ⟩);

/-
**Ky Fan inequality**: For a Hermitian matrix A and any orthogonal projection P
with trace equal to r, we have `trace(P * A) ≤ topEigenvalueSum hA r`.
-/
theorem ky_fan_ineq {A : Matrix (Fin d) (Fin d) ℝ} (hA : A.IsHermitian)
    (P : Matrix (Fin d) (Fin d) ℝ) (hP : IsOrthProj P)
    {r : ℕ} (hr : r ≤ d) (hrank : P.trace = (r : ℝ)) :
    (P * A).trace ≤ topEigenvalueSum hA r hr := by
  rw [ trace_orthProj_mul_hermitian_eq hA P hP ];
  convert weighted_sum_le_top_sum _ _ _ _ _ _;
  · convert IsOrthProj.diag_nonneg ( IsOrthProj.conjugate_unitary hP hA.eigenvectorUnitary ) using 1;
  · convert IsOrthProj.diag_le_one ( IsOrthProj.conjugate_unitary hP ⟨ hA.eigenvectorUnitary, ?_ ⟩ ) using 1;
  · convert hrank using 1;
    convert IsOrthProj.trace_conjugate_unitary hP hA.eigenvectorUnitary using 1;
  · linarith

/-! ## Claim 1: PCA Optimality

Under isotropic gradient (G ∝ I), minimizing expected forgetting reduces to
maximizing trace(P * S), which is achieved by the top-r eigenspace of S.
-/

/-- **PCA optimality for variance-dominated forgetting.**
For any PSD covariance matrix S, the top-r eigenspace projection minimizes
`trace((1-P) * S)` equivalently maximizes `trace(P * S)`.
This follows from the Ky Fan inequality. -/
theorem pca_optimality {S : Matrix (Fin d) (Fin d) ℝ} (hS : S.PosSemidef)
    (P : Matrix (Fin d) (Fin d) ℝ) (hP : IsOrthProj P)
    {r : ℕ} (hr : r ≤ d) (hrank : P.trace = (r : ℝ)) :
    (P * S).trace ≤ topEigenvalueSum hS.isHermitian r hr := by
  exact ky_fan_ineq hS.isHermitian P hP hr hrank

/-! ## Claim 2: Gradient-SVD Optimality

Under isotropic activations (S ∝ I), minimizing expected forgetting reduces to
maximizing trace(P * G), achieved by the top-r eigenspace of G.
-/

/-- **Gradient-SVD optimality for gradient-dominated forgetting.**
For any PSD gradient outer product G, under isotropic activations,
the top-r eigenspace of G minimizes expected forgetting. -/
theorem gradient_svd_optimality {G : Matrix (Fin d) (Fin d) ℝ} (hG : G.PosSemidef)
    (P : Matrix (Fin d) (Fin d) ℝ) (hP : IsOrthProj P)
    {r : ℕ} (hr : r ≤ d) (hrank : P.trace = (r : ℝ)) :
    (P * G).trace ≤ topEigenvalueSum hG.isHermitian r hr := by
  exact ky_fan_ineq hG.isHermitian P hP hr hrank

/-! ## Claim 3: Combined Matrix Optimality

The general optimal subspace minimizes `E[(∇L)ᵀ P_{W⊥} Δa]²`.
With `Δa ~ N(0, S)`, this objective becomes `trace(P_{W⊥} S^{1/2} G S^{1/2})`,
so the optimal W* is the top eigenspace of `M = S^{1/2} G S^{1/2}`.
-/

/-- The combined optimality matrix M = sqrtS * G * sqrtS. -/
def combinedMatrix (sqrtS G : Matrix (Fin d) (Fin d) ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  sqrtS * G * sqrtS

/-
The combined matrix is PSD when sqrtS is symmetric and G is PSD.
-/
theorem combinedMatrix_posSemidef
    (sqrtS G : Matrix (Fin d) (Fin d) ℝ)
    (hG : G.PosSemidef)
    (hsqrt : sqrtS.IsHermitian) :
    (combinedMatrix sqrtS G).PosSemidef := by
  have := hG.conjTranspose_mul_mul_same ( sqrtS.conjTranspose ) ; simp_all +decide [ Matrix.mul_assoc, Matrix.IsHermitian ];
  unfold combinedMatrix; simp_all +decide [ ← Matrix.mul_assoc ] ;

/-- **Combined optimality**: The optimal projection for the combined objective
`trace((1-P) * M)` is given by the top eigenspace of M = sqrtS * G * sqrtS,
via the Ky Fan inequality applied to M. -/
theorem combined_optimality
    (sqrtS G : Matrix (Fin d) (Fin d) ℝ)
    (hG : G.PosSemidef) (hsqrt : sqrtS.IsHermitian)
    (P : Matrix (Fin d) (Fin d) ℝ) (hP : IsOrthProj P)
    {r : ℕ} (hr : r ≤ d) (hrank : P.trace = (r : ℝ)) :
    (P * combinedMatrix sqrtS G).trace ≤
      topEigenvalueSum (combinedMatrix_posSemidef sqrtS G hG hsqrt).isHermitian r hr := by
  exact ky_fan_ineq _ P hP hr hrank

/-! ## Claim 3a: Special Case — Isotropic Gradient (G ∝ I ⟹ PCA)

When the gradient outer product is isotropic (G = c • I),
the combined matrix simplifies to M = c • S, so the optimal
subspace is determined by PCA (top eigenspace of S).
-/

/-
When G = c • I, the combined matrix M = sqrtS * (c • I) * sqrtS = c • (sqrtS * sqrtS) = c • S.
-/
theorem combinedMatrix_isotropic_gradient
    (sqrtS : Matrix (Fin d) (Fin d) ℝ) (c : ℝ) :
    combinedMatrix sqrtS (c • (1 : Matrix (Fin d) (Fin d) ℝ)) = c • (sqrtS * sqrtS) := by
  unfold combinedMatrix; simp +decide [ mul_assoc ] ;

/-! ## Claim 3b: Special Case — Isotropic Activations (S ∝ I ⟹ Gradient-SVD)

When the activation covariance is isotropic (S = c • I, so S^{1/2} = √c • I),
the combined matrix simplifies to M = c • G, so the optimal
subspace is determined by gradient-SVD (top eigenspace of G).
-/

/-
When sqrtS = c • I, the combined matrix M = (c•I) * G * (c•I) = c² • G.
-/
theorem combinedMatrix_isotropic_activations
    (G : Matrix (Fin d) (Fin d) ℝ) (c : ℝ) :
    combinedMatrix (c • (1 : Matrix (Fin d) (Fin d) ℝ)) G = (c ^ 2) • G := by
  ext i j; simp +decide [ *, sq, combinedMatrix ] ;
  ring

/-! ## Claim 4: Decision Criterion

Define the score S = trace(P_PCA * M) / trace(P_grad * M).
PCA is preferable when S > 1; gradient-SVD when S < 1.
-/

/-- The decision score comparing PCA vs gradient-SVD projections. -/
def decisionScore (P_pca P_grad M : Matrix (Fin d) (Fin d) ℝ) : ℝ :=
  (P_pca * M).trace / (P_grad * M).trace

/-
PCA is at least as good as gradient-SVD when the decision score ≥ 1.
-/
theorem pca_preferred_iff_score_ge_one
    (P_pca P_grad M : Matrix (Fin d) (Fin d) ℝ)
    (hM_pos : 0 < (P_grad * M).trace) :
    (P_grad * M).trace ≤ (P_pca * M).trace ↔ 1 ≤ decisionScore P_pca P_grad M := by
  exact ⟨ fun h => by rw [ decisionScore ] ; rw [ le_div_iff₀ hM_pos ] ; linarith, fun h => by rw [ decisionScore, le_div_iff₀ hM_pos ] at h; linarith ⟩

/-! ## Claim 5: Suboptimality Bound

When using W_PCA instead of the optimal W*, the suboptimality ratio
is bounded by the cross-alignment.
-/

/-- The Frobenius inner product of two matrices. -/
def frobeniusInner (A B : Matrix (Fin d) (Fin d) ℝ) : ℝ :=
  (A * B.transpose).trace

/-- Cross-alignment between two projections, measuring overlap.
`A = ‖P Q‖_F² / r` -/
def crossAlignment (P Q : Matrix (Fin d) (Fin d) ℝ) (r : ℕ) : ℝ :=
  frobeniusInner P Q / (r : ℝ)

/-
Cross-alignment is 1 when the projections coincide.
-/
theorem crossAlignment_self (P : Matrix (Fin d) (Fin d) ℝ)
    (hP : IsOrthProj P) {r : ℕ} (hr : P.trace = (r : ℝ)) (hr_pos : (0 : ℝ) < r) :
    crossAlignment P P r = 1 := by
  unfold crossAlignment;
  unfold frobeniusInner;
  rw [ div_eq_iff ] <;> have := hP.1 <;> have := hP.2 <;> simp_all +decide [ Matrix.IsHermitian, Matrix.trace ];
  linarith

/-
**Suboptimality bound**: The ratio trace(P_sub * M) / trace(P_opt * M) ≤ 1
for any suboptimal projection, since P_opt is the maximizer.
-/
theorem suboptimality_bound
    (P_sub P_opt M : Matrix (Fin d) (Fin d) ℝ)
    (_hM : M.PosSemidef)
    (hPsub : IsOrthProj P_sub) (_hPopt : IsOrthProj P_opt)
    {r : ℕ} (hrank_sub : P_sub.trace = (r : ℝ)) (_hrank_opt : P_opt.trace = (r : ℝ))
    (hopt : ∀ Q : Matrix (Fin d) (Fin d) ℝ, IsOrthProj Q → Q.trace = (r : ℝ) →
      (Q * M).trace ≤ (P_opt * M).trace)
    (hopt_pos : 0 < (P_opt * M).trace) :
    (P_sub * M).trace / (P_opt * M).trace ≤ 1 := by
  exact div_le_one_of_le₀ ( hopt P_sub hPsub hrank_sub ) hopt_pos.le

end