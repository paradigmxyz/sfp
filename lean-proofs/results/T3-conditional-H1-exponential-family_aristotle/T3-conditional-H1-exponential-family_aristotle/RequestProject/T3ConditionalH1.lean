/-
# Theorem T3: Conditional H1 Under Exponential Family

Formalization of the mathematical claims from T3-conditional-H1-exponential-family.txt.

We abstract the neural network setting as follows:
- E is a finite-dimensional inner product space (the activation space)
- W is a subspace of E, with P_W the orthogonal projection onto W
- The "old-task loss" L_old factors through P_W(a) via some function g
  (this is the consequence of the exponential family assumption with
   linear sufficient statistic T(a) = P_W(a))
- The key claims are about Lipschitz bounds, orthogonal complement
  irrelevance, and multi-layer extensions.
-/

import Mathlib

open scoped BigOperators NNReal
open Submodule

set_option maxHeartbeats 8000000

noncomputable section

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-! ## Orthogonal Projection Setup -/

/-- The orthogonal projection onto W, viewed as a continuous linear map E → E. -/
def projOnto (W : Submodule ℝ E) [W.HasOrthogonalProjection] : E →L[ℝ] E :=
  W.subtypeL.comp W.orthogonalProjection

/-- The orthogonal projection onto the orthogonal complement W⊥. -/
def projOrthComplement (W : Submodule ℝ E) [W.HasOrthogonalProjection]
    [Wᗮ.HasOrthogonalProjection] : E →L[ℝ] E :=
  Wᗮ.subtypeL.comp Wᗮ.orthogonalProjection

/-! ## Claim 1: Sufficiency of Projected Activations

Under the exponential family assumption with linear sufficient statistic
T(a) = P_W(a), the old-task loss depends on activations only through P_W(a).
That is, L_old(a) = g(P_W(a)) for some function g.

We formalize this as: if L_old factors through P_W, then L_old(a) = L_old(a')
whenever P_W(a) = P_W(a'). -/

/-
If the loss function factors through the orthogonal projection onto W,
then two activations with the same projection yield the same loss.
-/
theorem loss_depends_only_on_projection
    (W : Submodule ℝ E) [W.HasOrthogonalProjection]
    (g : E → ℝ)
    (L_old : E → ℝ)
    (hfactor : ∀ a : E, L_old a = g (projOnto W a))
    (a a' : E)
    (hproj : projOnto W a = projOnto W a') :
    L_old a = L_old a' := by
  rw [ hfactor, hfactor, hproj ]

/-! ## Claim 2: H1 as a Consequence (Lipschitz Bound)

Under the exponential family assumption with linear sufficient statistic,
if g is Lipschitz with constant C, then H1 holds:
  |L_old(θ) - L_old(θ₀)| ≤ C · ‖P_W(a(θ) - a(θ₀))‖ -/

/-
If L_old = g ∘ P_W and g is C-Lipschitz, then the H1 bound holds.
The orthogonal projection is linear, so P_W(a - a₀) = P_W(a) - P_W(a₀),
and the Lipschitz condition gives the bound.
-/
theorem H1_from_lipschitz
    (W : Submodule ℝ E) [W.HasOrthogonalProjection]
    (g : E → ℝ)
    (C : ℝ≥0)
    (hLip : LipschitzWith C g)
    (L_old : E → ℝ)
    (hfactor : ∀ a : E, L_old a = g (projOnto W a))
    (a a₀ : E) :
    |L_old a - L_old a₀| ≤ C * ‖projOnto W (a - a₀)‖ := by
  have := hLip.dist_le_mul ( projOnto W a ) ( projOnto W a₀ );
  simp_all +decide [ dist_eq_norm, projOnto ]

/-! ## Claim 3: Orthogonal Complement Irrelevance

The component P_{W⊥}(a) does not affect the old-task loss.
Formally: adding any vector from W⊥ does not change the loss. -/

/-
The orthogonal projection onto W kills vectors in W⊥.
-/
theorem projOnto_orthogonal_eq_zero
    (W : Submodule ℝ E) [W.HasOrthogonalProjection]
    (v : E) (hv : v ∈ Wᗮ) :
    projOnto W v = 0 := by
  simp [projOnto, Submodule.orthogonalProjection_mem_subspace_orthogonalComplement_eq_zero hv]

/-
If L_old factors through P_W, then adding a W⊥-component does not change the loss.
This means fine-tuning can freely modify the orthogonal complement without forgetting.
-/
theorem orthogonal_complement_irrelevance
    (W : Submodule ℝ E) [W.HasOrthogonalProjection]
    (g : E → ℝ)
    (L_old : E → ℝ)
    (hfactor : ∀ a : E, L_old a = g (projOnto W a))
    (a : E) (v : E) (hv : v ∈ Wᗮ) :
    L_old (a + v) = L_old a := by
  rw [ hfactor, hfactor ];
  rw [ projOnto, ContinuousLinearMap.coe_comp' ];
  simp [hv, orthogonalProjection_mem_subspace_orthogonalComplement_eq_zero]

/-! ## Claim 4: Diagnostic Test (R² Interpretation)

R² = 1 - Var(L_old | P_W(a)) / Var(L_old)

We formalize the key implication: if the conditional variance given P_W(a) is zero,
then L_old is measurable with respect to P_W(a), i.e., L_old factors through P_W. -/

/-
If the variance of L_old conditional on P_W(a) is zero (R² = 1),
then L_old is determined by P_W(a). We state this abstractly:
if L_old(a) = L_old(a') whenever P_W(a) = P_W(a'), then L_old factors through P_W.
-/
theorem R_squared_one_implies_factorization
    (W : Submodule ℝ E) [W.HasOrthogonalProjection]
    (L_old : E → ℝ)
    (hdet : ∀ a a' : E, projOnto W a = projOnto W a' → L_old a = L_old a') :
    ∃ g : E → ℝ, ∀ a : E, L_old a = g (projOnto W a) := by
  have h_idempotent : ∀ x : E, projOnto W (projOnto W x) = projOnto W x := by
    unfold projOnto;
    simp +decide [ Submodule.starProjection_eq_self_iff ];
  aesop

/-! ## Claim 5: Multi-layer Extension

For L layers, if each layer independently satisfies the condition,
then H1 holds jointly with a sum bound. -/

/-
Multi-layer H1 bound: if each layer ℓ has a Lipschitz function gℓ
such that the loss contribution of layer ℓ factors through P_{Wℓ},
and the total loss is the sum of layer contributions, then the
multi-layer H1 bound holds.
-/
theorem multi_layer_H1
    (n : ℕ)
    (E_layer : Fin n → Type*)
    [∀ i, NormedAddCommGroup (E_layer i)]
    [∀ i, InnerProductSpace ℝ (E_layer i)]
    (W : ∀ i, Submodule ℝ (E_layer i))
    [∀ i, (W i).HasOrthogonalProjection]
    (g : ∀ i, E_layer i → ℝ)
    (C : Fin n → ℝ≥0)
    (hLip : ∀ i, LipschitzWith (C i) (g i))
    (L_layer : ∀ i, E_layer i → ℝ)
    (hfactor : ∀ i a, L_layer i a = g i (projOnto (W i) a))
    (L_old : (∀ i, E_layer i) → ℝ)
    (hsum : ∀ a, L_old a = ∑ i : Fin n, L_layer i (a i))
    (a a₀ : ∀ i, E_layer i) :
    |L_old a - L_old a₀| ≤ ∑ i : Fin n, (C i : ℝ) * ‖projOnto (W i) (a i - a₀ i)‖ := by
  rw [ hsum, hsum, ← Finset.sum_sub_distrib ];
  refine' le_trans ( Finset.abs_sum_le_sum_abs _ _ ) ( Finset.sum_le_sum _ );
  intro i _; specialize hLip i
  have := hLip.dist_le_mul (projOnto (W i) (a i)) (projOnto (W i) (a₀ i))
  simp_all [dist_eq_norm, projOnto]

end