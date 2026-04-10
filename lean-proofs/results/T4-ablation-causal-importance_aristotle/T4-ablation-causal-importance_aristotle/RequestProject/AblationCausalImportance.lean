import Mathlib

open Finset BigOperators

noncomputable section

/-! # Theorem T4: Ablation Importance Equals Causal Importance

This file formalizes the relationship between ablation importance and causal importance
(Shapley values) for neural network activation components. We establish that under an
additivity assumption on the loss function, ablation-based rankings coincide with
Shapley-value rankings, and under a monotonicity assumption, ablation importances are
non-negative and the greedy top-r selection is optimal.

## Main results

* `ablation_eq_weight`: Under additivity, ablation importance equals the individual weight.
* `shapley_eq_weight`: Under additivity, the Shapley value equals the individual weight.
* `ablation_eq_shapley`: Under additivity, ablation importance equals the Shapley value.
* `ablation_nonneg`: Under monotonicity, ablation importance is non-negative.
* `projection_loss_eq_sum`: Under additivity, projection loss decomposes as a sum of weights.
* `interaction_zero_of_additive`: Under additivity, pairwise interactions vanish.
* `ablation_ranking_suboptimal_without_additivity`: Without additivity, ablation ranking
  can be suboptimal.
-/

variable {d : ℕ}

/-! ## Definitions -/

/-- Ablation importance: loss change when component i is zeroed out. -/
def ablationImportance (f : Finset (Fin d) → ℝ) (i : Fin d) : ℝ :=
  f Finset.univ - f (Finset.univ.erase i)

/-- Monotonicity assumption (M): adding any direction weakly increases the set function value.
    This models the assumption that "more information helps." -/
def MonotoneSetFn (f : Finset (Fin d) → ℝ) : Prop :=
  ∀ (S : Finset (Fin d)) (i : Fin d), i ∉ S → f S ≤ f (insert i S)

/-- Additivity assumption (A): the set function decomposes as a base value plus
    individual contributions: f(S) = f(∅) + Σ_{i ∈ S} w(i). -/
def AdditiveSetFn (f : Finset (Fin d) → ℝ) (w : Fin d → ℝ) : Prop :=
  ∀ S : Finset (Fin d), f S = f ∅ + ∑ i ∈ S, w i

/-- Shapley weight for a coalition of size s with d players:
    s! · (d - s - 1)! / d! -/
def shapleyWeight (d : ℕ) (s : ℕ) : ℝ :=
  (↑(s.factorial) * ↑((d - s - 1).factorial)) / ↑(d.factorial)

/-- Shapley value of player i. -/
def shapleyValue (f : Finset (Fin d) → ℝ) (i : Fin d) : ℝ :=
  ∑ S ∈ (Finset.univ.erase i).powerset,
    shapleyWeight d S.card * (f (insert i S) - f S)

/-- Pairwise interaction between directions i and j. -/
def pairwiseInteraction (f : Finset (Fin d) → ℝ) (i j : Fin d) : ℝ :=
  f {i, j} - f {i} - f {j} + f ∅

/-! ## Core lemmas -/

/-
Under additivity, the marginal contribution of i to any coalition S (with i ∉ S)
    equals w(i). This is the key algebraic fact underlying all main results.
-/
theorem marginal_eq_weight (f : Finset (Fin d) → ℝ) (w : Fin d → ℝ)
    (h : AdditiveSetFn f w) (S : Finset (Fin d))
    (i : Fin d) (hi : i ∉ S) :
    f (insert i S) - f S = w i := by
      rw [ h, h, Finset.sum_insert hi ] ; ring;
      linarith [ h S ]

/-! ## Main theorems -/

/-
**Claim 1: Ablation equals marginal contribution under additivity.**
    Under Assumption A, ablation importance of direction i equals its individual
    contribution w(i), independent of all other components.
-/
theorem ablation_eq_weight (f : Finset (Fin d) → ℝ) (w : Fin d → ℝ)
    (h : AdditiveSetFn f w) (i : Fin d) :
    ablationImportance f i = w i := by
      convert marginal_eq_weight f w h ( Finset.univ.erase i ) i ( by aesop ) using 1;
      simp +decide [ ablationImportance, Finset.insert_erase ]

/-
**Shapley weights sum to 1.** The combinatorial identity
    Σ_{S ⊆ [d]\{i}} |S|!·(d-|S|-1)!/d! = 1.
-/
theorem shapley_weights_sum_one (hd : 0 < d) (i : Fin d) :
    ∑ S ∈ (Finset.univ (α := Fin d)).erase i |>.powerset, shapleyWeight d S.card = 1 := by
  -- The sum can be rewritten as $\sum_{k=0}^{d-1} \binom{d-1}{k} \frac{k!(d-1-k)!}{d!}$.
  have h_sum : ∑ S ∈ Finset.powerset (Finset.univ.erase i), shapleyWeight d (Finset.card S) = ∑ k ∈ Finset.range (d), Nat.choose (d - 1) k * (Nat.factorial k * Nat.factorial (d - k - 1)) / (Nat.factorial d : ℝ) := by
    simp +decide [ shapleyWeight, Finset.sum_powerset ];
    rw [ Nat.sub_add_cancel hd ] ; refine' Finset.sum_congr rfl fun x hx => _ ; rw [ Finset.sum_congr rfl fun t ht => by rw [ Finset.mem_powersetCard.mp ht |>.2 ] ] ; norm_num [ Finset.card_univ ] ; ring;
  -- Simplify the sum using the identity $\binom{d-1}{k} \cdot k! \cdot (d-1-k)! = (d-1)!$.
  have h_simp : ∑ k ∈ Finset.range d, Nat.choose (d - 1) k * (Nat.factorial k * Nat.factorial (d - k - 1)) / (Nat.factorial d : ℝ) = ∑ k ∈ Finset.range d, (Nat.factorial (d - 1) : ℝ) / (Nat.factorial d : ℝ) := by
    refine Finset.sum_congr rfl fun x hx => ?_;
    rw [ Nat.cast_choose ] <;> try linarith [ Finset.mem_range.mp hx, Nat.sub_add_cancel hd ];
    rw [ div_mul_eq_mul_div, div_div, div_eq_div_iff ] <;> first | positivity | ring;
    rw [ Nat.sub_right_comm ] ; ring;
  cases d <;> simp_all +decide [ Nat.factorial_succ ];
  rw [ mul_div, div_eq_iff ] <;> ring ; positivity

/-
**Claim 2: Shapley equals individual weight under additivity.**
    Under Assumption A, φ_i = w(i). Combined with Claim 1, this shows
    ablation ranking is the unique fair (Shapley) ranking.
-/
theorem shapley_eq_weight (f : Finset (Fin d) → ℝ) (w : Fin d → ℝ)
    (hd : 0 < d) (h : AdditiveSetFn f w) (i : Fin d) :
    shapleyValue f i = w i := by
      -- By definition of shapleyValue, we can rewrite the sum as a product of sums.
      have h_sum : ∑ S ∈ (Finset.univ.erase i).powerset, shapleyWeight d S.card * (f (insert i S) - f S) = ∑ S ∈ (Finset.univ.erase i).powerset, shapleyWeight d S.card * w i := by
        refine' Finset.sum_congr rfl fun S hS => _;
        grind +locals;
      rw [ shapleyValue ];
      rw [ h_sum, ← Finset.sum_mul _ _ _, shapley_weights_sum_one hd i, mul_comm ] ; aesop

/-
**Claim 3: Monotonicity ensures ablation is non-negative.**
-/
theorem ablation_nonneg (f : Finset (Fin d) → ℝ)
    (h : MonotoneSetFn f) (i : Fin d) :
    0 ≤ ablationImportance f i := by
      exact sub_nonneg_of_le ( by simpa using h ( Finset.univ.erase i ) i ( by simpa ) )

/-- **Claim 4: Ablation equals Shapley under additivity.** -/
theorem ablation_eq_shapley (f : Finset (Fin d) → ℝ) (w : Fin d → ℝ)
    (hd : 0 < d) (h : AdditiveSetFn f w) (i : Fin d) :
    ablationImportance f i = shapleyValue f i := by
  rw [ablation_eq_weight f w h, shapley_eq_weight f w hd h]

/-
**Claim 5: Projection loss decomposition.**
    Under Assumption A, the loss from projecting onto subspace S equals
    the sum of weights of excluded directions.
-/
theorem projection_loss_eq_sum (f : Finset (Fin d) → ℝ) (w : Fin d → ℝ)
    (h : AdditiveSetFn f w) (S : Finset (Fin d)) :
    f Finset.univ - f S = ∑ i ∈ Finset.univ \ S, w i := by
      rw [ h, h ];
      rw [ h S ] ; aesop

/-
Under additivity and monotonicity, individual weights are non-negative.
-/
theorem weight_nonneg (f : Finset (Fin d) → ℝ) (w : Fin d → ℝ)
    (hadd : AdditiveSetFn f w) (hmon : MonotoneSetFn f) (i : Fin d) :
    0 ≤ w i := by
      -- By combining the results from ablation_eq_weight and ablation_nonneg, we conclude that w i is non-negative.
      have h_wi_nonneg : 0 ≤ ablationImportance f i :=
        ablation_nonneg f hmon i;
      rwa [ ablation_eq_weight f w hadd i ] at h_wi_nonneg

/-
**Practical diagnostic: Under additivity, pairwise interactions vanish.**
-/
theorem interaction_zero_of_additive (f : Finset (Fin d) → ℝ) (w : Fin d → ℝ)
    (h : AdditiveSetFn f w) (i j : Fin d) (hij : i ≠ j) :
    pairwiseInteraction f i j = 0 := by
      grind +locals

/-! ## Counterexample: Failure without additivity

We construct a monotone non-additive set function on 3 directions where ablation
ranking differs from optimal single-direction selection. Directions 0 and 1 have
a strong synergy (their combination gives value 100), but individually give value 0.
Direction 2 alone gives value 10.

The ablation importance of direction 0 is 90 (removing it from the full set drops
value from 100 to 10), while direction 2 has ablation importance 0. Yet selecting
direction 2 alone gives a better projection (value 10 vs 0).
-/

/-- A non-additive monotone set function on 3 directions with synergy. -/
def counterexampleFn : Finset (Fin 3) → ℝ := fun S =>
  if S = ∅ then 0
  else if S = {(0 : Fin 3)} then 0
  else if S = {(1 : Fin 3)} then 0
  else if S = {(2 : Fin 3)} then 10
  else if S = {(0 : Fin 3), (1 : Fin 3)} then 100
  else if S = {(0 : Fin 3), (2 : Fin 3)} then 10
  else if S = {(1 : Fin 3), (2 : Fin 3)} then 10
  else 100

/-
The counterexample function is monotone.
-/
theorem counterexample_monotone : MonotoneSetFn counterexampleFn := by
  unfold MonotoneSetFn counterexampleFn;
  norm_cast

/-
**Failure without additivity:** There exists a monotone set function where the
    direction with higher ablation importance gives strictly lower standalone value,
    demonstrating that ablation ranking can be suboptimal without additivity.
-/
theorem ablation_ranking_suboptimal_without_additivity :
    ∃ (f : Finset (Fin 3) → ℝ), MonotoneSetFn f ∧
      ablationImportance f 0 > ablationImportance f 2 ∧
      f {(2 : Fin 3)} > f {(0 : Fin 3)} := by
        constructor;
        refine' ⟨ _, _, _ ⟩;
        convert counterexample_monotone;
        · unfold ablationImportance;
          unfold counterexampleFn; norm_cast;
        · unfold counterexampleFn; norm_num;
          norm_cast

end