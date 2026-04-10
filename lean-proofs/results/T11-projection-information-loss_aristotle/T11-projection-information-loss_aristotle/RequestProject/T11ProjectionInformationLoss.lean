import Mathlib

/-!
# T11: Projection Information Loss (Rate-Distortion Bound)

We formalize results about information loss when projecting neural network activations
onto PCA subspaces. All results are expressed in the eigenbasis of the covariance matrix Σ,
where eigenvalues λ_1 ≥ λ_2 ≥ ... ≥ λ_d ≥ 0.

## Setup (Gaussian linear model in the eigenbasis)

- `eigenvalues i` = λ_i, the i-th eigenvalue of the covariance matrix (0-indexed, sorted
  decreasingly)
- `weights i` = w_i = ⟨w, v_i⟩, the component of the task-relevant direction w along
  eigenvector v_i
- `σ2` = σ², the noise variance
- `r` = the number of retained PCA dimensions (top-r projection)

Under Gaussian assumptions with Y = w^T a + ε, the mutual information has the closed form:
  I(Y; a) = (1/2) log(1 + Σ_i λ_i w_i² / σ²)

## Main results

1. **Data processing inequality**: I_lost ≥ 0
2. **Information loss identity**: closed-form expression for I_lost
3. **Information loss upper bound**: I_lost ≤ (1/2) log(1 + Σ_{i≥r} λ_i w_i² / σ²)
4. **Eigenvalue bound**: when eigenvalues are sorted, Σ_{i≥r} λ_i w_i² ≤ λ_r · Σ_{i≥r} w_i²
5. **Distortion**: E[‖a - P_W(a)‖²] = Σ_{i≥r} λ_i
6. **Explained variance ratio** and its connection to information retention
-/

open Real BigOperators

noncomputable section

/-- The "top-r" index set: indices 0, ..., r-1 (the retained PCA dimensions). -/
def topR (d : ℕ) (r : ℕ) : Finset (Fin d) :=
  Finset.univ.filter (fun i => (i : ℕ) < r)

/-- The "bottom" index set: indices r, ..., d-1 (the discarded PCA dimensions). -/
def botR (d : ℕ) (r : ℕ) : Finset (Fin d) :=
  Finset.univ.filter (fun i => r ≤ (i : ℕ))

/-- Gaussian mutual information for linear readout Y = wᵀa + ε in the eigenbasis.
    I(Y; a) = (1/2) · log(1 + (Σ_i λ_i · w_i²) / σ²) -/
def gaussianMI (d : ℕ) (eigenvalues : Fin d → ℝ) (weights : Fin d → ℝ) (σ2 : ℝ) : ℝ :=
  (1 / 2) * Real.log (1 + (∑ i, eigenvalues i * weights i ^ 2) / σ2)

/-- Gaussian mutual information after projecting onto the top-r PCA subspace.
    I(Y; P_W(a)) = (1/2) · log(1 + (Σ_{i<r} λ_i · w_i²) / σ²) -/
def gaussianMI_proj (d : ℕ) (r : ℕ) (eigenvalues : Fin d → ℝ)
    (weights : Fin d → ℝ) (σ2 : ℝ) : ℝ :=
  (1 / 2) * Real.log (1 + (∑ i ∈ topR d r, eigenvalues i * weights i ^ 2) / σ2)

/-- Information lost by projecting onto the top-r PCA subspace. -/
def infoLoss (d : ℕ) (r : ℕ) (eigenvalues : Fin d → ℝ)
    (weights : Fin d → ℝ) (σ2 : ℝ) : ℝ :=
  gaussianMI d eigenvalues weights σ2 - gaussianMI_proj d r eigenvalues weights σ2

/-- Sum of eigenvalue-weighted squared weights in the discarded subspace. -/
def discardedSignal (d : ℕ) (r : ℕ) (eigenvalues : Fin d → ℝ)
    (weights : Fin d → ℝ) : ℝ :=
  ∑ i ∈ botR d r, eigenvalues i * weights i ^ 2

/-- PCA distortion: E[‖a - P_W(a)‖²] = Σ_{i≥r} λ_i (sum of discarded eigenvalues). -/
def pcaDistortion (d : ℕ) (r : ℕ) (eigenvalues : Fin d → ℝ) : ℝ :=
  ∑ i ∈ botR d r, eigenvalues i

/-- Explained variance ratio: ρ(r) = (Σ_{i<r} λ_i) / (Σ_i λ_i). -/
def explainedVarianceRatio (d : ℕ) (r : ℕ) (eigenvalues : Fin d → ℝ) : ℝ :=
  (∑ i ∈ topR d r, eigenvalues i) / (∑ i, eigenvalues i)

/-- Eigenvalues are sorted in decreasing order. -/
def eigenvaluesSorted (d : ℕ) (eigenvalues : Fin d → ℝ) : Prop :=
  Antitone eigenvalues

/-- Eigenvalues are nonnegative (covariance matrix is PSD). -/
def eigenvaluesNonneg (d : ℕ) (eigenvalues : Fin d → ℝ) : Prop :=
  ∀ i, 0 ≤ eigenvalues i

-- Partition lemmas for topR and botR

theorem topR_disjoint_botR (d : ℕ) (r : ℕ) :
    Disjoint (topR d r) (botR d r) := by
  simp only [topR, botR, Finset.disjoint_filter]
  exact fun x _ h1 h2 => by omega

theorem topR_union_botR (d : ℕ) (r : ℕ) :
    topR d r ∪ botR d r = Finset.univ := by
  ext x; simp [topR, botR, Finset.mem_filter, Finset.mem_union]; omega

/-- The full sum splits into top-r and bottom parts. -/
theorem sum_split_topR_botR (d : ℕ) (r : ℕ) (f : Fin d → ℝ) :
    ∑ i, f i = ∑ i ∈ topR d r, f i + ∑ i ∈ botR d r, f i := by
  rw [← Finset.sum_union (topR_disjoint_botR d r), topR_union_botR]

-- Helper: nonneg sum of nonneg eigenvalue-weight products
theorem sum_eigenweight_nonneg (d : ℕ) (eigenvalues weights : Fin d → ℝ)
    (hev : eigenvaluesNonneg d eigenvalues) (s : Finset (Fin d)) :
    0 ≤ ∑ i ∈ s, eigenvalues i * weights i ^ 2 :=
  Finset.sum_nonneg fun i _ => mul_nonneg (hev i) (sq_nonneg _)

/-
Claim 1: Data Processing Inequality (I_lost ≥ 0)

The data processing inequality for Gaussian linear model:
    I(Y; P_W(a)) ≤ I(Y; a), equivalently I_lost ≥ 0.
-/
theorem infoLoss_nonneg (d : ℕ) (r : ℕ) (eigenvalues : Fin d → ℝ) (weights : Fin d → ℝ)
    (σ2 : ℝ) (hσ2 : 0 < σ2)
    (hev : eigenvaluesNonneg d eigenvalues) :
    0 ≤ infoLoss d r eigenvalues weights σ2 := by
  unfold infoLoss; norm_num;
  refine' mul_le_mul_of_nonneg_left ( Real.log_le_log _ _ ) _;
  · exact add_pos_of_pos_of_nonneg zero_lt_one ( div_nonneg ( Finset.sum_nonneg fun _ _ => mul_nonneg ( hev _ ) ( sq_nonneg _ ) ) hσ2.le );
  · gcongr;
    · exact fun i _ hi => mul_nonneg ( hev i ) ( sq_nonneg _ );
    · exact Finset.subset_univ _;
  · norm_num

-- Claim 1: Information Loss Identity

/-- The full signal power splits into retained and discarded parts. -/
theorem signal_split (d : ℕ) (r : ℕ) (eigenvalues : Fin d → ℝ) (weights : Fin d → ℝ) :
    ∑ i, eigenvalues i * weights i ^ 2 =
    (∑ i ∈ topR d r, eigenvalues i * weights i ^ 2) +
    discardedSignal d r eigenvalues weights := by
  exact sum_split_topR_botR d r _

/-
Information loss identity:
    I_lost = (1/2) · log((1 + full_signal/σ²) / (1 + retained_signal/σ²)).
-/
theorem infoLoss_eq (d : ℕ) (r : ℕ) (eigenvalues : Fin d → ℝ) (weights : Fin d → ℝ)
    (σ2 : ℝ) (hσ2 : 0 < σ2) (hev : eigenvaluesNonneg d eigenvalues) :
    infoLoss d r eigenvalues weights σ2 =
    (1 / 2) * Real.log ((1 + (∑ i, eigenvalues i * weights i ^ 2) / σ2) /
                         (1 + (∑ i ∈ topR d r, eigenvalues i * weights i ^ 2) / σ2)) := by
  rw [ infoLoss, gaussianMI, gaussianMI_proj ];
  rw [ Real.log_div, mul_sub ] <;> ring <;> norm_num [ ne_of_gt hσ2 ];
  · exact ne_of_gt ( add_pos_of_pos_of_nonneg zero_lt_one ( mul_nonneg ( Finset.sum_nonneg fun _ _ => mul_nonneg ( hev _ ) ( sq_nonneg _ ) ) ( inv_nonneg.2 hσ2.le ) ) );
  · exact ne_of_gt ( add_pos_of_pos_of_nonneg zero_lt_one ( mul_nonneg ( Finset.sum_nonneg fun _ _ => mul_nonneg ( hev _ ) ( sq_nonneg _ ) ) ( inv_nonneg.2 hσ2.le ) ) )

/-
Claim 1: Information Loss Upper Bound

Information loss upper bound:
    I_lost ≤ (1/2) · log(1 + discardedSignal / σ²).
-/
theorem infoLoss_le (d : ℕ) (r : ℕ) (eigenvalues : Fin d → ℝ) (weights : Fin d → ℝ)
    (σ2 : ℝ) (hσ2 : 0 < σ2) (hev : eigenvaluesNonneg d eigenvalues) :
    infoLoss d r eigenvalues weights σ2 ≤
    (1 / 2) * Real.log (1 + discardedSignal d r eigenvalues weights / σ2) := by
  -- Using the signal_split theorem, we can rewrite the full signal as the sum of the retained signal and the discarded signal.
  set S_full := ∑ i, eigenvalues i * weights i ^ 2
  set S_top := ∑ i ∈ topR d r, eigenvalues i * weights i ^ 2
  set S_bot := discardedSignal d r eigenvalues weights
  have h_split : S_full = S_top + S_bot := by
    exact signal_split d r eigenvalues weights;
  rw [ infoLoss_eq d r eigenvalues weights σ2 hσ2 hev ];
  gcongr;
  · exact div_pos ( add_pos_of_pos_of_nonneg zero_lt_one ( div_nonneg ( Finset.sum_nonneg fun _ _ => mul_nonneg ( hev _ ) ( sq_nonneg _ ) ) hσ2.le ) ) ( add_pos_of_pos_of_nonneg zero_lt_one ( div_nonneg ( Finset.sum_nonneg fun _ _ => mul_nonneg ( hev _ ) ( sq_nonneg _ ) ) hσ2.le ) );
  · rw [ div_le_iff₀ ];
    · rw [ add_div' ] <;> try positivity;
      rw [ add_div', div_mul_eq_mul_div, div_le_div_iff₀ ] <;> try positivity;
      rw [ add_div', mul_div_assoc' ];
      · rw [ div_mul_cancel₀ _ hσ2.ne' ] ; nlinarith [ show 0 ≤ S_top from Finset.sum_nonneg fun _ _ => mul_nonneg ( hev _ ) ( sq_nonneg _ ), show 0 ≤ S_bot from Finset.sum_nonneg fun _ _ => mul_nonneg ( hev _ ) ( sq_nonneg _ ) ];
      · positivity;
    · exact add_pos_of_pos_of_nonneg zero_lt_one ( div_nonneg ( Finset.sum_nonneg fun _ _ => mul_nonneg ( hev _ ) ( sq_nonneg _ ) ) hσ2.le )

/-
When eigenvalues are sorted, the discarded signal is bounded by the largest
    discarded eigenvalue times the squared norm of the discarded weight components.
-/
theorem discardedSignal_le_of_sorted (d : ℕ) (r : ℕ) (hr : r < d)
    (eigenvalues : Fin d → ℝ) (weights : Fin d → ℝ)
    (_hev : eigenvaluesNonneg d eigenvalues)
    (hsorted : eigenvaluesSorted d eigenvalues) :
    discardedSignal d r eigenvalues weights ≤
    eigenvalues ⟨r, hr⟩ * ∑ i ∈ botR d r, weights i ^ 2 := by
  rw [ Finset.mul_sum ];
  exact Finset.sum_le_sum fun i hi => mul_le_mul_of_nonneg_right ( hsorted <| Finset.mem_filter.mp hi |>.2 ) <| sq_nonneg _

-- Claim 2: Distortion and Rate-Distortion

/-- PCA distortion equals sum of discarded eigenvalues (by definition). -/
theorem pcaDistortion_eq (d : ℕ) (r : ℕ) (eigenvalues : Fin d → ℝ) :
    pcaDistortion d r eigenvalues = ∑ i ∈ botR d r, eigenvalues i := by
  rfl

/-
Distortion is nonneg when eigenvalues are nonneg.
-/
theorem pcaDistortion_nonneg (d : ℕ) (r : ℕ) (eigenvalues : Fin d → ℝ)
    (hev : eigenvaluesNonneg d eigenvalues) :
    0 ≤ pcaDistortion d r eigenvalues := by
  exact Finset.sum_nonneg fun _ _ => hev _

/-
More retained dimensions means less distortion.
-/
theorem pcaDistortion_antitone (d : ℕ) (eigenvalues : Fin d → ℝ)
    (hev : eigenvaluesNonneg d eigenvalues)
    (r1 r2 : ℕ) (h : r1 ≤ r2) :
    pcaDistortion d r2 eigenvalues ≤ pcaDistortion d r1 eigenvalues := by
  exact Finset.sum_le_sum_of_subset_of_nonneg ( fun x hx => Finset.mem_filter.mpr ⟨ Finset.mem_univ _, by linarith [ Finset.mem_filter.mp hx |>.2 ] ⟩ ) fun _ _ _ => hev _

/-- Total variance = retained variance + distortion. -/
theorem variance_split (d : ℕ) (r : ℕ) (eigenvalues : Fin d → ℝ) :
    ∑ i, eigenvalues i =
    (∑ i ∈ topR d r, eigenvalues i) + pcaDistortion d r eigenvalues := by
  exact sum_split_topR_botR d r _

/-
Claim 5: Explained Variance Ratio

Explained variance ratio is nonneg.
-/
theorem explainedVarianceRatio_nonneg (d : ℕ) (r : ℕ) (eigenvalues : Fin d → ℝ)
    (hev : eigenvaluesNonneg d eigenvalues)
    (htot : 0 < ∑ i, eigenvalues i) :
    0 ≤ explainedVarianceRatio d r eigenvalues := by
  exact div_nonneg ( Finset.sum_nonneg fun _ _ => hev _ ) htot.le

/-
Explained variance ratio is at most 1.
-/
theorem explainedVarianceRatio_le_one (d : ℕ) (r : ℕ) (eigenvalues : Fin d → ℝ)
    (hev : eigenvaluesNonneg d eigenvalues)
    (htot : 0 < ∑ i, eigenvalues i) :
    explainedVarianceRatio d r eigenvalues ≤ 1 := by
  exact div_le_one_of_le₀ ( Finset.sum_le_sum_of_subset_of_nonneg ( Finset.subset_univ _ ) fun _ _ _ => hev _ ) htot.le

/-
More retained dimensions means higher explained variance.
-/
theorem explainedVarianceRatio_monotone (d : ℕ) (eigenvalues : Fin d → ℝ)
    (hev : eigenvaluesNonneg d eigenvalues)
    (htot : 0 < ∑ i, eigenvalues i)
    (r1 r2 : ℕ) (h : r1 ≤ r2) :
    explainedVarianceRatio d r1 eigenvalues ≤ explainedVarianceRatio d r2 eigenvalues := by
  apply div_le_div_of_nonneg_right;
  · exact Finset.sum_le_sum_of_subset_of_nonneg ( fun i hi => Finset.mem_filter.mpr ⟨ Finset.mem_univ _, lt_of_lt_of_le ( Finset.mem_filter.mp hi |>.2 ) h ⟩ ) fun _ _ _ => hev _;
  · linarith

/-
Explained variance ratio = 1 - distortion / total_variance.
-/
theorem explainedVarianceRatio_eq (d : ℕ) (r : ℕ) (eigenvalues : Fin d → ℝ)
    (htot : ∑ i, eigenvalues i ≠ 0) :
    explainedVarianceRatio d r eigenvalues =
    1 - pcaDistortion d r eigenvalues / ∑ i, eigenvalues i := by
  have h_split : ∑ i, eigenvalues i = (∑ i ∈ topR d r, eigenvalues i) + pcaDistortion d r eigenvalues :=
    variance_split d r eigenvalues
  unfold explainedVarianceRatio pcaDistortion at *;
  grind

-- Claim 3: Task-Relevant Rate-Distortion Bound

/-- Task-relevant gradient alignment: γ_i = (∇_i)² / ‖∇‖². -/
def gradientAlignment (d : ℕ) (gradient : Fin d → ℝ) (i : Fin d) : ℝ :=
  gradient i ^ 2 / ∑ j, gradient j ^ 2

/-
Gradient alignments sum to 1 (when gradient is nonzero).
-/
theorem gradientAlignment_sum_eq_one (d : ℕ) (gradient : Fin d → ℝ)
    (hgrad : ∑ i, gradient i ^ 2 ≠ 0) :
    ∑ i, gradientAlignment d gradient i = 1 := by
  unfold gradientAlignment;
  rw [ ← Finset.sum_div, div_self hgrad ]

/-
Each gradient alignment is nonneg.
-/
theorem gradientAlignment_nonneg (d : ℕ) (gradient : Fin d → ℝ) (i : Fin d) :
    0 ≤ gradientAlignment d gradient i := by
  exact div_nonneg ( sq_nonneg _ ) ( Finset.sum_nonneg fun _ _ => sq_nonneg _ )

/-
Claim 4: Sufficient condition for target information retention

For information retained to be at least (1-ε) fraction of total information,
    it suffices that the information loss bound is at most ε times the total MI.
-/
theorem sufficient_r_for_retention (d : ℕ) (r : ℕ)
    (eigenvalues : Fin d → ℝ) (weights : Fin d → ℝ) (σ2 : ℝ) (ε : ℝ)
    (hσ2 : 0 < σ2) (hev : eigenvaluesNonneg d eigenvalues)
    (_hε : 0 < ε) (_hε1 : ε < 1)
    (_hMI : 0 < gaussianMI d eigenvalues weights σ2)
    (hbound : (1 / 2) * Real.log (1 + discardedSignal d r eigenvalues weights / σ2) ≤
              ε * gaussianMI d eigenvalues weights σ2) :
    gaussianMI_proj d r eigenvalues weights σ2 ≥
    (1 - ε) * gaussianMI d eigenvalues weights σ2 := by
  contrapose! hbound;
  convert infoLoss_le d r eigenvalues weights σ2 hσ2 hev |> fun x => x.trans_lt' _ using 1;
  unfold infoLoss;
  linarith

end