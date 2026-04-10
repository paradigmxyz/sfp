# Summary of changes for run c58816d4-2853-486f-8f52-779d0dbb2386
Successfully formalized all 5 claims from T5-pca-vs-gradient-svd.txt in `RequestProject/T5PcaVsGradientSvd.lean`. All theorems are fully proved with no `sorry` and only standard axioms (propext, Classical.choice, Quot.sound).

## What was formalized

The file works with `Matrix (Fin d) (Fin d) ℝ` and uses Mathlib's spectral theory for Hermitian matrices.

### Definitions
- **`IsOrthProj`**: Orthogonal projection matrices (symmetric + idempotent)
- **`topEigenvalueSum`**: Sum of the r largest eigenvalues (via supremum over r-element subsets)
- **`combinedMatrix`**: The matrix M = √Σ · G · √Σ
- **`decisionScore`**: Ratio trace(P_PCA · M) / trace(P_grad · M)
- **`crossAlignment`**: Frobenius inner product overlap measure

### Theorems proved (corresponding to the 5 claims)

1. **Trace complementarity** (`trace_compl_mul`): `trace((1-P)·S) = trace(S) - trace(P·S)`

2. **Ky Fan inequality** (`ky_fan_ineq`): For Hermitian A and rank-r projection P, `trace(P·A) ≤ sum of r largest eigenvalues`. This is the deep result underlying both PCA and gradient-SVD optimality. Proved via decomposition into helper lemmas:
   - Diagonal bounds for projections (`diag_nonneg`, `diag_le_one`)
   - Trace in eigenbasis (`trace_orthProj_mul_hermitian_eq`)
   - Unitary conjugation preserves projections (`conjugate_unitary`, `trace_conjugate_unitary`)
   - Weighted sum bound (`weighted_sum_le_top_sum`) — a combinatorial rearrangement result

3. **PCA optimality** (`pca_optimality`) and **Gradient-SVD optimality** (`gradient_svd_optimality`): Direct corollaries of Ky Fan

4. **Combined matrix optimality** (`combined_optimality`): Optimal subspace is top eigenspace of M = √Σ · G · √Σ, with special cases:
   - Isotropic gradient ⟹ PCA (`combinedMatrix_isotropic_gradient`: M = c·Σ)
   - Isotropic activations ⟹ gradient-SVD (`combinedMatrix_isotropic_activations`: M = c²·G)

5. **Decision criterion** (`pca_preferred_iff_score_ge_one`): PCA preferred iff decision score ≥ 1

6. **Suboptimality bound** (`suboptimality_bound`): trace ratio ≤ 1 for any suboptimal projection; **cross-alignment self-consistency** (`crossAlignment_self`): alignment = 1 for identical projections