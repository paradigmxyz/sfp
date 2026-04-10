# Summary of changes for run a941031a-ed66-42b7-8ae2-db8a7b06e132
I formalized all claims from `T1-forgetting-constant-bound.txt` into `RequestProject/ForgettingConstantBound.lean`. The file builds cleanly with no sorries and only standard axioms (propext, Classical.choice, Quot.sound).

## What was formalized

**Definitions:**
- `concentrationBound`: The bound ε = (σ_F · σ_D / v_min) · √(2 · log(2/δ) / n)
- `minSampleSize`: The minimum n for a target bound width
- `minSampleSize10pct`: The minimum n for 10% relative error

**Claim 1 — OLS Coefficient Identity** (`ols_coefficient_identity`):
Under a linear model F = C·D + ε with E[ε·D] = 0 and E[ε] = 0, proves C = Cov(F,D)/Var(D) algebraically from the expectation conditions.

**Claim 2 — Concentration Bound Properties:**
- `concentrationBound_nonneg`: The bound is non-negative under positivity assumptions.
- `concentrationBound_antitone_n`: The bound decreases monotonically as sample size n increases.
- `concentrationBound_sq`: ε² = (σ_F·σ_D/v_min)² · (2·log(2/δ)/n).

**Claim 3 — Non-Vacuity Condition** (`non_vacuity_iff`):
Proves ε < ε_target ⟺ n > 2(σ_F·σ_D)²·log(2/δ) / (v_min·ε_target)².

**Claim 4 — Layer Selection Optimality** (`sum_smallest_le_sum_of_card_eq`):
Selecting K layers with smallest forgetting constants minimizes the total sum over any K-element subset.

**Claim 5 — Structural Identity** (`structural_identity`, `concentration_bound_universal`):
Both Kyle's Lambda and the forgetting constant satisfy the same OLS formula and the same parametric concentration bound.

**Claim 6 — Numerical Verification:**
- `numerical_epsilon_sq`: With σ_F=σ_D=1, v_min=0.5: the prefactor (σ_F·σ_D/v_min)² = 4.
- `sqrt_n_scaling`: ε(4n) = ε(n)/2 (√n scaling verified for any parameters).
- `n_min_formula`: With the given parameters, n_min = 800 for 10% relative error.