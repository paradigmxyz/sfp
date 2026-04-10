# Summary of changes for run 1bd81190-f96c-48e4-87ec-80fa1f1e4786
Formalized all key claims from T8-bound-tightness.txt in `RequestProject/BoundTightness.lean`. All 9 theorems are fully proved (no sorries) and verified with only standard axioms (propext, Classical.choice, Quot.sound).

**Claims formalized:**

1. **Claim 1 – 1D Lower Bound Construction** (`forgetting_eq_sqrt_drift_1d`, `lower_bound_1d`): With L_old(a) = |a - a₀| and δ = (a - a₀)², forgetting = √δ exactly, achieving the lower bound with C = 1.

2. **Claim 2 – General Lower Bound** (`general_lower_bound_achievable`): For any C > 0 and δ > 0, there exists a C-Lipschitz function f and a point x with x² = δ such that |f(x) - f(0)| = C · √δ.

3. **Claim 3 – Tightness of √δ Rate** (`sqrt_dominates_higher_power`, `no_universal_higher_power_bound`): For any α > 1/2, the function δ^(1/2) dominates δ^α as δ → 0⁺. Consequently, no universal bound forgetting ≤ K · δ^α with α > 1/2 can hold for all 1-Lipschitz functions.

4. **Claim 4 – Quadratic Regime** (`quadratic_model_linear_scaling`, `quadratic_forgetting_over_sqrt_drift`): For the quadratic model f(x) = c·x², forgetting scales linearly with δ = x² (not as √δ), and forgetting/√δ = |c|·|x| → 0.

5. **Claim 5 – √δ vs δ Dichotomy** (`linear_model_tight`, `dichotomy_quadratic_small`): The linear model (nonzero gradient) gives forgetting/√δ = |c| > 0 (tight), while the quadratic model (critical point) gives forgetting/√δ < ε for small enough x (loose).