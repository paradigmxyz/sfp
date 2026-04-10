# Summary of changes for run 22b47952-213f-4d0e-bf97-609e2a657f83
I formalized all claims from `T4-ablation-causal-importance.txt` in `RequestProject/AblationCausalImportance.lean`. The file compiles successfully with no `sorry` statements and only standard axioms (`propext`, `Classical.choice`, `Quot.sound`).

## What was formalized

**Definitions:**
- `ablationImportance`: Loss change when component i is zeroed out
- `MonotoneSetFn`: Monotonicity assumption — adding any direction weakly increases loss
- `AdditiveSetFn`: Additivity assumption — loss decomposes as base + sum of individual contributions
- `shapleyWeight` / `shapleyValue`: Standard Shapley value from cooperative game theory
- `pairwiseInteraction`: Diagnostic for testing additivity

**Theorems proved (all sorry-free):**
1. `marginal_eq_weight`: Under additivity, the marginal contribution of any direction i to any coalition equals w(i)
2. `ablation_eq_weight` (Claim 1): Under additivity, ablation importance = individual weight w(i)
3. `shapley_weights_sum_one`: The combinatorial identity that Shapley weights sum to 1
4. `shapley_eq_weight` (Claim 2): Under additivity, Shapley value = individual weight w(i)
5. `ablation_nonneg` (Claim 3): Under monotonicity, ablation importance ≥ 0
6. `ablation_eq_shapley` (Claim 4): Under additivity, ablation importance = Shapley value (the main equivalence)
7. `projection_loss_eq_sum` (Claim 5): Under additivity, projection loss = sum of excluded weights
8. `weight_nonneg`: Under additivity + monotonicity, individual weights are non-negative
9. `interaction_zero_of_additive` (Claim 6): Under additivity, pairwise interactions vanish
10. `counterexample_monotone`: The counterexample function is monotone
11. `ablation_ranking_suboptimal_without_additivity` (Claim 5 failure): Concrete counterexample showing ablation ranking fails without additivity — a monotone non-additive function where direction 0 has higher ablation importance than direction 2, yet direction 2 alone gives strictly better standalone value