# Summary of changes for run 69c29962-b2ff-4f1e-acd8-945a2eeb5a18
Successfully formalized all 5 claims from `T3-conditional-H1-exponential-family.txt` in `RequestProject/T3ConditionalH1.lean`. The formalization abstracts the neural network setting into clean mathematical statements about orthogonal projections in inner product spaces:

**Definitions:**
- `projOnto W` — orthogonal projection onto subspace W, as a continuous linear map E → E
- `projOrthComplement W` — orthogonal projection onto W⊥

**Proven theorems (all sorry-free, standard axioms only):**

1. **`loss_depends_only_on_projection`** (Claim 1 — Sufficiency): If L_old factors through P_W, then activations with the same projection yield the same loss.

2. **`H1_from_lipschitz`** (Claim 2 — H1 bound): If L_old = g ∘ P_W and g is C-Lipschitz, then |L_old(a) - L_old(a₀)| ≤ C · ‖P_W(a - a₀)‖.

3. **`projOnto_orthogonal_eq_zero`** + **`orthogonal_complement_irrelevance`** (Claim 3): The projection P_W kills W⊥-vectors, so adding any W⊥-component to an activation does not change the loss.

4. **`R_squared_one_implies_factorization`** (Claim 4 — Diagnostic): If L_old is determined by P_W(a) (i.e., R² = 1), then L_old factors through P_W via some function g.

5. **`multi_layer_H1`** (Claim 5 — Multi-layer extension): For n layers with per-layer Lipschitz constants Cℓ and subspaces Wℓ, the joint H1 bound holds: |L_old(a) - L_old(a₀)| ≤ Σℓ Cℓ · ‖P_{Wℓ}(aℓ - a₀ℓ)‖.