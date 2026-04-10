# Summary of changes for run 3cf74b3d-05ef-46d5-a956-d1c57f0a7f37
Successfully formalized T6-adaptive-lambda-signal-decay.txt into `RequestProject/T6AdaptiveLambdaSignalDecay.lean`. All 8 theorems are fully proved with no `sorry` statements remaining.

## What was formalized

The file contains definitions and proofs for the mathematical claims about adaptive regularization schedules for neural network fine-tuning:

### Definitions
- `signalDecay` — exponential signal decay model s(t) = s₀·exp(-αt)
- `driftSeq` — discrete drift recurrence δ(t+1) = (1-2ηλ)·δ(t) + η·s(t)
- `driftExactDiscrete` — exact closed-form solution for the recurrence
- `steadyStateDrift` — steady-state drift η·s₀/(2λ)
- `optimalConstantLambda` — optimal constant λ = s₀·η/(2·δ_target)
- `adaptiveLambdaSchedule` — original document's schedule λ₀·exp(-αt/2)
- `correctAdaptiveSchedule` — corrected schedule λ₀·exp(-αt)
- `decayRateEstimator` — online α estimator from signal measurements
- `instantaneousDriftRate` — drift rate η·s(t) - 2·λ(t)·δ(t)

### Proved Theorems
1. **`drift_exact_discrete`** — Claim 1: The drift recurrence has exact closed-form solution δ(t) = r^t·δ₀ + η·s₀·(q^t - r^t)/(q - r) where r = 1-2ηλ, q = exp(-α). *(Note: The original document's formula used a continuous-time approximation with denominator (2λ - ηα); the exact discrete formula uses (exp(-α) - (1-2ηλ)) instead.)*
2. **`optimal_constant_lambda_correct`** — Claim 2: Setting steady-state drift equal to δ_target yields λ* = s₀·η/(2·δ_target).
3. **`exponential_schedule_constant_drift`** — Claim 3 (corrected): The schedule λ(t) = λ₀·exp(-αt) achieves zero instantaneous drift rate at constant δ. *(Note: The original document incorrectly states exp(-αt/2); this was formally disproved and corrected to exp(-αt).)*
4. **`decay_rate_estimator_exact`** — Claim 4: The log-ratio estimator α̂ = -(log s(t₂) - log s(t₁))/(t₂ - t₁) recovers α exactly under the exponential model.
5. **`adaptive_schedule_sensitivity`** — Claim 5: Schedule perturbation bound |λ(t;α̂) - λ(t;α)| ≤ λ₀·|α̂-α|·t/2 via the mean value theorem.
6. **`numerical_steady_state_drift`** — Claim 6a: With η=1e-4, s₀=1.0, λ=0.1: δ(∞) = 5e-4.
7. **`numerical_optimal_lambda`** — Claim 6b: With s₀=1.0, η=1e-4, δ_target=5e-4: λ* = 0.1.
8. **`signal_half_life_formula`** — Claim 6c: s(ln2/α) = s₀/2 (half-life formula).

## Corrections to the original document
Two errors in the original document were identified and corrected:
- **Claim 1**: The continuous-time approximation formula doesn't match the discrete recurrence exactly. The formalization uses the correct discrete closed-form.
- **Claim 3**: The schedule λ(t) = λ₀·exp(-αt/2) does NOT maintain zero drift rate; the correct schedule is λ(t) = λ₀·exp(-αt). This was formally disproved and the corrected version was proved.