# Task Description: Implementation of Adaptive Kalman Filter Noise Covariance Identification

**Reference Paper:** Zhang, L., Sidoti, D., Bienkowski, A., Pattipati, K. R., Bar-Shalom, Y., & Kleinman, D. L. (2020). *On the Identification of Noise Covariances and Adaptive Kalman Filtering: A New Look at a 50 Year-Old Problem*. IEEE Access, Vol. 8, pp. 59362–59388. DOI: 10.1109/ACCESS.2020.2982407

---

## 1. Article Summary

### 1.1 Problem Statement

The classical Kalman Filter (KF) is the optimal state estimator for linear dynamic systems driven by Gaussian white noise, **provided that the process noise covariance `Q` and measurement noise covariance `R` are known a priori**. In practice, these covariances are almost never known exactly — they must be estimated from data. Poorly chosen `Q` and `R` result in suboptimal state estimates, biased innovations, and inconsistent filters (incorrect uncertainty quantification).

This problem has been studied for ~50 years and is typically approached via four families of methods:

1. **Bayesian inference** — computationally intractable due to the curse of dimensionality.
2. **Maximum likelihood** — requires nonlinear programming; EM-based variants converge to local optima.
3. **Covariance matching** — no convergence proofs.
4. **Correlation methods** (Mehra 1970, Carew–Bélanger 1973, Odelson 2006 ALS) — the most successful historically, but suffer from assumptions that time averages equal ensemble averages, leading to divergence or large inaccuracies on finite data.

### 1.2 System Model

The paper considers the discrete-time linear Gauss–Markov system:

```
x(k+1) = F·x(k) + Γ·v(k)         (process equation)
z(k)   = H·x(k) + w(k)           (measurement equation)
```

where:
- `x(k)` is the `n_x`-dimensional state,
- `z(k)` is the `n_z`-dimensional measurement,
- `v(k) ~ N(0, Q)` is process noise (`n_v`-dimensional),
- `w(k) ~ N(0, R)` is measurement noise (`n_z`-dimensional),
- `F`, `H`, `Γ` are known system matrices,
- `Q` and `R` are the **unknown** positive-definite covariance matrices to be estimated.

### 1.3 Key Contributions of the Paper

1. **Necessary and sufficient identifiability condition for `Q` and `R`**, expressed as a rank condition on a *noise covariance identifiability matrix* `I`. This matrix is built from the coefficients of the **minimal polynomial of the closed-loop filter matrix** `F̄ = F(I − WH)`. Mehra's old observability + controllability condition is shown to be insufficient (via Odelson's counterexample).

2. **A novel six-step iterative algorithm** that successively estimates:
   - the steady-state Kalman gain `W`,
   - the innovation covariance `S`,
   - the measurement noise covariance `R`,
   - the process noise covariance `Q`,
   - the steady-state prediction/updated covariances `P̄` and `P`.
   The algorithm uses a gradient descent on a correlation-based objective with an **adaptive step size (bold driver method)**.

3. **Five equivalent formulas (`R1`–`R5`) for estimating `R`** using post-fit residuals `µ(k) = z(k) − H·x̂(k|k)`. This is novel — prior work used only innovations. The paper recommends **`R3`** because it guarantees positive definiteness (it is a continuous-time algebraic Riccati equation `G = R·S⁻¹·R`).

4. **Iterative coupled estimation of `Q` and `P`** that enforces structural constraints (diagonality, symmetry, positive definiteness) via a mask matrix and regularization parameter `λ_Q`.

5. **Special closed-form cases**: Wiener process (`F = I`, `H = I`) and `H = I` general case — these admit non-iterative solutions.

6. **Validation on five benchmark test cases** showing significant accuracy improvement (2×–9× lower RMSE) over Mehra's and Bélanger's methods, with guaranteed filter stability.

### 1.4 Core Mathematical Objects

- **Innovation:** `ν(k) = z(k) − H·x̂(k|k−1)`, with theoretical covariance `S = H·P̄·H' + R`.
- **Post-fit residual:** `µ(k) = z(k) − H·x̂(k|k) = (I − H·W)·ν(k)`, with covariance `G = R·S⁻¹·R`.
- **Sample autocovariance:** `Ĉ(i) = (1/(N−M))·Σⱼ ν(j)·ν(j+i)'` for lags `i = 0, 1, ..., M−1`.
- **Objective function** (normalized sum of off-zero-lag innovation autocorrelations):
  ```
  J = (1/2)·tr{ Σᵢ₌₁^(M−1) [diag(Ĉ(0))]^(−1/2) · Ĉ(i)' · [diag(Ĉ(0))]^(−1) · Ĉ(i) · [diag(Ĉ(0))]^(−1/2) }
  ```
  At the optimal `W`, the innovations are white, so `J → 0` as `N → ∞`.

---

## 2. Objective

Implement the six-step algorithm from the paper in code (Python) and reproduce the numerical results from the paper's five test cases:

1. **Case 1:** Second-order kinematic system (nearly-constant-velocity model), scalar measurement. Test effect of varying the number of lags `M ∈ {10, 20, 30, 40, 50, 100}`.
2. **Case 2:** Neethling (1974) 2-state system.
3. **Case 3:** Mehra/Bélanger 5-state system with diagonal `Q` and `R` — the most thorough comparison, including head-to-head with Mehra's and Bélanger's gain-update methods and sensitivity to sample size `N ∈ {1000, 2500, 5000, 10000}`.
4. **Case 4:** Odelson's detectable-but-unobservable 2-state system.
5. **Case 5:** Odelson's 3-state ill-conditioned system.

Each case runs 100 Monte Carlo trials (200 for Case 5) and reports: mean, 95% probability interval `[r, r̄]`, RMSE of each estimated quantity, plus averaged Normalized Innovation Squared (NIS) as a filter-consistency check.

---

## 3. Implementation Roadmap

### 3.1 Prerequisites & Foundations

Before coding the six-step loop, implement and unit-test these building blocks:

| Block | Equations | Notes |
|-------|-----------|-------|
| Standard Kalman filter (time + measurement update) | (3)–(9) | Use Joseph form (13) for numerical stability when debugging |
| Steady-state Riccati solver (for generating a stabilizing initial `W⁽⁰⁾`) | (10) | Kleinman iteration [28] or `scipy.linalg.solve_discrete_are` / MATLAB `dare` |
| Sample autocovariance `Ĉ(i)` | (50) | |
| Minimal-polynomial coefficients of `F̄` | (21) | Use `numpy.poly(eigvals(F̄))` then deduplicate |
| Identifiability matrix `I` and rank check | Algorithm 1, eqs. (28)–(36) | Required to confirm `Q`,`R` are identifiable before running the full algorithm |
| Lyapunov equation solver | (61) | `scipy.linalg.solve_discrete_lyapunov` |

### 3.2 The Six-Step Algorithm (Section VIII)

Let `r` index the outer iterations. Initialize: `W⁽⁰⁾` from DARE with any reasonable `Q⁽⁰⁾`, `R⁽⁰⁾`.

#### **Step 1 — Run the Kalman filter**
For `k = 1, …, N`, compute `x̂⁽ʳ⁾(k|k−1)`, `ν⁽ʳ⁾(k)`, `x̂⁽ʳ⁾(k|k)`, `µ⁽ʳ⁾(k)` using equations (128)–(131) with the current gain `W⁽ʳ⁾`.

#### **Step 2 — Compute sample autocovariances**
`Ĉ(i)` for `i = 0, 1, …, M−1` via equation (50). The lag count `M` should be `≥ n_x`; the paper uses `M = 40`–`100` typically.

#### **Step 3 — Update `W` via adaptive gradient descent**
Check the five termination conditions (gain change `δ_W < ζ_W`, gradient norm `< ζ_Δ`, `J < ζ_J`, patience exhausted, or max iterations `n_L` reached). If none triggered:

1. Compute `X = pinv([HF; HF̄F; …; HF̄^(M−1)F]) · [Ĉ(1); …; Ĉ(M−1)]` — eq. (63).
2. Solve the Lyapunov equation (61) for `Z`.
3. Compute the gradient `∇_W J` — eq. (60).
4. Update `W⁽ʳ⁺¹⁾ = W⁽ʳ⁾ − α⁽ʳ⁾·∇_W J` with the **bold-driver** step size (137): halve `α` if `J` increased, else multiply by 1.1 (capped at `c̄`).
5. Initial step size `α⁽⁰⁾` from eq. (136) with hyperparameters `c = 0.01`, `c_max = 0.2`, `β = 2`, `N_s = N`.
6. Keep track of the best-so-far `(W*, J*)`; select that on termination — eq. (139).

#### **Step 4 — Estimate `R`**
Given the converged `W` and `S = Ĉ(0)`, compute the post-fit residual covariance `G = (1/N)·Σ µ(k)·µ(k)'`. Estimate `R` using **`R3`** (recommended):
```
G = R·S⁻¹·R      →      solve for R as a CARE
```
Use Cholesky + eigendecomposition (Appendix F) or simultaneous diagonalization (Appendix G). If `R` is known to be diagonal, keep only `diag(R_estimated)`.

#### **Step 5 — Estimate `Q` and `P`**
Inner iteration (index `ℓ`):
1. Initialize `Γ·Q⁽⁰⁾·Γ' = W·S·W'` (the Wiener-process solution, eq. 165).
2. Solve the Lyapunov initialization (122) for `P⁽⁰⁾`.
3. Iterate eq. (123): `P⁽ℓ⁺¹⁾ = [(F·P⁽ℓ⁾·F' + Γ·Q⁽ᵗ⁾·Γ')⁻¹ + H'·R⁻¹·H]⁻¹` until `‖P⁽ℓ⁺¹⁾ − P⁽ℓ⁾‖ < tol`.
4. Update `Q⁽ᵗ⁺¹⁾ = Γ†·(P + W·S·W' − F·P·F' + λ_Q·I)·(Γ')†` — eqs. (124)–(127).
5. Apply the mask `Q⁽ᵗ⁺¹⁾ = A ⊙ Q⁽ᵗ⁺¹⁾` if structural constraints (e.g., diagonality) apply.
6. Repeat until `‖Q⁽ᵗ⁺¹⁾ − Q⁽ᵗ⁾‖ < tol`.
7. Finally compute `P̄` via eq. (111): `P̄ = F·P·F' + Γ·Q·Γ'`.

#### **Step 6 — Outer successive approximation**
Reinitialize the Kalman filter with the newly estimated `(Q, R)`, recompute `W⁽⁰⁾` by solving DARE, and repeat Steps 1–5. Track the best `J` across outer loops. Terminate when the improvement in `J` is below `ζ_J` or after 20 outer iterations.

### 3.3 Recommended Hyperparameters (from Section X)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `ζ_J`, `ζ_W`, `ζ_Δ` | `10⁻⁶` | Convergence tolerances |
| `c` | `0.01` | Initial step size base |
| `c_max` | `0.2` | Max step size |
| `β` | `2` | Step-size / sample-size exponent |
| `patience` | `5` | Epochs of no `J` improvement before stopping |
| `n_L` | `100`–`500` | Max inner iterations (scale with `n_x`) |
| Outer-loop limit | `20` | |
| `λ_Q` | `0.1`–`0.3` | `Q` regularization for ill-conditioned cases |
| `M` | `≥ n_x`, typically 40–100 | Lag count |

### 3.4 Suggested Code Architecture

```
noise_cov_id/
├── kalman.py              # Standard KF + DARE initialization
├── identifiability.py     # Minimal polynomial + matrix I (Algorithm 1)
├── estimate_W.py          # Step 3: adaptive gradient descent with bold driver
├── estimate_R.py          # Step 4: R1–R5, default R3 via Cholesky/eigendecomp
├── estimate_Q_P.py        # Step 5: inner loop for Q and P
├── six_step.py            # Top-level driver (Steps 1–6)
├── metrics.py             # RMSE, 95% PI, NIS computation
└── test_cases/
    ├── case1_kinematic.py
    ├── case2_neethling.py
    ├── case3_mehra5.py
    ├── case4_odelson2.py
    └── case5_odelson3.py
```

### 3.5 Validation Checklist

For each test case, verify:

1. **Identifiability:** `rank(I) − n_R ≥ n_Q` (eq. 37) — if this fails, the problem is ill-posed.
2. **Filter stability:** eigenvalues of `F̄ = F(I − WH)` are inside the unit circle at every iteration.
3. **Positive definiteness:** estimated `Q` and `R` have positive eigenvalues (guaranteed by `R3` and the masked `Q` update).
4. **Consistency:** averaged NIS lies within the 95% chi-squared confidence region for `n_z` degrees of freedom — eq. (191).
5. **Accuracy:** truth values lie within the 95% probability interval of the Monte Carlo estimate distribution.

### 3.6 Suggested Order of Attack

1. **Start with Case 2 (Neethling)** — it has the lowest condition number (`cond(I) = 2.3`), and the minimal-polynomial least squares also works, so you can cross-check.
2. **Move to Case 1** — lets you study the `M`-sensitivity curve and sanity-check with many MC runs.
3. **Case 3 (Mehra 5-state)** — the main benchmark. This is where the comparison against Mehra and Bélanger lives (Tables 5–8). Requires `N ≥ 5000` samples for stability.
4. **Cases 4 and 5** — ill-conditioned / unobservable edge cases that stress-test the regularization `λ_Q`.

### 3.7 Known Pitfalls

- **Pseudoinverse `Γ†`:** when `Γ` is tall and full column rank, use `(Γ'Γ)⁻¹·Γ'`. Numerically use `np.linalg.pinv` with a sensible `rcond`.
- **Minimal polynomial computation:** working from `eigvals(F̄)` with multiplicity detection is more stable than symbolic approaches; tolerance around `1e-8`.
- **`W⁽⁰⁾` must stabilize `F̄`.** If you pick it randomly, the KF may blow up. Always bootstrap from DARE.
- **Mehra's / Bélanger's baselines** require `N ≳ 5000` to produce stable gains (see Figs. 7–8). Don't report RMSE on runs where their `F̄` is unstable — filter those out or count them separately.
- **NIS normalization:** remember `ν'·S⁻¹·ν` should have mean ≈ `n_z`, so the plotted `ε̄(k)` in Fig. 4 is around 1 only because the paper plots `(1/n_z)·ν'·S⁻¹·ν` — check axis conventions before comparing.

---

## 4. Deliverables

1. Source code implementing the six-step algorithm with modular structure described above.
2. Scripts reproducing each of the five test cases with 100 (or 200) Monte Carlo runs.
3. Results tables matching Tables 3–10 of the paper: mean, `[r, r̄]` 95% PI, and RMSE for each estimated quantity (`R`, `Q`, `W`, `P̄`).
4. Plots analogous to Figs. 1–13: box plots vs. `M`, scatter plots of `(Q, R)` estimates, and averaged NIS over time with 95% consistency bands.
5. Short write-up comparing your reproduced numbers to the paper's values and noting any discrepancies.