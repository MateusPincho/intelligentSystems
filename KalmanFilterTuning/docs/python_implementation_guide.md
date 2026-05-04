# Python Implementation Guide: Six-Step Algorithm for Noise Covariance Identification



**Reference:** Zhang et al. (2020), IEEE Access, Vol. 8, pp. 59362–59388.

This document breaks down each step of the six-step algorithm into a concrete Python implementation plan. Each section specifies: inputs, outputs, equations used, recommended libraries, numerical caveats, and pseudocode.

---

## 0. Shared Infrastructure (build first)

### 0.1 Dependencies

```python
import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov, solve_lyapunov
from scipy.linalg import cholesky, eigh, pinv, inv
from numpy.linalg import matrix_rank, eigvals, norm
from dataclasses import dataclass, field
from typing import Optional, Callable
```

### 0.2 System container

Bundle the system matrices and estimates in a dataclass to avoid long argument lists:

```python
@dataclass
class LinearSystem:
    F: np.ndarray           # n_x × n_x state transition
    H: np.ndarray           # n_z × n_x measurement
    Gamma: np.ndarray       # n_x × n_v process-noise gain
    n_x: int = field(init=False)
    n_z: int = field(init=False)
    n_v: int = field(init=False)

    def __post_init__(self):
        self.n_x = self.F.shape[0]
        self.n_z = self.H.shape[0]
        self.n_v = self.Gamma.shape[1]

@dataclass
class FilterEstimate:
    W: np.ndarray           # Kalman gain
    S: np.ndarray           # innovation covariance
    R: np.ndarray           # measurement-noise covariance
    Q: np.ndarray           # process-noise covariance
    P_bar: np.ndarray       # prediction covariance
    P: np.ndarray           # updated covariance
    J_history: list = field(default_factory=list)
```

### 0.3 DARE-based stabilizing gain initializer

Needed to produce an initial `W⁽⁰⁾` that renders `F̄ = F(I − WH)` Schur-stable.

```python
def initial_gain_from_dare(sys: LinearSystem, Q0: np.ndarray, R0: np.ndarray) -> np.ndarray:
    """Solve DARE with guess (Q0, R0) and return the resulting steady-state gain."""
    # scipy's solve_discrete_are solves: X = F'·X·F - F'·X·H'·(R + H·X·H')⁻¹·H·X·F + Γ·Q·Γ'
    # Note scipy's convention: solve_discrete_are(A, B, Q, R) with A=F', B=H', so we transpose.
    P_bar = solve_discrete_are(sys.F.T, sys.H.T, sys.Gamma @ Q0 @ sys.Gamma.T, R0)
    S = sys.H @ P_bar @ sys.H.T + R0
    W = P_bar @ sys.H.T @ inv(S)
    return W
```

**Pitfall:** `scipy.linalg.solve_discrete_are` uses a transposed convention from control textbooks. Verify on a scalar example: for `F=0.9, H=1, Γ=1, Q=1, R=1` the steady-state `P̄` should satisfy `P̄ = 0.81·P̄ − 0.81·P̄²/(P̄+1) + 1`.

### 0.4 Closed-loop matrix helper

```python
def closed_loop_F_bar(sys: LinearSystem, W: np.ndarray) -> np.ndarray:
    return sys.F @ (np.eye(sys.n_x) - W @ sys.H)
```

---

## Step 1 — Run the Kalman Filter

**Goal:** Given the current gain `W⁽ʳ⁾`, run the filter across all `N` samples and store the innovation and post-fit-residual sequences.

### Equations
- (128) `x̂(k+1|k) = F·x̂(k|k)`
- (129) `ν(k+1) = z(k+1) − H·x̂(k+1|k)`
- (130) `x̂(k+1|k+1) = x̂(k+1|k) + W·ν(k+1)`
- (131) `µ(k+1) = z(k+1) − H·x̂(k+1|k+1)`

Note: in steady state, we use the **constant** gain `W`, not the time-varying one from the Riccati recursion. This is the whole point — we are testing how well a constant `W` whitens the innovations.

### Implementation

```python
def run_kalman_filter(sys: LinearSystem, W: np.ndarray, z: np.ndarray,
                      x0: Optional[np.ndarray] = None):
    """
    z : (N, n_z) array of measurements
    Returns innovations (N, n_z) and post-fit residuals (N, n_z).
    """
    N = z.shape[0]
    x_hat = np.zeros(sys.n_x) if x0 is None else x0.copy()
    nu = np.zeros((N, sys.n_z))
    mu = np.zeros((N, sys.n_z))

    for k in range(N):
        # Predict
        x_pred = sys.F @ x_hat
        # Innovation
        nu[k] = z[k] - sys.H @ x_pred
        # Update
        x_hat = x_pred + W @ nu[k]
        # Post-fit residual
        mu[k] = z[k] - sys.H @ x_hat

    return nu, mu
```

### Caveats
- **Burn-in:** discard the first `~20–50` samples before computing statistics — the constant-gain filter takes time to forget a bad `x₀`.
- **Vectorization:** for speed on large `N`, the loop can stay (it's already O(N)), but avoid allocating inside the loop.
- The truth `x₀` can be drawn from `N(0, P̄⁽⁰⁾)` in simulation studies.

---

## Step 2 — Sample Autocovariances

**Goal:** Compute `Ĉ(i)` for `i = 0, 1, ..., M − 1` from the innovation sequence.

### Equation
- (50) `Ĉ(i) = (1/(N − M)) · Σⱼ₌₁^(N−M) ν(j)·ν(j+i)'`

### Implementation

```python
def sample_autocovariances(nu: np.ndarray, M: int) -> np.ndarray:
    """
    nu : (N, n_z) innovations
    Returns C_hat : (M, n_z, n_z) stack with C_hat[i] = Ĉ(i).
    """
    N, n_z = nu.shape
    C_hat = np.zeros((M, n_z, n_z))
    for i in range(M):
        # ν(j) · ν(j+i)'  summed over j = 0, ..., N-M-1
        C_hat[i] = (nu[:N-M].T @ nu[i:N-M+i]) / (N - M)
    return C_hat
```

### Caveats
- Choose `M ≥ n_x`. In the paper, `M = 40` (Cases 3–4) or `M = 100` (Case 1, Case 4-alt). Case 5 uses `M = 15`.
- The estimator is biased for finite `N`; the paper uses `N − M` in the denominator (not `N`), matching the original Mehra convention. Stick with that.

---

## Step 3 — Adaptive Gradient Descent on `W`

**Goal:** Iteratively refine `W` by descending on the objective `J` in eq. (52) using the gradient (60). This is the most complex step.

### 3.1 Objective function `J`

```python
def compute_J(C_hat: np.ndarray) -> float:
    """Eq. (52): normalized sum of off-zero-lag autocorrelations."""
    M = C_hat.shape[0]
    diag_C0 = np.diag(np.diag(C_hat[0]))      # Hadamard product with identity
    E = np.diag(1.0 / np.sqrt(np.diag(C_hat[0])))   # E = diag(C(0))^(-1/2)
    E2 = E @ E                                 # = diag(C(0))^(-1)
    J = 0.0
    for i in range(1, M):
        Ci = C_hat[i]
        J += 0.5 * np.trace(E @ Ci.T @ E2 @ Ci @ E)
    return J
```

### 3.2 Theoretical correlation residual `X`

From eq. (63):
```
X = pinv([HF; HF̄F; ... ; HF̄^(M−1)F]) · [Ĉ(1); Ĉ(2); ...; Ĉ(M−1)]
```

```python
def compute_X(sys: LinearSystem, W: np.ndarray, C_hat: np.ndarray) -> np.ndarray:
    M = C_hat.shape[0]
    F_bar = closed_loop_F_bar(sys, W)
    # Build the stacked regressor
    blocks = []
    F_bar_power = np.eye(sys.n_x)
    for i in range(M - 1):           # corresponds to lags 1..M-1
        blocks.append(sys.H @ F_bar_power @ sys.F)
        F_bar_power = F_bar_power @ F_bar
    A = np.vstack(blocks)            # shape ((M-1)*n_z, n_x)
    rhs = np.vstack([C_hat[i] for i in range(1, M)])  # ((M-1)*n_z, n_z)
    X = pinv(A) @ rhs                # (n_x, n_z)
    return X
```

### 3.3 Solve the Lyapunov equation for `Z`

From eq. (61):
```
Z = F̄'·Z·F̄ + (1/2)·Σᵢ [Φ(i)'·E²·Ĉ(i)·E²·H + (Φ(i)'·E²·Ĉ(i)·E²·H)']
```
where `Φ(i) = H·F̄^(i−1)·F`.

```python
def solve_Z(sys: LinearSystem, W: np.ndarray, C_hat: np.ndarray) -> np.ndarray:
    M = C_hat.shape[0]
    F_bar = closed_loop_F_bar(sys, W)
    E2 = np.diag(1.0 / np.diag(C_hat[0]))
    rhs_sum = np.zeros((sys.n_x, sys.n_x))
    F_bar_power = np.eye(sys.n_x)
    for i in range(1, M):
        Phi_i = sys.H @ F_bar_power @ sys.F     # i.e. H·F̄^(i-1)·F
        term = Phi_i.T @ E2 @ C_hat[i] @ E2 @ sys.H
        rhs_sum += 0.5 * (term + term.T)
        F_bar_power = F_bar_power @ F_bar
    # Solve F̄'·Z·F̄ − Z + rhs_sum = 0   →   discrete Lyapunov:  Z − F̄'·Z·F̄ = rhs_sum
    Z = solve_discrete_lyapunov(F_bar.T, rhs_sum)
    return Z
```

### 3.4 Gradient `∇_W J`

Equation (60):
```
∇_W J = − Σᵢ Φ(i)'·E²·Ĉ(i)·E²·Ĉ(0) − F'·Z·F·X
        − Σᵢ Σ_{ℓ=0}^{i−2} [Ĉ(ℓ+1)·E²·Ĉ(i)'·E²·H·F̄^(i−ℓ−2)]'
```

```python
def compute_gradient(sys: LinearSystem, W: np.ndarray, C_hat: np.ndarray,
                     Z: np.ndarray, X: np.ndarray) -> np.ndarray:
    M = C_hat.shape[0]
    F_bar = closed_loop_F_bar(sys, W)
    E2 = np.diag(1.0 / np.diag(C_hat[0]))
    grad = np.zeros_like(W)

    # First two terms
    F_bar_power = np.eye(sys.n_x)
    for i in range(1, M):
        Phi_i = sys.H @ F_bar_power @ sys.F
        grad -= Phi_i.T @ E2 @ C_hat[i] @ E2 @ C_hat[0]
        F_bar_power = F_bar_power @ F_bar
    grad -= sys.F.T @ Z @ sys.F @ X

    # Cross term (third sum)
    for i in range(2, M):
        F_bar_power = np.eye(sys.n_x)
        for ell in range(i - 1):
            inner = C_hat[ell + 1] @ E2 @ C_hat[i].T @ E2 @ sys.H @ F_bar_power
            # Note: F_bar_power here is F̄^(i-ℓ-2). Build it incrementally.
            grad -= inner.T
            F_bar_power = F_bar_power @ F_bar

    return grad
```

**Warning:** the gradient formula has a nested double sum — transcribe (60) carefully and verify numerically with a finite-difference check on a tiny 2×2 system before trusting it.

### 3.5 Finite-difference gradient check (debugging)

```python
def finite_diff_grad(sys, W, z, M, eps=1e-6):
    grad_fd = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_plus = W.copy();  W_plus[i,j] += eps
            W_minus = W.copy(); W_minus[i,j] -= eps
            nu_p, _ = run_kalman_filter(sys, W_plus, z)
            nu_m, _ = run_kalman_filter(sys, W_minus, z)
            Jp = compute_J(sample_autocovariances(nu_p, M))
            Jm = compute_J(sample_autocovariances(nu_m, M))
            grad_fd[i,j] = (Jp - Jm) / (2*eps)
    return grad_fd
```
Run this once at the start of development to confirm your analytic gradient matches.

### 3.6 Bold-driver step size

Equations (136)–(138):

```python
def initial_step_size(N: int, N_s: int, c: float = 0.01, beta: float = 2.0) -> float:
    return min(c * (N / N_s) ** beta, c)

def update_step_size(alpha_prev: float, J_curr: float, J_prev: float,
                     N: int, N_s: int, c_max: float = 0.2, beta: float = 2.0) -> float:
    c_bar = min((N / N_s) ** beta, c_max)
    if J_curr > J_prev:
        return 0.5 * alpha_prev
    else:
        return min(1.1 * alpha_prev, c_bar)
```

### 3.7 Putting Step 3 together

```python
def estimate_W(sys, z, W0, M, n_L=100, patience=5,
               zeta_W=1e-6, zeta_J=1e-6, zeta_Delta=1e-6,
               c=0.01, c_max=0.2, beta=2.0, N_s=None, verbose=False):
    N = z.shape[0]
    N_s = N_s or N
    W = W0.copy()
    alpha = initial_step_size(N, N_s, c, beta)

    nu, _ = run_kalman_filter(sys, W, z)
    C_hat = sample_autocovariances(nu, M)
    J_prev = compute_J(C_hat)

    W_best, J_best = W.copy(), J_prev
    J_history = [J_prev]
    stagnation = 0

    for r in range(n_L):
        X = compute_X(sys, W, C_hat)
        Z = solve_Z(sys, W, C_hat)
        grad = compute_gradient(sys, W, C_hat, Z, X)

        # Termination: gradient norm
        if norm(grad) < zeta_Delta:
            break

        W_new = W - alpha * grad

        # Termination: stability check (reject update if unstable)
        if np.max(np.abs(eigvals(closed_loop_F_bar(sys, W_new)))) >= 1.0:
            alpha *= 0.5
            continue

        nu, _ = run_kalman_filter(sys, W_new, z)
        C_hat_new = sample_autocovariances(nu, M)
        J_new = compute_J(C_hat_new)
        J_history.append(J_new)

        # Termination: J tolerance
        if J_new < zeta_J:
            W, C_hat = W_new, C_hat_new
            break

        # Termination: gain change
        rel_change = norm((W_new - W) / (np.abs(W) + 1e-12))
        if rel_change < zeta_W:
            W, C_hat = W_new, C_hat_new
            break

        # Bold driver
        alpha = update_step_size(alpha, J_new, J_prev, N, N_s, c_max, beta)

        if J_new < J_best:
            W_best, J_best = W_new.copy(), J_new
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= patience:
                break

        W, J_prev, C_hat = W_new, J_new, C_hat_new

    # Final: return best-seen gain
    return W_best, J_best, J_history
```

---

## Step 4 — Estimate `R`

**Goal:** Given the converged `W` and innovation covariance `S`, solve for `R`.

### Equations (Proposition 2)
- (76) `R1: R = (I − H·W)·S`
- (77) `R2: R = (1/2)·{E[µ·ν'] + E[ν·µ']}`
- (78) `R3: G = R·S⁻¹·R` ← **recommended; preserves PD**
- (79) `R4: R = (1/2)·(G + S − H·W·S·W'·H')`
- (80) `R5: R = (1/2)·{G·(I − W'·H')⁻¹ + (I − H·W)⁻¹·G}`

The five are theoretically equal but numerically distinct.

### Implementation of `R3` via Cholesky + eigendecomposition (Appendix F)

```python
def estimate_R_from_G(G: np.ndarray, S: np.ndarray, diagonal: bool = False) -> np.ndarray:
    """
    Solve G = R · S⁻¹ · R for R (positive definite).
    """
    # Cholesky of S⁻¹:  S⁻¹ = L·L'
    S_inv = inv(S)
    # Make sure S_inv is symmetric
    S_inv = 0.5 * (S_inv + S_inv.T)
    L = cholesky(S_inv, lower=True)
    # (L'·R·L)² = L'·G·L
    M_sym = L.T @ G @ L
    M_sym = 0.5 * (M_sym + M_sym.T)
    eigvals_M, U = eigh(M_sym)
    # Numerical safety: clip tiny negatives
    eigvals_M = np.clip(eigvals_M, 0, None)
    sqrt_M = U @ np.diag(np.sqrt(eigvals_M)) @ U.T
    R_est = inv(L.T) @ sqrt_M @ inv(L)
    R_est = 0.5 * (R_est + R_est.T)
    if diagonal:
        R_est = np.diag(np.diag(R_est))
    return R_est
```

### Workflow

```python
def estimate_R(sys: LinearSystem, W: np.ndarray, nu: np.ndarray, mu: np.ndarray,
               diagonal: bool = False) -> tuple:
    S = (nu.T @ nu) / len(nu)              # innovation covariance estimate
    G = (mu.T @ mu) / len(mu)              # post-fit residual covariance
    S = 0.5 * (S + S.T)
    G = 0.5 * (G + G.T)
    R = estimate_R_from_G(G, S, diagonal=diagonal)
    return R, S, G
```

### Caveats
- **Positive-definiteness** of estimated `R` is not automatic for `R1`, `R2`, `R4`, `R5`. Only `R3` guarantees it (via the Riccati-style square root). This is why the paper recommends `R3`.
- When `R` is known to be diagonal, take `diag(R_est)` at the end — this is the Frobenius-norm least-squares projection onto the diagonal cone.
- If `G` has tiny negative eigenvalues from sampling noise, clip them to zero before the sqrt.

---

## Step 5 — Estimate `Q` and `P` Iteratively

**Goal:** Given `W`, `S`, `R`, find `Q` and `P` jointly via a double-loop fixed-point iteration.

### Equations
- (122) Lyapunov initializer for `P⁽⁰⁾`
- (123) Inner recursion: `P⁽ℓ⁺¹⁾ = [(F·P⁽ℓ⁾·F' + Γ·Q·Γ')⁻¹ + H'·R⁻¹·H]⁻¹`
- (124)–(125) Outer `Q` update: `Γ·Q·Γ' = P + W·S·W' − F·P·F'`
- (126) Mask for structural constraints
- (127) Regularization: `Q = Γ†·(D + λ_Q·I)·(Γ')†`

### Implementation

```python
def estimate_Q_and_P(sys: LinearSystem, W: np.ndarray, S: np.ndarray, R: np.ndarray,
                    mask: Optional[np.ndarray] = None,
                    lambda_Q: float = 0.0,
                    tol: float = 1e-8,
                    max_outer: int = 50,
                    max_inner: int = 200):
    F, H, G = sys.F, sys.H, sys.Gamma
    R_inv = inv(R)
    H_R_H = H.T @ R_inv @ H

    # Initial Q⁽⁰⁾ = W·S·W' projected through Γ†
    G_pinv = pinv(G)
    Q = G_pinv @ (W @ S @ W.T) @ G_pinv.T
    Q = 0.5 * (Q + Q.T)
    if mask is not None:
        Q = mask * Q

    F_bar = closed_loop_F_bar(sys, W)
    # Initial P from Lyapunov eq (122): solve  P - F̃·P·F̃' = WRW' + (I-WH)·Γ·Q·Γ'·(I-WH)'
    F_tilde = (np.eye(sys.n_x) - W @ H) @ F
    rhs = W @ R @ W.T + (np.eye(sys.n_x) - W @ H) @ G @ Q @ G.T @ (np.eye(sys.n_x) - W @ H).T
    P = solve_discrete_lyapunov(F_tilde, rhs)

    for t in range(max_outer):
        Q_prev = Q.copy()

        # Inner loop: iterate (123) until P converges
        for _ in range(max_inner):
            P_pred = F @ P @ F.T + G @ Q @ G.T
            P_pred_inv = inv(P_pred)
            P_new = inv(P_pred_inv + H_R_H)
            P_new = 0.5 * (P_new + P_new.T)
            if norm(P_new - P) < tol:
                P = P_new
                break
            P = P_new

        # Outer Q update (124)–(127)
        D = P + W @ S @ W.T - F @ P @ F.T
        Q_new = G_pinv @ (D + lambda_Q * np.eye(sys.n_x)) @ G_pinv.T
        Q_new = 0.5 * (Q_new + Q_new.T)
        if mask is not None:
            Q_new = mask * Q_new

        if norm(Q_new - Q_prev) < tol:
            Q = Q_new
            break
        Q = Q_new

    # P_bar from eq. (111)
    P_bar = F @ P @ F.T + G @ Q @ G.T
    return Q, P, P_bar
```

### Caveats
- `Γ† @ D @ (Γ')†` does *not* necessarily yield a PSD `Q` if `D` isn't in the range of `Γ`. The mask + regularization helps, but for pathological cases you may need to project onto the PSD cone (eigen-decompose, clip negatives to 0, recompose).
- `lambda_Q` is case-dependent: paper uses `0` for Cases 1–3, `0.1` for Case 4, `0.3` for Case 5.
- The mask `A` is, for example, `np.eye(n_v)` to enforce diagonal `Q`, or `np.ones((n_v, n_v))` for unrestricted.
- **Special cases** (Section IX): if `F = I` and `H = I` (Wiener process), skip this iteration entirely and use `Q = W·S·W'`, `P̄ = W·S`.

---

## Step 6 — Outer Successive-Approximation Loop

**Goal:** Re-initialize the Kalman filter using the just-estimated `(Q, R)`, get a new stabilizing gain, and repeat Steps 1–5 until `J` stops improving across outer iterations.

### Implementation

```python
def six_step_algorithm(sys: LinearSystem, z: np.ndarray,
                       Q0: np.ndarray, R0: np.ndarray,
                       M: int = 40,
                       max_outer: int = 20,
                       mask_Q: Optional[np.ndarray] = None,
                       diagonal_R: bool = False,
                       lambda_Q: float = 0.0,
                       zeta_J: float = 1e-6,
                       **inner_kwargs) -> FilterEstimate:

    # Step 1 outer init: DARE-based stabilizing gain
    W = initial_gain_from_dare(sys, Q0, R0)

    best = None
    J_best = np.inf
    J_outer_history = []

    for outer in range(max_outer):
        # Steps 1–3: refine W
        W, J_final, _ = estimate_W(sys, z, W, M, **inner_kwargs)

        # Step 1 rerun to get fresh nu and mu at the converged W
        nu, mu = run_kalman_filter(sys, W, z)

        # Step 4: estimate R
        R, S, G = estimate_R(sys, W, nu, mu, diagonal=diagonal_R)

        # Step 5: estimate Q, P, P_bar
        Q, P, P_bar = estimate_Q_and_P(sys, W, S, R, mask=mask_Q, lambda_Q=lambda_Q)

        J_outer_history.append(J_final)

        if J_final < J_best:
            J_best = J_final
            best = FilterEstimate(W=W.copy(), S=S.copy(), R=R.copy(),
                                  Q=Q.copy(), P_bar=P_bar.copy(), P=P.copy(),
                                  J_history=list(J_outer_history))

        # Convergence across outer loops
        if outer > 0 and abs(J_outer_history[-1] - J_outer_history[-2]) < zeta_J:
            break

        # Re-seed W for next outer iteration using the fresh (Q, R)
        try:
            W = initial_gain_from_dare(sys, Q, R)
        except Exception:
            # Keep current W if DARE fails (e.g., R became near-singular)
            pass

    return best
```

### Caveats
- **DARE re-seeding can fail** if estimated `R` is nearly singular or `Q` is indefinite. Wrap in a try/except and fall back to the current `W`.
- Track the **best** `(W, Q, R)` across outer iterations by `J`, not the final one — `J` can oscillate near the optimum.

---

## 7. Driver: Monte Carlo Harness

Each test case needs 100+ runs with different noise realizations, then aggregation into tables matching the paper's format.

```python
def simulate_trajectory(sys: LinearSystem, Q_true: np.ndarray, R_true: np.ndarray,
                        N: int, x0: Optional[np.ndarray] = None, rng=None):
    rng = rng or np.random.default_rng()
    x = np.zeros(sys.n_x) if x0 is None else x0.copy()
    z = np.zeros((N, sys.n_z))
    # Cholesky factors for sampling
    L_Q = cholesky(Q_true, lower=True)
    L_R = cholesky(R_true, lower=True)
    for k in range(N):
        v = L_Q @ rng.standard_normal(sys.n_v)
        w = L_R @ rng.standard_normal(sys.n_z)
        x = sys.F @ x + sys.Gamma @ v
        z[k] = sys.H @ x + w
    return z

def monte_carlo(sys, Q_true, R_true, Q0, R0, N, n_runs, **alg_kwargs):
    results = []
    for run in range(n_runs):
        rng = np.random.default_rng(seed=run)
        z = simulate_trajectory(sys, Q_true, R_true, N, rng=rng)
        est = six_step_algorithm(sys, z, Q0, R0, **alg_kwargs)
        results.append(est)
    return results
```

### Reporting helpers

```python
def summarize(estimates, true_value, name="R11"):
    """Return mean, 95% PI, RMSE for a scalar quantity across runs."""
    values = np.array([e for e in estimates])
    lower, upper = np.percentile(values, [2.5, 97.5])
    mean = values.mean()
    rmse = np.sqrt(((values - true_value) ** 2).mean())
    return {"name": name, "truth": true_value, "lower": lower, "mean": mean,
            "upper": upper, "rmse": rmse}

def averaged_NIS(sys, W, S, z_batches):
    """Eq. (191): average NIS across MC runs at each time step."""
    per_run = []
    S_inv = inv(S)
    for z in z_batches:
        nu, _ = run_kalman_filter(sys, W, z)
        nis = np.einsum('ki,ij,kj->k', nu, S_inv, nu)
        per_run.append(nis)
    return np.array(per_run).mean(axis=0)
```

---

## 8. Identifiability Pre-Check (Algorithm 1 of the paper)

Run this **before** the main algorithm to confirm `Q` and `R` are identifiable for the system.

```python
def minimal_polynomial_coeffs(F_bar: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Return the coefficients [a_0=1, a_1, ..., a_m] of the minimal polynomial of F_bar.
    For most practical systems this equals the characteristic polynomial
    (distinct eigenvalues), so np.poly is sufficient.
    """
    # Characteristic polynomial — use np.poly(eigvals) then deduplicate repeated roots.
    eigs = eigvals(F_bar)
    # Cluster nearly-equal eigenvalues
    unique = []
    for e in eigs:
        if not any(abs(e - u) < tol for u in unique):
            unique.append(e)
    coeffs = np.poly(unique).real
    return coeffs

def build_identifiability_matrix(sys, W):
    """Implement Algorithm 1 of the paper, returning matrix I."""
    F_bar = closed_loop_F_bar(sys, W)
    a = minimal_polynomial_coeffs(F_bar)
    m = len(a) - 1
    n_v, n_z = sys.n_v, sys.n_z

    # Build B_l and G_l, eqs. (28)–(30)
    B = [np.zeros((n_z, n_v)) for _ in range(m + 1)]
    G = [np.zeros((n_z, n_z)) for _ in range(m + 1)]
    G[0] = np.eye(n_z)
    for l in range(1, m + 1):
        accum = np.zeros_like(sys.F)
        F_bar_power = np.eye(sys.n_x)
        for i in range(l):
            accum += a[i] * F_bar_power           # a_i · F̄^(l-i-1)? — check indexing
            F_bar_power = F_bar_power @ F_bar
        # Actually from (28): Σ_{i=0}^{l-1} a_i · F̄^(l-i-1)
        # Reset accum with correct power sequence:
        accum = sum(a[i] * np.linalg.matrix_power(F_bar, l - i - 1) for i in range(l))
        B[l] = sys.H @ accum @ sys.Gamma
        G[l] = a[l] * np.eye(n_z) - sys.H @ accum @ sys.F @ W

    # Build I column-by-column following Algorithm 1
    n_Q = n_v * (n_v + 1) // 2
    n_R = n_z * (n_z + 1) // 2
    I_mat = np.zeros(((m + 1) * n_z * n_z, n_Q + n_R))

    for j in range(m + 1):
        r = j * n_z * n_z
        k = 0
        # Columns for Q
        for l in range(n_v):
            # Diagonal element q_ll
            b = sum(np.outer(B[i][:, l], B[i - j][:, l])
                    for i in range(j + 1, m + 1))
            I_mat[r:r + n_z * n_z, k] = b.flatten()
            k += 1
            for p in range(l + 1, n_v):
                d = sum(np.outer(B[i][:, l], B[i - j][:, p]) +
                        np.outer(B[i][:, p], B[i - j][:, l])
                        for i in range(j + 1, m + 1))
                I_mat[r:r + n_z * n_z, k] = d.flatten()
                k += 1
        # Columns for R
        for l in range(n_z):
            g = sum(np.outer(G[i][:, l], G[i - j][:, l]) for i in range(j, m + 1))
            I_mat[r:r + n_z * n_z, k] = g.flatten()
            k += 1
            for p in range(l + 1, n_z):
                f = sum(np.outer(G[i][:, l], G[i - j][:, p]) +
                        np.outer(G[i][:, p], G[i - j][:, l])
                        for i in range(j, m + 1))
                I_mat[r:r + n_z * n_z, k] = f.flatten()
                k += 1

    return I_mat, matrix_rank(I_mat), n_Q + n_R
```

### Usage

```python
I_mat, rank_I, n_unknowns = build_identifiability_matrix(sys, W0)
if rank_I < n_unknowns:
    print(f"WARNING: underdetermined (rank {rank_I} < unknowns {n_unknowns})")
    print("Consider imposing diagonal-Q or diagonal-R constraints.")
```

---

## 9. Minimal End-to-End Example (Case 1 sketch)

```python
T = 0.1
F = np.array([[1, T], [0, 1]])
Gamma = np.array([[0.5 * T**2], [T]])
H = np.array([[1, 0]])
sys = LinearSystem(F=F, H=H, Gamma=Gamma)

Q_true = np.array([[0.0025]])
R_true = np.array([[0.01]])
Q0, R0 = np.array([[0.1]]), np.array([[0.1]])

rng = np.random.default_rng(0)
z = simulate_trajectory(sys, Q_true, R_true, N=1000, rng=rng)

est = six_step_algorithm(sys, z, Q0, R0,
                         M=100,
                         mask_Q=np.ones((1, 1)),   # Q is scalar
                         diagonal_R=False,
                         lambda_Q=0.0,
                         n_L=100, N_s=1000)

print("Estimated W:\n", est.W)
print("Estimated R:", est.R)
print("Estimated Q:", est.Q)
print("Estimated P_bar:\n", est.P_bar)
```

Expected values (from Table 3 of the paper): `R ≈ 0.01`, `Q ≈ 0.0025`, `W ≈ [0.0952, 0.0476]'`.

---

## 10. Development Order & Testing Checklist

1. **Build shared infrastructure** (`LinearSystem`, `run_kalman_filter`, `sample_autocovariances`, `initial_gain_from_dare`).
2. **Unit test the KF**: simulate a scalar AR(1) + noise, check innovations are white and have correct variance.
3. **Implement `compute_J` and finite-difference gradient check** on a tiny system.
4. **Implement `compute_X`, `solve_Z`, `compute_gradient`** and verify analytical vs. finite-difference gradients match to ~1e-5 relative error.
5. **Implement `estimate_W`** with bold-driver step size; test on Case 2 (Neethling) which has `cond(I) = 2.3`.
6. **Implement `estimate_R`** via `R3`; cross-check by plugging the true `W` and verifying recovery to within MC noise.
7. **Implement `estimate_Q_and_P`** inner/outer loop; test the Wiener-process special case first where closed-form is available.
8. **Assemble `six_step_algorithm`** and run Case 2 single-shot; compare to Table 4.
9. **Build MC harness**; reproduce Case 1 Table 3.
10. **Tackle Case 3** (5-state); this is where bugs in the gradient or identifiability matrix will show up.
11. **Cases 4 and 5**; tune `lambda_Q`.

### Sanity checks at every level
- Symmetrize all covariance updates: `M ← 0.5·(M + M.T)`.
- Check Schur-stability of `F̄` after every gain update; reject unstable updates and halve the step size.
- Clip negative eigenvalues arising from sampling noise when taking matrix square roots.
- Set `np.random.default_rng(seed)` per MC run for reproducibility.
- Log `J` per iteration; if it ever increases by more than 2× in one step, your gradient or step size is wrong.
