"""
estimate_W.py — Step 3: adaptive gradient descent on the Kalman gain W.

The objective J measures the non-whiteness of the innovation sequence:

    J = (1/2) Σ_{i=1}^{M-1} tr{ E Ĉ(i)' E² Ĉ(i) E }

where E = diag(Ĉ(0))^{-1/2}.  The optimal W makes innovations white (J→0).

Gradient descent uses a bold-driver adaptive step size (eqs. 136–138).
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm, eigvals, pinv
from scipy.linalg import solve_discrete_lyapunov
from typing import Optional

from .kalman import (LinearSystem, closed_loop_F_bar,
                     run_kalman_filter, sample_autocovariances)


# ---------------------------------------------------------------------------
# Objective J
# ---------------------------------------------------------------------------

def compute_J(C_hat: np.ndarray) -> float:
    """Eq. (52): normalised sum of off-zero-lag innovation autocorrelations.

    J = (1/2) Σ_{i=1}^{M-1} tr{ E Ĉ(i)' E² Ĉ(i) E }
    """
    d = np.diag(C_hat[0])
    if np.any(d <= 0):
        # Degenerate innovations — return a large value
        return 1e30
    E = np.diag(1.0 / np.sqrt(d))
    E2 = E @ E
    M = C_hat.shape[0]
    J = 0.0
    for i in range(1, M):
        Ci = C_hat[i]
        J += 0.5 * np.trace(E @ Ci.T @ E2 @ Ci @ E)
    return float(J)


# ---------------------------------------------------------------------------
# Correlation residual X (eq. 63)
# ---------------------------------------------------------------------------

def compute_X(sys: LinearSystem,
              W: np.ndarray,
              C_hat: np.ndarray) -> np.ndarray:
    """Least-squares correlation residual matrix X (eq. 63).

    Solves:
        [H F; H F̄ F; ...; H F̄^{M-2} F] X = [Ĉ(1); ...; Ĉ(M-1)]

    Returns X of shape (n_x, n_z).
    """
    M = C_hat.shape[0]
    F_bar = closed_loop_F_bar(sys, W)
    blocks_A = []
    blocks_b = []
    F_bar_pow = np.eye(sys.n_x)
    for i in range(M - 1):
        blocks_A.append(sys.H @ F_bar_pow @ sys.F)
        blocks_b.append(C_hat[i + 1])
        F_bar_pow = F_bar_pow @ F_bar
    A = np.vstack(blocks_A)      # ((M-1)*n_z, n_x)
    rhs = np.vstack(blocks_b)    # ((M-1)*n_z, n_z)
    X = pinv(A) @ rhs            # (n_x, n_z)
    return X


# ---------------------------------------------------------------------------
# Lyapunov equation for Z (eq. 61)
# ---------------------------------------------------------------------------

def solve_Z(sys: LinearSystem,
            W: np.ndarray,
            C_hat: np.ndarray) -> np.ndarray:
    """Solve the discrete Lyapunov equation for Z (eq. 61).

    Equation:  Z = F̄' Z F̄ + Q_rhs
    where
        Q_rhs = (1/2) Σ_{i=1}^{M-1} [Φ(i)' E² Ĉ(i) E² H + transpose]
        Φ(i)  = H F̄^{i-1} F

    Returns Z of shape (n_x, n_x).
    """
    M = C_hat.shape[0]
    F_bar = closed_loop_F_bar(sys, W)
    d = np.diag(C_hat[0])
    E2 = np.diag(1.0 / np.maximum(d, 1e-30))

    rhs_sum = np.zeros((sys.n_x, sys.n_x))
    F_bar_pow = np.eye(sys.n_x)
    for i in range(1, M):
        Phi_i = sys.H @ F_bar_pow @ sys.F   # Φ(i) = H F̄^{i-1} F
        term = Phi_i.T @ E2 @ C_hat[i] @ E2 @ sys.H
        rhs_sum += 0.5 * (term + term.T)
        F_bar_pow = F_bar_pow @ F_bar

    # Solve Z − F̄' Z F̄ = rhs_sum
    Z = solve_discrete_lyapunov(F_bar.T, rhs_sum)
    return Z


# ---------------------------------------------------------------------------
# Gradient ∇_W J (eq. 60)
# ---------------------------------------------------------------------------

def compute_gradient(sys: LinearSystem,
                     W: np.ndarray,
                     C_hat: np.ndarray,
                     Z: np.ndarray,
                     X: np.ndarray) -> np.ndarray:
    """Analytic gradient of J w.r.t. W (eq. 60).

    Three-term formula:
        ∇J = − Σ_i Φ(i)' E² Ĉ(i) E² Ĉ(0)
             − F' Z F X
             − Σ_{i=2}^{M-1} Σ_{ℓ=0}^{i-2} [Ĉ(ℓ+1) E² Ĉ(i)' E² H F̄^{i-ℓ-2}]'

    The third term uses power F̄^{i-ℓ-2}, **not** F̄^ℓ.
    We pre-compute all needed powers for correctness.
    """
    M = C_hat.shape[0]
    F_bar = closed_loop_F_bar(sys, W)
    d = np.diag(C_hat[0])
    E2 = np.diag(1.0 / np.maximum(d, 1e-30))

    grad = np.zeros_like(W)

    # Term 1: − Σ_i Φ(i)' E² Ĉ(i) E² Ĉ(0)
    # Term 2: − F' Z F X  (added after the loop)
    F_bar_pow = np.eye(sys.n_x)
    for i in range(1, M):
        Phi_i = sys.H @ F_bar_pow @ sys.F
        grad -= Phi_i.T @ E2 @ C_hat[i] @ E2 @ C_hat[0]
        F_bar_pow = F_bar_pow @ F_bar

    grad -= sys.F.T @ Z @ sys.F @ X

    # Term 3: nested double sum with corrected power indexing
    # Pre-compute F̄^0, F̄^1, …, F̄^{M-2}
    max_power = max(M - 2, 0)
    F_bar_pows = [np.eye(sys.n_x)]
    for _ in range(max_power):
        F_bar_pows.append(F_bar_pows[-1] @ F_bar)

    for i in range(2, M):
        for ell in range(i - 1):          # ell = 0, …, i-2
            power = i - ell - 2           # required power of F̄
            F_bar_p = F_bar_pows[power]
            inner = (C_hat[ell + 1] @ E2 @ C_hat[i].T @ E2
                     @ sys.H @ F_bar_p)
            grad -= inner.T

    return grad


# ---------------------------------------------------------------------------
# Finite-difference gradient (for validation / debugging)
# ---------------------------------------------------------------------------

def finite_diff_grad(sys: LinearSystem,
                     W: np.ndarray,
                     z: np.ndarray,
                     M: int,
                     eps: float = 1e-6) -> np.ndarray:
    """Central-difference approximation of ∇_W J.

    Used only for debugging — O(2 n_x n_z N) per call.
    """
    grad_fd = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp = W.copy(); Wp[i, j] += eps
            Wm = W.copy(); Wm[i, j] -= eps
            nu_p, _ = run_kalman_filter(sys, Wp, z)
            nu_m, _ = run_kalman_filter(sys, Wm, z)
            Jp = compute_J(sample_autocovariances(nu_p, M))
            Jm = compute_J(sample_autocovariances(nu_m, M))
            grad_fd[i, j] = (Jp - Jm) / (2 * eps)
    return grad_fd


# ---------------------------------------------------------------------------
# Bold-driver step size (eqs. 136–138)
# ---------------------------------------------------------------------------

def initial_step_size(N: int, N_s: int,
                      c: float = 0.01,
                      beta: float = 2.0) -> float:
    """Eq. (136): initial step size."""
    return min(c * (N / N_s) ** beta, c)


def update_step_size(alpha_prev: float,
                     J_curr: float,
                     J_prev: float,
                     N: int,
                     N_s: int,
                     c_max: float = 0.2,
                     beta: float = 2.0) -> float:
    """Eqs. (137)–(138): bold-driver step size update."""
    c_bar = min((N / N_s) ** beta, c_max)
    if J_curr > J_prev:
        return 0.5 * alpha_prev
    else:
        return min(1.1 * alpha_prev, c_bar)


# ---------------------------------------------------------------------------
# Main gradient-descent loop (Step 3)
# ---------------------------------------------------------------------------

def estimate_W(sys: LinearSystem,
               z: np.ndarray,
               W0: np.ndarray,
               M: int,
               n_L: int = 100,
               patience: int = 5,
               zeta_W: float = 1e-6,
               zeta_J: float = 1e-6,
               zeta_Delta: float = 1e-6,
               c: float = 0.01,
               c_max: float = 0.2,
               beta: float = 2.0,
               N_s: Optional[int] = None,
               verbose: bool = False):
    """Iteratively refine W by descending on the innovation-whiteness objective J.

    Termination conditions (any one):
    1. Gradient norm < zeta_Delta
    2. Objective J < zeta_J
    3. Relative gain change < zeta_W
    4. Patience exhausted (stagnation counter >= patience)
    5. Maximum iterations n_L reached

    Returns (W_best, J_best, J_history).
    """
    N = z.shape[0]
    N_s = N_s if N_s is not None else N
    W = W0.copy()
    alpha = initial_step_size(N, N_s, c, beta)

    nu, _ = run_kalman_filter(sys, W, z)
    C_hat = sample_autocovariances(nu, M)
    J_prev = compute_J(C_hat)

    W_best = W.copy()
    J_best = J_prev
    J_history = [J_prev]
    stagnation = 0

    for r in range(n_L):
        X = compute_X(sys, W, C_hat)
        Z = solve_Z(sys, W, C_hat)
        grad = compute_gradient(sys, W, C_hat, Z, X)

        # Termination: gradient norm
        if norm(grad) < zeta_Delta:
            if verbose:
                print(f"  [W] stop: grad norm {norm(grad):.2e} < {zeta_Delta:.2e}")
            break

        W_new = W - alpha * grad

        # Stability check: reject update if F̄ becomes unstable
        eig_max = np.max(np.abs(eigvals(closed_loop_F_bar(sys, W_new))))
        if eig_max >= 1.0:
            alpha *= 0.5
            if verbose:
                print(f"  [W] iter {r}: unstable update (|eig|={eig_max:.4f}), halve α→{alpha:.2e}")
            continue

        nu_new, _ = run_kalman_filter(sys, W_new, z)
        C_hat_new = sample_autocovariances(nu_new, M)
        J_new = compute_J(C_hat_new)
        J_history.append(J_new)

        # Termination: J below tolerance
        if J_new < zeta_J:
            W, C_hat = W_new, C_hat_new
            if verbose:
                print(f"  [W] stop: J={J_new:.2e} < {zeta_J:.2e}")
            break

        # Termination: relative gain change
        rel_change = norm(W_new - W) / (norm(W) + 1e-12)
        if rel_change < zeta_W:
            W, C_hat = W_new, C_hat_new
            if verbose:
                print(f"  [W] stop: rel ΔW={rel_change:.2e} < {zeta_W:.2e}")
            break

        # Bold-driver step update
        alpha = update_step_size(alpha, J_new, J_prev, N, N_s, c_max, beta)

        # Track best seen
        if J_new < J_best:
            W_best = W_new.copy()
            J_best = J_new
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= patience:
                if verbose:
                    print(f"  [W] stop: patience exhausted at iter {r}")
                break

        W, J_prev, C_hat = W_new, J_new, C_hat_new

    if verbose:
        print(f"  [W] final J_best={J_best:.4e}")

    return W_best, J_best, J_history
