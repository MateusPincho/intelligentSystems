"""
kalman.py — Foundation module for the six-step adaptive noise covariance identification.

Provides:
- LinearSystem: dataclass for system matrices
- FilterEstimate: dataclass for algorithm outputs
- initial_gain_from_dare: DARE-based stabilizing Kalman gain initializer
- closed_loop_F_bar: closed-loop state transition matrix
- run_kalman_filter: constant-gain steady-state KF pass
- sample_autocovariances: sample lag-i autocovariance matrices
- simulate_trajectory: synthetic trajectory generator
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.linalg import solve_discrete_are, cholesky
from numpy.linalg import inv


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class LinearSystem:
    """Discrete-time linear system x(k+1) = F x(k) + Γ v(k), z(k) = H x(k) + w(k)."""
    F: np.ndarray       # (n_x, n_x) state transition
    H: np.ndarray       # (n_z, n_x) measurement
    Gamma: np.ndarray   # (n_x, n_v) process-noise gain
    n_x: int = field(init=False)
    n_z: int = field(init=False)
    n_v: int = field(init=False)

    def __post_init__(self):
        self.F = np.atleast_2d(np.array(self.F, dtype=float))
        self.H = np.atleast_2d(np.array(self.H, dtype=float))
        self.Gamma = np.atleast_2d(np.array(self.Gamma, dtype=float))
        self.n_x = self.F.shape[0]
        self.n_z = self.H.shape[0]
        self.n_v = self.Gamma.shape[1]


@dataclass
class FilterEstimate:
    """Output of the six-step algorithm."""
    W: np.ndarray           # (n_x, n_z) Kalman gain
    S: np.ndarray           # (n_z, n_z) innovation covariance
    R: np.ndarray           # (n_z, n_z) measurement-noise covariance
    Q: np.ndarray           # (n_v, n_v) process-noise covariance
    P_bar: np.ndarray       # (n_x, n_x) prediction covariance
    P: np.ndarray           # (n_x, n_x) updated covariance
    J_history: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def closed_loop_F_bar(sys: LinearSystem, W: np.ndarray) -> np.ndarray:
    """Return F̄ = F (I − W H)."""
    return sys.F @ (np.eye(sys.n_x) - W @ sys.H)


def initial_gain_from_dare(sys: LinearSystem,
                           Q0: np.ndarray,
                           R0: np.ndarray) -> np.ndarray:
    """Solve the DARE and return the steady-state Kalman gain.

    scipy convention: solve_discrete_are(A, B, Q, R) solves
        X = A' X A − A' X B (R + B' X B)^{-1} B' X A + Q
    To match the standard KF DARE we pass A=F', B=H', Q=Γ Q Γ', R=R0.
    """
    Q0 = np.atleast_2d(np.array(Q0, dtype=float))
    R0 = np.atleast_2d(np.array(R0, dtype=float))
    GQG = sys.Gamma @ Q0 @ sys.Gamma.T
    # Symmetrize to avoid numerical issues
    GQG = 0.5 * (GQG + GQG.T)
    R0s = 0.5 * (R0 + R0.T)
    P_bar = solve_discrete_are(sys.F.T, sys.H.T, GQG, R0s)
    P_bar = 0.5 * (P_bar + P_bar.T)
    S = sys.H @ P_bar @ sys.H.T + R0s
    W = P_bar @ sys.H.T @ inv(S)
    return W


# ---------------------------------------------------------------------------
# Kalman filter pass (constant gain, steady-state)
# ---------------------------------------------------------------------------

def run_kalman_filter(sys: LinearSystem,
                      W: np.ndarray,
                      z: np.ndarray,
                      x0: Optional[np.ndarray] = None):
    """Run the constant-gain steady-state Kalman filter.

    Equations (128)–(131):
        x̂(k+1|k)   = F x̂(k|k)
        ν(k+1)      = z(k+1) − H x̂(k+1|k)
        x̂(k+1|k+1) = x̂(k+1|k) + W ν(k+1)
        µ(k+1)      = z(k+1) − H x̂(k+1|k+1)

    Parameters
    ----------
    z : (N, n_z) measurement array
    Returns (nu, mu), each (N, n_z).
    """
    N = z.shape[0]
    x_hat = np.zeros(sys.n_x) if x0 is None else x0.copy()
    nu = np.zeros((N, sys.n_z))
    mu = np.zeros((N, sys.n_z))

    for k in range(N):
        x_pred = sys.F @ x_hat
        nu[k] = z[k] - sys.H @ x_pred
        x_hat = x_pred + W @ nu[k]
        mu[k] = z[k] - sys.H @ x_hat

    return nu, mu


# ---------------------------------------------------------------------------
# Sample autocovariances
# ---------------------------------------------------------------------------

def sample_autocovariances(nu: np.ndarray, M: int) -> np.ndarray:
    """Compute lag-0 through lag-(M-1) sample autocovariance matrices.

    Equation (50):
        Ĉ(i) = (1 / (N − M)) Σ_{j=0}^{N-M-1} ν(j) ν(j+i)'

    Parameters
    ----------
    nu  : (N, n_z) innovations
    M   : number of lags (must satisfy M < N)

    Returns
    -------
    C_hat : (M, n_z, n_z)
    """
    N, n_z = nu.shape
    C_hat = np.zeros((M, n_z, n_z))
    denom = N - M
    for i in range(M):
        # Vectorised outer-product sum: rows 0..N-M-1 of nu times rows i..N-M+i-1
        C_hat[i] = (nu[:denom].T @ nu[i:denom + i]) / denom
    return C_hat


# ---------------------------------------------------------------------------
# Trajectory simulator
# ---------------------------------------------------------------------------

def simulate_trajectory(sys: LinearSystem,
                        Q_true: np.ndarray,
                        R_true: np.ndarray,
                        N: int,
                        x0: Optional[np.ndarray] = None,
                        rng=None) -> np.ndarray:
    """Simulate N measurements from the true system.

    Returns z : (N, n_z).
    """
    rng = rng if rng is not None else np.random.default_rng()
    Q_true = np.atleast_2d(np.array(Q_true, dtype=float))
    R_true = np.atleast_2d(np.array(R_true, dtype=float))

    L_Q = cholesky(Q_true, lower=True)
    L_R = cholesky(R_true, lower=True)

    x = np.zeros(sys.n_x) if x0 is None else x0.copy()
    z = np.zeros((N, sys.n_z))

    for k in range(N):
        v = L_Q @ rng.standard_normal(sys.n_v)
        w = L_R @ rng.standard_normal(sys.n_z)
        x = sys.F @ x + sys.Gamma @ v
        z[k] = sys.H @ x + w

    return z
