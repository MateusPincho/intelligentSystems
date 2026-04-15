"""
estimate_Q_P.py — Step 5: iterative estimation of Q and P.

Double-loop fixed-point iteration:
  Outer loop: update Q from P via eqs. (124)–(127)
  Inner loop: update P given Q via eq. (123)

Initialization:
  Q^{(0)} from Wiener-process solution (eq. 165)
  P^{(0)} from discrete Lyapunov eq. (122)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv, norm, pinv
from scipy.linalg import solve_discrete_lyapunov
from typing import Optional

from .kalman import LinearSystem, closed_loop_F_bar


def estimate_Q_and_P(sys: LinearSystem,
                     W: np.ndarray,
                     S: np.ndarray,
                     R: np.ndarray,
                     mask: Optional[np.ndarray] = None,
                     lambda_Q: float = 0.0,
                     tol: float = 1e-8,
                     max_outer: int = 50,
                     max_inner: int = 200):
    """Estimate Q and P via a double fixed-point iteration.

    Parameters
    ----------
    W        : (n_x, n_z) converged Kalman gain
    S        : (n_z, n_z) innovation covariance estimate
    R        : (n_z, n_z) measurement noise covariance estimate
    mask     : (n_v, n_v) structural mask for Q (e.g. np.eye for diagonal Q)
               Use None or np.ones for unconstrained Q.
    lambda_Q : regularisation parameter (0 for Cases 1–3)
    tol      : convergence tolerance for both loops

    Returns (Q, P, P_bar).
    """
    F, H, Gam = sys.F, sys.H, sys.Gamma
    n_x, n_v, n_z = sys.n_x, sys.n_v, sys.n_z

    G_pinv = pinv(Gam)
    R_inv = inv(R)
    H_R_H = H.T @ R_inv @ H   # (n_x, n_x), used in inner P loop

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    # Q^{(0)}: Wiener-process solution (eq. 165): Γ Q Γ' = W S W'
    Q = G_pinv @ (W @ S @ W.T) @ G_pinv.T
    Q = 0.5 * (Q + Q.T)
    if mask is not None:
        Q = mask * Q

    # P^{(0)}: solve discrete Lyapunov for the updated covariance (eq. 122)
    # P − F̃ P F̃' = W R W' + (I − W H) Γ Q Γ' (I − W H)'
    # where F̃ = (I − W H) F   (NOTE: different from F̄ = F (I − W H))
    IWH = np.eye(n_x) - W @ H
    F_tilde = IWH @ F
    rhs0 = W @ R @ W.T + IWH @ Gam @ Q @ Gam.T @ IWH.T
    rhs0 = 0.5 * (rhs0 + rhs0.T)
    P = solve_discrete_lyapunov(F_tilde, rhs0)
    P = 0.5 * (P + P.T)

    # ------------------------------------------------------------------
    # Double loop
    # ------------------------------------------------------------------
    for _ in range(max_outer):
        Q_prev = Q.copy()

        # Inner loop: iterate P (eq. 123)
        for _ in range(max_inner):
            P_pred = F @ P @ F.T + Gam @ Q @ Gam.T
            P_pred = 0.5 * (P_pred + P_pred.T)
            try:
                P_pred_inv = inv(P_pred)
            except np.linalg.LinAlgError:
                break
            P_new = inv(P_pred_inv + H_R_H)
            P_new = 0.5 * (P_new + P_new.T)
            if norm(P_new - P) < tol:
                P = P_new
                break
            P = P_new

        # Outer Q update (eqs. 124–127)
        D = P + W @ S @ W.T - F @ P @ F.T
        D = 0.5 * (D + D.T)
        Q_new = G_pinv @ (D + lambda_Q * np.eye(n_x)) @ G_pinv.T
        Q_new = 0.5 * (Q_new + Q_new.T)
        if mask is not None:
            Q_new = mask * Q_new

        # Project onto PSD cone (clip negative eigenvalues)
        eig_vals, eig_vecs = np.linalg.eigh(Q_new)
        eig_vals = np.clip(eig_vals, 0.0, None)
        Q_new = eig_vecs @ np.diag(eig_vals) @ eig_vecs.T
        Q_new = 0.5 * (Q_new + Q_new.T)

        if norm(Q_new - Q_prev) < tol:
            Q = Q_new
            break
        Q = Q_new

    # Final prediction covariance (eq. 111)
    P_bar = F @ P @ F.T + Gam @ Q @ Gam.T
    P_bar = 0.5 * (P_bar + P_bar.T)

    return Q, P, P_bar
