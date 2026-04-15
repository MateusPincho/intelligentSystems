"""
estimate_R.py — Step 4: estimate measurement noise covariance R.

Uses the R3 formula (eq. 78):
    G = R S^{-1} R
where G is the post-fit residual covariance and S is the innovation covariance.

This is solved via Cholesky decomposition + eigendecomposition (Appendix F),
which guarantees the estimate is positive definite.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv
from scipy.linalg import cholesky, eigh

from .kalman import LinearSystem


def estimate_R_from_G(G: np.ndarray,
                      S: np.ndarray,
                      diagonal: bool = False) -> np.ndarray:
    """Solve G = R S^{-1} R for R (positive definite) via Appendix F.

    Steps:
    1. Cholesky factor L of S^{-1}: S^{-1} = L L'
    2. Form M_sym = L' G L  (should be PSD)
    3. Eigendecompose M_sym = U Λ U',  clip Λ ≥ 0
    4. √M_sym = U √Λ U'
    5. R = L'^{-1} √M_sym L^{-1}
    """
    # Symmetrize inputs
    G = 0.5 * (G + G.T)
    S = 0.5 * (S + S.T)

    S_inv = inv(S)
    S_inv = 0.5 * (S_inv + S_inv.T)

    L = cholesky(S_inv, lower=True)          # S^{-1} = L L'

    M_sym = L.T @ G @ L
    M_sym = 0.5 * (M_sym + M_sym.T)

    eig_vals, U = eigh(M_sym)
    eig_vals = np.clip(eig_vals, 0.0, None)  # numerical safety

    sqrt_M = U @ np.diag(np.sqrt(eig_vals)) @ U.T

    # R = L'^{-1} √M L^{-1}
    L_T_inv = inv(L.T)
    L_inv = inv(L)
    R_est = L_T_inv @ sqrt_M @ L_inv
    R_est = 0.5 * (R_est + R_est.T)

    if diagonal:
        R_est = np.diag(np.diag(R_est))

    return R_est


def estimate_R(sys: LinearSystem,
               W: np.ndarray,
               nu: np.ndarray,
               mu: np.ndarray,
               diagonal: bool = False):
    """Estimate R from innovation and post-fit residual sequences.

    Parameters
    ----------
    nu : (N, n_z) innovations  (after burn-in has been applied by caller)
    mu : (N, n_z) post-fit residuals

    Returns (R, S, G).
    """
    N = len(nu)
    S = (nu.T @ nu) / N
    G = (mu.T @ mu) / N
    S = 0.5 * (S + S.T)
    G = 0.5 * (G + G.T)

    R = estimate_R_from_G(G, S, diagonal=diagonal)
    return R, S, G
