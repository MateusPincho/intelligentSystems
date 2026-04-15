"""
six_step.py — Top-level driver for the six-step adaptive noise covariance ID.

Outer successive-approximation loop (Step 6):
  For each outer iteration r:
    1. Run Kalman filter with current W (Step 1)
    2. Compute sample autocovariances (Step 2)
    3. Update W via gradient descent (Step 3)
    4. Estimate R from post-fit residuals (Step 4)
    5. Estimate Q and P via inner Lyapunov iteration (Step 5)
  Reinitialise W from DARE with updated (Q, R) and repeat.
  Track the best (W, Q, R, P) by the achieved objective J.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .kalman import (LinearSystem, FilterEstimate,
                     initial_gain_from_dare, run_kalman_filter)
from .estimate_W import estimate_W
from .estimate_R import estimate_R
from .estimate_Q_P import estimate_Q_and_P


def six_step_algorithm(sys: LinearSystem,
                       z: np.ndarray,
                       Q0: np.ndarray,
                       R0: np.ndarray,
                       M: int = 40,
                       max_outer: int = 20,
                       mask_Q: Optional[np.ndarray] = None,
                       diagonal_R: bool = False,
                       lambda_Q: float = 0.0,
                       zeta_J: float = 1e-6,
                       burn_in: int = 50,
                       verbose: bool = False,
                       **inner_kwargs) -> FilterEstimate:
    """Run the full six-step adaptive Kalman filter identification algorithm.

    Parameters
    ----------
    z         : (N, n_z) measurement sequence
    Q0, R0    : initial guesses for noise covariances (used to seed DARE)
    M         : number of lags for autocovariance (>= n_x)
    max_outer : maximum number of outer successive-approximation iterations
    mask_Q    : structural mask for Q  (None = unconstrained)
    diagonal_R: if True, enforce diagonal structure on R
    lambda_Q  : regularisation for ill-conditioned Q estimation
    zeta_J    : outer convergence tolerance on J
    burn_in   : samples to discard before computing S and G
    **inner_kwargs : forwarded to estimate_W (n_L, patience, c, c_max, etc.)

    Returns FilterEstimate with the best (lowest J) values.
    """
    Q0 = np.atleast_2d(np.array(Q0, dtype=float))
    R0 = np.atleast_2d(np.array(R0, dtype=float))

    # Initialise W from DARE with the user-supplied guesses
    W = initial_gain_from_dare(sys, Q0, R0)

    best: Optional[FilterEstimate] = None
    J_best = np.inf
    J_outer_history: list = []

    for outer in range(max_outer):
        if verbose:
            print(f"[six_step] outer iteration {outer + 1}/{max_outer}")

        # Steps 1–3: refine W
        W, J_final, _ = estimate_W(sys, z, W, M, verbose=verbose, **inner_kwargs)

        # Step 1 rerun with converged W to get fresh (nu, mu)
        nu, mu = run_kalman_filter(sys, W, z)

        # Apply burn-in before estimating covariances
        nu_trim = nu[burn_in:]
        mu_trim = mu[burn_in:]

        if len(nu_trim) < M + 1:
            # Not enough data after burn-in; skip R/Q update
            break

        # Step 4: estimate R
        R, S, G = estimate_R(sys, W, nu_trim, mu_trim, diagonal=diagonal_R)

        # Step 5: estimate Q, P, P_bar
        Q, P, P_bar = estimate_Q_and_P(
            sys, W, S, R,
            mask=mask_Q,
            lambda_Q=lambda_Q
        )

        J_outer_history.append(J_final)

        if J_final < J_best:
            J_best = J_final
            best = FilterEstimate(
                W=W.copy(),
                S=S.copy(),
                R=R.copy(),
                Q=Q.copy(),
                P_bar=P_bar.copy(),
                P=P.copy(),
                J_history=list(J_outer_history),
            )

        if verbose:
            print(f"  J={J_final:.4e}  R={R.flatten()}  Q={Q.flatten()}")

        # Convergence across outer loops
        if outer > 0 and abs(J_outer_history[-1] - J_outer_history[-2]) < zeta_J:
            if verbose:
                print("  [six_step] outer converged.")
            break

        # Re-seed W for next outer iteration using the fresh (Q, R)
        try:
            W = initial_gain_from_dare(sys, Q, R)
        except Exception:
            # DARE failed (e.g., R nearly singular) — keep current W
            pass

    return best
