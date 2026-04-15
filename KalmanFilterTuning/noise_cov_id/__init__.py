"""
noise_cov_id — Adaptive Kalman filter noise covariance identification.

Implements the six-step algorithm from:
    Zhang et al. (2020). On the Identification of Noise Covariances and
    Adaptive Kalman Filtering: A New Look at a 50 Year-Old Problem.
    IEEE Access, Vol. 8, pp. 59362–59388.
"""

from .kalman import (
    LinearSystem,
    FilterEstimate,
    closed_loop_F_bar,
    initial_gain_from_dare,
    run_kalman_filter,
    sample_autocovariances,
    simulate_trajectory,
)
from .identifiability import (
    minimal_polynomial_coeffs,
    build_identifiability_matrix,
)
from .estimate_W import (
    compute_J,
    compute_X,
    solve_Z,
    compute_gradient,
    finite_diff_grad,
    estimate_W,
)
from .estimate_R import (
    estimate_R_from_G,
    estimate_R,
)
from .estimate_Q_P import estimate_Q_and_P
from .six_step import six_step_algorithm
from .metrics import summarize, averaged_NIS

__all__ = [
    # kalman
    "LinearSystem", "FilterEstimate",
    "closed_loop_F_bar", "initial_gain_from_dare",
    "run_kalman_filter", "sample_autocovariances", "simulate_trajectory",
    # identifiability
    "minimal_polynomial_coeffs", "build_identifiability_matrix",
    # estimate_W
    "compute_J", "compute_X", "solve_Z", "compute_gradient",
    "finite_diff_grad", "estimate_W",
    # estimate_R
    "estimate_R_from_G", "estimate_R",
    # estimate_Q_P
    "estimate_Q_and_P",
    # six_step
    "six_step_algorithm",
    # metrics
    "summarize", "averaged_NIS",
]
