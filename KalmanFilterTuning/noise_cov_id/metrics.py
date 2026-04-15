"""
metrics.py — Reporting utilities for Monte Carlo experiments.

- summarize : mean, 95% probability interval, RMSE for a scalar quantity
- averaged_NIS : averaged Normalized Innovation Squared across MC runs
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv
from typing import Sequence

from .kalman import LinearSystem, run_kalman_filter
from .kalman import FilterEstimate


def summarize(values: Sequence[float],
              true_value: float,
              name: str = "param") -> dict:
    """Compute summary statistics for a scalar estimated quantity.

    Parameters
    ----------
    values     : list of scalar estimates from MC runs
    true_value : ground-truth value
    name       : label for the quantity

    Returns a dict with keys: name, truth, lower, mean, upper, rmse.
    """
    arr = np.array(values, dtype=float)
    lower, upper = np.percentile(arr, [2.5, 97.5])
    mean = float(arr.mean())
    rmse = float(np.sqrt(((arr - true_value) ** 2).mean()))
    return {
        "name": name,
        "truth": true_value,
        "lower": lower,
        "mean": mean,
        "upper": upper,
        "rmse": rmse,
    }


def averaged_NIS(sys: LinearSystem,
                 results: Sequence[FilterEstimate],
                 z_list: Sequence[np.ndarray]) -> np.ndarray:
    """Compute the averaged NIS across MC runs (eq. 191).

    For each run i:
        ε_i(k) = ν_i(k)' S_i^{-1} ν_i(k)

    Returns ε̄(k) = (1/n_runs) Σ_i ε_i(k), shape (N,).

    The paper plots ε̄(k)/n_z, so divide by sys.n_z before plotting.
    """
    per_run = []
    for est, z in zip(results, z_list):
        S_inv = inv(est.S)
        nu, _ = run_kalman_filter(sys, est.W, z)
        # ε(k) = ν(k)' S^{-1} ν(k)  (scalar per time step)
        nis = np.einsum('ki,ij,kj->k', nu, S_inv, nu)
        per_run.append(nis)
    return np.array(per_run).mean(axis=0)
