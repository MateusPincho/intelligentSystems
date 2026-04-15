"""
identifiability.py — Identifiability pre-check for Q and R.

Implements Algorithm 1 from Zhang et al. (2020):
- minimal_polynomial_coeffs: minimal polynomial of the closed-loop matrix F̄
- build_identifiability_matrix: constructs the identifiability matrix I and
  checks the rank condition rank(I) >= n_Q + n_R
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import eigvals, matrix_rank, matrix_power

from .kalman import LinearSystem, closed_loop_F_bar


def minimal_polynomial_coeffs(F_bar: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Return monic minimal-polynomial coefficients of F_bar.

    For systems with distinct eigenvalues the minimal polynomial equals the
    characteristic polynomial.  We cluster nearly-equal eigenvalues (within
    absolute tolerance *tol*) so that repeated roots are not counted twice.

    Returns coefficients [1, a_1, …, a_m] (length m+1).
    """
    eigs = eigvals(F_bar)
    unique: list = []
    for e in eigs:
        if not any(abs(e - u) < tol for u in unique):
            unique.append(e)
    coeffs = np.poly(unique).real  # monic, length = len(unique)+1
    return coeffs


def build_identifiability_matrix(sys: LinearSystem,
                                  W: np.ndarray):
    """Build the noise-covariance identifiability matrix (Algorithm 1).

    Returns
    -------
    I_mat      : ((m+1)*n_z^2, n_Q + n_R) identifiability matrix
    rank_I     : numerical rank of I_mat
    n_unknowns : n_Q + n_R (total unknowns)
    """
    F_bar = closed_loop_F_bar(sys, W)
    a = minimal_polynomial_coeffs(F_bar)  # [1, a_1, ..., a_m]
    m = len(a) - 1                        # degree of minimal polynomial
    n_v, n_z, n_x = sys.n_v, sys.n_z, sys.n_x

    # -----------------------------------------------------------------------
    # Build B_l and G_l matrices for l = 0, …, m
    # From eqs. (28)–(30):
    #   B_l = H  [Σ_{i=0}^{l-1} a_i F̄^{l-i-1}]  Γ   (B_0 = 0)
    #   G_l = a_l I_{n_z} − H [Σ_{i=0}^{l-1} a_i F̄^{l-i-1}] F W
    # -----------------------------------------------------------------------
    B = [np.zeros((n_z, n_v)) for _ in range(m + 1)]
    G_mat = [np.zeros((n_z, n_z)) for _ in range(m + 1)]
    G_mat[0] = np.eye(n_z)  # G_0 = a_0 * I = I (monic polynomial, a_0=1)

    for l in range(1, m + 1):
        # accum = Σ_{i=0}^{l-1} a_i * F̄^{l-i-1}
        accum = sum(a[i] * matrix_power(F_bar, l - i - 1) for i in range(l))
        B[l] = sys.H @ accum @ sys.Gamma
        G_mat[l] = a[l] * np.eye(n_z) - sys.H @ accum @ sys.F @ W

    # -----------------------------------------------------------------------
    # Number of unknowns
    # n_Q = number of independent entries in symmetric n_v×n_v matrix
    # n_R = number of independent entries in symmetric n_z×n_z matrix
    # -----------------------------------------------------------------------
    n_Q = n_v * (n_v + 1) // 2
    n_R = n_z * (n_z + 1) // 2
    n_unknowns = n_Q + n_R

    n_rows = (m + 1) * n_z * n_z
    I_mat = np.zeros((n_rows, n_unknowns))

    for j in range(m + 1):
        r = j * n_z * n_z
        k = 0

        # Columns for Q (upper-triangular independent entries)
        for p in range(n_v):
            # Diagonal entry q_{pp}
            indices_Q = list(range(j + 1, m + 1))
            b = (sum(np.outer(B[i][:, p], B[i - j][:, p]) for i in indices_Q)
                 if indices_Q else np.zeros((n_z, n_z)))
            I_mat[r:r + n_z * n_z, k] = b.flatten()
            k += 1
            # Off-diagonal entries q_{pq} for q > p
            for q in range(p + 1, n_v):
                d = (sum(np.outer(B[i][:, p], B[i - j][:, q])
                         + np.outer(B[i][:, q], B[i - j][:, p])
                         for i in indices_Q)
                     if indices_Q else np.zeros((n_z, n_z)))
                I_mat[r:r + n_z * n_z, k] = d.flatten()
                k += 1

        # Columns for R (upper-triangular independent entries)
        for p in range(n_z):
            # Diagonal entry r_{pp}
            indices_R = list(range(j, m + 1))
            g = (sum(np.outer(G_mat[i][:, p], G_mat[i - j][:, p]) for i in indices_R)
                 if indices_R else np.zeros((n_z, n_z)))
            I_mat[r:r + n_z * n_z, k] = g.flatten()
            k += 1
            # Off-diagonal entries r_{pq} for q > p
            for q in range(p + 1, n_z):
                f = (sum(np.outer(G_mat[i][:, p], G_mat[i - j][:, q])
                         + np.outer(G_mat[i][:, q], G_mat[i - j][:, p])
                         for i in indices_R)
                     if indices_R else np.zeros((n_z, n_z)))
                I_mat[r:r + n_z * n_z, k] = f.flatten()
                k += 1

    rank_I = matrix_rank(I_mat)
    return I_mat, rank_I, n_unknowns
