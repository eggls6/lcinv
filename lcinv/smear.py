"""Exposure-time smearing: averaging the model flux over each exposure.

A detector integrates flux, so for an exposure of length T centred on t_i

    m_i = -2.5 log10[ (1/T) * integral_{t_i - T/2}^{t_i + T/2} L(t) dt ] + offsets.

During an exposure only the rotation angle changes appreciably (the Sun/observer directions
move by ~1e-5 of their daily arc in 30 s), sweeping Delta phi = 2 pi T / P. A rotational
harmonic k of the lightcurve is attenuated by sinc(k Delta phi / 2) = sin(x) / x, x = k Delta phi / 2.

The integral is evaluated by Gauss-Legendre quadrature in time with Q nodes:
    t_iq = t_i + (T_i / 2) x_q,   L_bar_i = sum_q (w_q / 2) L(t_iq),
with the Sun/observer unit vectors held at their mid-exposure values. For a periodic signal
whose highest relevant harmonic is k_max, the Gauss-Legendre error of the exposure average is
    E_Q = x^(2Q) 2^(2Q) (Q!)^4 / [(2Q + 1) ((2Q)!)^3],   x = k_max Delta phi / 2
(the standard remainder with |f^(2Q)| <= x^(2Q); Q = 1 gives the midpoint error x^2 / 6),
which `choose_nodes` keeps below a tolerance (relative flux, i.e. ~1.09 tol mag). Rows with
T = 0 put all nodes at t_i, so the instantaneous model is recovered exactly; with Q = 1 the
computation is identical to the unsmeared one.
"""
import math

import numpy as np

SEC_PER_DAY = 86400.0


def gauss_legendre(Q):
    """Gauss-Legendre nodes on [-1, 1] and weights normalised to sum 1.

    Parameters
    ----------
    Q : int
        Number of nodes.

    Returns
    -------
    x, w : ndarray
    """
    x, w = np.polynomial.legendre.leggauss(int(Q))
    return x, w / 2.0


def phase_sweep(exptime_s, P_h):
    """Rotation angle [rad] swept during an exposure of `exptime_s` [s] at period `P_h` [h]."""
    return 2.0 * np.pi * np.asarray(exptime_s, float) / (P_h * 3600.0)


def gl_error(x, Q):
    """Error bound of the Q-node Gauss-Legendre average of ``cos(x s)`` over ``s`` in [-1, 1].

    Parameters
    ----------
    x : float or ndarray
        Half the phase swept by the highest harmonic, ``k_max * dphi / 2``.
    Q : int
        Number of nodes.

    Returns
    -------
    float or ndarray
        Relative flux error.
    """
    f2q = math.factorial(2 * Q)
    return x ** (2 * Q) * 2.0 ** (2 * Q) * math.factorial(Q) ** 4 / ((2 * Q + 1) * f2q ** 3)


def choose_nodes(exptime_s, P_h, k_max, tol=1e-4, q_max=8):
    """Number of Gauss-Legendre nodes per exposure.

    The smallest Q whose error bound `gl_error` (``k_max dphi / 2``, Q) is below `tol` for the
    longest exposure; this is the automatic ``--smear-nodes``.

    Parameters
    ----------
    exptime_s : float or ndarray
        Exposure times [s].
    P_h : float
        Rotation period [h].
    k_max : int
        Highest rotational harmonic to resolve (`k_max_for_lmax`).
    tol : float, optional
        Tolerance in relative flux (about 1.09 tol mag).
    q_max : int, optional
        Upper limit.

    Returns
    -------
    int
        1 (the midpoint, i.e. no smearing) when no exposure time is given.
    """
    T = np.nanmax(np.asarray(exptime_s, float)) if np.size(exptime_s) else 0.0
    if not np.isfinite(T) or T <= 0:
        return 1
    xk = k_max * phase_sweep(T, P_h) / 2.0
    for Q in range(1, q_max + 1):
        if gl_error(xk, Q) < tol:
            return Q
    return q_max


def node_times(t, exptime_s, Q):
    """Times of the quadrature nodes of every exposure.

    Parameters
    ----------
    t : ndarray
        Mid-exposure times [d].
    exptime_s : float or ndarray
        Exposure times [s].
    Q : int
        Nodes per exposure.

    Returns
    -------
    times : ndarray
        Shape (N * Q,), ordered row-major (row i, then node q).
    weights : ndarray
        Shape (Q,), summing to 1.
    """
    x, w = gauss_legendre(Q)
    half = 0.5 * np.nan_to_num(np.asarray(exptime_s, float)) / SEC_PER_DAY
    tq = np.asarray(t, float)[:, None] + half[:, None] * x[None, :]
    return tq.reshape(-1), w


def sinc_attenuation(k, exptime_s, P_h):
    """Exact attenuation of rotational harmonic `k` by averaging over each exposure.

    Returns ``sinc(k dphi / 2)`` per row, with ``dphi`` from `phase_sweep`; used by the period
    search (`lcinv.period.fit_fourier`).
    """
    x = k * phase_sweep(exptime_s, P_h) / 2.0
    return np.sinc(x / np.pi)   # numpy sinc(u) = sin(pi u) / (pi u)


def k_max_for_lmax(lmax):
    """Highest rotational harmonic to resolve for a shape of degree `lmax`, with a margin for the
    nonlinearity of terminator and limb.
    """
    return 2 * int(lmax) + 2
