"""Azimuthal harmonics of a quantity around the LMC centre: v(PA) = c0 + sum_m [a_m cos(m PA) + b_m sin(m PA)].

PA is the position angle about the centre, N through E, in degrees. The fit is unweighted least squares truncated at
harmonic mmax (default 2); errors assume independent residuals with a common variance, estimated from the residual scatter.
Used for position-angle profiles of the proper motions
and the centre fits (scripts/fit_centre.py).
"""
import numpy as np


def design(pa_deg, mmax=2):
    """Design matrix of the harmonic model.

    Parameters
    ----------
    pa_deg : array_like, shape (N,)
        Position angles [deg].
    mmax : int
        Highest harmonic.

    Returns
    -------
    ndarray, shape (N, 1 + 2 mmax)
        Columns 1, cos(PA), sin(PA), cos(2 PA), sin(2 PA), ...
    """
    pa = np.deg2rad(pa_deg)
    return np.column_stack([np.ones_like(pa)] + [f(m * pa) for m in range(1, mmax + 1) for f in (np.cos, np.sin)])


def fit(pa_deg, v, mmax=2):
    """Least-squares harmonic fit. Returns (coefficients [c0, a1, b1, a2, b2, ...], their errors from the residual scatter).

    Parameters
    ----------
    pa_deg : array_like, shape (N,)
        Position angles [deg].
    v : array_like, shape (N,)
        Values (unweighted).
    mmax : int
        Highest harmonic.

    Returns
    -------
    c, err : ndarray, shape (1 + 2 mmax,)
        Coefficients in the units of v and their standard errors, (X^T X)^-1 times the residual variance with
        N - (1 + 2 mmax) degrees of freedom.
    """
    X = design(pa_deg, mmax)
    c = np.linalg.lstsq(X, v, rcond=None)[0]
    cov = np.linalg.inv(X.T @ X) * np.var(v - X @ c, ddof=X.shape[1])
    return c, np.sqrt(np.diag(cov))


def amplitude_phase(c, m):
    """Amplitude and the position angle [deg] of the maximum of the m-th harmonic, in [0, 360/m).

    a cos(m PA) + b sin(m PA) = A cos(m (PA - PA_max)) with A = hypot(a, b) and m PA_max = atan2(b, a).

    Parameters
    ----------
    c : array_like
        Coefficients from fit, [c0, a1, b1, ...].
    m : int
        Harmonic order, 1 .. mmax.

    Returns
    -------
    A : float
        Amplitude, in the units of v.
    pa_max : float
        Position angle of the first maximum [deg].
    """
    a, b = c[2 * m - 1], c[2 * m]
    return np.hypot(a, b), (np.rad2deg(np.arctan2(b, a)) / m) % (360 / m)
