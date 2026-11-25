"""Pixel maps on the sky: a per-pixel statistic of a quantity, blank where a pixel has too few stars.

Model: none; these are descriptive statistics. pixel_map bins stars into square pixels of equal size in x and y (any
projected sky coordinates, e.g. degrees on a tangent plane) and returns a statistic (median by default) of a per-star quantity,
NaN where a pixel holds fewer than nmin finite values. robust_var assumes the core of the distribution is Gaussian, so that
1.4826 MAD equals sigma. wquantile spreads each star's weight over its rank.

Used by the map and profile figures (scripts/paper_maps.py, paper_pm_maps.py,
paper_feh_profile.py, paper_mdf_literature.py).
"""
import numpy as np
import pandas as pd


def robust_var(v):
    """Variance from the median absolute deviation (Gaussian-equivalent).

    Parameters
    ----------
    v : array_like
        Values (NaN not removed).

    Returns
    -------
    float
        (1.4826 MAD)^2, in units of v squared; 1.4826 MAD equals sigma for a Gaussian.
    """
    v = np.asarray(v)
    return (1.4826 * np.median(np.abs(v - np.median(v)))) ** 2


def wquantile(x, w, q):
    """Weighted quantiles q of x (each star's weight spread over its rank, so equal weights give the usual median).

    Parameters
    ----------
    x : ndarray, shape (N,)
        Values.
    w : ndarray, shape (N,)
        Non-negative weights.
    q : float or array_like
        Quantiles in [0, 1].

    Returns
    -------
    float or ndarray
        Linear interpolation of sorted x at cumulative weight q x sum(w), each star sitting at the middle of its weight.
    """
    i = np.lexsort((w, x))   # by x, ties by w: the same order whatever the order of the rows (identical (x, w) are interchangeable)
    c = np.cumsum(w[i])
    return np.interp(np.asarray(q) * c[-1], c - .5 * w[i], x[i])


def pixel_map(x, y, v, edges, nmin=5, stat=np.median):
    """stat(v) in square pixels with the given edges (same for x and y); NaN where fewer than nmin finite values.
    Returns an array indexed [ix, iy].

    Parameters
    ----------
    x, y : array_like, shape (N,)
        Pixel coordinates of the stars (e.g. lmcold.footprint.lmc_xy, deg).
    v : array_like, shape (N,)
        Quantity; NaN values are dropped.
    edges : array_like
        Pixel edges, used for both axes; stars outside are dropped.
    nmin : int
        Smallest number of finite values in a pixel.
    stat : callable or str
        Statistic accepted by pandas GroupBy.agg (np.median by default).

    Returns
    -------
    ndarray, shape (len(edges) - 1, len(edges) - 1)
        First index x, second y (transpose for imshow with origin='lower').
    """
    v = np.asarray(v)
    ok = np.isfinite(v)
    ix, iy = np.digitize(np.asarray(x)[ok], edges) - 1, np.digitize(np.asarray(y)[ok], edges) - 1
    n = len(edges) - 1
    inside = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    g = pd.DataFrame(dict(ix=ix[inside], iy=iy[inside], v=v[ok][inside])).groupby(["ix", "iy"]).v.agg([stat, "size"])
    g = g[g["size"] >= nmin]
    m = np.full((n, n), np.nan)
    m[g.index.get_level_values(0), g.index.get_level_values(1)] = g.iloc[:, 0]
    return m
