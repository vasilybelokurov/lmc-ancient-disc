"""Selection groups pooled from fine to coarse.

Stars the survey treated alike are grouped by keys (e.g. sky patch x magnitude bin x carton class); groups with too few stars
are merged into coarser ones. Used by scripts/boss_selection.py, boss_mock.py, build_giant_mass.py and lmcold.selection.
"""
import numpy as np
import pandas as pd


def assign(keys, nmin):
    """Selection-group label per star.

    keys: list of per-star key arrays, finest first; the last level takes every star still unassigned. A star takes the finest
    level whose group holds >= nmin of the stars not assigned at a finer level, so every star belongs to exactly one group.

    Parameters
    ----------
    keys : list of array_like, each of shape (N,)
        Group keys per level (any type; compared as strings).
    nmin : int
        Smallest group size accepted at levels other than the last.

    Returns
    -------
    ndarray of object, shape (N,)
        Label "<level>:<key>", the level index and the key as a string.
    """
    label = np.full(len(keys[0]), "", dtype=object)
    todo = np.ones(len(keys[0]), bool)
    for i, key in enumerate(keys):
        k = pd.Series(np.asarray(key)[todo]).astype(str)  # keys of the stars still unassigned
        # Group size among those stars, per star; the last level takes everything left.
        take = (k.map(k.value_counts()) >= nmin).to_numpy() | (i == len(keys) - 1)
        idx = np.flatnonzero(todo)[take]
        label[idx] = f"{i}:" + k[take]
        todo[idx] = False
    return label


def bin_index(x, edges):
    """Bin of x in edges, 0 .. len(edges) - 2; values on or beyond the outer edges go to the end bins.

    Parameters
    ----------
    x : array_like
        Values (NaN goes to the last bin).
    edges : array_like
        Increasing bin edges; bins are [edges[i], edges[i + 1]).

    Returns
    -------
    ndarray of int
    """
    return np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
