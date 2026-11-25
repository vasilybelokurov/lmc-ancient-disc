"""Weights of a spectroscopic sample drawn from a parent catalogue (BOSS), with a pass rate that depends on metallicity.

Selection group (index s per star): stars the survey treated alike (for BOSS: sky patch x G bin x carton class), with N_elig
parent stars, N_obs observed and N_good with a good spectrum. Within a group the targeting is taken to ignore metallicity, so
N_elig / N_obs weights the observed stars to the parent; the group weight of a good spectrum, N_elig / N_good, also assumes that
failed spectra ignore metallicity. They do not: weak-lined metal-poor spectra fail more often at faint G. The pass factor puts
the failures back. Pass groups (pass_groups) are G bin x XP [M/H] bin, with stars without XP [M/H] in a bin of their own; for BOSS
also x ring (boss_pass_groups), since spectra fail more often in the crowded centre; in
each pass group
  f = sum_obs N_elig / N_obs  /  sum_good N_elig / N_good,
and the final weight of a good spectrum is N_elig / N_good * f. Within a pass group the weighted good spectra then add up to the
weighted observed stars, so the pass step is removed whatever its cause, and the targeting step is left to be tested against
the parent. Selection groups with fewer than NMIN_GOOD good spectra are uncovered (weight 0). XP [M/H] below -1.75 is one bin:
narrower metal-poor bins left too few good spectra per pass group at faint G (factors 2.1 +- 0.9, 2.5 +- 0.5). Every count takes per-star
multiplicities m (a 0/1 mask for a held-out subset, bootstrap counts), so no normalisation leaks across subsets.

Used by scripts/boss_selection.py and scripts/apogee_selection.py (the weights written to the tier-B weight catalogues,
data/sources.yaml keys release_boss_weights and release_apogee_weights; APOGEE through lmcold.apogee_target.weights), the mock
(scripts/boss_mock.py) and the mass (scripts/build_giant_mass.py, lmcold.mass_samples). Shown in figures/paper_selection.png.
Arrays are per star of the parent catalogue: s (selection-group index, 0 .. n - 1), obs and good (bool), pg (pass-group
index), m (multiplicity).
"""
import numpy as np
import pandas as pd

from lmcold.constants import LMC
from lmcold.footprint import angular_distance
from lmcold.groups import bin_index

PG_EDGES = [12.0, 16.0, 16.75, 17.1, 17.5]  # G [mag] edges of the BOSS pass groups (carton limit G <= 17.5)
XP_EDGES = [-4, -1.75, -1.5, -1.25, -1, -0.5, 1]  # XP [M/H] [dex] edges (Andrae et al. 2023); see module docstring for -1.75
NMIN_GOOD = 3  # good spectra a selection group needs to be covered
RING_EDGES = [3.0, 6.0]  # deg from the LMC centre: BOSS pass groups


def pass_groups(mag, mh, edges=PG_EDGES):
    """Pass-group index per star: magnitude bin (G for BOSS by default) x XP [M/H] bin, with one extra XP bin for stars
    without XP [M/H].

    Parameters
    ----------
    mag : array_like, shape (N,)
        Magnitude [mag] (G for BOSS, H for APOGEE with its own edges).
    mh : array_like, shape (N,)
        XP [M/H] [dex]; NaN where missing.
    edges : array_like
        Magnitude bin edges; values beyond go to the end bins.

    Returns
    -------
    ndarray of int, shape (N,)
        magnitude bin x len(XP_EDGES) + XP bin (XP bin len(XP_EDGES) - 1 for missing [M/H]).
    """
    nx = len(XP_EDGES)  # XP bins 0 .. nx - 2, missing = nx - 1
    return bin_index(mag, edges) * nx + np.where(np.isfinite(mh), bin_index(mh, XP_EDGES), nx - 1)


def boss_pass_groups(g, mh, ra, dec):
    """BOSS pass-group index per star: pass_groups(G, XP [M/H]) x ring (RING_EDGES).

    Parameters
    ----------
    g : array_like
        Gaia G [mag].
    mh : array_like
        XP [M/H] [dex] (NaN where missing).
    ra, dec : array_like
        Positions [deg]; the ring is set by the angular distance from lmcold.constants.LMC: < 3, 3-6, >= 6 deg.

    Returns
    -------
    ndarray of int
    """
    ring = np.digitize(angular_distance(np.asarray(ra, float), np.asarray(dec, float), LMC["ra0"], LMC["dec0"]), RING_EDGES)
    return pass_groups(np.asarray(g, float), np.asarray(mh, float)) * (len(RING_EDGES) + 1) + ring


def _per_star(s, x, m):
    """Sum of m x over each selection group, returned per star (the sum of its own group)."""
    return np.bincount(s, m * x, s.max() + 1)[s]


def group_weights(s, good, m):
    """N_elig / N_good for good spectra in covered selection groups, 0 elsewhere.

    Parameters
    ----------
    s : ndarray of int, shape (N,)
        Selection-group index per parent star.
    good : ndarray of bool, shape (N,)
        Good spectrum.
    m : ndarray, shape (N,)
        Multiplicity of each star in the counts.

    Returns
    -------
    ndarray, shape (N,)
        Weight; covered means N_good >= NMIN_GOOD.
    """
    n_g = _per_star(s, good, m)
    return np.where(good & (n_g >= NMIN_GOOD), _per_star(s, np.ones(len(s)), m) / np.where(n_g > 0, n_g, 1), 0.0)


def pass_factor(s, obs, good, pg, m):
    """Pass factor per star, from its pass group pg (covered selection groups only; 1 where a pass group has no good spectra).

    Parameters
    ----------
    s : ndarray of int, shape (N,)
        Selection-group index per parent star.
    obs, good : ndarray of bool, shape (N,)
        Observed; good spectrum (good implies observed).
    pg : ndarray of int, shape (N,)
        Pass-group index (pass_groups or boss_pass_groups).
    m : ndarray, shape (N,)
        Multiplicities.

    Returns
    -------
    ndarray, shape (N,)
        f of the module docstring for the pass group of each star.
    """
    # Observed stars weighted to the parent (N_elig / N_obs) and good spectra (N_elig / N_good), both in covered groups only.
    n_e, n_o = _per_star(s, np.ones(len(s)), m), _per_star(s, obs, m)
    w_good = group_weights(s, good, m)
    w_obs = np.where(obs & (_per_star(s, good, m) >= NMIN_GOOD), n_e / np.where(n_o > 0, n_o, 1), 0.0)
    n = pg.max() + 1
    o, gd = np.bincount(pg, m * w_obs, n), np.bincount(pg, m * w_good, n)  # weighted observed and good per pass group
    return np.where(gd > 0, o / np.where(gd > 0, gd, 1), 1.0)[pg]


def coverage_factor(keys, covered, nmin, m):
    """Per star: parent / covered parent in its fill cell, which puts back the parent stars of uncovered selection groups.

    keys: per-star cell keys, finest first (e.g. region x G x XP bin, ..., G). A star takes the finest level whose cell holds
    >= nmin covered stars among the stars not yet assigned; the last level takes the rest. The factor is used on the weights of
    covered good spectra, so the gap stars of a cell are assumed to share its covered stars' metallicities and member fraction.
    m: per-star multiplicities (bootstrap counts; ones for the data).

    Parameters
    ----------
    keys : list of array_like, each of shape (N,)
        Cell keys per level, finest first.
    covered : ndarray of bool, shape (N,)
        Parent star in a covered selection group.
    nmin : int
        Covered stars (counted with multiplicity) a cell needs below the last level.
    m : ndarray, shape (N,)
        Multiplicities.

    Returns
    -------
    ndarray, shape (N,)
        Parent / covered parent in the star's fill cell (1 where the cell is fully covered).

    Raises
    ------
    AssertionError
        If a cell of the last level contains stars with m > 0 but no covered star.
    """
    f = np.ones(len(covered))
    todo = np.ones(len(covered), bool)
    for i, key in enumerate(keys):
        k = pd.factorize(pd.Series(np.asarray(key)[todo]).astype(str))[0]
        n_cov = np.bincount(k, (m * covered)[todo])
        take = (n_cov[k] >= nmin) | (i == len(keys) - 1)
        kt = k[take]
        idx = np.flatnonzero(todo)[take]
        n_all, n_c = np.bincount(kt, m[idx], len(n_cov)), np.bincount(kt, (m * covered)[idx], len(n_cov))
        assert np.all(n_c[kt][m[idx] > 0] > 0), 'a fill cell has no covered stars'
        f[idx] = n_all[kt] / np.where(n_c[kt] > 0, n_c[kt], 1)
        todo[idx] = False
    return f
