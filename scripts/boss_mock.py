"""Mock of the BOSS selection on the real design, with known true [Fe/H], to test the weights of lmcold.selection.

Real: eligible parent, selection groups, observed flags, G, which stars have XP [M/H] (data/boss_weights.parquet).
Mock:
  true [Fe/H]: per eligible star, the BOSSNet [Fe/H] of a good spectrum from the same nside-16 pixel drawn with probability
    proportional to its weight, de-compressed about FEH_PIV by 1 / COMP, plus N(0, 0.05);
  pass: P = p_s T, T = 1 - A(G) clip((-1 - [Fe/H]) / 1.5, 0, 1)^1.5, A(G) = 0.7 clip((G - 16.25) / 1.25, 0, 1) (shaped like the
    measured deficit, not like the estimator); p_s keeps the real pass rate of the selection group (capped so P <= 1);
  XP [M/H]: true [Fe/H] + N(0, XP_SD) where the real star has XP [M/H] (XP_SD = robust scatter of Andrae et al. 2023 about
    APOGEE DR17 for the LMC giants);
  variant obs_bias: observation itself favours [Fe/H] < -1.5 by a factor 1.3 within each selection group (keeping its N_obs),
    which violates the assumption of the weights.
For NREAL realisations per variant: weighted count of good spectra / true count of covered eligible stars, in bins of true
[Fe/H] (FEH_EDGES) and below -1.5 and -2, for the group weights alone and the final weights (group weights x pass factor,
lmcold.selection, as scripts/boss_selection.py).
Fill test (NFILL realisations per failure model; failure 'central' doubles A(G) within 3 deg, as the real spectra fail more in the
crowded centre): final weights x coverage factor (lmcold.selection.coverage_factor) over the true count of ALL eligible stars,
below -1.5 and -2 and in total, per ring (0-3, 3-6, 6-12 deg). Fills: 'ring' (keys ring x G, G; the current one) and 'xp'
(region x G x XP bin, region x G, ring x G, G; regions = 4 rings x quadrants, FILL_NMIN covered stars per cell). Coverage:
'standard' (>= NMIN_GOOD good spectra) and 'sparse_gap' (also good / parent >= SPARSE: sparse groups filled like gaps).
Writes results/boss_mock.json.

Input: data/boss_weights.parquet (release_boss_weights, scripts/boss_selection.py). The mock pass model T is floored at 0.05.
results/boss_mock.json: feh_edges; obs_bias_1.0 and obs_bias_1.3, each with group/final x binned (per FEH_EDGES bin) and
cumulative (below -1.5, -2) mean and sd over NREAL realisations of recovered / true; fill: real and central failure models,
each with '<coverage>/<fill>' -> ring0..ring2 -> [ratio below -1.5, below -2, all stars] mean and sd over NFILL realisations.
The mock serves the BOSS truth-mock panel of figures/paper_selection_checks.png (appendix, Sect. 2.2).

Run: .venv/bin/python scripts/boss_mock.py   (after boss_selection.py)
"""
import json

import numpy as np
import pandas as pd
import healpy as hp

from lmcold import selection
from lmcold.constants import LMC
from lmcold.footprint import angular_distance, lmc_xy
from lmcold.groups import bin_index
from lmcold.sources import DATA, ROOT

# %% [markdown]
# ### Settings
# Numbers of realisations and the parameters of the mock.

# %%
NREAL = 50  # realisations per observation-bias variant
NFILL = 20  # realisations per failure model in the fill test
FILL_NMIN = 20  # covered stars needed in a fill cell (lmcold.selection.coverage_factor)
SPARSE = 0.01  # 'sparse_gap' coverage: groups with good / parent < 1% are treated as gaps
FILL_G_EDGES = [12.0, 15.5, 16.25, 16.75, 17.1, 17.5]  # G bins [mag] of the fill cells
FEH_PIV, COMP = -0.6, 0.85  # [dex], -: BOSSNet [Fe/H] is stretched about FEH_PIV by 1 / COMP to make the true [Fe/H]
XP_SD = 0.11  # dex; Andrae+23 XP [M/H] scatter about APOGEE for LMC giants (correlation 0.85)
CUTS = (-1.5, -2.0)  # [Fe/H] thresholds [dex] of the cumulative ratios
FEH_EDGES = [-3.0, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.5, 0.5]  # true [Fe/H] bins [dex]

# %% [markdown]
# ### Load the data
# The BOSS weights `data/boss_weights.parquet` (`scripts/boss_selection.py`): the real parent,
# selection groups, observed flags and the stars with XP [M/H].

# %%
# The real design: one row per eligible parent star.
c = pd.read_parquet(DATA / "boss_weights.parquet")

# %% [markdown]
# ### The real design
# Group index, real pass rate, donors of true [Fe/H], rings and fill regions of each star.

# %%
# s: integer selection-group index per star; g: Gaia G [mag]; real observed and good-spectrum flags
s = pd.factorize(c.group)[0]
g, obs_real, good_real = c.g.to_numpy(), c.observed.to_numpy(), c.good.to_numpy()
# Stars with XP [M/H]; pix: their HEALPix nside-16 nested pixel (donor pools of true_feh)
has_xp = c.mh_xp.notna().to_numpy()
pix = hp.ang2pix(16, np.radians(90 - c.dec.to_numpy()), np.radians(c.ra.to_numpy()), nest=True)
# Real pass rate N_good / N_obs of each star's selection group.
p_real = (np.bincount(s, good_real) / np.maximum(np.bincount(s, obs_real), 1))[s]
# Donors of true [Fe/H]: good spectra with positive weight, labelled by nside-16 pixel.
donors = c[c.good & (c.w > 0)].assign(pix=pix[(c.good & (c.w > 0)).to_numpy()])
# rho: angular distance [deg] from the adopted LMC centre; ring: 3-ring index of the fill test
rho = angular_distance(c.ra.to_numpy(), c.dec.to_numpy(), LMC["ra0"], LMC["dec0"])
ring = np.digitize(rho, [3, 6])  # 0, 1, 2 for 0-3, 3-6, >= 6 deg
# Projected coordinates [deg] (x east, y north) about the LMC centre
x, y = lmc_xy(c.ra.to_numpy(), c.dec.to_numpy())
# Fill region: 4 rings (0-3, 3-6, 6-9, >= 9 deg) x 4 quadrants of the projected (x east, y north) plane.
region = 4 * np.clip(np.digitize(rho, [3, 6, 9]), 0, 3) + (x >= 0) + 2 * (y >= 0)
# G bin of each star for the fill cells
gfill = bin_index(g, FILL_G_EDGES)
n_parent = np.bincount(s)[s]  # eligible stars in each star's selection group


# %% [markdown]
# ### Mock ingredients
# `true_feh` draws a true [Fe/H] for every star; `observe` returns the observed flags, optionally
# biased towards metal-poor stars.

# %%
def true_feh(rng):
    """Draw a true [Fe/H] for every eligible star.

    Parameters
    ----------
    rng : numpy.random.Generator

    Returns
    -------
    numpy.ndarray
        True [Fe/H] [dex] per star: BOSSNet [Fe/H] of a weight-drawn donor in the same nside-16 pixel (any donor if the pixel
        has none), stretched about FEH_PIV by 1 / COMP, plus N(0, 0.05 dex).
    """
    out = np.full(len(c), np.nan)
    # For each pixel with donors: draw its stars' values from those donors, probability proportional to weight w
    for p, d in donors.groupby("pix"):
        k = np.flatnonzero(pix == p)
        out[k] = rng.choice(d.feh.to_numpy(), len(k), p=(d.w / d.w.sum()).to_numpy())
    # Stars in pixels without donors draw from all donors
    left = np.isnan(out)
    out[left] = rng.choice(donors.feh.to_numpy(), left.sum(), p=(donors.w / donors.w.sum()).to_numpy())
    # Stretch about FEH_PIV by 1 / COMP and add N(0, 0.05 dex) scatter
    return FEH_PIV + (out - FEH_PIV) / COMP + rng.normal(0, 0.05, len(c))


def observe(f, rng, bias):
    """Real observed flags, or the same number per selection group drawn with weight `bias` for [Fe/H] < -1.5.

    Parameters
    ----------
    f : numpy.ndarray
        True [Fe/H] per star [dex].
    rng : numpy.random.Generator
    bias : float
        Relative probability of observing a star with [Fe/H] < -1.5; 1 returns the real flags.

    Returns
    -------
    numpy.ndarray of bool
        Observed flag per star; N_obs per selection group equals the real one.
    """
    if bias == 1:
        return obs_real
    key = np.log(rng.random(len(c))) / np.where(f < -1.5, bias, 1.0)  # Efraimidis-Spirakis weighted sampling keys
    # Within each group, the N_obs stars with the largest keys are observed.
    rank = pd.Series(-key).groupby(s).rank(method="first").to_numpy()
    return rank <= np.bincount(s, obs_real)[s]


# %% [markdown]
# ### One realisation
# Failures that depend on $G$ and true [Fe/H], mock XP [M/H], the weights, and the recovered over
# true counts.

# %%
def realisation(seed, bias, central=False, fill=False):
    """One mock realisation of the BOSS selection and the recovered / true ratios.

    Parameters
    ----------
    seed : int
        Random seed.
    bias : float
        Observation bias for [Fe/H] < -1.5 (see observe).
    central : bool
        Double the failure amplitude A(G) within 3 deg of the centre.
    fill : bool
        Run the fill (coverage) test instead of the weight test.

    Returns
    -------
    dict
        fill=False: {'group'|'final': {'binned': ratio per FEH_EDGES bin, 'cumulative': ratio below each of CUTS}}, with the
        truth = covered eligible stars. fill=True: {'<coverage>/<fill>': {'ring<r>': [ratio < -1.5, ratio < -2, ratio all]}},
        with the truth = all eligible stars in the ring.
    """
    rng = np.random.default_rng(seed)
    # True [Fe/H] [dex] and observed flags of this realisation
    f = true_feh(rng)
    obs = observe(f, rng, bias)
    # Pass model: amplitude A(G) rises from 0 at G = 16.25 to 0.7 at G >= 17.5; T falls with [Fe/H] below -1.
    a = 0.7 * np.clip((g - 16.25) / 1.25, 0, 1) * np.where(central & (rho < 3), 2.0, 1.0)
    t = np.clip(1 - a * np.clip((-1 - f) / 1.5, 0, 1) ** 1.5, 0.05, 1)
    # Normalise T within each group so the expected pass rate of the observed stars equals the real one.
    mean_t = np.bincount(s, obs * t) / np.maximum(np.bincount(s, obs), 1)
    # Good spectrum: observed and passing with probability P = p_s T, capped at 1
    good = obs & (rng.random(len(c)) < np.minimum(1, p_real / np.maximum(mean_t[s], 1e-9) * t))
    # Mock XP [M/H] only where the real star has one.
    mh = np.where(has_xp, f + rng.normal(0, XP_SD, len(c)), np.nan)
    # Weights as in scripts/boss_selection.py: w0 group weights alone; w = w0 x pass factor (final weights)
    ones = np.ones(len(c))
    w0 = selection.group_weights(s, good, ones)
    w = w0 * selection.pass_factor(s, obs, good, selection.boss_pass_groups(g, mh, c.ra, c.dec), ones)
    # Good spectra in each star's group; covered: groups with >= NMIN_GOOD good spectra
    n_good = np.bincount(s, good)[s]
    covered = n_good >= selection.NMIN_GOOD
    if fill:
        # Fill keys, finest first: integer codes of (ring, G) or (region, G, XP bin); stars without XP get their own bin.
        xpbin = np.where(np.isfinite(mh), bin_index(mh, selection.XP_EDGES), len(selection.XP_EDGES))
        keys = dict(ring=[ring * 10 + gfill, gfill], xp=[(region * 10 + gfill) * 10 + xpbin, region * 10 + gfill, ring * 10 + gfill, gfill])
        # For each coverage rule and fill: final weights x coverage factor, summed per ring below each cut and in total,
        # over the true count of all eligible stars there
        out = {}
        for cov_name, cov in (("standard", covered), ("sparse_gap", covered & (n_good / n_parent >= SPARSE))):
            for fill_name, k in keys.items():
                ww = w * cov * selection.coverage_factor(k, cov, FILL_NMIN, ones)
                out[f"{cov_name}/{fill_name}"] = {f"ring{r}": [float(np.sum(ww[(ring == r) & (f < x)]) / np.sum((ring == r) & (f < x)))
                                                               for x in CUTS] + [float(np.sum(ww[ring == r]) / np.sum(ring == r))]
                                                  for r in range(3)}
        return out
    k = np.digitize(f, FEH_EDGES)  # 0 below the first edge; bins 1 .. len - 1
    # True number of covered eligible stars per FEH_EDGES bin
    true = np.bincount(k[covered], minlength=len(FEH_EDGES) + 1)[1:len(FEH_EDGES)]
    out = {}
    # Recovered / true per bin and below each of CUTS, for the group and the final weights
    for lab, ww in (("group", w0), ("final", w)):
        out[lab] = dict(binned=(np.bincount(k, ww, len(FEH_EDGES) + 1)[1:len(FEH_EDGES)] / true).tolist(),
                        cumulative=[float(np.sum(ww[f < x]) / np.sum(covered & (f < x))) for x in CUTS])
    return out


# %% [markdown]
# ### Weight test
# Realisations without and with an observation bias.

# %%
# NREAL realisations (seeds 0 .. NREAL - 1) without (1.0) and with (1.3) observation bias; mean and sd to 3 decimals
res = dict(feh_edges=FEH_EDGES)
for bias in (1.0, 1.3):
    r = [realisation(seed, bias) for seed in range(NREAL)]
    res[f"obs_bias_{bias}"] = {lab: {kind: dict(mean=np.round(np.mean([q[lab][kind] for q in r], 0), 3).tolist(),
                                                sd=np.round(np.std([q[lab][kind] for q in r], 0), 3).tolist())
                                     for kind in ("binned", "cumulative")} for lab in ("group", "final")}

# %% [markdown]
# ### Coverage test
# Realisations of the fill test for the real and the central failure models.

# %%
# Column names of the fill ratios; NFILL realisations (seeds from 1000) for each failure model, mean and sd
res["fill"] = dict(columns=[f"<{x}" for x in CUTS] + ["all"])
for central in (False, True):
    r = [realisation(1000 + seed, 1.0, central, fill=True) for seed in range(NFILL)]
    res["fill"]["central" if central else "real"] = {v: {ring_: dict(mean=np.round(np.mean([q[v][ring_] for q in r], 0), 3).tolist(),
                                                                    sd=np.round(np.std([q[v][ring_] for q in r], 0), 3).tolist())
                                                         for ring_ in r[0][v]} for v in r[0]}

# %% [markdown]
# ### Write the results
# `results/boss_mock.json`.

# %%
# Write results/boss_mock.json and print it
(ROOT / "results" / "boss_mock.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
