"""BOSS selection function: weights for the good spectra of the eligible parent of mwm_magcloud_rgb_boss.

Inputs: data/boss_parent.parquet (eligible), data/boss_observed.parquet (observed, spectrum_ok, giant, BOSSNet feh),
data/boss_cartons.parquet (carton memberships; SDSS-V XP [M/H], kept as mh_xp_sdss for comparison), data/xp_andrae.parquet
(XP [M/H] of Andrae et al. 2023 for cool giants, scripts/build_xp_andrae.py).
Class: lmcold.magcloud.carton_class. Selection groups: HEALPix (nested) x G bin x class, pooled from fine to coarse within the class
(lmcold.groups.assign, NMIN eligible stars): (nside 64, G) -> (64) -> (32) -> (16) -> (8) -> class.
Weights (lmcold.selection): w_group = N_elig / N_good per selection group; w = w_group x pass factor of the star's pass group
(G x XP [M/H] x ring, selection.boss_pass_groups), which puts back the failed spectra: weak-lined metal-poor stars fail more often at faint G, and
more often in the crowded centre (a pass factor common to all rings under-corrects at rho < 3 deg and over-corrects beyond 6 deg).
Checks (results/boss_selection.json):
  C1 closure: w-weighted good spectra minus covered eligible stars, fractions of XP [M/H] below -1, -1.5, -2, for w and
     w_group; errors from a bootstrap over nside-16 pixels that recomputes all counts and factors (NBOOT).
  H  held-out sky: pass factors from the even nside-8 pixels applied to the odd ones (with their own group counts); closure there.
  R  closure per ring (0-1.5, 1.5-3, 3-6, 6-12 deg): w-weighted good spectra over covered eligible stars, fractions of XP [M/H]
     below -1 and -1.5; for the final weights and for a pass factor common to all rings (selection.pass_groups).
  P  pass factor per pass group with its bootstrap error (quoted in the text).
Writes data/boss_weights.parquet (one row per eligible star).

sources.yaml: reads release_boss_parent, release_boss_observed, release_boss_cartons, release_xp_andrae;
writes release_boss_weights. Output columns: source_id, ra, dec [deg], g [mag], cls (carton class), group (selection-group
label 'level:key', lmcold.groups.assign), observed, good (observed and spectrum_ok), giant, w (final weight; 0 for stars that are
not good spectra or lie in uncovered groups), w_group (N_elig / N_good), pass_factor (NaN where w_group = 0), mh_xp (Andrae+23,
cool giants only) and mh_xp_sdss [dex], feh (raw BOSSNet [dex]). The weights serve figures/paper_selection.png (b), the
selection checks in the appendix and the mass estimate (build_giant_mass.py).

Run: .venv/bin/python scripts/boss_selection.py   (after build_boss_observed.py, build_boss_cartons.py, build_xp_andrae.py)
"""
import json

import healpy as hp
import numpy as np
import pandas as pd

from lmcold import magcloud, selection
from lmcold.constants import LMC
from lmcold.footprint import angular_distance
from lmcold.sources import DATA, ROOT
from lmcold.groups import assign, bin_index

# %% [markdown]
# ### Settings
# Pooling threshold of the selection groups, bootstrap size, $G$ bins, pooling levels and closure thresholds.

# %%
NMIN = 50  # minimum number of eligible stars in a selection group at a given level of pooling
NBOOT = 200  # bootstrap draws over nside-16 HEALPix pixels
GEDGES = [12.0, 15.5, 16.25, 16.75, 17.1, 17.5]  # G bins [mag] of the finest selection groups (to the carton limit 17.5)
LEVELS = [(64, True), (64, False), (32, False), (16, False), (8, False)]  # (HEALPix nside, split by G?), finest first
CUTS = (-1.0, -1.5, -2.0)  # XP [M/H] thresholds [dex] of the closure checks

# %% [markdown]
# ### Load the data
# The eligible parent (`data/boss_parent.parquet`), its observed and
# good-spectrum flags (`data/boss_observed.parquet`), carton memberships (`data/boss_cartons.parquet`)
# and XP [M/H] (`data/xp_andrae.parquet`), joined on the Gaia source_id.

# %%
# One row per eligible parent star; xp_andrae is a left join (stars without Andrae+23 [M/H] get NaN mh_xp).
c = (pd.read_parquet(DATA / "boss_parent.parquet", columns=["source_id", "ra", "dec", "g", "eligible"]).query("eligible")
     .merge(pd.read_parquet(DATA / "boss_observed.parquet"), on="source_id", validate="1:1")
     .merge(pd.read_parquet(DATA / "boss_cartons.parquet"), on="source_id", validate="1:1").rename(columns={"mh_xp": "mh_xp_sdss"})
     .merge(pd.read_parquet(DATA / "xp_andrae.parquet", columns=["source_id", "mh_xp"]), on="source_id", how="left", validate="1:1"))

# %% [markdown]
# ### Classes and selection groups
# The fibre-priority class of each star, and the selection groups pooled from
# fine to coarse until each holds at least NMIN parent stars.

# %%
c["cls"] = magcloud.carton_class(c.cartons)
c["good"] = c.observed & c.spectrum_ok
# HEALPix angles: colatitude theta = 90 deg - dec, longitude phi = ra (equatorial pixelisation).
theta, phi = np.radians(90 - c.dec.to_numpy()), np.radians(c.ra.to_numpy())
gbin = pd.Series(bin_index(c.g, GEDGES), c.index).astype(str)
# Keys finest first: class_pix64_Gbin, class_pix64, class_pix32, class_pix16, class_pix8, class.
keys = [c.cls + "_" + pd.Series(hp.ang2pix(nside, theta, phi, nest=True), c.index).astype(str) + ("_" + gbin if by_g else "")
        for nside, by_g in LEVELS] + [c.cls]
c["group"] = assign(keys, NMIN)
s = pd.factorize(c.group)[0]  # integer selection-group index per star

# %% [markdown]
# ### Rings, pass groups and bootstrap pixels
# The arrays the weights and checks use.

# %%
obs, good, mh = c.observed.to_numpy(), c.good.to_numpy(), c.mh_xp.to_numpy()
xp = np.isfinite(mh)  # star has an Andrae+23 XP [M/H] (the closure checks use these only)
# Ring index 0-3 for 0-1.5, 1.5-3, 3-6, >= 6 deg from the LMC centre (check R).
rr = np.digitize(angular_distance(c.ra.to_numpy(), c.dec.to_numpy(), LMC["ra0"], LMC["dec0"]), [1.5, 3.0, 6.0])
# Pass group: G bin x XP [M/H] bin x ring (0-3, 3-6, >= 6 deg), selection.boss_pass_groups.
pg = selection.boss_pass_groups(c.g, mh, c.ra, c.dec)
npg = pg.max() + 1
pix8, pix16 = hp.ang2pix(8, theta, phi, nest=True), hp.ang2pix(16, theta, phi, nest=True)


# %% [markdown]
# ### Helpers
# `final` gives the weights, `closure` compares the weighted XP [M/H] fractions with those of the covered
# parent, `factor_per_pass_group` returns the pass factors; all take per-star multiplicities for the bootstrap.

# %%
def final(m):
    """Final weight per star, N_elig / N_good x pass factor, for multiplicities m.

    Parameters
    ----------
    m : numpy.ndarray
        Per-star multiplicity (ones for the data, bootstrap counts, or a 0/1 mask).

    Returns
    -------
    numpy.ndarray
        Weight per star; 0 unless the star is a good spectrum in a covered selection group.
    """
    # Group weight N_elig / N_good (0 outside covered groups) times the pass factor of the star's pass group.
    return selection.group_weights(s, good, m) * selection.pass_factor(s, obs, good, pg, m)


def closure(w, m):
    """w-weighted good spectra minus covered eligible stars: fractions of XP [M/H] below CUTS (multiplicities m).

    Parameters
    ----------
    w : numpy.ndarray
        Weight per star.
    m : numpy.ndarray
        Per-star multiplicity.

    Returns
    -------
    numpy.ndarray
        One difference of fractions per threshold in CUTS (0 = perfect closure). The reference is every eligible star with XP
        [M/H] in a selection group holding >= NMIN_GOOD good spectra (covered groups).
    """
    # Weighted sample: good spectra with a weight and an XP [M/H].
    sel = (w > 0) & xp
    # Reference: eligible stars with XP [M/H] in covered groups (>= NMIN_GOOD good spectra, counted with multiplicities m).
    ref = (np.bincount(s, m * good)[s] >= selection.NMIN_GOOD) & xp
    # Weighted fraction below x minus the reference fraction below x, for each x in CUTS [dex].
    return np.array([np.sum((m * w)[sel & (mh < x)]) / np.sum((m * w)[sel]) - np.sum(m[ref & (mh < x)]) / np.sum(m[ref])
                     for x in CUTS])


def factor_per_pass_group(m):
    """Pass factor of each pass group, for multiplicities m.

    Parameters
    ----------
    m : numpy.ndarray
        Per-star multiplicity.

    Returns
    -------
    numpy.ndarray
        Length npg; NaN for pass groups with no stars.
    """
    # Per-star pass factor, then scattered to one value per pass group (all stars of a group share it).
    f = selection.pass_factor(s, obs, good, pg, m)
    out = np.full(npg, np.nan)
    out[pg] = f
    return out


# %% [markdown]
# ### Weights
# Group weight, pass factor and final weight of every eligible star.

# %%
# Weights of the data (all multiplicities 1).
ones = np.ones(len(c))
c["w_group"] = selection.group_weights(s, good, ones)
c["pass_factor"] = np.where(c.w_group > 0, selection.pass_factor(s, obs, good, pg, ones), np.nan)
c["w"] = final(ones)  # final weight = w_group x pass factor

# %% [markdown]
# ### Bootstrap
# Redraw the HEALPix pixels and recompute the closure and the pass factors.

# %%
# Bootstrap over nside-16 pixels: m = number of times a star's pixel is drawn; counts and factors are recomputed per draw.
# Each row of boot: closure of final weights (3), closure of group weights (3), pass factor per pass group (npg).
rng = np.random.default_rng(1)  # fixed seed for reproducibility
upix = np.unique(pix16)  # occupied nside-16 pixels, sorted
boot = []
for _ in range(NBOOT):
    m = np.bincount(np.searchsorted(upix, rng.choice(upix, len(upix))), minlength=len(upix))[np.searchsorted(upix, pix16)].astype(float)
    boot.append(np.concatenate([closure(final(m), m), closure(selection.group_weights(s, good, m), m), factor_per_pass_group(m)]))
boot = np.array(boot)  # shape (NBOOT, 6 + npg)
err = boot.std(0)  # bootstrap error of every entry of a row

# %% [markdown]
# ### Checks and outputs
# Closure overall, on the held-out half of the sky and per ring, and the pass factors; written to
# `data/boss_weights.parquet` and `results/boss_selection.json`.

# %%
# Held-out sky: even/odd nside-8 pixels (nested numbering); pass factors from the even half, applied to the odd half.
even = (pix8 % 2 == 0).astype(float)
w_odd = selection.group_weights(s, good, 1 - even)
# Check results; each entry is described in the module docstring (C1, H, R, P).
tests = dict(
    # C1: closure for the final weights (error columns 0-2) and the group weights alone (error columns 3-5).
    C1_closure={k: {f"<{x}": dict(diff=round(float(d), 4), err=round(float(e), 4)) for x, d, e in zip(CUTS, v, err[i:i + 3])}
                for (k, v), i in ((("final", closure(c.w.to_numpy(), ones)), 0), (("group", closure(c.w_group.to_numpy(), ones)), 3))},
    # H: odd half weighted by its own group counts times the even-half pass factors.
    H_held_out=dict(final=np.round(closure(w_odd * factor_per_pass_group(even)[pg], 1 - even), 4).tolist(),
                    group=np.round(closure(w_odd, 1 - even), 4).tolist()),
    # Per ring: weighted fraction below x divided by the fraction among covered eligible stars (1 = closure).
    # 'common' uses pass groups without rings (selection.pass_groups).
    R_ring_closure={lab: {f"{lo}-{hi}": {f"<{x}": round(float(np.sum(w[(w > 0) & xp & (rr == k) & (mh < x)]) / np.sum(w[(w > 0) & xp & (rr == k)])
                                                           / np.mean(mh[(np.bincount(s, good)[s] >= selection.NMIN_GOOD) & xp & (rr == k)] < x)), 3)
                                         for x in (-1.0, -1.5)} for k, (lo, hi) in enumerate(zip([0, 1.5, 3, 6], [1.5, 3, 6, 12]))}
                    for lab, w in (("final", c.w.to_numpy()),
                                   ("common", c.w_group.to_numpy() * selection.pass_factor(s, obs, good, selection.pass_groups(c.g.to_numpy(), mh), ones)))},
    # P: pass factor per pass group, bootstrap error (columns 6 onward), and good spectra in covered groups per pass group.
    P_pass_factor=dict(g_edges=selection.PG_EDGES, xp_edges=selection.XP_EDGES, ring_edges=selection.RING_EDGES, index="(G bin x XP bin) x ring + ring", factor=np.round(factor_per_pass_group(ones), 3).tolist(),
                       err=np.round(err[6:], 3).tolist(), n_good=np.bincount(pg[c.w_group > 0], minlength=npg).tolist()),
    # Counts: eligible, good, in covered groups, sums of weights, stars with XP [M/H], stars per class.
    summary=dict(eligible=len(c), good=int(good.sum()), covered=int((np.bincount(s, good)[s] >= selection.NMIN_GOOD).sum()),
                 sum_w_strata=round(float(c.w_group.sum()), 1), sum_w=round(float(c.w.sum()), 1), with_xp=int(xp.sum()),
                 classes=c.cls.value_counts().to_dict()))
# One row per eligible star (columns in the module docstring), then the checks as JSON.
c[["source_id", "ra", "dec", "g", "cls", "group", "observed", "good", "giant", "w", "w_group", "pass_factor", "mh_xp", "mh_xp_sdss",
   "feh"]].to_parquet(DATA / "boss_weights.parquet", index=False)
(ROOT / "results" / "boss_selection.json").write_text(json.dumps(tests, indent=1))
print(json.dumps({k: v for k, v in tests.items() if k != "P_pass_factor"}, indent=1))
