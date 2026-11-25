"""Paper figure (Sect. 2.2): the selection functions of the three samples.

(a) RRab completeness against distance from the LMC centre: Gaia = fraction of the RRab found by OGLE's own search that Gaia DR3
also classifies as RRab; OGLE = fraction of the Gaia RRab inside the OGLE-IV footprint found by OGLE's own search (stars from
OGLE-LMC-RRLYR-39960 on were added from Gaia); union = n_union / N, with N the Lincoln-Petersen estimate n_O n_G / n_both made in
magnitude quartiles (results/rrab_independence.csv, scripts/check_rrab_independence.py). Binomial errors; bins with >= 20 stars.
(b, c) Share of the spectroscopic sample in each bin of Gaia XP [M/H] (Andrae et al. 2023, cool giants; measured for observed and
unobserved stars alike) divided by the share of the parent: good spectra unweighted (open) and with the final weights (filled).
(b) BOSS: parent = eligible stars of mwm_magcloud_rgb_boss in covered selection groups; w = selection-group weight x pass factor
(scripts/boss_selection.py). (c) APOGEE: parent = covered candidates of the LMC1-35 RGB targeting (scripts/apogee_selection.py).
Errors: bootstrap over nside-16 pixels (weights fixed); bins with >= NMIN sample stars.
Writes figures/paper_selection.png.

Inputs: data/rrab.parquet (release_rrab; flags ogle_own, in_gaia, rho [deg]), results/rrab_independence.csv
(scripts/check_rrab_independence.py; per ring rho_lo, rho_hi, C_union_by_mag), data/boss_weights.parquet (release_boss_weights;
BOSS parent with group, good, w, mh_xp), data/apogee_weights.parquet (release_apogee_weights; APOGEE candidates with covered,
quality, w, mh_xp).
Panels (1 x 3, square; legends above the panels):
(a) x: rho [deg], 0-16; y: completeness, 0-1.04; Gaia (blue) and OGLE own search (orange) in 1-deg bins with >= 20 stars,
    binomial errors; green squares: union (capture-recapture), x error bars the ring half-widths.
(b) x: XP [M/H], -2.6 to 0.1; y: sample / parent share per bin for BOSS (good spectra), unweighted (open) and with the final
    weights w (filled, offset +0.03); errors: standard deviation of NBOOT nside-16 pixel bootstraps; bins with >= NMIN sample
    stars; grey line at 1.
(c) the same for APOGEE (good-quality spectra among the covered candidates).
Placement: Sect. 2.2, main text (docs/figures.md). Run: .venv/bin/python scripts/paper_selection.py   (about 4 s on a laptop, docs/run_times.txt)
"""
import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmcold import ogle
from lmcold.layout import panel_grid
from lmcold.sources import DATA, ROOT

plt.style.use(ROOT / "src" / "lmcold" / "paper.mplstyle")
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"   # colours: Gaia / BOSS, OGLE / APOGEE, union
# XP [M/H] bin edges [dex] of panels (b) BOSS and (c) APOGEE
EDGES = {"BOSS": np.array([-2.5, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, 0.0]),
         "APOGEE": np.array([-2.5, -1.5, -1.0, -0.75, -0.5, 0.0])}  # APOGEE has ~50 metal-poor stars with XP
NMIN = 10  # minimum number of sample stars in an XP bin
NBOOT = 200  # pixel bootstrap draws
LEG_TOP = 1.36  # legends above the panels, tops aligned

# %% [markdown]
# ### Load the data
# The RRab catalogue `data/rrab.parquet` and the selection weights of the two spectroscopic samples, `data/boss_weights.parquet`
# and `data/apogee_weights.parquet` (Zenodo bundle 1); the union completeness per ring from `results/rrab_independence.csv`
# (written by `scripts/check_rrab_independence.py`).

# %%
# data/rrab.parquet: all RRab within 20 deg, with the flags ogle_own, in_gaia and rho [deg]
t = pd.read_parquet(DATA / "rrab.parquet")
# data/boss_weights.parquet: the BOSS parent with selection group, good-spectrum flag, weight w and XP [M/H]
b = pd.read_parquet(DATA / "boss_weights.parquet")
# data/apogee_weights.parquet: the APOGEE candidates (only the covered ones kept) with quality flag, weight w and XP [M/H]
a = pd.read_parquet(DATA / "apogee_weights.parquet").query("covered")
# results/rrab_independence.csv: union completeness per ring (rho_lo, rho_hi, C_union_by_mag), from magnitude-quartile estimates
u = pd.read_csv(ROOT / "results" / "rrab_independence.csv")

# %% [markdown]
# ### The samples
# All RRab within 20 deg with the OGLE-IV footprint flag; the BOSS parent in covered selection groups.

# %%
t = t[t.rho < 20].copy()  # the whole RRab catalogue (Gaia query radius 20 deg), not only members
# True for RRab inside the OGLE-IV footprint (lmcold.ogle)
t["in_fp"] = ogle.in_footprint(t.ra.to_numpy(), t.dec.to_numpy())
b = b[b.group.map(b.groupby("group").good.sum()) >= 3]  # covered selection groups (>= 3 good spectra, selection.NMIN_GOOD)


# %% [markdown]
# ### Completeness and share ratios
# `binned` gives a completeness per bin of $\rho$; `ratios` gives the share of the sample in each XP [M/H] bin over
# the share of the parent, with a bootstrap over HEALPix pixels.

# %%
def binned(s, flag, col, edges):
    """Fraction of s with `flag` in bins of `col`, binomial error; NaN where fewer than 20 stars.

    Parameters
    ----------
    s : pandas.DataFrame
        Stars.
    flag : str
        Boolean column whose mean is the fraction.
    col : str
        Binned column (here rho [deg]).
    edges : numpy.ndarray
        Bin edges in `col`.

    Returns
    -------
    centres, fraction, error : numpy.ndarray
        Bin centres, fraction with `flag` and sqrt(f (1 - f) / n) per bin.
    """
    # Bin index of each star (-1 or len(edges) - 1 outside); n: stars per bin
    k = np.digitize(s[col], edges) - 1
    n = np.array([(k == i).sum() for i in range(len(edges) - 1)])
    # Fraction with flag per bin (NaN below 20 stars); returned with bin centres and binomial errors
    f = np.array([s[flag][k == i].mean() if n[i] >= 20 else np.nan for i in range(len(edges) - 1)])
    return 0.5 * (edges[1:] + edges[:-1]), f, np.sqrt(f * (1 - f) / np.maximum(n, 1))


def ratios(t, inc, weights, edges):
    """Sample / parent share per XP [M/H] bin for each weight column (None = unweighted), pixel-bootstrap errors, sample count
    per bin.

    Parameters
    ----------
    t : pandas.DataFrame
        Parent stars with ra, dec [deg], mh_xp and the columns named in `inc` and `weights`; stars without XP [M/H] are dropped.
    inc : str
        Boolean column marking the sample (good spectra).
    weights : list of (str or None)
        Weight columns of the sample; None for unit weights.
    edges : numpy.ndarray
        XP [M/H] bin edges; stars outside them are ignored.

    Returns
    -------
    ratio : numpy.ndarray, shape (len(weights), len(edges) - 1)
        (weighted sample in bin / weighted sample) / (parent in bin / parent).
    error : numpy.ndarray, same shape
        Standard deviation over NBOOT draws of nside-16 (nested) pixels with replacement.
    n : numpy.ndarray
        Unweighted sample count per bin.
    """
    # Stars with XP [M/H]; k: XP bin index, ok: inside the edges
    t = t[t.mh_xp.notna()]
    k = np.digitize(t.mh_xp, edges) - 1
    ok = (k >= 0) & (k < len(edges) - 1)
    # HEALPix nside-16 nested pixel of each star (colatitude, longitude in rad), the bootstrap unit
    pix = hp.ang2pix(16, np.radians(90 - t.dec.to_numpy()), np.radians(t.ra.to_numpy()), nest=True)
    # s: sample stars in a bin; ws: one weight array per entry of weights (ones for None)
    s = t[inc].to_numpy() & ok
    ws = [np.ones(len(t)) if w is None else t[w].to_numpy() for w in weights]
    # Occupied pixels and a seeded generator (seed 1: the same errors on every run)
    upix, rng = np.unique(pix), np.random.default_rng(1)

    def one(m):
        """Ratios for star multiplicities m (pixel draw counts; ones for the data)."""
        # Parent count per bin, then for each weighting the sample share over the parent share
        par = np.bincount(k[ok], m[ok], len(edges) - 1)
        return [np.bincount(k[s], (m * x)[s], len(edges) - 1) / np.sum((m * x)[s]) / (par / par.sum()) for x in ws]
    # Each draw: NBOOT times, pixels drawn with replacement; a star's multiplicity = the times its pixel was drawn
    boot = [one(np.bincount(np.searchsorted(upix, rng.choice(upix, len(upix))), minlength=len(upix))[np.searchsorted(upix, pix)]
                .astype(float)) for _ in range(NBOOT)]
    # Ratios of the data (multiplicity 1), bootstrap standard deviation, unweighted sample count per bin
    return np.array(one(np.ones(len(t)))), np.array(boot).std(0), np.bincount(k[s], minlength=len(edges) - 1)


# %% [markdown]
# ### The figure
# (a) RR Lyrae completeness of each survey and of their union; (b, c) BOSS and APOGEE share ratios, unweighted and
# with the final weights.

# %%
# Three equal panels; (a) RRab completeness
fig, ax = panel_grid(1, 3, hgap=0.5, top=0.75)
p = ax[0, 0]
# Gaia: fraction of the OGLE own-search RRab that Gaia has; OGLE: fraction of the Gaia RRab in the
# OGLE-IV footprint found by OGLE's own search; 1-deg bins of rho from 0 to 16 deg
for s, flag, col, lab in ((t[t.ogle_own], "in_gaia", BLUE, "Gaia"), (t[t.in_gaia & t.in_fp], "ogle_own", ORANGE, "OGLE (own search)")):
    x, f, e = binned(s, flag, "rho", np.arange(0, 16.1, 1.0))
    p.errorbar(x, f, yerr=e, fmt="o-", ms=2.5, lw=1, color=col, label=lab)
# Union completeness per ring (capture-recapture), x bars the ring half-widths
p.errorbar(0.5 * (u.rho_lo + u.rho_hi), u.C_union_by_mag, xerr=0.5 * (u.rho_hi - u.rho_lo), fmt="s", ms=3, lw=1, color=GREEN,
           label="OGLE or Gaia")
# Axes, legend above the panel, panel letter
p.set(xlim=(0, 16), ylim=(0, 1.04), xlabel=r"$\rho$ [deg]", ylabel="completeness")
p.legend(loc="upper left", bbox_to_anchor=(0, LEG_TOP), ncol=1, handlelength=1.5, title="RRab", alignment="left")
p.text(0.95, 0.05, "(a)", transform=p.transAxes, ha="right")
# (b) BOSS and (c) APOGEE: (parent table, sample flag, label, colour, panel letter)
for p, (s, inc, lab, col, tag) in zip(ax[0, 1:], ((b, "good", "BOSS", BLUE, "(b)"), (a, "quality", "APOGEE", ORANGE, "(c)"))):
    # Share ratios unweighted (row 0) and with weights w (row 1); x: bin centres; ok: bins with >= NMIN sample stars
    r, e, n = ratios(s, inc, [None, "w"], EDGES[lab])
    x, ok = 0.5 * (EDGES[lab][1:] + EDGES[lab][:-1]), n >= NMIN
    for i, style in enumerate((dict(fmt="o", mfc="white", lw=0.8, label="unweighted"), dict(fmt="o-", lw=1, label="final weights"))):
        # Unweighted offset by -0.03, weighted by +0.03 dex
        p.errorbar(x[ok] + 0.03 * (2 * i - 1), r[i][ok], yerr=e[i][ok], ms=3, color=col, **style)
    # Grey line at ratio 1 (sample follows the parent); y range wide enough for every error bar
    p.axhline(1, color="0.6", lw=0.6)
    p.set(xlabel=r"[M/H]$_{\rm XP}$", ylabel="sample / parent", xlim=(-2.6, 0.1),
          ylim=(0.4, max(1.7, float(np.nanmax((r + e)[:, ok])) * 1.05)))
    p.legend(loc="upper left", bbox_to_anchor=(0, LEG_TOP), ncol=1, handlelength=1.5, title=lab, alignment="left")
    p.text(0.95, 0.93, tag, transform=p.transAxes, ha="right", va="top")
fig.savefig(ROOT / "figures" / "paper_selection.png")
