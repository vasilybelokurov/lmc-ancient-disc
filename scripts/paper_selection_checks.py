"""Paper figure (appendix, Sect. 2.2): checks of the selection functions of scripts/paper_selection.py.

RRab, from the overlap of OGLE-IV (own discoveries only: stars from OGLE-LMC-RRLYR-39960 on were added from Gaia) and Gaia DR3:
C_Gaia = fraction of OGLE-own RRab that Gaia also classifies as RRab; C_OGLE = fraction of Gaia RRab inside the OGLE-IV footprint
found by OGLE itself. (a) C_Gaia against mean magnitude in three rings; (b, c) C_Gaia and C_OGLE in 1 deg pixels (>= 5 stars in
both), with the OGLE-IV footprint outline. (d) BOSS truth mock on the real design (scripts/boss_mock.py): weighted count / true
count in bins of true [Fe/H], selection-group weights alone and final weights; mean and sd of the realisations.
Writes figures/paper_selection_checks.png and results/rrab_selection.csv (per pixel).

Inputs: data/rrab.parquet (release_rrab; all RRab within 20 deg, flags ogle_own, in_gaia, OGLE mean I), results/boss_mock.json
(scripts/boss_mock.py; key obs_bias_1.0 = mock without an observation bias).
Panels (1 x 4, square):
(a) x: I + 0.6 mag (~ Gaia G for LMC RRab) [mag], 17.8-20.6, 0.25-mag bins with >= 20 stars; y: Gaia completeness of the
    OGLE-own RRab, binomial errors; rings 0-2, 2-4, 4-12 deg.
(b, c) x (east, increasing to the left), y (north) [deg], +-12.5: C_Gaia and C_OGLE in 1-deg pixels of the orthographic
    projection, colour 0.4-1 (shared bar above both); only pixels inside the OGLE-IV footprint with >= 5 stars found by both;
    grey contour: OGLE-IV footprint.
(d) x: true [Fe/H] [dex] (mock bins); y: recovered / true weighted count, 0.5-1.7; grey: selection-group weights alone, blue:
    final weights; mean and sd over the mock realisations.
Output CSV, one row per 1-deg pixel: x, y [deg] (centre), n_ogle, n_gaia, n_both, C_gaia = n_both / n_ogle, C_ogle =
n_both / n_gaia, C_union = (n_ogle + n_gaia - n_both) / (n_ogle n_gaia / n_both).
Placement: appendix (Sect. 2.2; docs/figures.md). Run: .venv/bin/python scripts/paper_selection_checks.py   (about 3 s on a laptop, docs/run_times.txt)
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_closing, binary_fill_holes
import numpy as np
import pandas as pd

from lmcold import ogle
from lmcold.constants import LMC
from lmcold.footprint import lmc_xy
from lmcold.layout import panel_grid
from lmcold.sources import DATA, ROOT

plt.style.use(ROOT / "src" / "lmcold" / "paper.mplstyle")
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"   # line colours (rings of panel a; final weights of panel d)
L = 12.5  # deg, half width of the map panels

# %% [markdown]
# ### Load the data
# The RRab catalogue `data/rrab.parquet` (Zenodo bundle 1) and the BOSS truth mock `results/boss_mock.json` (written by
# `scripts/boss_mock.py`).

# %%
# data/rrab.parquet: all RRab within 20 deg, with the flags ogle_own, in_gaia and the OGLE mean I magnitude
t = pd.read_parquet(DATA / "rrab.parquet")
# results/boss_mock.json: recovered / true weighted counts per true [Fe/H] bin over the mock realisations
mock = json.loads((ROOT / "results" / "boss_mock.json").read_text())

# %% [markdown]
# ### The RR Lyrae
# OGLE-own RRab and *Gaia* RRab inside the OGLE-IV footprint, in the orthographic projection about the LMC centre.

# %%
t = t[t.rho < 20].copy()   # rho [deg] from the LMC centre
t["in_fp"] = ogle.in_footprint(t.ra.to_numpy(), t.dec.to_numpy())   # inside the OGLE-IV fields
# Orthographic x (east), y (north) [deg] about the LMC centre.
t["x"], t["y"] = lmc_xy(t.ra.to_numpy(), t.dec.to_numpy())
og, ga = t[t.ogle_own], t[t.in_gaia & t.in_fp]  # OGLE-own RRab; Gaia RRab inside the OGLE-IV footprint


def binned(s, flag, col, edges):
    """Fraction of s with `flag` in bins of `col`, binomial error; NaN where fewer than 20 stars.

    Parameters
    ----------
    s : pandas.DataFrame
        Stars.
    flag : str
        Boolean column whose mean is the fraction.
    col : str
        Binned column (here mag [mag]).
    edges : numpy.ndarray
        Bin edges in `col`.

    Returns
    -------
    centres, fraction, error : numpy.ndarray
        Bin centres, fraction with `flag` and sqrt(f (1 - f) / n) per bin.
    """
    k = np.digitize(s[col], edges) - 1   # bin of each star
    # Stars per bin, and the fraction with `flag` in bins of at least 20 stars.
    n = np.array([(k == i).sum() for i in range(len(edges) - 1)])
    f = np.array([s[flag][k == i].mean() if n[i] >= 20 else np.nan for i in range(len(edges) - 1)])
    return 0.5 * (edges[1:] + edges[:-1]), f, np.sqrt(f * (1 - f) / np.maximum(n, 1))


# %% [markdown]
# ### Panel (a): *Gaia* completeness against magnitude
# In three rings of $\rho$; OGLE $I + 0.6$ stands in for *Gaia* $G$.

# %%
fig, ax = panel_grid(1, 4, hgap=0.55, top=0.7)   # four equal panels in a row
a = ax[0, 0]
m = og.assign(mag=og.I + 0.6)  # OGLE I + 0.6 ~ Gaia G for LMC RRab
# Rings in rho [deg]; 0.25-mag bins from 17.75 mag.
for (lo, hi), col in zip(((0, 2), (2, 4), (4, 12)), (BLUE, ORANGE, GREEN)):
    x, f, e = binned(m[(m.rho >= lo) & (m.rho < hi)], "in_gaia", "mag", np.arange(17.75, 20.6, 0.25))
    a.errorbar(x, f, yerr=e, fmt="o-", ms=2.5, lw=1, color=col, label=f"{lo} < $\\rho$ < {hi} deg")
a.set(xlim=(17.8, 20.6), ylim=(0, 1.04), xlabel=r"$I + 0.6 \approx G$ [mag]", ylabel="Gaia completeness")
a.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=1, borderaxespad=0.1, handlelength=1.5)
a.text(0.95, 0.05, "(a)", transform=a.transAxes, ha="right")

# %% [markdown]
# ### Panels (b, c): completeness on the sky
# Counts per 1-deg pixel (written to `results/rrab_selection.csv`) and the outline of the OGLE-IV footprint.

# %%
# Completeness per 1-deg pixel of the orthographic x, y inside the OGLE-IV footprint (pixels with >= 5 stars found by both).
e = np.arange(-12, 12.01, 1.0)   # deg, pixel edges in x and y
rows = []   # one dict per pixel kept
for i in range(len(e) - 1):
    for j in range(len(e) - 1):
        # Stars of pixel (i, j) inside the footprint; OGLE-own, Gaia and both.
        s = t[t.in_fp & (t.x >= e[i]) & (t.x < e[i + 1]) & (t.y >= e[j]) & (t.y < e[j + 1])]
        nO, nG, nB = int(s.ogle_own.sum()), int(s.in_gaia.sum()), int((s.ogle_own & s.in_gaia).sum())
        if nB >= 5:
            rows.append(dict(ix=i, iy=j, x=0.5 * (e[i] + e[i + 1]), y=0.5 * (e[j] + e[j + 1]), n_ogle=nO, n_gaia=nG, n_both=nB,
                             C_gaia=nB / nO, C_ogle=nB / nG, C_union=(nO + nG - nB) / (nO * nG / nB)))
p = pd.DataFrame(rows)   # ix, iy: pixel indices for the maps; dropped from the csv
p.drop(columns=["ix", "iy"]).to_csv(ROOT / "results" / "rrab_selection.csv", index=False, float_format="%.4f")
# Footprint outline: a 400 x 400 grid of (x, y) mapped back to (ra, dec) with rho = hypot(x, y) and PA = atan2(x, y). This is
# the inverse of an azimuthal equidistant projection; lmc_xy is orthographic (x = sin(rho) sin(PA)), so near rho = 12.5 deg the
# outline lies about 0.1 deg farther from the centre than the stars it encloses (sin(12.6 deg) = 12.5 deg in radians).
g = np.linspace(-L, L, 400)   # deg, grid in x and y
gx, gy = np.meshgrid(g, g)
r0 = np.deg2rad(np.hypot(gx, gy))   # rad, angular distance from the centre
pa = np.arctan2(gx, gy)   # rad, position angle (north through east)
# Spherical trigonometry: (ra, dec) at distance r0 and position angle pa from the LMC centre (lmcold.constants).
dec0, ra0 = np.deg2rad(LMC["dec0"]), np.deg2rad(LMC["ra0"])
dec = np.arcsin(np.sin(dec0) * np.cos(r0) + np.cos(dec0) * np.sin(r0) * np.cos(pa))
ra = ra0 + np.arctan2(np.sin(pa) * np.sin(r0) * np.cos(dec0), np.cos(r0) - np.sin(dec0) * np.sin(dec))
# (400, 400) bool map: grid point inside the OGLE-IV footprint.
fp = ogle.in_footprint(np.rad2deg(ra).ravel(), np.rad2deg(dec).ravel()).reshape(gx.shape)
fp = binary_fill_holes(binary_closing(fp, np.ones((5, 5))))  # close the ~0.1 deg gaps between field rectangles (grid step 0.06 deg)
CMAP = ListedColormap(plt.cm.Blues(np.linspace(0.95, 0.15, 256)))  # completeness 1 is light blue, not white
# (b) and (c): map of the pixel completeness (NaN: pixel not kept) and the footprint outline.
for a, col, lab in ((ax[0, 1], "C_gaia", "(b) Gaia"), (ax[0, 2], "C_ogle", "(c) OGLE, own search")):
    mm = np.full((len(e) - 1, len(e) - 1), np.nan)
    mm[p.ix, p.iy] = p[col]
    im = a.pcolormesh(e, e, mm.T, cmap=CMAP, vmin=0.4, vmax=1.0, rasterized=True)
    a.contour(g, g, fp.astype(float), levels=[0.5], colors="0.3", linewidths=0.6)
    a.set(xlim=(L, -L), ylim=(-L, L), aspect="equal", xticks=[10, 5, 0, -5, -10], yticks=[-10, -5, 0, 5, 10],
          xlabel=r"$x$ (east) [deg]")
    a.text(0.0, 1.02, lab, transform=a.transAxes, fontsize=7)
ax[0, 1].set_ylabel(r"$y$ (north) [deg]")
ax[0, 2].tick_params(labelleft=False)
# One colour bar spanning panels (b) and (c): two axes widths plus the gap (hgap 0.55 inch) in units of the axes width.
cax = ax[0, 1].inset_axes([0.0, 1.16, 2.0 + 0.55 / (ax[0, 1].get_position().width * fig.get_size_inches()[0]), 0.05])
fig.colorbar(im, cax=cax, orientation="horizontal")
cax.xaxis.set_ticks_position("top")
cax.tick_params(labelsize=6)
cax.set_title("completeness per 1 deg pixel; grey line: OGLE-IV footprint", fontsize=7, pad=9)
# %% [markdown]
# ### Panel (d): the BOSS mock
# Recovered over true counts per bin of true [Fe/H] (`results/boss_mock.json`, written by `scripts/boss_mock.py`).

# %%
a = ax[0, 3]
me = np.array(mock["feh_edges"])   # dex, true [Fe/H] bin edges of the mock
# Selection-group weights (grey) and final weights (blue), offset by -+0.02 dex; mean and sd over the realisations.
for key, lab, col, dx in (("group", "selection-group weights", "0.55", -0.02), ("final", "final weights", BLUE, 0.02)):
    q = mock["obs_bias_1.0"][key]["binned"]
    a.errorbar(0.5 * (me[1:] + me[:-1]) + dx, q["mean"], yerr=q["sd"], fmt="o-", ms=3, lw=1, color=col, label=lab)
a.axhline(1, color="0.6", lw=0.6)
a.set(xlabel="true [Fe/H]", ylabel="recovered / true", xlim=(-3.1, 0.6), ylim=(0.5, 1.7))
a.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=1, borderaxespad=0.1, handlelength=1.5, title="BOSS mock", alignment="left")
a.text(0.95, 0.05, "(d)", transform=a.transAxes, ha="right")
fig.savefig(ROOT / "figures" / "paper_selection_checks.png")
# Median completeness over the kept pixels.
print(p[["C_gaia", "C_ogle", "C_union"]].median().round(3).to_dict())
