"""Paper figure: RR Lyrae membership. (a) sky to 20 deg: members, stars in the SMC mask, stars outside the distance window;
(b) relative distance modulus against distance from the LMC centre; (c) foreground left in the distance window against radius,
with, inset, the fraction of stars with SMC-like proper motion against distance from the SMC, which sets the mask radius.
Reads data/rrab.parquet and results/rrab_membership.csv; writes figures/paper_rrab_membership.png.

Inputs: data/rrab.parquet (sources.yaml release_rrab, scripts/build_rrab.py), RRab with a relative distance modulus dmu (stars
without one are dropped here), smc_masked flag, rho [deg] and Gaia proper motions; results/rrab_membership.csv
(scripts/fig_rrab_membership.py): per radial bin rho_lo-rho_hi [deg] the foreground fraction left in the distance window, contam,
and its error contam_err, from the density of stars in the sidebands 0.9 < |dmu| < 2 mag assumed flat in dmu.
Limits from lmcold.footprint: DMU_WINDOW = 0.6 mag, RHO_MAX = 10 deg, SMC_MASK_RADIUS = 13 deg.
Panels (1 x 3, square):
(a) x (east, increasing to the left), y (north) [deg], orthographic about the LMC centre, +-25 deg: members (blue, outside the
    SMC mask and |dmu| < 0.6), SMC-masked stars (orange), stars outside the window (yellow); circles: SMC mask (orange),
    10-deg science radius (dashed), 20-deg limit of the data (grey).
(b) x: rho [deg], 0-20; y: dmu [mag], -2 to 2 (axis to 2.4 to leave room for the colour bar); 2D histogram (0.4 deg x 0.05 mag,
    log colour) of stars outside the SMC mask; orange lines: the +-0.6 mag window; dashed: RHO_MAX.
(c) x: rho [deg]; y: foreground in the window [per cent] (log, 0.4-150), x error bars the bin half-widths. Inset: fraction of
    window stars whose proper motion is closer to the SMC systemic motion than to the LMC one, in 1-deg bins of distance from
    the SMC centre [deg]; orange line: mask radius.
Placement: appendix (Sect. 2.1; docs/figures.md). Run: .venv/bin/python scripts/paper_rrab_membership.py   (about 1 s on a laptop, docs/run_times.txt)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from lmcold.constants import LMC, SMC
from lmcold.footprint import DMU_WINDOW, RHO_MAX, SMC_MASK_RADIUS, angular_distance, lmc_xy, small_circle
from lmcold.layout import panel_grid
from lmcold.sources import DATA, ROOT

plt.style.use(ROOT / "src" / "lmcold" / "paper.mplstyle")
# Colours: members, SMC mask, stars outside the distance window, 20-deg data limit.
BLUE, ORANGE, YELLOW, GREY = "#2a78d6", "#eb6834", "#eda100", "0.35"

# %% [markdown]
# ### Load the data
# The RRab catalogue `data/rrab.parquet` (Zenodo bundle 1) and the foreground fraction per radial bin from
# `results/rrab_membership.csv` (written by `scripts/fig_rrab_membership.py`).

# %%
# data/rrab.parquet: RRab with relative distance modulus dmu, smc_masked flag, rho [deg] and Gaia proper motions
t = pd.read_parquet(DATA / "rrab.parquet")
# results/rrab_membership.csv: per radial bin (rho_lo, rho_hi) the foreground fraction left in the window, contam, contam_err
c = pd.read_csv(ROOT / "results" / "rrab_membership.csv")

# %% [markdown]
# ### The stars with a distance
# RRab with a relative distance modulus, in the orthographic projection, and those inside the distance window.

# %%
t = t[t.dmu.notna()]
# Orthographic x (east), y (north) [deg] about the LMC centre.
t["x"], t["y"] = lmc_xy(t.ra.to_numpy(), t.dec.to_numpy())
win = t.dmu.abs() < DMU_WINDOW  # inside the distance window

# %% [markdown]
# ### Panel (a): the sky
# Members, SMC-masked stars and stars outside the distance window.

# %%
# One row of three equal panels.
fig, (ax,) = panel_grid(1, 3, top=0.1)

a = ax[0]
# Three groups, members drawn last (on top) with smaller points.
for sel, col, lab, s in ((~t.smc_masked & ~win, YELLOW, r"$|\Delta\mu| > 0.6$", 0.6), (t.smc_masked, ORANGE, "SMC", 0.6),
                         (~t.smc_masked & win, BLUE, r"$|\Delta\mu| < 0.6$", 0.15)):
    a.scatter(t.x[sel], t.y[sel], s=s, color=col, lw=0, rasterized=True, label=lab)
# Circles: the SMC mask, the 10-deg science radius RHO_MAX and the 20-deg data limit, projected about the LMC centre.
for (ra0, dec0, r), col, ls in (((SMC["ra0"], SMC["dec0"], SMC_MASK_RADIUS), ORANGE, "-"), ((LMC["ra0"], LMC["dec0"], RHO_MAX), "k", "--"),
                                ((LMC["ra0"], LMC["dec0"], 20.0), GREY, "-")):
    cx, cy = lmc_xy(*small_circle(ra0, dec0, r))
    front = np.hypot(cx, cy) <= 20.01  # draw only the part of each circle within the 20 deg data circle
    a.plot(np.where(front, cx, np.nan), np.where(front, cy, np.nan), color=col, ls=ls, lw=0.8)
a.set(xlim=(25, -25), ylim=(-25, 25), aspect="equal", xlabel=r"$x$ (east) [deg]", ylabel=r"$y$ (north) [deg]")
a.legend(loc="lower left", ncol=3, fontsize=6, markerscale=6, handletextpad=0.05, borderaxespad=0.1, columnspacing=0.6,
         handlelength=0.8)  # one row in the strip below y = -21, outside the 20 deg data circle
a.text(0.96, 0.04, "(a)", transform=a.transAxes, ha="right")

# %% [markdown]
# ### Panel (b): relative distance modulus against $\rho$

# %%
a = ax[1]
# 2D histogram of the stars outside the SMC mask: 0.4-deg bins in rho, 0.05-mag bins in dmu, log colour scale.
o = ~t.smc_masked
h = a.hist2d(t.rho[o], t.dmu[o], bins=[np.arange(0, 20.01, 0.4), np.arange(-2.0, 2.01, 0.05)], norm=LogNorm(), cmap="Blues",
             rasterized=True)
# The +-DMU_WINDOW distance window (orange) and RHO_MAX (dashed).
for s in (1, -1):
    a.axhline(s * DMU_WINDOW, color=ORANGE, lw=0.8)
a.plot([RHO_MAX, RHO_MAX], [-2.0, 2.0], color="k", ls="--", lw=0.8)
a.set(xlabel=r"$\rho$ [deg]", ylabel=r"$\Delta\mu$ [mag]")
a.set_ylim(-2.0, 2.4)  # an empty strip above the data (|dmu| <= 2) holds the colour bar
# Horizontal colour bar inside the empty strip at the top of the panel.
cax = a.inset_axes([0.6, 0.855, 0.36, 0.045])
fig.colorbar(h[3], cax=cax, orientation="horizontal")
cax.tick_params(labelsize=5, pad=1, length=2)
a.text(0.78, 0.94, "stars/pixel", transform=a.transAxes, ha="center", va="center", fontsize=6)
a.text(0.96, 0.04, "(b)", transform=a.transAxes, ha="right")

# %% [markdown]
# ### Panel (c): foreground in the window
# With, inset, the fraction of stars with SMC-like proper motion against distance from the SMC.

# %%
a = ax[2]
# Bin centres [deg]; foreground fraction in per cent, x error bars the bin half-widths.
rc = 0.5 * (c.rho_lo + c.rho_hi)
a.errorbar(rc, 100 * c.contam, yerr=100 * c.contam_err, xerr=0.5 * (c.rho_hi - c.rho_lo), fmt="o", ms=2.5, color=BLUE, lw=0.8)
a.axvline(RHO_MAX, color="k", ls="--", lw=0.8)
a.set(yscale="log", ylim=(0.4, 150), xlabel=r"$\rho$ [deg]", ylabel=r"foreground in the window [%]")
a.text(0.96, 0.04, "(c)", transform=a.transAxes, ha="right")
# Inset: stars in the distance window with a Gaia proper motion.
ins = a.inset_axes([0.17, 0.66, 0.27, 0.30])
d = t[win & t.pmra.notna()]
# SMC-like: proper motion [mas/yr] nearer to the SMC systemic motion than to the LMC one (lmcold.constants).
smclike = np.hypot(d.pmra - SMC["pmra0"], d.pmdec - SMC["pmdec0"]) < np.hypot(d.pmra - LMC["pmra0"], d.pmdec - LMC["pmdec0"])
# Distance from the SMC centre [deg] in 1-deg bins, 0-22 deg; the SMC-like fraction per bin, and the mask radius (orange line).
dS = angular_distance(d.ra, d.dec, SMC["ra0"], SMC["dec0"])
b = np.arange(0, 22.1, 1.0)
i = np.digitize(dS, b)
ins.plot(0.5 * (b[1:] + b[:-1]), [smclike[i == k].mean() for k in range(1, len(b))], "o-", ms=1.5, lw=0.8, color=BLUE)
ins.axvline(SMC_MASK_RADIUS, color=ORANGE, lw=0.8)
ins.set_xlabel("from SMC [deg]", fontsize=6, labelpad=1)
ins.set_ylabel("SMC-like", fontsize=6, labelpad=2)
ins.tick_params(labelsize=6)
ins.set(ylim=(0, 1.05), xticks=[0, 10, 20], yticks=[0, 0.5, 1])

fig.savefig(ROOT / "figures" / "paper_rrab_membership.png")
