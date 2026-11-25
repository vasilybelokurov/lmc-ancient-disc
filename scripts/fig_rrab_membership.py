"""RRab membership: the SMC mask, the distance window, and what is left behind.

(a) sky: members, stars removed by the SMC mask, stars removed by the distance window; (b) fraction of window stars with SMC-like
proper motion against distance from the SMC (closer to the SMC systemic PM than to the LMC one), which sets the mask radius;
(c) dmu in radial bins outside the mask, with the window and the sidebands; (d) residual contamination inside the window,
estimated from the sideband density (assumed flat in dmu); (e) counts against radius at each step.
Writes figures/rrab_membership.png and results/rrab_membership.csv.

Diagnostic behind the membership limits of lmcold.footprint (SMC mask radius 13 deg, |dmu| < 0.6 mag, rho < 10 deg); the paper
version is scripts/paper_rrab_membership.py (appendix, docs/figures.md), which reads results/rrab_membership.csv for its panel (c);
scripts/paper_text_data.py takes the foreground fractions of the text from the same table.

Input: data/rrab.parquet (sources.yaml key release_rrab): ra, dec, rho [deg], dmu [mag, relative distance modulus], pmra, pmdec
[mas/yr], smc_masked, member. Stars without dmu are dropped. Reference proper motions: lmcold.constants.LMC and SMC.

Figure (2 x 3 grid, matplotlib constrained layout):
(a) (left column, both rows) sky in orthographic coordinates about the LMC centre (lmcold.footprint.lmc_xy), x east [deg]
    (increasing to the left), y north [deg]: members (blue), stars in the SMC mask (orange), stars outside it but outside the
    distance window (yellow); orange curve: the mask edge; grey circle: radius 20 in these coordinates (the 20-deg catalogue
    edge lies at 19.6).
(b) x: angular distance from the SMC centre [deg], 1-deg bins; y: fraction of window stars with Gaia proper motions closer to the
    SMC systemic proper motion than to the LMC one, with binomial errors; vertical line: mask radius.
(c) x: dmu [mag]; y (log): normalised histograms outside the SMC mask in four rho ranges [deg]; grey band: the window, light
    orange bands: the sidebands SIDE.
(d) x: rho [deg] (bins RB); y (log): residual contamination of the window [per cent], with Poisson errors from the
    sideband counts.
(e) x: rho [deg]; y (log): RRab per bin with dmu (grey), removed by the SMC mask (orange), members (blue; window and outside the
    mask, before the rho < 10 deg cut).

results/rrab_membership.csv, per rho bin: rho_lo, rho_hi [deg], N (with dmu), masked, members (window, outside the mask), side
(sideband stars outside the mask), contam and contam_err (expected foreground in the window / members).

Run: .venv/bin/python scripts/fig_rrab_membership.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmcold.constants import LMC, SMC
from lmcold.footprint import DMU_WINDOW, SMC_MASK_RADIUS, angular_distance
from lmcold.footprint import lmc_xy
from lmcold.sources import DATA, ROOT

# %% [markdown]
# ### Settings
# The $\Delta\mu$ sidebands and the radial bins.

# %%
SIDE = (0.9, 2.0)  # sideband |dmu| range [mag], beyond the LMC distribution
RB = np.array([0, 2, 4, 6, 8, 10, 12, 14, 17, 20])  # rho bins [deg]


xy = lmc_xy  # (ra, dec) [deg] -> orthographic (x, y) [deg] about the LMC centre


# %% [markdown]
# ### Load the data
# The RR Lyrae catalogue `data/rrab.parquet` (`scripts/build_rrab.py`).

# %%
# one row per RRab; columns listed in the module docstring
t = pd.read_parquet(DATA / "rrab.parquet")

# %% [markdown]
# ### Window and sidebands
# Stars with a distance, their projected position and distance from the SMC, and the
# window and sideband flags.

# %%
# drop stars without a relative distance modulus
t = t[t.dmu.notna()]
t["x"], t["y"] = xy(t.ra, t.dec)  # orthographic coordinates [deg]
t["dS"] = angular_distance(t.ra, t.dec, SMC["ra0"], SMC["dec0"])  # angular distance from the SMC centre [deg]
win = t.dmu.abs() < DMU_WINDOW  # distance window, |dmu| < DMU_WINDOW = 0.6 mag (lmcold.footprint)
side = (t.dmu.abs() > SIDE[0]) & (t.dmu.abs() < SIDE[1])  # sidebands, 0.9 < |dmu| < 2.0 mag

# %% [markdown]
# ### Foreground in the window
# Per radial bin, the sideband counts scaled to the window width give the foreground left among
# the members; written to `results/rrab_membership.csv`.

# %%
rows = []
for lo, hi in zip(RB[:-1], RB[1:]):
    # stars in this rho bin, and those of them outside the SMC mask
    r = (t.rho >= lo) & (t.rho < hi)
    ok = r & ~t.smc_masked
    n_side = (ok & side).sum()
    # Sideband counts scaled by width: window 2 x 0.6 = 1.2 mag against sidebands 2 x (2.0 - 0.9) = 2.2 mag.
    expect = n_side * (2 * DMU_WINDOW) / (2 * (SIDE[1] - SIDE[0]))  # foreground expected inside the window
    # contam: expected foreground / members; contam_err: Poisson error of the sideband count propagated to contam
    # (max(..., 1) avoids division by zero in empty bins)
    rows.append(dict(rho_lo=lo, rho_hi=hi, N=r.sum(), masked=(r & t.smc_masked).sum(), members=(ok & win).sum(),
                     side=n_side, contam=expect / max((ok & win).sum(), 1), contam_err=np.sqrt(max(n_side, 1)) * expect /
                     max(n_side, 1) / max((ok & win).sum(), 1)))
res = pd.DataFrame(rows)
res.to_csv(ROOT / "results" / "rrab_membership.csv", index=False, float_format="%.4f")
print(res.to_string(index=False, float_format="%.3f"))

# %% [markdown]
# ### Panel (a): the sky
# Members, stars in the SMC mask and stars outside the distance window.

# %%
# 2 x 3 grid: panel (a) fills the left column
fig = plt.figure(figsize=(17, 9.5), constrained_layout=True)
g = fig.add_gridspec(2, 3)
a = fig.add_subplot(g[:, 0])
# per class: selection, colour, label, marker size
for sel, c, lab, s in ((~t.smc_masked & ~win, "#eda100", "outside the distance window", 2),
                       (t.smc_masked, "#eb6834", "SMC mask", 2), (t.member, "#2a78d6", "members", 0.5)):
    a.scatter(t.x[sel], t.y[sel], s=s, color=c, label=f"{lab} ({sel.sum()})", rasterized=True)
# grey circle of radius 20 deg in the orthographic coordinates
th = np.linspace(0, 2 * np.pi, 400)
a.plot(20 * np.sin(th), 20 * np.cos(th), color="0.3", lw=0.8)
ra_m, dec_m = [], []
# Same small-circle formula as lmcold.footprint.small_circle.
for p in th:  # mask edge: points 13 deg from the SMC centre
    # d: mask radius; r0, a0: SMC centre dec and ra [rad]; p: position angle on the circle [rad]
    d, r0, a0 = np.deg2rad(SMC_MASK_RADIUS), np.deg2rad(SMC["dec0"]), np.deg2rad(SMC["ra0"])
    dec = np.arcsin(np.sin(r0) * np.cos(d) + np.cos(r0) * np.sin(d) * np.cos(p))
    ra_m.append(np.rad2deg(a0 + np.arctan2(np.sin(p) * np.sin(d) * np.cos(r0), np.cos(d) - np.sin(r0) * np.sin(dec))))
    dec_m.append(np.rad2deg(dec))
# mask edge in the orthographic coordinates
mx, my = xy(np.array(ra_m), np.array(dec_m))
a.plot(mx, my, color="#eb6834", lw=1.5)
# x axis reversed: east to the left
a.set(xlim=(21, -21), ylim=(-21, 21), aspect="equal", xlabel="x east [deg]", ylabel="y north [deg]", title="(a) membership")
a.legend(frameon=False, fontsize=8, loc="upper right", markerscale=4)

# %% [markdown]
# ### Panel (b): the SMC mask radius
# Fraction of window stars with SMC-like proper motions against distance
# from the SMC centre.

# %%
a = fig.add_subplot(g[0, 1])
# SMC-like: proper motion closer (Euclidean, in mas/yr) to the SMC systemic proper motion than to the LMC one.
smclike = np.hypot(t.pmra - SMC["pmra0"], t.pmdec - SMC["pmdec0"]) < np.hypot(t.pmra - LMC["pmra0"], t.pmdec - LMC["pmdec0"])
# 1-deg bins of distance from the SMC centre [deg]
b = np.arange(0, 22.1, 1.0)
# bin index of each window star with a Gaia proper motion
i = np.digitize(t.dS[win & t.pmra.notna()], b)
# f: fraction SMC-like per bin; n: stars per bin
f = [smclike[win & t.pmra.notna()][i == k].mean() for k in range(1, len(b))]
n = np.array([(i == k).sum() for k in range(1, len(b))])
# binomial error sqrt(f (1 - f) / n)
a.errorbar(0.5 * (b[1:] + b[:-1]), f, yerr=np.sqrt(np.array(f) * (1 - np.array(f)) / np.maximum(n, 1)), fmt="o-", color="#2a78d6")
a.axvline(SMC_MASK_RADIUS, color="#eb6834", lw=1.5, label=f"mask radius {SMC_MASK_RADIUS:g} deg")
a.set(xlabel="distance from the SMC centre [deg]", ylabel="fraction with SMC-like PM", title="(b) setting the mask radius", ylim=(0, 1.05))
a.legend(frameon=False)

# %% [markdown]
# ### Panel (c): relative distance modulus
# Histograms of $\Delta\mu$ outside the SMC mask in four ranges of
# $\rho$, with the window and the sidebands.

# %%
a = fig.add_subplot(g[0, 2])
# dmu bins [mag], 0.05 mag wide
h = np.arange(-2.2, 2.21, 0.05)
# four rho ranges [deg], each with its colour
for (lo, hi), c in zip(((0, 6), (6, 10), (10, 14), (14, 20)), ("#2a78d6", "#1baf7a", "#eda100", "#eb6834")):
    sel = ~t.smc_masked & (t.rho >= lo) & (t.rho < hi)
    a.hist(t.dmu[sel], h, histtype="step", lw=1.8, color=c, density=True, label=f"rho {lo}-{hi} ({sel.sum()})")
a.axvspan(-DMU_WINDOW, DMU_WINDOW, color="0.85", zorder=0, label="window")
# sidebands on both sides of the window
for s in (1, -1):
    a.axvspan(s * SIDE[0], s * SIDE[1], color="#fbe3c4", zorder=0)
a.set(yscale="log", ylim=(1e-3, 5), xlabel="dmu [mag]", ylabel="density", title="(c) dmu outside the SMC mask (sidebands shaded)")
a.legend(frameon=False, fontsize=8)

# %% [markdown]
# ### Panels (d, e): foreground and counts
# The foreground left in the window, and the counts at each step, against $\rho$.

# %%
# centres of the rho bins [deg]
rc = 0.5 * (RB[1:] + RB[:-1])
a = fig.add_subplot(g[1, 1])
# contamination in per cent, with its Poisson error
a.errorbar(rc, 100 * res.contam, yerr=100 * res.contam_err, fmt="o-", color="#2a78d6")
a.set(xlabel="rho [deg]", ylabel="residual contamination in the window [%]", yscale="log", title="(d) contamination from sidebands")

a = fig.add_subplot(g[1, 2])
# counts per rho bin from the table above, drawn as steps
for col, c, lab in (("N", "0.4", "all with dmu"), ("masked", "#eb6834", "removed by the SMC mask"), ("members", "#2a78d6", "members")):
    a.step(RB, np.r_[res[col], res[col].iloc[-1]], where="post", color=c, lw=2, label=lab)
a.set(yscale="log", xlabel="rho [deg]", ylabel="RRab per bin", title="(e) counts")
a.legend(frameon=False, fontsize=8)
fig.savefig(ROOT / "figures" / "rrab_membership.png", dpi=150)
