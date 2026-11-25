"""Diagnostic: Milky Way foreground of the BOSS giants against radius (§2.3).

(a) Foreground fraction of the model (lmcold.foreground; data/giant_foreground.parquet from build_giant_mass.py), weighted
mean per 1-deg ring, for three metallicity ranges.
(b) Test of its assumption, a uniform foreground surface density: weighted surface density of BOSS giants whose proper motion
(Magellanic Stream frame) lies more than SIDE mas/yr from the local LMC motion (median of the [Fe/H] > -1 giants in the same
nside-16 pixel). The carton keeps only stars within 1.2 mas/yr of a fixed LMC motion, so these sideband stars are a fixed
fraction of the foreground only if its proper-motion distribution does not change across the field; they measure its radial
shape, not its amount. Poisson errors from the raw counts.
Writes figures/fig_giant_foreground.png and results/giant_foreground_sideband.csv.

Paper figure: appendix (Sect. 2.3), docs/figures.md row figures/fig_giant_foreground.png.

Inputs: the weighted BOSS giants of lmcold.mass_samples.samples (release_boss_weights and the files listed there);
data/giant_foreground.parquet (rho [deg], coverage fill, model foreground fraction f_mw; scripts/build_giant_mass.py);
data/boss_parent.parquet (release_boss_parent: Gaia DR3 pmra, pmdec).

Figure (two equal square panels):
(a) x: angular distance rho from the LMC centre [deg], 1-deg rings (lmcold.foreground.RINGS) plotted at their centres; y (log):
    model foreground fraction, the mean of f_mw weighted by w x fill. Orange: [Fe/H] < -1.5; green: -1.5 to -1; blue: > -1.
(b) x: rho [deg] in the rings EDGES (horizontal bars: ring width); y: surface density of sideband giants with [Fe/H] < -1.5,
    sum of w x fill per deg^2, with Poisson errors (density / sqrt(raw count)). Dashed line: the inverse-variance weighted mean
    (the uniform-density hypothesis); its chi2 is printed.

results/giant_foreground_sideband.csv: rho_lo, rho_hi [deg], n_raw (sideband giants), density [deg^-2], err [deg^-2].

Run: .venv/bin/python scripts/fig_giant_foreground.py
"""
import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmcold import magcloud
from lmcold.foreground import RINGS
from lmcold.layout import panel_grid
from lmcold.mass_samples import samples
from lmcold.sources import DATA, ROOT

# %% [markdown]
# ### Settings

# %%
SIDE = 0.8  # mas/yr: minimum distance from the local LMC proper motion (MS frame) for a sideband star
EDGES = [0, 3, 6, 9, 12]  # rings of panel (b) [deg]

# %% [markdown]
# ### Load the data
# The BOSS mass sample through `lmcold.mass_samples.samples()` (built from the catalogues of Zenodo bundle 1), joined on Gaia
# source_id with the foreground fraction of each giant (`data/giant_foreground.parquet`, written by `scripts/build_giant_mass.py`)
# and with its Gaia proper motion (`data/boss_parent.parquet`, Zenodo bundle 1).

# %%
# lmcold.mass_samples.samples(): BOSS giants with calibrated [Fe/H] and selection weight w
b, _, _ = samples()
# data/giant_foreground.parquet: rho [deg], coverage fill and Milky Way foreground fraction f_mw of each BOSS giant
b = b.merge(pd.read_parquet(DATA / "giant_foreground.parquet")[["source_id", "rho", "fill", "f_mw"]], on="source_id")
# data/boss_parent.parquet: Gaia proper motions pmra, pmdec [mas/yr]
b = b.merge(pd.read_parquet(DATA / "boss_parent.parquet", columns=["source_id", "pmra", "pmdec"]), on="source_id")

# %% [markdown]
# ### The sideband sample
# BOSS giants with their model foreground fraction, and their proper motion relative to the local LMC motion.

# %%
# Proper motions in the Magellanic Stream frame (Nidever et al. 2008): pm_L cos B, pm_B [mas/yr].
_, _, b["pl"], b["pb"] = magcloud.ms_coordinates(b.ra.to_numpy(), b.dec.to_numpy(), b.pmra.to_numpy(), b.pmdec.to_numpy())
# Local LMC motion: median MS-frame proper motion of the [Fe/H] > -1 giants in the same HEALPix nside-16 (NESTED, ~3.7 deg)
# pixel; pixels without such giants give NaN (those stars are dropped from panel b).
b["pix"] = hp.ang2pix(16, b.ra.to_numpy(), b.dec.to_numpy(), lonlat=True, nest=True)
b = b.join(b[b.feh > -1].groupby("pix")[["pl", "pb"]].median(), on="pix", rsuffix="_lmc")
b["side"] = np.hypot(b.pl - b.pl_lmc, b.pb - b.pb_lmc) > SIDE

# %% [markdown]
# ### Panel (a): model fraction per ring; panel (b): sideband density
# The sideband densities go to `results/giant_foreground_sideband.csv`.

# %%
plt.style.use(ROOT / "src/lmcold/paper.mplstyle")
fig, ax = panel_grid(1, 2, width=5.0, left=0.6, right=0.1, hgap=0.7, bottom=0.45, top=0.55)
cols = {"[Fe/H] < −1.5": ("#d75a20", b.feh < -1.5), "−1.5 to −1": ("#33976b", (b.feh >= -1.5) & (b.feh < -1)),
        "> −1": ("#2878b5", b.feh >= -1)}
centre = 0.5 * (RINGS[1:] + RINGS[:-1])
for lab, (col, m) in cols.items():
    # Weighted mean of the per-star model fraction f_mw in each ring, weights w x fill.
    k = np.digitize(b.rho[m], RINGS) - 1
    w = (b.w * b["fill"])[m].to_numpy()
    ax[0, 0].plot(centre, np.bincount(k, w * b.f_mw[m], len(centre)) / np.bincount(k, w, len(centre)), "o-", ms=3, lw=0.8,
                  color=col, label=lab)
ax[0, 0].set(yscale="log", ylim=(1e-3, 1), xlim=(0, 12), xlabel=r"$\rho$ [deg]", ylabel="foreground fraction (model)", title="(a) Model")
ax[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.12), borderaxespad=0, ncol=3)
# Solid angle of each ring [deg^2]: 2 pi (cos rho_in - cos rho_out) sr x (180/pi)^2.
area = 2 * np.pi * -np.diff(np.cos(np.radians(EDGES))) * (180 / np.pi) ** 2
mp = b[(b.feh < -1.5) & b.side & b.pl_lmc.notna()]
k = np.digitize(mp.rho, EDGES) - 1
dens = np.bincount(k, (mp.w * mp["fill"]).to_numpy(), len(area)) / area
n = np.bincount(k, minlength=len(area))
err = dens / np.sqrt(np.maximum(n, 1))
c = 0.5 * (np.array(EDGES[1:]) + np.array(EDGES[:-1]))
ax[0, 1].errorbar(c, dens, yerr=err, xerr=1.5, fmt="o", ms=3, color="#d75a20", lw=0.8)
mean = np.sum(dens / err ** 2) / np.sum(1 / err ** 2)
ax[0, 1].axhline(mean, color="0.4", ls="--", lw=0.7)
ax[0, 1].set(xlim=(0, 12), ylim=(0, None), xlabel=r"$\rho$ [deg]", ylabel=r"sideband giants per deg$^2$",
             title=r"(b) [Fe/H] $<-1.5$, $|\Delta\mu| > 0.8$ mas yr$^{-1}$")
fig.savefig(ROOT / "figures/fig_giant_foreground.png")
t = pd.DataFrame(dict(rho_lo=EDGES[:-1], rho_hi=EDGES[1:], n_raw=n, density=dens, err=err))
t.to_csv(ROOT / "results/giant_foreground_sideband.csv", index=False)
chi2 = float(np.sum(((dens - mean) / err) ** 2))
print(t.round(3).to_string(index=False), f"\nchi2 about a constant {chi2:.1f} for {len(dens) - 1} dof")
