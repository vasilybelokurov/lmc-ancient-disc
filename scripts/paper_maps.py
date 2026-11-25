"""Paper figure: the three samples as observed on the sky (RR Lyrae, APOGEE giants, BOSS giants). One row per sample; columns:
star counts, median [Fe/H], median proper motions (raw: bulk motion and perspective dominate), and the sample's own third
coordinate: relative distance modulus for the RR Lyrae, line-of-sight velocity for the giants. Orthographic projection about the
adopted LMC centre, used only as a reference point (no kinematic model enters). Overlays: the RR Lyrae science radius (10 deg) and
SMC mask edge; the 12 deg cone of the giant samples. One [Fe/H] scale for all rows: the 5th-95th percentiles of the giants'
[Fe/H]. Writes figures/paper_maps.png.

Inputs: data/rrab.parquet (sources.yaml release_rrab, scripts/build_rrab.py; RRab with member flag, adopted photometric feh, Gaia
pmra/pmdec and dmu, the relative distance modulus with dmu = 0 at the mean LMC distance), data/giants_apogee.parquet and
data/giants_boss.parquet (release_giants_apogee, release_giants_boss, scripts/build_giants.py; BOSS feh and vlos on the APOGEE
scale). RR Lyrae: members only (rho < 10 deg, outside the SMC mask, |dmu| < 0.6 mag); giants: all rows of the release catalogues.
Panels (3 rows x 5 columns, all square; rows: RR Lyrae, APOGEE giants, BOSS giants). Axes: orthographic x (east, increasing
to the left) and y (north) [deg] about lmcold.constants.LMC (ra0, dec0) (lmcold.footprint.lmc_xy), -12.5 to 12.5 deg, 0.5-deg
pixels. Columns: (1) stars per pixel, log scale 1-500; (2) median [Fe/H] [dex], shared scale; (3) median mu_alpha* [mas/yr],
1.3-2.5; (4) median mu_delta [mas/yr], -0.6-1.2; (5) median dmu [mag], -0.25 to 0.25, for the RR Lyrae, and median v_los
[km/s], vlos0 +- 60 km/s, for the giants, on one colour scale (bar with both axes). Medians are drawn only in pixels with >= 5
stars (lmcold.maps.pixel_map). Colour bars sit above the top row. Paper: Sect. 2.1, main text (docs/figures.md).
Run: .venv/bin/python scripts/paper_maps.py   (about 3 s on a laptop, docs/run_times.txt)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from lmcold.constants import LMC, SMC
from lmcold.footprint import RHO_MAX, SMC_MASK_RADIUS, lmc_xy, small_circle
from lmcold.layout import panel_grid
from lmcold.maps import pixel_map
from lmcold.sources import DATA, ROOT

# %% [markdown]
# ### Load the data
# The RR Lyrae catalogue `data/rrab.parquet` (built by `scripts/build_rrab.py`) and the two giant catalogues
# `data/giants_apogee.parquet` and `data/giants_boss.parquet` (built by `scripts/build_giants.py`), all from Zenodo bundle 1.

# %%
# data/rrab.parquet: RRab positions, member flag, photometric [Fe/H], Gaia proper motions, relative distance modulus dmu
t = pd.read_parquet(DATA / "rrab.parquet")
# data/giants_apogee.parquet, data/giants_boss.parquet: the giants with [Fe/H], v_los (BOSS on the APOGEE scale), proper motions
apogee, boss = (pd.read_parquet(DATA / f"giants_{s}.parquet") for s in ("apogee", "boss"))

# %% [markdown]
# ### Settings
# Panel size, pixel edges, the shared [Fe/H] colour scale (5th-95th percentiles of the giants), the colour columns and
# the guide circles (RR Lyrae science radius, SMC mask, the cone of the giants).

# %%
plt.style.use(ROOT / "src" / "lmcold" / "paper.mplstyle")
L = 12.5  # deg, half width of each panel
EDGES = np.arange(-L, L + 0.01, 0.5)  # deg, 0.5-deg pixel edges in x and y
FEH_LIM = tuple(np.percentile(np.r_[apogee.feh, boss.feh], [5, 95]))
COLUMNS = [  # (column, colour-bar label, colour map, limits)
    ("counts", "stars per pixel", "Blues", None),
    ("feh", "median [Fe/H]", "viridis", FEH_LIM),
    ("pmra", r"median $\mu_{\alpha*}$ [mas yr$^{-1}$]", "viridis", (1.3, 2.5)),
    ("pmdec", r"median $\mu_\delta$ [mas yr$^{-1}$]", "viridis", (-0.6, 1.2)),
]
# Third coordinate of each sample: (colour-bar label, colour map, limits); v_los limits: LMC vlos0 +- 60 km/s.
THIRD = {"dmu": (r"median $\Delta\mu$ [mag] (RR Lyrae)", "RdBu_r", (-0.25, 0.25)),
         "vlos": (r"median $v_{\rm los}$ [km s$^{-1}$]", "RdBu_r", (LMC["vlos0"] - 60, LMC["vlos0"] + 60))}
# Guide circles: (x, y) of a small circle of radius r [deg] about (ra0, dec0), with its plot keywords. RR Lyrae: the science
# radius RHO_MAX = 10 deg (dashed) and the 13-deg SMC mask edge (orange); giants: the 12-deg cone of the spectroscopic samples.
circle = lambda ra0, dec0, r, **kw: (lmc_xy(*small_circle(ra0, dec0, r)), kw)  # noqa: E731
rrl_guides = [circle(LMC["ra0"], LMC["dec0"], RHO_MAX, color="k", lw=0.6, ls="--"),
              circle(SMC["ra0"], SMC["dec0"], SMC_MASK_RADIUS, color="#eb6834", lw=0.8)]
giant_guides = [circle(LMC["ra0"], LMC["dec0"], 12.0, color="k", lw=0.6, ls="--")]

# %% [markdown]
# ### The three samples
# The RR Lyrae members and all giants of the two release catalogues.

# %%
SAMPLES = [("RR Lyrae", t[t.member], "dmu", rrl_guides), ("APOGEE giants", apogee, "vlos", giant_guides),
           ("BOSS giants", boss, "vlos", giant_guides)]
print(f"[Fe/H] scale {FEH_LIM[0]:.2f} to {FEH_LIM[1]:.2f}; N: " + ", ".join(f"{n} {len(s)}" for n, s, _, _ in SAMPLES))

# %% [markdown]
# ### The figure
# One row per sample: star counts, then pixel medians of [Fe/H], the two proper-motion components and the third
# coordinate of the sample (relative distance modulus or line-of-sight velocity).

# %%
fig, ax = panel_grid(len(SAMPLES), len(COLUMNS) + 1, left=0.45, hgap=0.1, vgap=0.1, top=0.9)
for i, (name, s, third, guides) in enumerate(SAMPLES):
    x, y = lmc_xy(s.ra.to_numpy(), s.dec.to_numpy())
    for j, (col, label, cmap, lim) in enumerate(COLUMNS + [(third, *THIRD[third])]):
        a = ax[i, j]
        if col == "counts":
            n, _, _ = np.histogram2d(x, y, [EDGES, EDGES])  # n[ix, iy]; transposed below for pcolormesh (rows = y)
            im = a.pcolormesh(EDGES, EDGES, np.where(n > 0, n, np.nan).T, cmap=cmap, norm=LogNorm(vmin=1, vmax=500), rasterized=True)
        else:
            im = a.pcolormesh(EDGES, EDGES, pixel_map(x, y, s[col].to_numpy(), EDGES).T, cmap=cmap, vmin=lim[0], vmax=lim[1],
                              rasterized=True)
        for (gx, gy), kw in guides:
            inside = (np.abs(gx) < L) & (np.abs(gy) < L)  # blank the parts of a circle outside the panel
            a.plot(np.where(inside, gx, np.nan), np.where(inside, gy, np.nan), **kw)
        a.set(xlim=(L, -L), ylim=(-L, L), aspect="equal", xticks=[5, 0, -5], yticks=[-5, 0, 5])
        a.set_xlabel(r"$x$ (east) [deg]" if i == len(SAMPLES) - 1 else "")
        if j == 0:
            a.set_ylabel(f"{name}\n" + r"$y$ (north) [deg]")
        else:
            a.tick_params(labelleft=False)
        if i < len(SAMPLES) - 1:
            a.tick_params(labelbottom=False)
        if i == 0:  # colour bar above each column, on the top row only
            cax = a.inset_axes([0.05, 1.2, 0.9, 0.05])
            fig.colorbar(im, cax=cax, orientation="horizontal")
            cax.xaxis.set_ticks_position("top")
            cax.tick_params(labelsize=6)
            cax.set_title(label, fontsize=7, pad=9)
            if col == "dmu":  # the same colours read as v_los for the giant rows: second scale under the bar
                (d0, d1), (v0, v1) = THIRD["dmu"][2], THIRD["vlos"][2]
                to_v = lambda d: v0 + (d - d0) * (v1 - v0) / (d1 - d0)  # noqa: E731
                to_d = lambda v: d0 + (v - v0) * (d1 - d0) / (v1 - v0)  # noqa: E731
                sec = cax.secondary_xaxis("bottom", functions=(to_v, to_d))
                sec.tick_params(labelsize=6, pad=1)
                sec.set_xlabel(THIRD["vlos"][0] + " (giants)", fontsize=6, labelpad=1)
fig.savefig(ROOT / "figures" / "paper_maps.png")
