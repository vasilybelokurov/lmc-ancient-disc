"""Does the BOSS parent lose stars to dust as the mass conversion assumes? Infrared test in the central LMC against 2MASS.

Cells are HEALPix nside=1024 pixels (3.4 arcmin, the finest box of the Skowron et al. 2021 map) within 4.5 deg of the centre and
inside the map (SFD saturates in the LMC, so no fallback):
  N_G: parent stars of the re-implemented carton (data/boss_parent.parquet, eligible);
  N_H: 2MASS stars in the APOGEE giant polygon with 12.35 <= H <= 13.3 (data/twomass_centre.parquet); A_H = 0.20 A_G.
The model ratio f = n_G / n_H per Msun formed is from lmcold.isochrones.count_grid for a fixed population (ages 2-12 Gyr equal
mass, [M/H] = -1.0, -0.7, -0.4) on a grid of true and carton E(B-V). Each cell is predicted twice:
  'cell':   f at the cell's own reddening (Skowron E(V-I), its SFD-based carton reddening);
  'region': f at the median reddening of its conversion region (results/mass_regions.csv), as build_giant_mass.py does.
In bins of cell E(V-I), q = sum(N_G) / sum(N_H f), normalised to cells with E(V-I) < 0.1. A complete parent with the right dust model
gives q_cell = 1 at all E(V-I). Errors: bootstrap over cells. Per region, sum(N_H f_cell) / sum(N_H f_region) is the parent fraction
the cell-by-cell model expects relative to the regional median: the regional mass is high by its inverse.
Run build_giant_mass.py first. Writes figures/paper_parent_extinction.png, results/parent_extinction.csv (per E(V-I) bin) and data/parent_extinction_cells.parquet.

Inputs: data/boss_parent.parquet (release_boss_parent; eligible stars, SFD E(B-V) per star), data/twomass_centre.parquet
(sources.yaml twomass_centre), Skowron et al. (2021) map (skowron21_table1 via lmcold.extinction.rc_map), PARSEC isochrones
(parsec_gaiaEDR3, parsec_2mass_spitzer), results/mass_regions.csv (scripts/build_giant_mass.py: median true and carton E(B-V)
of each of the 16 conversion regions, index 4 x ring + quadrant).
Model population: every isochrone of the grid with [M/H] in MH and age 2-12 Gyr counts once (the grid is uniform in log age,
step 0.05 dex), i.e. n_box_raw summed over those isochrones.
Outputs:
- data/parent_extinction_cells.parquet, one row per cell: pix (HEALPix nside 1024, ring order), n_g, n_h, ra, dec [deg],
  region, evi [mag], f_cell, f_region.
- results/parent_extinction.csv, one row per model (cell, region) and E(V-I) bin: evi_lo, evi_hi, n_cells, n_g, q, q_p16, q_p84.
Panels (1 x 2):
(a) x (east, increasing to the left), y (north) [deg] about the LMC centre, +-4.5 deg; cells coloured by Skowron E(V-I) [mag],
    0-0.4, colour bar above.
(b) x: E(V-I) of the cell [mag], 0-0.6 (bin centres, offset +-0.004 by model); y: q (normalised observed / predicted parent
    count), 0-1.6; blue: model at each cell's reddening, orange: model at the region median; bars: bootstrap 16th-84th
    percentiles; bins with >= 50 parent stars.
Placement: appendix (Sect. 2.3; docs/figures.md). Run: .venv/bin/python scripts/paper_parent_extinction.py
Run time: 27 s on a MacBook (M-series), docs/run_times.txt.
"""
import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from lmcold import apogee_target, magcloud
from lmcold.constants import LMC
from lmcold.extinction import EVI_PER_EBV, rc_map, true_ebv
from lmcold.footprint import angular_distance, lmc_xy
from lmcold.isochrones import count_grid
from lmcold.layout import panel_grid
from lmcold.sources import DATA, ROOT, path

# %% [markdown]
# ### Settings and the model ratio
# Cells, tested area, the fixed isochrone population, and the model ratio f = n_G / n_H as a function of the true and the
# carton reddening.

# %%
NSIDE, RMAX = 1024, 4.5  # HEALPix resolution of the cells; radius [deg] of the tested area
RINGS = [0, 3, 6, 9, 12]  # deg, ring edges of the conversion regions (as in build_giant_mass.py)
EBV = np.arange(0, 1.01, .025)  # mag, grid of true E(B-V) for the model ratio
EBV_CARTON = np.array([0, .01, .02, .03, .05, .08, .12, .2, .35, .6, 1.])  # mag, grid of the carton's E(B-V)
MH, AGES = [-1.0, -0.7, -0.4], (2., 12.)  # isochrone [M/H] and age range [Gyr] of the fixed model population
EVI_EDGES = np.array([0, .05, .1, .15, .2, .25, .3, .4, .5, .8])  # mag, bins of cell E(V-I)


def model_ratio():
    """f(E(B-V), carton E(B-V)) = parent-box / H-box giants per Msun, summed over the fixed population.

    Returns
    -------
    scipy.interpolate.RegularGridInterpolator
        Linear interpolator of the ratio on the (EBV, EBV_CARTON) grid [mag]; called with an (N, 2) array of (true E(B-V),
        carton E(B-V)). The 2MASS count depends on the true E(B-V) only (A_H, E(J-Ks) from count_grid).
    """
    def pick(g):
        """Rows of isochrone grid g with [M/H] in MH and age (from logAge, Gyr) within AGES."""
        age = 10**g.logAge / 1e9
        return g[np.isin(g.MH.round(2), MH) & (age >= AGES[0]) & (age <= AGES[1])]
    # PARSEC grids of the fixed population: Gaia EDR3 bands (BOSS parent box) and 2MASS bands (H box).
    boss, twomass = pick(pd.read_parquet(path('parsec_gaiaEDR3'))), pick(pd.read_parquet(path('parsec_2mass_spitzer')))
    # Giants per Msun in the H box at each true E(B-V), shape (len(EBV),); in the parent box on the (EBV, EBV_CARTON) grid.
    h = np.array([count_grid(twomass, 'APOGEE', e, e, 0.).n_box_raw.sum() for e in EBV])
    g = np.array([[count_grid(boss, 'BOSS', e, c, 0.).n_box_raw.sum() for c in EBV_CARTON] for e in EBV])
    return RegularGridInterpolator((EBV, EBV_CARTON), g / h[:, None])


def xyz(ra, dec):
    """Unit vectors, shape (N, 3), of positions ra, dec [deg]."""
    r, d = np.radians(ra), np.radians(dec)
    return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])


# %% [markdown]
# ### Load the data
# The BOSS parent `data/boss_parent.parquet`, the 2MASS stars of the LMC centre `data/twomass_centre.parquet`, and the median
# reddening of each conversion region from `results/mass_regions.csv` (written by `scripts/build_giant_mass.py`). Two further
# inputs are read where they are used: the PARSEC isochrone grids inside `model_ratio()` above, and the Skowron et al. (2021)
# red-clump reddening map by `lmcold.extinction.rc_map()` in the next cell.

# %%
# data/boss_parent.parquet: the BOSS parent (eligible stars kept) with ra, dec [deg] and SFD E(B-V) (column ebv)
p = pd.read_parquet(DATA / 'boss_parent.parquet').query('eligible')
# data/twomass_centre.parquet: 2MASS stars of the LMC centre with J, H, Ks magnitudes (j_m, h_m, k_m)
t = pd.read_parquet(DATA / 'twomass_centre.parquet')
# results/mass_regions.csv: per conversion region the median true and carton E(B-V)
reg = pd.read_csv(ROOT / 'results/mass_regions.csv', index_col='region')

# %% [markdown]
# ### Counts per cell
# Parent stars and 2MASS giants per cell, the conversion region and reddening of each cell, and the model ratio at the
# cell's reddening and at its region's median.

# %%
# 2MASS stars in the APOGEE giant colour-magnitude polygon (maglim 14.5) with H <= 13.3.
t = t[(apogee_target.rgb_class((t.j_m - t.k_m).to_numpy(), t.h_m.to_numpy(), 14.5) > 0) & (t.h_m <= 13.3)]
# HEALPix pixels (ring order) within RMAX of the LMC centre: the cells.
cells = hp.query_disc(NSIDE, hp.ang2vec(LMC['ra0'], LMC['dec0'], lonlat=True), np.radians(RMAX))
count = lambda s: np.bincount(hp.ang2pix(NSIDE, s.ra.to_numpy(), s.dec.to_numpy(), lonlat=True), minlength=hp.nside2npix(NSIDE))[cells]  # noqa: E731
d = pd.DataFrame(dict(pix=cells, n_g=count(p), n_h=count(t)))   # parent and 2MASS counts per cell
d['ra'], d['dec'] = hp.pix2ang(NSIDE, cells, lonlat=True)        # cell centres [deg]
# Distance from the centre [deg] and tangent-plane x (east), y (north) [deg] of each cell.
rho = angular_distance(d.ra.to_numpy(), d.dec.to_numpy(), LMC['ra0'], LMC['dec0'])
x, y = lmc_xy(d.ra.to_numpy(), d.dec.to_numpy())
# Conversion region as in build_giant_mass.py: 4 x ring + (x >= 0) + 2 (y >= 0); x east, y north.
d['region'] = 4 * np.clip(np.digitize(rho, RINGS) - 1, 0, 3) + (x >= 0) + 2 * (y >= 0)
sfd = p.ebv.to_numpy()[cKDTree(xyz(p.ra.to_numpy(), p.dec.to_numpy())).query(xyz(d.ra.to_numpy(), d.dec.to_numpy()))[1]]  # nearest star
ebv = true_ebv(d.ra.to_numpy(), d.dec.to_numpy(), np.full(len(d), np.nan), rc_map())  # NaN outside the RC map
inside = np.isfinite(ebv)   # cells inside the Skowron map (SFD saturates in the LMC, so no fallback)
print(f'{(~inside).sum()} of {len(d)} cells outside the RC map, dropped')
d, sfd, ebv, x, y = d[inside].reset_index(drop=True), sfd[inside], ebv[inside], x[inside], y[inside]
d['evi'] = ebv * EVI_PER_EBV  # back to Skowron E(V-I) [mag]
f = model_ratio()   # interpolator of n_G / n_H on (true E(B-V), carton E(B-V))
# Model ratio at the cell's own true and carton E(B-V), and at the median values of its conversion region.
d['f_cell'] = f(np.column_stack([ebv, magcloud.carton_ebv(d.ra, d.dec, sfd)]))
d['f_region'] = f(np.column_stack([reg.ebv.loc[d.region], reg.ebv_carton.loc[d.region]]))
# One row per cell (columns in the module docstring).
d.to_parquet(DATA / 'parent_extinction_cells.parquet', index=False)
binned_path = ROOT / 'results/parent_extinction.csv'

# %% [markdown]
# ### Observed over predicted, per E(V-I) bin
# Normalised to the cells with E(V-I) < 0.1, with a bootstrap over cells; written to `results/parent_extinction.csv`.

# %%
rng = np.random.default_rng(3)   # fixed seed of the bootstrap over cells
d['bin'] = np.digitize(d.evi, EVI_EDGES) - 1  # E(V-I) bin; len(EVI_EDGES) - 1 for evi >= 0.8 (not reported)


def q(s, col):
    """sum N_G / sum N_H f per E(V-I) bin, normalised to E(V-I) < 0.1.

    Parameters
    ----------
    s : pandas.DataFrame
        Cells with bin, evi, n_g, n_h and f_cell, f_region.
    col : str
        Model: 'cell' or 'region'.

    Returns
    -------
    numpy.ndarray
        q per bin, length len(EVI_EDGES) (the last entry is the overflow bin); NaN where a bin is empty.
    """
    # Observed parent counts and predicted counts N_H f summed per E(V-I) bin (overflow bin included).
    num, den = np.bincount(s.bin, s.n_g, len(EVI_EDGES)), np.bincount(s.bin, s.n_h * s['f_' + col], len(EVI_EDGES))
    low = s.evi < .1   # normalisation cells
    # Ratio per bin divided by the same ratio over the low-reddening cells.
    return num / den / (s.n_g[low].sum() / (s.n_h * s['f_' + col])[low].sum())


rows = []
# For each model: q per bin and the 16th-84th percentiles over the bootstrap draws; the overflow bin (evi >= 0.8) is not reported.
for col in ('cell', 'region'):
    boot = np.array([q(d.iloc[rng.integers(0, len(d), len(d))], col) for _ in range(200)])  # 200 bootstrap draws of cells
    for j, (value, lo, hi) in enumerate(zip(q(d, col), *np.nanpercentile(boot, [16, 84], axis=0))):
        if j < len(EVI_EDGES) - 1:
            rows.append(dict(model=col, evi_lo=EVI_EDGES[j], evi_hi=EVI_EDGES[j + 1], n_cells=int((d.bin == j).sum()),
                             n_g=int(d.n_g[d.bin == j].sum()), q=value, q_p16=lo, q_p84=hi))
# Table per model and E(V-I) bin, written to results/parent_extinction.csv.
binned = pd.DataFrame(rows)
binned.to_csv(binned_path, index=False)
print(binned.round(3).to_string(index=False))
# Per region: median and 99th percentile of cell E(V-I), and the parent fraction the cell-by-cell model expects relative to the
# regional-median model.
per_region = d.groupby('region').apply(lambda s: pd.Series(dict(
    evi_median=s.evi.median(), evi_p99=s.evi.quantile(.99),
    expected_cell_over_region=(s.n_h * s.f_cell).sum() / (s.n_h * s.f_region).sum())), include_groups=False)
print(per_region.round(3).to_string())

# %% [markdown]
# ### The figure

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
fig, axes = panel_grid(1, 2, width=7, left=.55, right=.1, hgap=.7, bottom=.42, top=.6)
# (a) Cells on the sky coloured by Skowron E(V-I) [mag], 0-0.4; east to the left; colour bar above the panel.
sc = axes[0, 0].scatter(x, y, c=d.evi, s=.6, marker='s', cmap='magma_r', vmin=0, vmax=.4, lw=0, rasterized=True)
axes[0, 0].set(xlim=(RMAX, -RMAX), ylim=(-RMAX, RMAX), xlabel='$x$ (east) [deg]', ylabel='$y$ (north) [deg]')
axes[0, 0].set_title('(a) Skowron+21 $E(V-I)$ in 3.4$^\\prime$ cells', pad=24)
fig.colorbar(sc, cax=axes[0, 0].inset_axes([0, 1.03, 1, .04]), orientation='horizontal').ax.xaxis.set_ticks_position('top')
# (b) q against cell E(V-I) for the two models (offset -+0.004 mag in x), bins with >= 50 parent stars only.
ax = axes[0, 1]
for (col, lab), color, dx in zip([('cell', 'model at the cell $E(V-I)$'), ('region', 'model at the region median')], ['#2878b5', '#d75a20'], [-.004, .004]):
    b = binned[(binned.model == col) & (binned.n_g >= 50)]
    c = (b.evi_lo + b.evi_hi) / 2 + dx
    ax.errorbar(c, b.q, [b.q - b.q_p16, b.q_p84 - b.q], fmt='o-', ms=3, color=color, label=lab)
ax.axhline(1, color='0.3', ls='--', lw=.7)   # q = 1: complete parent with the right dust model
ax.set(xlabel=r'$E(V-I)$ of the cell', ylabel=r'$\sum N_G\,/\,\sum N_H f$, normalised', xlim=(0, .6), ylim=(0, 1.6))
ax.set_title('(b) Gaia parent against 2MASS giants', pad=24)
ax.legend(loc='lower left')
fig.savefig(ROOT / 'figures/paper_parent_extinction.png')
print('Saved paper_parent_extinction.png')
