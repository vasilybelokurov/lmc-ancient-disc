"""Are the BOSS line-of-sight velocity errors right? Test on the giants observed by both surveys.

For the 1,926 stars in both release catalogues (data/giants_apogee.parquet, data/giants_boss.parquet; BOSS velocities calibrated
to APOGEE by scripts/build_giants.py), z = (v_BOSS - v_APOGEE) / sqrt(e_BOSS^2 + e_APOGEE^2). If the quoted errors are right, the
robust standard deviation of z (1.4826 x median absolute deviation) is 1. APOGEE errors (median 0.04 km/s) are negligible, so this
tests the BOSS errors. Binaries and real velocity changes between the epochs only widen z.

Writes figures/fig_boss_velocity_errors.png: three equal panels against BOSS S/N, G and Teff bins (bins with >= 20 stars):
robust sd of z (filled; 1 = errors right), and robust sd of v_BOSS - v_APOGEE [km/s] next to the median quoted BOSS error [km/s]
(open). Prints the numbers. Diagnostic figure (not in the paper). Limitation: the overlap has few stars at G > 17.1 (BOSS sample
extends to 17.5).

The bins of BINS are read from this file's source by scripts/paper_text_kin_support.py, which recomputes the robust sd of z per bin
for the KB page 'Rotational support, centre and sensitivity' (section 'Metallicity errors and the two surveys'); keep BINS a plain
literal.

Inputs: the release giant catalogues data/giants_apogee.parquet and data/giants_boss.parquet (keys release_giants_apogee and
release_giants_boss of data/sources.yaml; Zenodo bundle 1).

Run: .venv/bin/python scripts/fig_boss_velocity_errors.py   (run time not recorded)
"""
# %% [markdown]
# ### Settings
# The bins of BOSS signal-to-noise ratio, *Gaia* $G$ magnitude and BOSS effective temperature in which the test is repeated, and
# their axis labels.

# %%
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmcold.layout import panel_grid
from lmcold.sources import ROOT, path

# Bin edges per column of the merged table (suffix _b: the BOSS row): S/N, G [mag], Teff [K].
BINS = dict(snr_b=[16, 20, 25, 30, 40, 60, 200], G_b=[12, 15.5, 16.25, 16.75, 17.1, 17.5], teff_b=[3500, 4000, 4300, 4500, 5500])
# Axis label of each binning column.
LABELS = dict(snr_b='BOSS S/N', G_b='G [mag]', teff_b=r'BOSS $T_{\rm eff}$ [K]')

# %% [markdown]
# ### Load the data
# The two release giant catalogues, joined on the *Gaia* `source_id`: one row per star observed by both surveys, with the APOGEE
# columns suffixed `_a` and the BOSS columns `_b`.

# %%
m = pd.read_parquet(path('release_giants_apogee')).merge(pd.read_parquet(path('release_giants_boss')), on='source_id', suffixes=('_a', '_b'))

# %% [markdown]
# ### Normalised velocity differences
# The difference of the two line-of-sight velocities $\Delta v$ (km/s) and its ratio $z$ to the combined quoted error, with the
# robust standard deviation of each over all common stars.

# %%
dv = m.vlos_b - m.vlos_a  # km/s, BOSS minus APOGEE
z = dv / np.hypot(m.evlos_a, m.evlos_b)  # dimensionless; unit spread if the quoted errors [km/s] are right
rsd = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))  # noqa: E731  (robust standard deviation)
print(f'N {len(m)}: robust sd z {rsd(z):.2f}, of dv {rsd(dv):.2f} km/s; median BOSS error {m.evlos_b.median():.2f} km/s')

# %% [markdown]
# ### The figure
# One panel per binning column: the robust sd of $z$ in each bin with at least 20 stars (left axis), and the robust sd of
# $\Delta v$ and the median quoted BOSS error in km/s (right axis).

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
# Three equal panels in one row (sizes in inches), one per entry of BINS.
fig, ax = panel_grid(1, 3, width=7.0, left=0.5, right=0.5, hgap=0.85, top=0.35, bottom=0.45)
for a, (col, edges) in zip(ax[0], BINS.items()):
    rows = []  # per bin: centre, robust sd of z, robust sd of dv [km/s], median BOSS error [km/s], N
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (m[col] >= lo) & (m[col] < hi)  # stars in this bin
        if k.sum() >= 20:  # bins with fewer stars are skipped
            rows.append(((lo + hi) / 2, rsd(z[k]), rsd(dv[k]), m.evlos_b[k].median(), k.sum()))
            print(f'{col} {lo}-{hi}: N {k.sum()}, sd z {rows[-1][1]:.2f}, sd dv {rows[-1][2]:.2f}, median e_BOSS {rows[-1][3]:.2f}')
    # Left axis: robust sd of z per bin, with the line z sd = 1 (errors right).
    x, sz, sdv, ev, _ = np.array(rows).T
    a.plot(x, sz, 'o-', color='k', ms=4, label='robust sd of z')
    a.axhline(1, color='0.6', lw=0.5)
    a.set(xlabel=LABELS[col], ylabel='robust sd of z', ylim=(0, 1.5))
    # Right axis [km/s]: robust sd of dv (open squares) and median quoted BOSS error (open triangles).
    b = a.twinx()
    b.plot(x, sdv, 's--', color='C0', mfc='none', ms=4, label=r'sd($v_{\rm BOSS}-v_{\rm APOGEE}$)')
    b.plot(x, ev, '^:', color='C3', mfc='none', ms=4, label='median BOSS error')
    b.set(ylabel='km/s', ylim=(0, 7))
    # One legend for both axes, above the first panel.
    if col == 'snr_b':
        h = a.get_legend_handles_labels()[0] + b.get_legend_handles_labels()[0]
        a.legend(h, [x.get_label() for x in h], loc='lower left', bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=5)
fig.savefig(ROOT / 'figures/fig_boss_velocity_errors.png', dpi=200)
