"""Paper figure for Sect. 2.1.2: the BOSS -> APOGEE calibration (Eqs 1, 2) on the stars observed by both surveys.

Run build_giants.py first (writes data/giants_overlap.parquet and results/boss_calibration.json).
(a) raw BOSSNet [Fe/H] against APOGEE [Fe/H], with the fitted relation at the pivot (t = s = 0) and the one-to-one line;
(b) calibrated BOSS minus APOGEE [Fe/H] against APOGEE [Fe/H]; (c) BOSS minus APOGEE velocity against BOSS Teff, with Eq. 1.
Lines and bands in (b, c): median and 16th-84th percentiles per bin. In-sample: all stars of the overlap enter the fit.

Inputs (both derived, not in data/sources.yaml; made by scripts/build_giants.py from data/spec_giants.parquet and
data/gaia_giants.parquet):
- data/giants_overlap.parquet: giants with good spectra in both APOGEE DR17 and BOSS DR20 (same Gaia DR3 source_id); columns
  source_id, teff_b [K], snr_b, v_b [km/s], feh_b (raw BOSSNet) from BOSS and v_a [km/s], feh_a from APOGEE (ASPCAP).
- results/boss_calibration.json, key 'calibration': coefficients d0, d1 (velocity, Eq. 1) and c0-c3 ([Fe/H], Eq. 2) of
  lmcold.giants.fit_calibration, fitted on all overlap stars.
Method: lmcold.giants.calibrate inverts Eq. 2 to put each BOSS [Fe/H] on the APOGEE scale; with t = (Teff_BOSS - 4200 K)/1000 K
and s = 20/(S/N)_BOSS - 1 the relation is [Fe/H]_BOSS = c0 + c1 [Fe/H]_APOGEE + c2 t + c3 s, and v_BOSS - v_APOGEE = d0 + d1 t.

Panels (1 x 3, equal square panels; points: individual overlap stars):
(a) x: APOGEE [Fe/H] [dex], y: raw BOSSNet [Fe/H] [dex], both -2.7 to 0.4; orange line Eq. 2 at t = s = 0 (c0 + c1 x),
    dashed one-to-one line.
(b) x: APOGEE [Fe/H] [dex], y: calibrated BOSS minus APOGEE [Fe/H] [dex], -1 to 1; running median and 16th-84th percentile
    band in 0.25-dex bins from -2.5 to 0.5 with >= 10 stars.
(c) x: BOSS Teff [K], 3600-5500, y: v_BOSS - v_APOGEE [km/s], -15 to 20; running median and band in 150-K bins from 3700 K;
    orange line Eq. 1 (d0 + d1 t).
Output: figures/paper_boss_calibration.png (appendix B, see docs/figures.md).
Run: .venv/bin/python scripts/paper_boss_calibration.py
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmcold import giants
from lmcold.layout import panel_grid
from lmcold.sources import DATA, ROOT

# %% [markdown]
# ### Load the data
# The stars observed by both surveys, `data/giants_overlap.parquet` (Zenodo bundle 1), and the fitted coefficients of
# `results/boss_calibration.json` (both written by `scripts/build_giants.py`).

# %%
# data/giants_overlap.parquet: stars with both spectra; BOSS (_b) and APOGEE (_a) [Fe/H], velocity, BOSS Teff and S/N
m = pd.read_parquet(DATA / 'giants_overlap.parquet')
# results/boss_calibration.json: coefficients c0, c1, ... (Eq. 2, [Fe/H]) and d0, d1 (Eq. 1, velocity) of the BOSS calibration
cal = json.load(open(ROOT / 'results/boss_calibration.json'))['calibration']

# %% [markdown]
# ### The stars observed by both surveys
# The BOSS [Fe/H] is put on the APOGEE scale.

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
# calibrate returns (velocity, [Fe/H]) on the APOGEE scale; only the calibrated [Fe/H] is kept (panel b).
_, m['feh_cal'] = giants.calibrate(cal, m.teff_b, m.snr_b, m.v_b, m.feh_b)
# Colours: blue for the running percentiles, orange for the fitted relations
blue, orange = '#2878b5', '#d75a20'


# %% [markdown]
# ### Running percentiles
# `running` draws the median and the 16th-84th percentile band of a quantity in bins, used in panels (b) and (c).

# %%
def running(ax, x, y, edges, nmin=10):
    """Median and 16th-84th percentiles of y in bins of x with at least nmin stars.

    Draws the percentile band and the median line on `ax` at the bin centres; bins with fewer than `nmin` stars are skipped.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Panel to draw on.
    x, y : numpy.ndarray
        Binned quantity and the quantity summarised (same units as the panel axes).
    edges : numpy.ndarray
        Bin edges in x.
    nmin : int, optional
        Minimum number of stars for a bin to be drawn.

    Returns
    -------
    None
    """
    i = np.digitize(x, edges) - 1  # bin index; -1 or len(edges)-1 for stars outside the edges, which no bin selects
    # Per bin with >= nmin stars: (bin centre, 16th, 50th, 84th percentiles of y)
    rows = [(0.5 * (edges[k] + edges[k + 1]), *np.percentile(y[i == k], [16, 50, 84])) for k in range(len(edges) - 1)
            if (i == k).sum() >= nmin]
    # Band between the 16th and 84th percentiles and the median line
    c, lo, med, hi = np.array(rows).T
    ax.fill_between(c, lo, hi, color=blue, alpha=.2, lw=0)
    ax.plot(c, med, '-', color=blue)


# %% [markdown]
# ### The figure
# (a) raw BOSSNet against APOGEE [Fe/H]; (b) residual after calibration; (c) velocity offset against BOSS $T_{\rm eff}$.

# %%
# Three equal panels (sizes in inches); dot: style of the individual stars
fig, axes = panel_grid(1, 3, width=7, left=.5, right=.1, hgap=.55, bottom=.42, top=.3)
dot = dict(s=1.5, color='0.45', alpha=.5, lw=0, rasterized=True)
x = np.linspace(-2.7, .4, 2)  # end points of the straight lines in (a) [dex]

# (a) Raw BOSSNet against APOGEE [Fe/H]; Eq. 2 at the pivot (Teff = 4200 K, S/N = 20), whose slope c1 is the BOSSNet scale factor.
ax = axes[0, 0]
# Stars, one-to-one line and Eq. 2 at t = s = 0
ax.scatter(m.feh_a, m.feh_b, **dot)
ax.plot(x, x, '--', color='0.3', lw=.7, label='one to one')
ax.plot(x, cal['c0'] + cal['c1'] * x, color=orange, label=rf'Eq. 2, $t=s=0$: slope {cal["c1"]:.2f}')
ax.set(xlim=(-2.7, .4), ylim=(-2.7, .4), xlabel=r'[Fe/H]$_{\rm APOGEE}$', ylabel=r'[Fe/H]$_{\rm BOSS}$ (BOSSNet)',
       title=f'(a) {len(m):,} stars in both surveys')
ax.legend(loc='upper left')

# (b) Residual of the calibrated BOSS [Fe/H] against APOGEE [Fe/H].
ax = axes[0, 1]
# Residuals [dex] of each star and their running percentiles in 0.25-dex bins from -2.5
ax.scatter(m.feh_a, m.feh_cal - m.feh_a, **dot)
running(ax, m.feh_a.to_numpy(), (m.feh_cal - m.feh_a).to_numpy(), np.arange(-2.5, .5, .25))
ax.axhline(0, color='0.3', ls='--', lw=.7)
ax.set(xlim=(-2.7, .4), ylim=(-1, 1), xlabel=r'[Fe/H]$_{\rm APOGEE}$', ylabel='calibrated BOSS $-$ APOGEE [Fe/H] (dex)',
       title='(b) [Fe/H] after calibration')

# (c) Raw velocity difference against BOSS Teff, with the fitted linear trend (Eq. 1), d1 per 1000 K about T_PIVOT = 4200 K.
ax = axes[0, 2]
dv = (m.v_b - m.v_a).to_numpy()  # km/s
# Velocity differences and running percentiles in 150-K bins from 3700 K
ax.scatter(m.teff_b, dv, **dot)
running(ax, m.teff_b.to_numpy(), dv, np.arange(3700, 5600, 150))
t = np.linspace(3600, 5500, 2)  # K
# Eq. 1: v_BOSS - v_APOGEE = d0 + d1 (Teff - T_PIVOT) / 1000 K [km/s]
ax.plot(t, cal['d0'] + cal['d1'] * (t - giants.T_PIVOT) / 1000, color=orange, label='Eq. 1')
ax.axhline(0, color='0.3', ls='--', lw=.7)
ax.set(xlim=(3600, 5500), ylim=(-15, 20), xlabel=r'$T_{\rm eff,BOSS}$ (K)', ylabel=r'$v_{\rm BOSS}-v_{\rm APOGEE}$ (km s$^{-1}$)',
       title='(c) Velocity offset')
ax.legend(loc='upper right')
fig.savefig(ROOT / 'figures/paper_boss_calibration.png')
print('Saved paper_boss_calibration.png')
