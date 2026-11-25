"""Density centres of the RR Lyrae and of the giants by [Fe/H], to compare with the kinematic centre.

Centre: the weighted mean position of the stars within APERTURES[j] deg of the current centre, iterated from the adopted centre
until it moves less than 1e-4 deg; the centre of point symmetry of the counts within that radius (an inclined disc projects to
ellipses, symmetric about its centre). Positions on the tangent plane of the adopted centre (lmcold.footprint.lmc_xy).
Samples: RRab members (complete within 10 deg; lmcold.samples); the photometric red giants that the BOSS carton selects from
Gaia (data/boss_parent.parquet, `eligible`: G <= 17.5, colour, magnitude and proper-motion cuts, all [Fe/H]); and the BOSS giants
of lmcold.mass_samples weighted for selection, coverage and Milky Way foreground (w x fill x (1 - f_mw)), in the [Fe/H] bins of
lmcold.samples.FEH_BINS.
Errors: 16th-84th percentiles of NBOOT bootstrap resamples. Writes results/density_centre.csv and figures/fig_density_centre.png
(offset east and north against aperture, with the kinematic centres of results/kin_mcmc_data_both.csv, stage 1, in the margin).

Diagnostic for the kinematic centre (Sect. 3); not a paper figure. results/density_centre.csv is read by scripts/paper_kin_centre.py
(the density centres of figure paper_kin_centre, KB chapter 7) and scripts/paper_text_kin_support.py (distances of the kinematic
centres from the density centres).

Inputs: data/rrab.parquet (sources.yaml key release_rrab, via lmcold.samples.rrab); data/boss_parent.parquet
(release_boss_parent); the weighted BOSS giants of lmcold.mass_samples.samples (release_boss_weights and the files listed there);
data/giant_foreground.parquet (coverage fill and Milky Way foreground fraction f_mw per BOSS giant, scripts/build_giant_mass.py);
results/kin_mcmc_data_both.csv (scripts/fit_kinematics_mcmc.py; columns stage, population, param, median, p16, p84).

Figure (two equal square panels; legend right of (b)): x: aperture radius [deg]; each population is shifted by 0.06 deg in x for
legibility. y: density-centre offset east (left panel) and north (right panel) of the adopted centre [deg] on the orthographic
tangent plane, with 16th-84th bootstrap percentiles. Stars right of the dotted line at x = 5.6: stage-1 kinematic centre (median
and 16th-84th percentiles of dx or dy) of the same population, where results/kin_mcmc_data_both.csv has one.

results/density_centre.csv: population, aperture [deg], N (stars within the aperture of the final centre), dx, dy [deg] and their
16th and 84th bootstrap percentiles (dx_p16, dx_p84, dy_p16, dy_p84).

Run: .venv/bin/python scripts/fig_density_centre.py   (run time not recorded)
"""
# %% [markdown]
# ### Settings
# The aperture radii and the number of bootstrap resamples.

# %%
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmcold import samples
from lmcold.footprint import lmc_xy
from lmcold.layout import panel_grid
from lmcold.mass_samples import samples as mass_samples
from lmcold.sources import DATA, ROOT

APERTURES = [1.5, 2.5, 3.5, 5.]  # deg
NBOOT = 200  # bootstrap resamplings of the stars


# %% [markdown]
# ### The density centre within one aperture
# `centre` iterates the weighted mean position of the stars within radius `r` of the current centre until it stops moving.

# %%
def centre(x, y, w, r):
    """Iterated weighted mean position within radius r, starting from the origin (the adopted centre).

    Parameters
    ----------
    x, y : numpy.ndarray
        Tangent-plane coordinates east and north of the adopted centre [deg].
    w : numpy.ndarray
        Weights per star.
    r : float
        Aperture radius [deg].

    Returns
    -------
    numpy.ndarray, shape (2,)
        (x, y) of the centre [deg]: the first iterate that moves less than 1e-4 deg, or the 200th.
    """
    c = np.zeros(2)                    # start at the adopted centre [deg]
    for _ in range(200):               # at most 200 iterations
        # Stars within r of the current centre and their weighted mean position [deg].
        k = np.hypot(x - c[0], y - c[1]) < r
        new = np.array([np.average(x[k], weights=w[k]), np.average(y[k], weights=w[k])])
        # Converged when the centre moves by less than 1e-4 deg.
        if np.hypot(*(new - c)) < 1e-4:
            break
        c = new
    return new


# %% [markdown]
# ### Load the data
# The RR Lyrae members (`lmcold.samples.rrab()`, `data/rrab.parquet`), the Gaia red giants the BOSS carton selects
# (`data/boss_parent.parquet`), and the BOSS giants with their selection weights (`lmcold.mass_samples`) joined to their coverage
# fill and Milky Way foreground fraction (`data/giant_foreground.parquet`); all from Zenodo bundle 1. The giants are split into the
# five [Fe/H] bins of the kinematic fits.

# %%
rr = samples.rrab()   # dict of RRab member arrays (ra, dec [deg], ...)
# Gaia stars the BOSS carton selects (lmcold.magcloud.eligible), unweighted, all metallicities.
par = pd.read_parquet(DATA / 'boss_parent.parquet').query('eligible')
# Populations: (label, ra [deg], dec [deg], weight per star); RR Lyrae and Gaia giants unweighted.
pops = [('RR Lyrae', rr['ra'], rr['dec'], np.ones(len(rr['ra']))),
        ('red giants, Gaia G<17.5', par.ra.to_numpy(), par.dec.to_numpy(), np.ones(len(par)))]
# BOSS giants with selection weights w (first element of lmcold.mass_samples.samples), joined to fill and f_mw per star.
b = mass_samples()[0].merge(pd.read_parquet(DATA / 'giant_foreground.parquet')[['source_id', 'fill', 'f_mw']], on='source_id',
                            validate='one_to_one')
# Weight: selection weight x coverage fill x (1 - Milky Way foreground fraction).
b = b.assign(W=b.w * b['fill'] * (1 - b.f_mw))
# One population per [Fe/H] bin of lmcold.samples.FEH_BINS; the open end bins have edges -9 and 9, labelled as one-sided.
for lo, hi in samples.FEH_BINS:
    s = b[(b.feh >= lo) & (b.feh < hi)]
    label = f"[Fe/H] < {hi}" if lo < -5 else (f"[Fe/H] > {lo}" if hi > 5 else f"{lo} < [Fe/H] < {hi}")
    pops.append((f'giants {label}', s.ra.to_numpy(), s.dec.to_numpy(), s.W.to_numpy()))

# %% [markdown]
# ### Density centres and their bootstrap errors
# For each population and aperture: the centre, and the 16th-84th percentiles of the centres of NBOOT bootstrap resamples.

# %%
rows = []                          # one dict per (population, aperture)
rng = np.random.default_rng(1)     # fixed seed of the bootstrap
for name, ra, dec, w in pops:
    x, y = lmc_xy(ra, dec)   # deg east and north of the adopted centre, tangent plane
    for r in APERTURES:
        c = centre(x, y, w, r)
        # Bootstrap: resample stars with replacement and repeat the iteration.
        boot = np.array([centre(x[i], y[i], w[i], r) for i in (rng.integers(0, len(x), len(x)) for _ in range(NBOOT))])
        lo, hi = np.percentile(boot, [16, 84], axis=0)
        rows.append(dict(population=name, aperture=r, N=int((np.hypot(x - c[0], y - c[1]) < r).sum()), dx=c[0], dy=c[1],
                         dx_p16=lo[0], dx_p84=hi[0], dy_p16=lo[1], dy_p84=hi[1]))
        print(rows[-1], flush=True)
# Table of centres, written to results/density_centre.csv.
t = pd.DataFrame(rows)
t.to_csv(ROOT / 'results/density_centre.csv', index=False, float_format='%.4f')

# %% [markdown]
# ### The figure
# Density-centre offsets east and north against aperture radius, with the stage-1 kinematic centres of
# `results/kin_mcmc_data_both.csv` (from `scripts/fit_kinematics_mcmc.py data both`) in the margin.

# %%
kin = pd.read_csv(ROOT / 'results/kin_mcmc_data_both.csv').query('stage == 1')   # kinematic fits: centre offsets dx, dy [deg]
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
fig, axes = panel_grid(1, 2, width=7, left=.6, right=1.9, hgap=.6, bottom=.5, top=.3, aspect=1.)
# One colour per population, in the order of pops: RR Lyrae, Gaia red giants, five giant [Fe/H] bins.
colours = ['#7b3294', 'k', '#d7191c', '#fdae61', '#a6d96a', '#1a9641', '#2878b5']
for ax, c in zip(axes[0], ('dx', 'dy')):
    for j, (name, colour) in enumerate(zip(t.population.unique(), colours)):
        s = t[t.population == name]
        # Aperture shifted by 0.06 deg per population for legibility; error bars are the bootstrap 16th-84th percentiles.
        a = s.aperture + .06 * (j - 3)
        ax.errorbar(a, s[c], [s[c] - s[f'{c}_p16'], s[f'{c}_p84'] - s[c]], fmt='o-', ms=4, lw=1, color=colour, label=name)
        # Stage-1 kinematic centre of the same population (if fitted), drawn as a star at x = 5.9 + 0.06 j.
        k = kin[(kin.population == name) & (kin.param == c)]
        if len(k):
            k = k.iloc[0]
            ax.errorbar(5.9 + .06 * j, k['median'], [[k['median'] - k.p16], [k.p84 - k['median']]], fmt='*', ms=8, color=colour)
    # Zero offset (adopted centre) and the dotted divider between apertures and kinematic centres at x = 5.6.
    ax.axhline(0, color='.5', lw=.6)
    ax.axvline(5.6, color='.7', lw=.6, ls=':')
    ax.set(xlim=(1, 6.4), xlabel='aperture radius [deg]  (stars: kinematic)', ylabel=f"density centre offset {'east' if c == 'dx' else 'north'} [deg]")
axes[0, 1].legend(loc='upper left', bbox_to_anchor=(1.03, 1), frameon=False, fontsize=7)
fig.savefig(ROOT / 'figures/fig_density_centre.png', dpi=200)
