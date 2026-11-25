"""Corner plots of a kinematic posterior: argument RUN, samples data/kin_mcmc_<RUN>_samples.npz from scripts/fit_kinematics_mcmc.py.

figures/fig_kin_mcmc_corner_<RUN>_shared.png: the sampled parameters the joint fit shares (of centre offset, bulk proper motion,
line-of-sight systemic velocity, inclination, line of nodes, lopsided part), stage 1 for each population in its own colour and
stage 2 (all populations together) in black; 1 and 2 sigma contours. For RUN data_stream also figures/paper_kin_corner.png
(paper appendix): the same for the parameters PAPER only (centre, proper motion north, orientation, m = 2 part of v_phi). figures/fig_kin_mcmc_corner_<RUN>_<i>.png: all sampled
parameters of stage 1 for population i (outlier fraction shown as f_out, not its logit).

RUN is '<kind>_<model>' of fit_kinematics_mcmc.py, e.g. data_centre, data_lopsided, data_both (suffix _dist for the runs with RR
Lyrae distance moduli). The npz keys are
'stage|population|site': '1|<i>|<site>' for the stage-1 fit of population i (its own parameters have the site suffix _0,
since each stage-1 fit holds one population) and '2|all|<site>' for the joint fit, each of shape (chains, draws) (flat in runs
before 2026-10-29; reshape(-1) reads both); 'names' lists the populations in order. The per-population and full shared plots are online
material. Contours: 2D levels enclosing 39% and 86% of the mass (1 and 2 sigma of a 2D Gaussian); bins 40,
smoothing 1 bin. In the shared plot each 1D histogram is normalised to unit area, and the axis ranges are the 0.2-99.8 percentiles
of all fits together.

Used in KB chapter 6 ('Correlations of the geometric parameters'), run with data_stream. Its input for that run,
data/kin_mcmc_data_stream_samples.npz, is in Zenodo bundle 1, so the figure is made without rerunning the MCMC fit.

Run: .venv/bin/python scripts/paper_kin_corner.py data_stream
"""
import sys

import matplotlib
matplotlib.use('Agg')
import corner
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from lmcold import kinfit
from lmcold.sources import DATA, ROOT

LABELS = dict(dx=r'$\Delta x$ E [deg]', dy=r'$\Delta y$ N [deg]', pmra0=r'$\mu_{\alpha*,0}$ [mas/yr]',
              pmdec0=r'$\mu_{\delta,0}$ [mas/yr]', inc=r'$i$ [deg]', theta=r'$\theta$ [deg]', v0=r'$v_0$ [km/s]',
              R0=r'$R_0$ [kpc]', sigma=r'$\sigma$ [km/s]', k=r'$k$', logit_fout=r'$f_{\rm out}$',
              vlos0=r'$v_{\rm los,0}$ [km/s]', sigma_los=r'$\sigma_{\rm los}$ [km/s]', m1c=r'$m_{1c}$', m1s=r'$m_{1s}$',
              dmu0=r'$\Delta\mu_0$ [mag]', sigma_mu=r'$\sigma_\mu$ [mag]', m2c=r'$m_{2c}$', m2s=r'$m_{2s}$',
              vr0=r'$v_{R,0}$', r2c=r'$r_{2c}$', r2s=r'$r_{2s}$')
LEVELS = 1 - np.exp(-.5 * np.array([1., 4.]))  # 1 and 2 sigma in 2D: mass fractions 0.39 and 0.86 (1 - exp(-n^2 / 2))
GIANT_COLOURS = ['#d7191c', '#fdae61', '#a6d96a', '#1a9641', '#2878b5']  # [Fe/H] bins, metal-poor first; RR Lyrae purple
RUN = sys.argv[1]  # '<kind>_<model>', e.g. data_both
PAPER = ('dx', 'dy', 'pmdec0', 'inc', 'theta', 'm2c', 'm2s')  # parameters of the paper's reduced shared corner plot

# %% [markdown]
# ### Load the data
# The posterior samples of the run given as argument (`data_stream` for the paper): `data/kin_mcmc_<RUN>_samples.npz` (from
# `scripts/fit_kinematics_mcmc.py`) holds the draws of every stage-1 fit and of the joint stage-2 fit, and the population names.

# %%
# Posterior draws of run RUN: keys 'stage|population|site', each of shape (chains, draws), plus 'names'.
s = np.load(DATA / f'kin_mcmc_{RUN}_samples.npz')
names = list(s['names'])  # the populations, in the order of the run

# %% [markdown]
# ### Colours and shared parameters
# Each population gets a fixed colour, and the shared parameters sampled in every fit are listed.

# %%
# One colour per population, fixed by name so that it does not depend on the population order in the run.
colours = [GIANT_COLOURS[[m for m in names if m != 'RR Lyrae'].index(n)] if n != 'RR Lyrae' else '#7b3294' for n in names]
# (draws, len(params)) array of the samples under key.format(name) for each parameter name.
stack = lambda key, params: np.column_stack([s[key.format(n)].reshape(-1) for n in params])  # noqa: E731
# Shared parameters drawn in every fit of this run (e.g. vlos0 is held in the stage-1 RR Lyrae fit, which has no velocities).
SHARED = [n for n in kinfit.SHARED if all(f'{k}|{n}' in s for k in ['2|all'] + [f'1|{i}' for i in range(len(names))])]  # sampled in every fit


def overlay(params, fname, dpi):
    """Corner plot of params: each stage-1 fit in its colour, then the joint fit in black, on common axis ranges."""
    shared = [stack(f'1|{i}|{{}}', params) for i in range(len(names))] + [stack('2|all|{}', params)]
    lo, hi = np.percentile(np.vstack(shared), [.2, 99.8], axis=0)
    fig = None
    for x, colour in zip(shared, colours + ['k']):
        fig = corner.corner(x, fig=fig, range=list(zip(lo, hi)), bins=40, smooth=1., levels=LEVELS, color=colour,
                            plot_datapoints=False, plot_density=False, fill_contours=False, hist_kwargs=dict(density=True),
                            labels=[LABELS[n] for n in params], label_kwargs=dict(fontsize=11))
    for j in range(len(params)):  # 1D histograms of different widths: rescale each diagonal panel to its tallest curve
        ax = fig.axes[j * (len(params) + 1)]
        ax.relim(), ax.autoscale_view(scalex=False)
    fig.legend([Line2D([], [], color=c) for c in colours + ['k']], names + ['joint (stage 2)'], loc='upper right',
               bbox_to_anchor=(.98, .98), frameon=False, fontsize=12)
    fig.savefig(ROOT / f'figures/{fname}', dpi=dpi)
    plt.close(fig)


# %% [markdown]
# ### The shared parameters, all fits overlaid
# All shared parameters (online material) and, for the sky-aligned run, the paper's reduced set `PAPER`.

# %%
overlay(SHARED, f'fig_kin_mcmc_corner_{RUN}_shared.png', 110)
if RUN == 'data_stream':  # the paper's appendix figure: the geometry and m = 2 terms only
    overlay([n for n in PAPER if n in SHARED], 'paper_kin_corner.png', 110)

# %% [markdown]
# ### Every parameter of each stage-1 fit
# One full corner plot per population (online material).

# %%
# One full corner plot per population (stage 1), with medians and 16th/84th percentiles in the panel titles.
for i, name in enumerate(names):
    key = lambda n: f'1|{i}|{n}' if n in kinfit.SHARED else f'1|{i}|{n}_0'  # noqa: E731  (own parameters: site suffix _0)
    params = [n for n in kinfit.PARAMS if key(n) in s]
    x = np.column_stack([s[key(n)].reshape(-1) for n in params])
    j = params.index('logit_fout')
    x[:, j] = 1 / (1 + np.exp(-x[:, j]))  # logit -> outlier fraction f_out
    fig = corner.corner(x, bins=40, smooth=1., levels=LEVELS, color=colours[i], plot_datapoints=False, quantiles=[.16, .5, .84],
                        show_titles=True, title_fmt='.3g', labels=[LABELS[n] for n in params], label_kwargs=dict(fontsize=11))
    fig.suptitle(f'{name}, stage 1', fontsize=14, x=.7, y=.9)
    fig.savefig(ROOT / f'figures/fig_kin_mcmc_corner_{RUN}_{i}.png', dpi=90)
    plt.close(fig)
