"""Disc orientation and shape of the velocity ellipsoid against metallicity: each group fitted on its own (stage 1) compared with the
joint fit (stage 2) of the velocity-ellipsoid model (paper, kinematics section).

Values: posterior draws of scripts/fit_kinematics_mcmc.py 'data ellipsoid' (giants, 2 < rho < 9 deg) and 'data ellipsoid outer'
(RR Lyrae, 4 < rho < 9 deg, where one Gaia error scale describes their errors; scripts/paper_kin_support_ellipsoid.py), read from
data/kin_mcmc_data_<run>_samples.npz (stage-1 keys '1|<i>|<param>[_0]', stage-2 keys '2|all|<param>[_<i>]'). Medians and 16th-84th
percentiles; the axis ratios sigma_phi / sigma_R and sigma_z / sigma_R from paired draws (at R_SIG = 4 kpc). x: median [Fe/H] of each
group (results/kin_support_ellipsoid.csv), horizontal bars its 16th-84th percentiles.

Writes figures/paper_kin_geometry_feh.png and results/kin_geometry_feh.csv (population, stage, inc, theta, ratio_phi, ratio_z, each
median, _p16, _p84). Panels (2 x 2, equal; legend above; letters lower right): (a) inclination i [deg], (b) position angle of the line of nodes theta [deg],
(c) sigma_phi / sigma_R, (d) sigma_z / sigma_R. Giants black circles joined by thin lines, RR Lyrae blue stars (filled: on its own;
no sigma_z / sigma_R, proper motions only); joint fit: grey band (16th-84th percentiles of the shared i and theta of the giants' run)
in (a, b), open symbols in (c, d) (each group's own ellipsoid with the shared geometry).

Run: .venv/bin/python scripts/paper_kin_geometry_feh.py   (after fit_kinematics_mcmc.py data ellipsoid, and data ellipsoid outer)
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmcold.layout import panel_grid
from lmcold.sources import DATA, ROOT

RUNS = dict(giants='ellipsoid', rrl='ellipsoid_outer')  # sample runs, as in Fig. kin_support
QUANT = ('inc', 'theta', 'ratio_phi', 'ratio_z')  # i [deg], theta [deg], sigma_phi / sigma_R, sigma_z / sigma_R: one panel each


def draws(z, i, stage):
    """Draw arrays of QUANT for population i of npz z, from its own fit (stage 1) or the joint one (stage 2)."""
    # g(n): flattened draws (chains x draws) of parameter n; own=True for per-population sites (suffix _0 in stage 1, _<i> in
    # stage 2), own=False for sites shared in stage 2 (inc, theta), stored without a suffix.
    if stage == 1:
        g = lambda n, own=True: z[f'1|{i}|{n}_0' if own else f'1|{i}|{n}'].reshape(-1)  # noqa: E731
    else:
        g = lambda n, own=True: z[f'2|all|{n}_{i}' if own else f'2|all|{n}'].reshape(-1)  # noqa: E731
    return dict(inc=g('inc', False), theta=g('theta', False), ratio_phi=g('sigma_phi') / g('sigma_R'),
                ratio_z=g('sigma_z') / g('sigma_R'))


# %% [markdown]
# ### Load the data
# The median and 16th-84th percentiles of [Fe/H] of each group from `results/kin_support_ellipsoid.csv` (written by
# `scripts/paper_kin_support_ellipsoid.py`). The posterior draws are read one run at a time in the loop below:
# `data/kin_mcmc_data_ellipsoid_samples.npz` (giants) and `data/kin_mcmc_data_ellipsoid_outer_samples.npz` (RR Lyrae).

# %%
# [Fe/H] spread of each group (feh_med, feh_p16, feh_p84), indexed by population.
feh = pd.read_csv(ROOT / 'results/kin_support_ellipsoid.csv').set_index('population')

# %% [markdown]
# ### Orientation and ellipsoid shape per group
# For each group, from the posterior draws of its own fit (stage 1) and of the joint fit (stage 2): the inclination $i$, the
# position angle $\theta$ of the line of nodes and the axis ratios $\sigma_\phi/\sigma_R$, $\sigma_z/\sigma_R$ (paired draws), as
# medians and 16th-84th percentiles, with the [Fe/H] spread of the group. Written to `results/kin_geometry_feh.csv`.

# %%
rows = []  # one dict per population and stage
for kind, run in RUNS.items():
    # Posterior draws of one run (module docstring for the key layout); z['names'] lists the populations in index order.
    z = np.load(DATA / f'kin_mcmc_data_{run}_samples.npz')
    for i, pop in enumerate(z['names']):
        # Giants from the 2-9 deg run, RR Lyrae from the outer run only.
        if (pop == 'RR Lyrae') != (kind == 'rrl'):
            continue
        for stage in (1, 2):
            r = dict(population=pop, stage=stage)
            # Median and 16th/84th percentiles of each quantity: columns n, n_p16, n_p84.
            for n, v in draws(z, i, stage).items():
                r |= dict(zip((n, f'{n}_p16', f'{n}_p84'), np.percentile(v, [50, 16, 84])))
            rows.append(r)
# Add the [Fe/H] median and percentiles of each population (x coordinate of the figure), then write and print.
t = pd.DataFrame(rows).join(feh[['feh_med', 'feh_p16', 'feh_p84']], on='population')
t.to_csv(ROOT / 'results/kin_geometry_feh.csv', index=False, float_format='%.4f')
print(t[['population', 'stage'] + list(QUANT)].round(3).to_string(index=False))

# %% [markdown]
# ### The figure
# Four panels against [Fe/H]: $i$, $\theta$, $\sigma_\phi/\sigma_R$ and $\sigma_z/\sigma_R$. Filled: own fits; grey band and open
# symbols: the joint fit.

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
# 2 x 2 equal panels (sizes in inches), shared [Fe/H] axis per column.
fig, ax = panel_grid(2, 2, width=4.6, left=0.55, right=0.1, hgap=0.55, vgap=0.12, top=0.35, bottom=0.42, aspect=0.8)
# Asymmetric error bars [lower, upper]: of quantity n (err) and of [Fe/H] (fx), from the 16th-84th percentiles.
err = lambda s, n: [s[n] - s[f'{n}_p16'], s[f'{n}_p84'] - s[n]]  # noqa: E731
fx = lambda s: [s.feh_med - s.feh_p16, s.feh_p84 - s.feh_med]  # noqa: E731
# Own fits (stage 1) and joint fit (stage 2), each split into giants (g) and RR Lyrae (r).
own, joint = t[t.stage == 1], t[t.stage == 2]
g1, r1 = own[own.population != 'RR Lyrae'], own[own.population == 'RR Lyrae']
g2, r2 = joint[joint.population != 'RR Lyrae'], joint[joint.population == 'RR Lyrae']
DX = 0.03  # [Fe/H] offset of the joint-fit symbols
for a, n, lab, letter in zip(ax.ravel(), QUANT, (r'$i$ [deg]', r'$\theta$ [deg]', r'$\sigma_\phi/\sigma_R$', r'$\sigma_z/\sigma_R$'), 'abcd'):
    # Giants, own fits: black circles joined by a thin line.
    a.plot(g1.feh_med, g1[n], '-', color='k', lw=0.6, alpha=0.6)
    a.errorbar(g1.feh_med, g1[n], xerr=fx(g1), yerr=err(g1, n), fmt='o', color='k', ms=4, lw=0.8, label='giants, own fit')
    if n != 'ratio_z':  # RR Lyrae sigma_z rests on the proper motions alone and is not measurable
        a.errorbar(r1.feh_med, r1[n], xerr=fx(r1), yerr=err(r1, n), fmt='*', color='C0', ms=8, lw=0.8, label=r'RR Lyrae, own fit ($4<\rho<9^\circ$)')
    if n in ('inc', 'theta'):  # shared by all groups in the joint fit: one band (the giants' run, 2-9 deg)
        s = g2.iloc[0]
        a.axhspan(s[f'{n}_p16'], s[f'{n}_p84'], color='0.7', alpha=0.6, lw=0, label='joint fit, shared geometry')
        a.axhline(s[n], color='0.45', lw=0.6)
    # Axis ratios in the joint fit: open symbols, shifted by DX in [Fe/H].
    else:
        a.errorbar(g2.feh_med + DX, g2[n], yerr=err(g2, n), fmt='o', color='k', mfc='white', ms=4, lw=0.8, label='joint fit')
        if n != 'ratio_z':
            a.errorbar(r2.feh_med + DX, r2[n], yerr=err(r2, n), fmt='*', color='C0', mfc='white', ms=8, lw=0.8)
    a.set(ylabel=lab, xlim=(-2.2, -0.2))
    a.text(0.9, 0.06, f'({letter})', transform=a.transAxes, fontsize=7)  # lower right: empty in every panel
# x labels on the bottom row only; common y range for the two axis-ratio panels.
for a in ax[0]:
    a.tick_params(labelbottom=False)
for a in ax[1]:
    a.set_xlabel('[Fe/H]')
ax[1, 0].set_ylim(0.4, 1.1)
ax[1, 1].set_ylim(0.4, 1.1)
ax[0, 0].errorbar([], [], fmt='o', color='k', mfc='white', ms=4, lw=0.8, label='joint fit, own ellipsoid')  # open symbols of (c, d)
ax[0, 0].legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, fontsize=6, handletextpad=0.3, columnspacing=1.0)
fig.savefig(ROOT / 'figures/paper_kin_geometry_feh.png', dpi=300)
