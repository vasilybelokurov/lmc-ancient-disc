"""Paper figure: the kinematic centres of the old population against density centres (kinematics section, summary of the centre).

One square panel, offsets east (to the left) and north [deg] of the vdMK14 centre (lmcold.constants.LMC) on the orthographic plane
(lmcold.footprint.lmc_xy). Kinematic centres (posterior medians, 16th-84th percentile bars) from scripts/fit_kinematics_mcmc.py:
coloured circles, stage 1 of the Level-2 velocity-ellipsoid model (each population on its own; giants at 2 < rho < 9 deg,
results/kin_mcmc_data_ellipsoid.csv; RR Lyrae at 4 < rho < 9 deg, results/kin_mcmc_data_ellipsoid_outer.csv, as for their quoted
kinematics); black filled star, stage 2 (joint) of that model at 2 < rho < 9 deg; black open star, stage 2 of Level 1
(results/kin_mcmc_data_centre.csv, sky-aligned dispersions). Density centres
(results/density_centre.csv, scripts/fig_density_centre.py: iterated aperture means) of the RR Lyrae and of the Gaia red giants
G < 17.5, one point per aperture (1.5, 2.5, 3.5, 5 deg) joined by a line from the smallest aperture (open marker at its end). Published
centres (positions from the cited papers): open squares,
kinematic; open diamonds, density; open triangles, density of old stars; black +, the vdMK14 centre.
Writes figures/paper_kin_centre.png. Run: .venv/bin/python scripts/paper_kin_centre.py   (1 s in the run of 2026-11-22, docs/run_times.txt)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmcold.footprint import lmc_xy
from lmcold.layout import panel_grid
from lmcold.sources import ROOT

COLOURS = ['#d7191c', '#fdae61', '#a6d96a', '#1a9641', '#2878b5', '#7b3294']  # giant bins metal-poor first, RR Lyrae
# colour of the density centres of each population
DENSITY_COLOURS = {'RR Lyrae': '#7b3294', 'red giants, Gaia G<17.5': '0.4'}
# label: ((RA, Dec) [deg] of each paper of the group), marker, label position (east, north [deg]) and alignment; a thin leader
# line joins the label to the group mean.
PUBLISHED = {
    'Gaia PM (Luri+21)': ([(81.07, -69.41)], 's', (0.62, 0.42, 'center')),
    'Gaia/VMC PM (Choi+22, Niederhofer+22,\nKacharov+24, Vijayasree+25)': (
        [(80.443, -69.272), (79.95, -69.31), (80.29, -69.25), (80.30, -69.27)], 's', (0.12, 0.62, 'center')),
    'HI (vdMK14)': ([(78.77, -69.01)], 's', (-0.38, 0.68, 'center')),
    'bar (vdM01, El Youssoufi+19)': ([(81.28, -69.78), (81.24, -69.73)], 'D', (0.55, -0.45, 'center')),
    'outer isophotes (vdM01)': ([(82.25, -69.50)], 'D', (0.85, 0.22, 'center')),
    'old stars (Jacyszyn-Dobrzeniecka+17,\nCusano+21, Rathore+25)': ([(80.38, -69.61), (80.615, -69.580), (80.27, -69.65)], '^', (-0.15, -0.5, 'center')),
}

# %% [markdown]
# ### Load the data
# Kinematic centres (offsets `dx` east and `dy` north of the centre of van der Marel & Kallivayalil 2014, deg) from the posterior
# summary tables of `scripts/fit_kinematics_mcmc.py`: the velocity-ellipsoid model at $2<\rho<9^\circ$
# (`results/kin_mcmc_data_ellipsoid.csv`) and at $4<\rho<9^\circ$ (`results/kin_mcmc_data_ellipsoid_outer.csv`), and Level 1 with
# sky-aligned dispersions (`results/kin_mcmc_data_centre.csv`); density centres from `results/density_centre.csv`.

# %%
l2 = pd.read_csv(ROOT / 'results/kin_mcmc_data_ellipsoid.csv')  # ellipsoid model, 2-9 deg: stage 1 per group and stage 2 (joint)
rr = pd.read_csv(ROOT / 'results/kin_mcmc_data_ellipsoid_outer.csv')  # ellipsoid model, 4-9 deg (used for the RR Lyrae)
l1 = pd.read_csv(ROOT / 'results/kin_mcmc_data_centre.csv')  # Level 1, sky-aligned dispersions
dens = pd.read_csv(ROOT / 'results/density_centre.csv')  # density centres per population and aperture

# %% [markdown]
# ### The centres
# Stage 1 of each group (giants at $2<\rho<9^\circ$, RR Lyrae at $4<\rho<9^\circ$); stage 2 (joint) of the velocity-ellipsoid model
# and of Level 1.

# %%
# stage-1 centres: giant bins from the 2-9 deg fits, RR Lyrae from the 4-9 deg fit
own = pd.concat([l2[(l2.stage == 1) & (l2.population != 'RR Lyrae')], rr[(rr.stage == 1) & (rr.population == 'RR Lyrae')]])

# %% [markdown]
# ### The figure
# One panel, east to the left: kinematic centres with 16th-84th percentile bars, joint fits as stars, density centres per aperture
# joined by lines, and the published centres in grey.

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
# one square panel; top=1.25 leaves room for the legend above it
fig, ax = panel_grid(1, 1, width=3.4, left=0.55, right=0.1, top=1.25)
a = ax[0, 0]
# q(table, population, parameter): the row (median, p16, p84) of that population and parameter
q = lambda t, pop, n: t[(t.population == pop) & (t.param == n)].iloc[0]  # noqa: E731
# kinematic centres of the groups, one colour each (COLOURS)
for pop, c in zip(own[own.param == 'dx'].population, COLOURS):
    # offsets east (dx) and north (dy) of the vdMK14 centre [deg]
    x, y = q(own, pop, 'dx'), q(own, pop, 'dy')
    # median with asymmetric 16th-84th percentile bars
    a.errorbar(x['median'], y['median'], xerr=[[x['median'] - x.p16], [x.p84 - x['median']]],
               yerr=[[y['median'] - y.p16], [y.p84 - y['median']]], fmt='o', ms=3.5, color=c, lw=0.8, zorder=6,
               label=pop.replace('giants ', 'giants, ') + (r' ($4<\rho<9^\circ$)' if pop == 'RR Lyrae' else ''))
# stage-2 (joint, population 'shared') centres of the ellipsoid model (filled star) and of Level 1 (open star)
for t, mfc, lab in ((l2, 'k', 'joint, Level 2'), (l1, 'w', 'joint, Level 1')):
    x, y = q(t[t.stage == 2], 'shared', 'dx'), q(t[t.stage == 2], 'shared', 'dy')
    a.plot(x['median'], y['median'], '*', ms=9, mfc=mfc, mec='k', mew=0.8, label=lab, zorder=4)  # below the circles of the groups
# density centres: one line per population through the apertures, smallest first; the largest aperture open
for pop, t in dens.groupby('population', sort=False):
    if pop not in DENSITY_COLOURS:
        continue
    t = t.sort_values('aperture')
    a.plot(t.dx, t.dy, '-', color=DENSITY_COLOURS[pop], lw=0.8)
    a.plot(t.dx, t.dy, 'D', ms=2.5, color=DENSITY_COLOURS[pop], label=f'density, {pop.replace("red giants, ", "giants ")}')
    a.plot(t.dx.iloc[-1], t.dy.iloc[-1], 'D', ms=4, mfc='w', mec=DENSITY_COLOURS[pop], mew=0.8)
# published centres: (RA, Dec) -> offsets [deg], marker per paper, label joined to the group mean by a leader line
for lab, (pos, m, (lx, ly, ha)) in PUBLISHED.items():
    x, y = lmc_xy(np.array([p[0] for p in pos]), np.array([p[1] for p in pos]))
    a.plot(x, y, m, ms=3.5, mfc='none', mec='0.3', mew=0.6)
    a.annotate(lab, (x.mean(), y.mean()), xytext=(lx, ly), fontsize=4.5, color='0.3', ha=ha, va='center',
               arrowprops=dict(arrowstyle='-', color='0.6', lw=0.4, shrinkA=1, shrinkB=3))
# the vdMK14 centre itself, at the origin
a.plot(0, 0, '+', color='k', ms=9, mew=1.0)
a.annotate('vdMK14', (0, 0), xytext=(0.0, 0.05), fontsize=4.5, ha='center')
# east to the left (x axis reversed); equal scales in x and y
a.set(xlim=(1.15, -0.55), ylim=(-0.75, 0.95), aspect='equal', xlabel='offset east [deg]', ylabel='offset north [deg]')
# legend above the panel
a.legend(loc='lower left', bbox_to_anchor=(-0.15, 1.02), ncol=2, fontsize=4.5, frameon=False, handletextpad=0.3)
fig.savefig(ROOT / 'figures/paper_kin_centre.png')
