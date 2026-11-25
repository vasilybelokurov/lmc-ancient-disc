"""Paper figure (Section 6.6, Fig. sims_sidm): the centre of an LMC-mass halo in cold and self-interacting dark matter, against the TNG50
progenitors.

Inputs: results/sims_sidm_shapes.csv (scripts/sims_sidm_shapes.py: Concerto Halo104, CDM and GroupSIDM, z = 4, 3, 2, 1.5, 0) and
results/sims_halo_settle.csv (scripts/sims_halo_settle.py: TNG50, 60 progenitors).
(a) eps2 at 2 kpc against redshift: Concerto CDM and SIDM, all of l = 2 (filled)
and the part not axisymmetric about the minor axis of the 1.5-2.5 kpc shell (open); TNG50 DM, the part not axisymmetric about the
stellar disc axis (median and 16th-84th percentiles over the progenitors of the per-galaxy median in redshift bins of 0.5, z >= 1,
bins with >= 20 galaxies).
(b) c/a of the host in ellipsoidal shells against the shell's major-axis radius (shells with >= NMIN particles); (c) eps2 =
|Phi_2|/|Phi_0| of the potential against radius. In (b) and (c) CDM in greys and SIDM in reds, light at z = 4 to dark at z = 0.
Writes figures/paper_sims_sidm.png; prints the TNG50 bins.
Run: .venv/bin/python scripts/manifest.py run paper_sims_sidm.py   (after scripts/sims_sidm_shapes.py and sims_halo_settle.py; 1 s in
the recorded run of docs/run_times.txt)"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from lmcold.layout import panel_grid
from lmcold.sources import ROOT


# %% [markdown]
# ### Load the data
# The figure reads two result tables and no simulation; the simulations are needed only to remake them:
# `results/sims_sidm_shapes.csv` (`scripts/sims_sidm_shapes.py`, from the Concerto Halo104 runs in cold and self-interacting dark
# matter) and `results/sims_halo_settle.csv` (`scripts/sims_halo_settle.py`, from the dark matter of 60 TNG50 progenitors).

# %%
S = pd.read_csv(ROOT / 'results/sims_sidm_shapes.csv')   # Concerto Halo104, CDM and SIDM: shell axis ratios and eps2 at five redshifts
T = pd.read_csv(ROOT / 'results/sims_halo_settle.csv', dtype={'halo': str})   # TNG50 progenitors: eps2 at 2 kpc per snapshot

# %% [markdown]
# ### The Concerto halo and the TNG50 band
# `S`: axis ratios and $\epsilon_2$ of Halo104 in ellipsoidal shells, CDM and SIDM, at five redshifts (rows with `N = 0` are the values
# at 2 kpc). The TNG50 band is $\epsilon_2$ of the progenitors (`T`) in redshift bins.

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
NMIN = 10000   # particles per shell for a shape to be drawn in (b) and (c)
Z_BIN, MIN_GAL, Z_MIN = 0.5, 20, 1.0   # TNG50 band: redshift bins, galaxies per bin, lowest redshift

T['zb'] = (T.z / Z_BIN).round() * Z_BIN
g = T.groupby(['zb', 'halo']).eps2.median().groupby(level=0)
tng = pd.DataFrame({'m': g.median(), 'lo': g.quantile(.16), 'hi': g.quantile(.84), 'n': g.size()})
tng = tng[(tng.n >= MIN_GAL) & (tng.index >= Z_MIN)]


# %% [markdown]
# ### The figure
# (a) $\epsilon_2$ at 2 kpc against redshift; (b) $c/a$ and (c) $\epsilon_{2,\rm all}$ against radius, light to dark from $z=4$ to 0.

# %%
fig, ax = panel_grid(1, 3, width=7.0, left=0.5, right=0.1, hgap=0.62, top=0.3)
ax = ax[0]
zs = sorted(S.z.unique(), reverse=True)
shade = np.linspace(0.35, 1.0, len(zs))   # light at z = 4 to dark at z = 0
COL = {'CDM': dict(zip(zs, plt.cm.Greys(shade))), 'SIDM': dict(zip(zs, plt.cm.Reds(0.3 + 0.65 * (shade - 0.35) / 0.65)))}
for z in zs:
    for run in ('CDM', 'SIDM'):
        w = S[(S.z == z) & (S.run == run) & (S.N >= NMIN)]
        ax[1].plot(w.r, w.s, '-', marker='o', ms=2, lw=1, c=COL[run][z])
        ax[2].plot(w.r, w.eps2, '-', marker='o', ms=2, lw=1, c=COL[run][z])
zh = [tuple(plt.Line2D([], [], c=COL[run][z], lw=1.5) for run in ('CDM', 'SIDM')) for z in zs]
ax[2].legend(zh, [f'$z={z:.1f}$' for z in zs], handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, frameon=False, fontsize=5, ncol=2,
             loc='lower right', handlelength=2.4, columnspacing=0.7, handletextpad=0.4)   # empty corner (PI)
ax[1].set(xscale='log', ylim=(0, 1.02), xlabel='major-axis radius [kpc]', ylabel='minor-to-major axis ratio $c/a$')
ax[2].set(xscale='log', yscale='log', xlabel='radius [kpc]', ylabel=r'$\epsilon_{2,\rm all}=|\Phi_2|/|\Phi_0|$')
ax[0].fill_between(tng.index, tng.lo, tng.hi, color='0.5', alpha=0.25, lw=0)
ax[0].plot(tng.index, tng.m, '-', c='0.4', lw=1.5, label='TNG50 (with baryons)')
for run, c in (('CDM', 'k'), ('SIDM', 'C3')):
    e = S[(S.run == run) & (S.N == 0)].sort_values('z')
    ax[0].plot(e.z, e.eps2, '-o', c=c, ms=3, lw=1, label=f'Concerto {run}')
    ax[0].plot(e.z, e.eps2_na, ':o', c=c, ms=3, lw=0.8, mfc='white')
ax[0].set(yscale='log', xlim=(4.2, -0.2), ylim=(1.5e-4, 0.06), xlabel='redshift', ylabel=r'$\epsilon_{2,\rm all}$ (filled), $\epsilon_2$ (open) at 2 kpc')
ax[0].legend(frameon=False, fontsize=5, loc='lower left', handlelength=1.8, handletextpad=0.4)   # empty corner (PI)
for a, t in zip(ax, ('(a) quadrupole at 2 kpc', '(b) shape of the halo', '(c) its potential quadrupole')):
    a.set_title(t, loc='left', fontsize=8)
fig.savefig(ROOT / 'figures/paper_sims_sidm.png', dpi=300)
print(tng.round(4).to_string())
