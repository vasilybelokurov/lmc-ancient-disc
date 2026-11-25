"""The scrambling experiment of Letter Fig. 4 for one TNG50 cohort, runnable without the simulation (KB page of Letter Fig. 4).

The stars of the cohort (progenitor 670104, born at snapshot 33, z = 2.0; results/scramble_demo_*, scripts/build_scramble_demo.py)
are moved as test particles for 0.5 Gyr through two AGAMA potentials of their progenitor's gravity at the birth snapshot, held
fixed (frozen at birth), as the variants E and D0_fr of scripts/sims_scramble.py:
  E      all the gravity (dark matter, gas and stars in one Multipole expansion, lmax 6);
  D0_fr  the dark matter and the baryons in two expansions with only the m = 0 terms kept: axisymmetric about the cohort's spin.
The frame is the full run's: origin at the stellar centre, z along the cohort's spin at birth. Because the origin follows the stars,
each potential gets the full run's UniformAcceleration term, minus the mean force on 500 stars within 2 kpc (constant for a frozen
potential). At the snapshot times after birth, v_phi/sigma_z of the stars about the z axis (sims_scramble.support: mass-weighted
mean azimuthal velocity over the vertical velocity dispersion) is compared with vs_sim_birth of results/sims_scramble.csv; the
script stops if any value differs by more than TOL.
Reads results/scramble_demo_*.{ini,csv} and results/sims_scramble.csv. Writes results/scramble_demo.csv (variant, t [Gyr since the
birth snapshot], vs: v_phi/sigma_z) and figures/scramble_demo.png.
Run: .venv/bin/python scripts/manifest.py run scramble_demo.py   (needs AGAMA; a few seconds)
"""
# %% [markdown]
# ### Settings
# The cohort, the tolerance of the comparison with the full run, and the functions of the full experiment
# (`scripts/sims_scramble.py`; importing it needs AGAMA but no simulation files).

# %%
import matplotlib

matplotlib.use('Agg')
import agama
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmcold.layout import panel_grid
from lmcold.sources import ROOT
from sims_scramble import KPC_KMS_GYR, accel, support

COHORT = ('670104', 33)   # progenitor (z = 0 subhalo id), birth snapshot: as scripts/build_scramble_demo.py
N_GRID = 101             # times of the curves in the figure, 0-0.46 Gyr
TOL = 0.01                # largest allowed |difference| of v_phi/sigma_z from the full run (its CSV is rounded to 5 decimals)
RES = ROOT / 'results'

# %% [markdown]
# ### Load the data
# The cohort's stars in the integration frame (positions in kpc, velocities in km/s, masses in Msun), the 500 stars that define the
# frame's acceleration, the snapshot times after birth, the coefficient files of the two potentials, and the full run's values.

# %%
stars = pd.read_csv(RES / 'scramble_demo_stars.csv')          # the stars followed in v_phi/sigma_z (r < 3, R > 0.5 kpc at birth)
cen = pd.read_csv(RES / 'scramble_demo_centre.csv').values    # (500, 3) kpc: stars within 2 kpc at birth
times = pd.read_csv(RES / 'scramble_demo_times.csv')          # t: AGAMA time unit (1 kpc / (km/s) = 0.9778 Gyr); t_gyr: Gyr
p_all = agama.Potential(str(RES / 'scramble_demo_E.ini'))             # all particles, all terms
p_dm = agama.Potential(str(RES / 'scramble_demo_D0_fr_dm.ini'))       # dark matter, m = 0 terms
p_bar = agama.Potential(str(RES / 'scramble_demo_D0_fr_bar.ini'))     # gas + stars, m = 0 terms
full = pd.read_csv(RES / 'sims_scramble.csv', dtype={'halo': str}).query('halo == @COHORT[0] and s_birth == @COHORT[1]')

# %% [markdown]
# ### The two potentials
# Each frozen potential plus the frame's acceleration: AGAMA's UniformAcceleration term with minus the mean force on the 500 stars,
# given at the first and last times (the same value: the potential does not change), as in `sims_scramble.run_cohort`.

# %%
t = times.t.values
ic = stars[['x', 'y', 'z', 'vx', 'vy', 'vz']].values          # (N, 6) initial conditions at the birth snapshot
m = stars.m.values
p_axi = agama.Potential(p_dm, p_bar)
pots = {'E': agama.Potential(p_all, accel([p_all, p_all], [cen, cen], t[[0, -1]])),
        'D0_fr': agama.Potential(p_dm, p_bar, accel([p_axi, p_axi], [cen, cen], t[[0, -1]]))}

# %% [markdown]
# ### Orbits and rotational support
# The orbits are integrated to the last time and read from AGAMA's interpolated trajectory, as in the full run: at the snapshot
# times (for the comparison) and on a grid of N_GRID times (for the curves); $v_\phi/\sigma_z$ about the cohort's spin at birth (the
# z axis) at each time.

# %%
tt = np.union1d(np.linspace(0, t[-1], N_GRID), t)              # AGAMA time units: the grid and the snapshot times
rows = []
for var, pot in pots.items():
    orbs = agama.orbit(potential=pot, ic=ic, time=t[-1], dtype=object, verbose=False)
    w6 = np.stack([o(tt) for o in orbs], axis=1)              # (time, star, 6): kpc, km/s
    rows += [dict(variant=var, t=tk * KPC_KMS_GYR, vs=support(s[:, :3], s[:, 3:], m, np.array([0., 0., 1.]))) for tk, s in zip(tt, w6)]
demo = pd.DataFrame(rows)
demo.to_csv(RES / 'scramble_demo.csv', index=False, float_format='%.5f')

# %% [markdown]
# ### Comparison with the full run
# The same quantity in `results/sims_scramble.csv` (`vs_sim_birth`), matched by variant and time.

# %%
key = lambda d: d.assign(it=(d.t * 1e5).round().astype(int))   # time in units of 1e-5 Gyr: the full run's rounding
cmp = key(demo).merge(key(full)[['variant', 'it', 'vs_sim_birth']], on=['variant', 'it'])   # the snapshot times only
dev = (cmp.vs - cmp.vs_sim_birth).abs()
print(cmp.assign(diff=cmp.vs - cmp.vs_sim_birth)[['variant', 't', 'vs', 'vs_sim_birth', 'diff']].to_string(index=False))
print(f'largest |difference| from the full run: {dev.max():.2e} (tolerance {TOL})')
assert len(cmp) == 2 * len(t) and dev.max() < TOL, 'the demo does not reproduce the full run'

# %% [markdown]
# ### Figure
# $v_\phi/\sigma_z$ against time since birth: the test particles of this run (lines), and at the snapshot times those of the full
# run (open circles) and the real stars (black squares). The figure is saved at the end of this cell.

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
fig, a = panel_grid(1, 1, width=3.4, top=0.75, aspect=0.75)   # room above the panel for the legend
a = a[0, 0]
VAR = {'E': ('all gravity, frozen at birth (E)', 'C1'), 'D0_fr': ('axisymmetric about the spin (D0_fr)', 'C4')}
for var, (lab, col) in VAR.items():
    d, f = demo[demo.variant == var], full[full.variant == var]
    a.plot(d.t, d.vs, '-', color=col, lw=1.2, label=lab)
    a.plot(f.t, f.vs_sim_birth, 'o', color=col, mfc='none', ms=5)
real = full[full.variant == 'E']                               # the real stars' values are repeated in every variant's rows
a.plot(real.t, real.vs_meas_birth, 'ks--', ms=3, lw=0.8, label='real stars (TNG50)')
a.plot([], [], 'o', color='0.4', mfc='none', ms=5, label='full run (sims_scramble.csv)')   # legend entry of the open circles
a.axhline(0, color='0.6', lw=0.5)
a.set(xlabel='time since birth [Gyr]', ylabel=r'$v_\phi/\sigma_z$', xlim=(-0.02, 0.5), ylim=(-2.2, 3.2))
a.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=2, fontsize=5, frameon=False)   # above the panel
fig.savefig(ROOT / 'figures/scramble_demo.png', dpi=250)
print('wrote figures/scramble_demo.png')
