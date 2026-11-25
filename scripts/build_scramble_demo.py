"""Inputs of the runnable one-cohort scrambling experiment of the knowledge base (KB page of Letter Fig. 4, scripts/scramble_demo.py).

scripts/sims_scramble.py moves the stars of every early TNG50 cohort as test particles through AGAMA potentials of their progenitor
and needs the TNG50 particle cutouts. This script extracts, for one cohort (COHORT: progenitor 670104, born at snapshot 33,
z = 2.0), the few things that the frozen-at-birth variants E and D0_fr of that experiment need, so that scripts/scramble_demo.py can
repeat them without the cutouts. It makes them with the functions of scripts/sims_scramble.py and the same steps as its run_cohort
(the cohort, the integration frame, the expansions at the birth snapshot); these steps are restated here, not changed.

Reads: the TNG50 full cutouts of the progenitor at snapshots 21-40 and its SubLink tree (lmcold.sims.progenitor, main_branch;
scripts/fetch_tng50_progenitors.py), in ~/data.
Writes (integration frame of sims_scramble.py: origin at the stellar centre, z along the cohort's spin at its birth snapshot;
kpc, km/s, Msun):
  results/scramble_demo_E.ini                 AGAMA Multipole coefficients of all particles (dark matter, gas, stars) within 60 kpc
                                              at the birth snapshot: variant E (all gravity, frozen at birth);
  results/scramble_demo_D0_fr_dm.ini,         the dark matter and the baryons (gas + stars) in separate expansions with only the
  results/scramble_demo_D0_fr_bar.ini         m = 0 terms kept: variant D0_fr (all gravity axisymmetric about the cohort's spin);
  results/scramble_demo_stars.csv             the cohort members followed in v_phi/sigma_z (x, y, z, vx, vy, vz, m);
  results/scramble_demo_centre.csv            the 500 stars within 2 kpc whose mean force defines the frame's acceleration (x, y, z);
  results/scramble_demo_times.csv             the times of the snapshots after birth at which v_phi/sigma_z is measured (t in
                                              AGAMA's unit 1 kpc / (km/s) = 0.9778 Gyr, t_gyr in Gyr).
Run: .venv/bin/python scripts/manifest.py run build_scramble_demo.py   (needs the TNG50 cutouts and AGAMA)
"""
# %% [markdown]
# ### Settings
# The cohort, named by its progenitor (z = 0 subhalo id of TNG50-1) and its birth snapshot, and the functions of the full experiment.

# %%
import shutil
import tempfile

import numpy as np
import pandas as pd

from fetch_tng50_progenitors import FULL_SNAPS
from lmcold import sims
from lmcold.sources import ROOT
from sims_scramble import (DTMAX, KPC_KMS_GYR, M0, R_COHORT, R_MEMBER, WINDOW, expansion, frame, in_ring, keep_terms, load,
                           particles)

COHORT = ('670104', 33)   # progenitor, birth snapshot: its real stars and variant E lose their rotation, variant D0_fr keeps it
OUT = ROOT / 'results'    # output directory of the demo inputs
FMT = '%.17g'             # full float64 precision, so that the demo starts from the same numbers as the full run

# %% [markdown]
# ### Load the data
# The progenitor's particles at every snapshot 21-40 (z = 4-1.5), centred on its stars as in the full run (`sims_scramble.load`).

# %%
name, s_birth = COHORT
# SubLink main branch of the progenitor: its subhalo id at each snapshot.
branch = sims.main_branch(name)
# One dict per snapshot with a cutout (keys used below: 's' snapshot, 'lb' lookback time [Gyr], 'a' scale factor, 'st' stars).
snaps = [d for d in (load(name, s, branch) for s in FULL_SNAPS) if d is not None]
# Index and data of the birth snapshot.
i0 = next(i for i, c in enumerate(snaps) if c['s'] == s_birth)
c0 = snaps[i0]

# %% [markdown]
# ### The cohort and its frame
# As in `sims_scramble.one` and `run_cohort`: the stars formed in the 25 Myr before the birth snapshot within 3 kpc; their members
# within 10 kpc at birth; the subset at r < 3 kpc and cylindrical R > 0.5 kpc whose $v_\phi/\sigma_z$ is followed (`in_ring`); the
# frame with z along the members' spin; the snapshots up to 0.5 Gyr after the mean formation time.

# %%
st = c0['st']
age = st['age_form'] - c0['lb']                                                    # Gyr formed before the birth snapshot
k = (age >= 0) & (age < WINDOW) & (np.linalg.norm(st['x'], axis=1) < R_COHORT)     # the cohort
t_form = float(np.average(st['age_form'][k], weights=st['m'][k]))                  # mean formation time (lookback, Gyr)
w0 = np.isin(st['ids'], st['ids'][k]) & (np.linalg.norm(st['x'], axis=1) < R_MEMBER)
xs, vs, ms = st['x'][w0], st['v'][w0], st['m'][w0]                                 # members at birth (kpc, km/s, Msun)
sel = in_ring(xs, vs, ms)                                                          # followed in v_phi/sigma_z
n = (ms[:, None] * np.cross(xs, vs)).sum(0)
n /= np.linalg.norm(n)                                                             # cohort's spin at birth
R = frame(n)                                                                       # rows: integration-frame axes
later = [i for i in range(i0, len(snaps)) if t_form - snaps[i]['lb'] <= DTMAX + 1e-6]
t = np.array([(c0['lb'] - snaps[i]['lb']) / KPC_KMS_GYR for i in later])           # AGAMA time units since birth
xc = st['x'][np.linalg.norm(st['x'], axis=1) < 2] @ R.T                            # stars within 2 kpc, integration frame
cen = xc[np.random.default_rng(0).choice(len(xc), min(500, len(xc)), replace=False)]   # as run_cohort's pick (seed 0)
print(f'{name} snapshot {s_birth}: z = {1 / c0["a"] - 1:.2f}, {len(ms)} members, {sel.sum()} followed, {len(t)} times')

# %% [markdown]
# ### Write the inputs
# The expansions at the birth snapshot (all particles; dark matter and baryons separately with only m = 0 kept), the followed
# stars in the integration frame, the frame-acceleration stars and the times.

# %%
# Variant E: one Multipole expansion of all particles (integration frame R), written to a temporary file and copied to results/.
with tempfile.TemporaryDirectory() as tmp:
    shutil.copy(expansion(*particles(c0, ('dm', 'gas', 'st'), R), f'{tmp}/all.ini'), OUT / 'scramble_demo_E.ini')
    # Variant D0_fr: dark matter and baryons expanded separately, then only the m = 0 terms kept (M0, sims_scramble.keep_terms).
    for part, parts in (('dm', ('dm',)), ('bar', ('gas', 'st'))):
        f = expansion(*particles(c0, parts, R), f'{tmp}/{part}.ini')
        shutil.copy(keep_terms(f, M0, 'D0_fr'), OUT / f'scramble_demo_D0_fr_{part}.ini')
# Followed members rotated into the integration frame: positions [kpc] and velocities [km/s], shape (N, 3) each.
x, v = xs[sel] @ R.T, vs[sel] @ R.T
pd.DataFrame(np.column_stack([x, v, ms[sel]]), columns=['x', 'y', 'z', 'vx', 'vy', 'vz', 'm']).to_csv(
    OUT / 'scramble_demo_stars.csv', index=False, float_format=FMT)
# Frame-acceleration stars [kpc] and the measurement times (AGAMA units and Gyr).
pd.DataFrame(cen, columns=['x', 'y', 'z']).to_csv(OUT / 'scramble_demo_centre.csv', index=False, float_format=FMT)
pd.DataFrame({'t': t, 't_gyr': t * KPC_KMS_GYR}).to_csv(OUT / 'scramble_demo_times.csv', index=False, float_format=FMT)
print('wrote results/scramble_demo_*')
