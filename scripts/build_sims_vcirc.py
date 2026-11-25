"""Circular speed at R_ORB in the TNG50 progenitors when the early cohorts of Fig. sims_heating (c) were born: the orbital period that
sets the time axis of the letter's scrambling figure (scripts/paper_letter_scramble.py).

For every (galaxy, birth snapshot) of the 'full' cohorts of results/sims_heating_coherence.csv (scripts/paper_sims_heating.py: FULL
sample, cohorts born at z = 1.5-4), the main progenitor's cutout with stars, gas and dark matter (lmcold.sims.progenitor, kind 'full';
positions in physical kpc about SubhaloPos, the potential minimum): M(< R_ORB) summed over the three, v_c = sqrt(G M / R_ORB) (spherical
approximation; black holes are not in the cutouts) and the circular orbital period T_orb = 2 pi R_ORB / v_c.
Reads results/sims_heating_coherence.csv (scripts/build_sims_heating.py) and the full cutouts of the FULL progenitors
(scripts/fetch_tng50_progenitors.py).
Writes results/sims_vcirc.csv: halo, snap, lookback [Gyr], m_in [Msun], vc [km/s], t_orb [Gyr].
Run: .venv/bin/python scripts/manifest.py run build_sims_vcirc.py   (2 s in the recorded run of docs/run_times.txt)
"""
# %% [markdown]
# ### Settings
# The radius at which the circular speed is measured, and the constants for $v_c$ and the period.

# %%
import numpy as np
import pandas as pd

from lmcold import sims
from lmcold.constants import G, KPC_GYR
from lmcold.sources import ROOT

R_ORB = 2.0  # kpc, the radius at which the halo quadrupole eps2 is measured (scripts/sims_halo_settle.py)

# %% [markdown]
# ### Load the data
# The birth snapshots of the early cohorts, from `results/sims_heating_coherence.csv` (set 'full'); the cutouts are read in the next
# step.

# %%
coh = pd.read_csv(ROOT / 'results/sims_heating_coherence.csv', dtype={'halo': str})

# %% [markdown]
# ### Circular speed and orbital period at each birth snapshot
# The mass of stars, gas and dark matter within `R_ORB` gives $v_c=\sqrt{GM(<R_{\rm orb})/R_{\rm orb}}$ and
# $T_{\rm orb}=2\pi R_{\rm orb}/v_c$.

# %%
rows = []
for name, g in coh[coh.set == 'full'].groupby('halo'):
    b = sims.main_branch(name)
    for snap in sorted(g.s_birth.unique()):
        p = sims.progenitor(name, int(snap), 'full', b)
        m = sum(p[k]['m'][np.linalg.norm(p[k]['x'], axis=1) < R_ORB].sum() for k in ('stars', 'gas', 'dm') if k in p)   # Msun
        vc = np.sqrt(G * m / R_ORB)   # km/s (G in kpc (km/s)^2 / Msun, lmcold.constants)
        rows.append(dict(halo=name, snap=int(snap), lookback=p['lookback'], m_in=m, vc=vc, t_orb=2 * np.pi * R_ORB / vc * KPC_GYR))   # Gyr: 1 kpc/(km/s) = KPC_GYR Gyr
t = pd.DataFrame(rows)
t.to_csv(ROOT / 'results/sims_vcirc.csv', index=False, float_format='%.4g')
print(t.describe().round(3).to_string())
