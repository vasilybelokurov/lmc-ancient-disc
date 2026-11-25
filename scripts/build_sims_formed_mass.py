"""Total formed stellar mass of every simulated analogue: results/sims_formed_mass.csv, the x-axis scale of Fig. sims_kinematics
(scripts/paper_sims_kinematics.py).

For each analogue of lmcold.sims.names() (681 TNG50, 5 Auriga, 3 FIRE-2): the stars within sims.R_MAX of the centre today
(lmcold.sims.load); formed mass = present mass / surviving fraction (lmcold.surviving_mass on the PARSEC grid, data/sources.yaml
key parsec_gaiaEDR3) for TNG50 and FIRE-2, GFM_InitialMass for Auriga; summed.
Reads the star particles of every analogue (scripts/fetch_tng50.py, fetch_auriga.py, fetch_fire2.py) and the PARSEC grid.
Writes results/sims_formed_mass.csv: suite, name, m_formed [Msun].

Run: .venv/bin/python scripts/manifest.py run build_sims_formed_mass.py   (reads the particles of all 689 analogues; 13 s in the
recorded run of docs/run_times.txt)
"""
# %% [markdown]
# ### Settings
# Only the simulation reader and the surviving-mass grid are needed.

# %%
import pandas as pd

from lmcold import sims, surviving_mass
from lmcold.sources import ROOT, path

if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data
    # The PARSEC grid gives the fraction of the formed mass of a stellar population still in stars at its age and metallicity; the
    # particles of each analogue are read in the next step by `lmcold.sims.load`.

    # %%
    survival = surviving_mass.survival_grid(pd.read_parquet(path('parsec_gaiaEDR3')))

    # %% [markdown]
    # ### Formed mass of every analogue
    # `lmcold.sims.load` returns each particle's formed mass `m_formed` [M☉] (the snapshot's birth mass for Auriga, present mass over
    # surviving fraction for TNG50 and FIRE-2); the sum over the particles within `sims.R_MAX` is the analogue's total.

    # %%
    m = pd.DataFrame([dict(suite=a, name=n, m_formed=sims.load(a, n, survival)['m_formed'].sum()) for a, n in sims.names()])
    m.to_csv(ROOT / 'results/sims_formed_mass.csv', index=False, float_format='%.4g')
