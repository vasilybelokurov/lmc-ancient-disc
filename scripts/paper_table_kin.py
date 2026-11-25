"""Paper table (Section 4, tab:kin): rotation and velocity ellipsoid of each group fitted on its own (stage 1), at R_sigma = 4 kpc.

Rows: the five giant [Fe/H] groups (2 < rho < 9 deg) and the RR Lyrae (4 < rho < 9 deg); then the APOGEE giants of the three
metal-rich groups alone. Columns: N; v_rot(4 kpc), sigma_R, sigma_phi, sigma_z [km/s] and v_rot/sigma_R, v_rot/sigma_z (posterior
median and 16th-84th percentiles of the draws, results/kin_support_ellipsoid.csv from scripts/paper_kin_support_ellipsoid.py);
gamma = dlnsig_dR [1/kpc], k and the outlier fraction f [%] (stage-1 posterior, results/kin_mcmc_data_ellipsoid.csv, RR Lyrae
results/kin_mcmc_data_ellipsoid_outer.csv). APOGEE rows: maximum-likelihood values with inverse-Hessian errors (the *_apogee
columns of kin_support_ellipsoid.csv). Format: lmcold.tex.pm (1 decimal for velocities, 3 for gamma, 2 for ratios and k, 1 for f).
The RR Lyrae have no line-of-sight velocities: sigma_z and v_rot/sigma_z are '--'.

Writes results/table_kin.tex (the rows; copied to the paper's tables/table_kin.tex).
Run: .venv/bin/python scripts/manifest.py run paper_table_kin.py
"""
import numpy as np
import pandas as pd

from lmcold.sources import ROOT
from lmcold.tex import GROUP_LABEL, count, fout_percent, pm, pm_sym

# %% [markdown]
# ### Load the data
# The velocity-ellipsoid summary per population and the stage-1 posteriors of the giants' and the RR Lyrae fits.

# %%
# One row per population: n, posterior medians and p16/p84 of vrot4, sigma_R, sigma_phi, sigma_z, ratio_R, ratio_z [km/s or
# dimensionless], and the APOGEE-only maximum-likelihood values (*_apogee, *_apogee_err; NaN for the other groups).
S = pd.read_csv(ROOT / 'results/kin_support_ellipsoid.csv').set_index('population')
# Stage-1 posterior summaries (columns stage, population, param, median, p16, p84) of the giants' and the RR Lyrae fits.
FITS = {'giants': pd.read_csv(ROOT / 'results/kin_mcmc_data_ellipsoid.csv'),
        'RR Lyrae': pd.read_csv(ROOT / 'results/kin_mcmc_data_ellipsoid_outer.csv')}


def stage1(pop, param):
    """(median, p16, p84) of a stage-1 parameter of population pop."""
    t = FITS['RR Lyrae' if pop == 'RR Lyrae' else 'giants']   # the table holding pop
    r = t[(t.stage == 1) & (t.population == pop) & (t.param == param)].iloc[0]
    return r['median'], r.p16, r.p84


# %% [markdown]
# ### Write the rows
# One row per population (MCMC posteriors), then the APOGEE-only rows of the metal-rich groups (maximum likelihood).

# %%
rows = []   # LaTeX table rows
for pop, s in S.iterrows():
    # q(column, decimals): median and its 16th, 84th percentiles from the summary table, formatted by lmcold.tex.pm.
    q = lambda c, d: pm(s[c], s[f'{c}_p16'], s[f'{c}_p84'], d)
    rr = pop == 'RR Lyrae'   # no line-of-sight velocities: sigma_z columns are '--'
    # Columns: group, N, vrot4, sigma_R, sigma_phi, sigma_z, gamma, vrot/sigma_R, vrot/sigma_z, k, outlier fraction [%].
    rows.append(' & '.join([GROUP_LABEL[pop], count(s.n), q('vrot4', 1), q('sigma_R', 1), q('sigma_phi', 1),
                            '--' if rr else q('sigma_z', 1), pm(*stage1(pop, 'dlnsig_dR'), 3), q('ratio_R', 2),
                            '--' if rr else q('ratio_z', 2), pm(*stage1(pop, 'k'), 2),
                            f'{fout_percent(stage1(pop, "logit_fout")[0]):.1f}']) + r' \\')
rows.append(r'\hline')
# APOGEE-only rows: groups with a finite APOGEE fit; value +- inverse-Hessian error; gamma, k and f left empty.
for pop, s in S[np.isfinite(S.vrot4_apogee)].iterrows():
    a = lambda c, d: pm_sym(s[f'{c}_apogee'], s[f'{c}_apogee_err'], d)
    rows.append(' & '.join(['APOGEE ' + GROUP_LABEL[pop], count(s.n_apogee), a('vrot4', 1), a('sigma_R', 1), a('sigma_phi', 1),
                            a('sigma_z', 1), '', a('ratio_R', 2), a('ratio_z', 2), '', '']) + r' \\')
(ROOT / 'results/table_kin.tex').write_text('\n'.join(rows) + '\n')
print('\n'.join(rows))
