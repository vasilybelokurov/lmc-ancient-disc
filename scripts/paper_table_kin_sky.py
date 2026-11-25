"""Paper table (Appendix, tab:kin_sky): Level-2 parameters of each group with sky-aligned dispersions, fitted on its own (stage 1).

Rows: the five giant [Fe/H] groups and the RR Lyrae at 2 < rho < 9 deg; then the APOGEE giants of the three metal-rich groups
alone. Columns: N; v_0 [km/s], sigma_los [km/s], v_0/sigma_los (results/kin_support.csv from scripts/paper_kin_support.py: posterior
median and 16th-84th percentiles, the ratio from the draws); R_0 [kpc] and the outlier fraction f [%] (stage-1 posterior,
results/kin_mcmc_data_stream.csv); sigma_PM [km/s] and v_0/sigma_PM from the 4 < rho < 9 deg fits (kin_support.csv columns sigma,
ratio_pm). APOGEE rows: maximum-likelihood v_0, sigma_los and their ratio with inverse-Hessian errors. Format: lmcold.tex.pm
(1 decimal for velocities, 2 for R_0 and ratios, 1 for f). The RR Lyrae have no line-of-sight velocities ('--').

Writes results/table_kin_sky.tex (the rows; copied to the paper's tables/table_kin_sky.tex).
Run: .venv/bin/python scripts/manifest.py run paper_table_kin_sky.py
"""
import numpy as np
import pandas as pd

from lmcold.sources import ROOT
from lmcold.tex import GROUP_LABEL, count, fout_percent, pm, pm_sym

# %% [markdown]
# ### Load the data
# The rotational-support summary per population and the stage-1 posteriors of the sky-aligned fits.

# %%
# One row per population: v0, sigma_los, sigma [km/s], ratios, with percentiles; APOGEE-only ML columns *_apogee.
S = pd.read_csv(ROOT / 'results/kin_support.csv').set_index('population')
# Stage-1 posterior summaries (each population on its own) of the sky-aligned 2-9 deg fit: population, param, median, p16, p84.
F = pd.read_csv(ROOT / 'results/kin_mcmc_data_stream.csv').query('stage == 1')


def stage1(pop, param):
    """(median, p16, p84) of a stage-1 parameter of population pop in the 2-9 deg sky-aligned fit."""
    r = F[(F.population == pop) & (F.param == param)].iloc[0]   # the one row of this population and parameter
    return r['median'], r.p16, r.p84


# %% [markdown]
# ### Write the rows
# One row per population (MCMC posteriors), then the APOGEE-only rows of the metal-rich groups (maximum likelihood).

# %%
rows = []   # LaTeX table rows
for pop, s in S.iterrows():
    # q(column, decimals): median with its percentile columns (suffixes _p16/_p84, or _lo/_hi for the ratios) via lmcold.tex.pm.
    q = lambda c, d, lo='_p16', hi='_p84': pm(s[c], s[c + lo], s[c + hi], d)
    rr = pop == 'RR Lyrae'   # no line-of-sight velocities: '--' in the sigma_los and ratio columns
    # Columns: group, N, v0, R0, sigma_los, v0/sigma_los, sigma_PM, v0/sigma_PM, outlier fraction [%].
    rows.append(' & '.join([GROUP_LABEL[pop], count(s.n), q('v0', 1), pm(*stage1(pop, 'R0'), 2),
                            '--' if rr else q('sigma_los', 1), '--' if rr else q('ratio', 2, '_lo', '_hi'), q('sigma', 1),
                            q('ratio_pm', 2, '_lo', '_hi'), f'{fout_percent(stage1(pop, "logit_fout")[0]):.1f}']) + r' \\')
rows.append(r'\hline')
# APOGEE-only rows (populations with a finite v0_apogee): N, ML v0, sigma_los, ratio with symmetric errors; other columns empty.
for pop, s in S[np.isfinite(S.v0_apogee)].iterrows():
    rows.append(' & '.join(['APOGEE ' + GROUP_LABEL[pop], count(s.n_apogee), pm_sym(s.v0_apogee, s.v0_apogee_err, 1), '',
                            pm_sym(s.sigma_los_apogee, s.sigma_los_apogee_err, 1), pm_sym(s.ratio_apogee, s.ratio_apogee_err, 2),
                            '', '', '']) + r' \\')
# Write the rows (copied to the paper's tables/table_kin_sky.tex) and print them.
(ROOT / 'results/table_kin_sky.tex').write_text('\n'.join(rows) + '\n')
print('\n'.join(rows))
