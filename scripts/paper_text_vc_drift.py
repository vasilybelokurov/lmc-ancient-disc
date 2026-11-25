"""Knowledge-base text, section 'Rotation and the circular speed' of the kinematics chapter: every number as a LaTeX macro (prefix
\\vd), from the tables of the shared-v_c + drift model.

Inputs: results/kin_vc_drift_data.csv, kin_vc_drift_mock.csv, kin_vc_drift_rrl.csv (scripts/kin_vc_drift.py),
kin_vc_drift_ext.csv, kin_vc_drift_deficit.csv (scripts/kin_vc_drift_extensions.py), kin_vc_drift_profiles.csv
(scripts/fig_kin_vc_drift.py); settings of scripts/kin_vc_drift.py; lmcold.kinfit.R_SIG.
Writes results/macros_vcdrift.tex.
Run: .venv/bin/python scripts/manifest.py run paper_text_vc_drift.py"""

# %% [markdown]
# ### Settings
# Imports, settings of scripts/kin_vc_drift.py and the macro store M.

# %%
import numpy as np
import pandas as pd

from kin_vc_drift import CURVE, H_FIT, MOCK_VC
from lmcold.kinfit import R_SIG
from lmcold.sources import ROOT
from textnum import num

R = ROOT / 'results'   # folder of the result tables
M = {}   # macro name (without the \vd prefix) -> text


def f1(x):
    """x with one decimal."""
    return f'{x:.1f}'


# %% [markdown]
# ### Free and drift fits
# Rotation speeds, circular speed, lag and scale heights of the free and shared-v_c + drift fits to the data.

# %%
d = pd.read_csv(R / 'kin_vc_drift_data.csv')   # one row per (model, population): vrot4, vc4 [km/s], h [kpc], ...
# Rows of the three models, each in population order (most metal-poor first).
free, drift, hfree = (d[d.model == m].reset_index(drop=True) for m in ('free', 'drift', 'drift, h free'))
lag = drift.vc4 - drift.vrot4   # km/s, circular speed minus rotation at R_SIG, per population
# R_SIG [kpc]; range of the free rotation speeds; v_c of the drift fit (measured and free h); lag of the first and last population;
# range of the exponential fits [kpc]; scale heights h [kpc]; NPar: rotation-curve parameters (CURVE) of all but one population.
M.update(RSig=num(R_SIG), VrotLo=f1(free.vrot4.min()), VrotHi=f1(free.vrot4.max()), VcFour=f1(drift.vc4.iloc[0]),
         VcHfree=f1(hfree.vc4.iloc[0]), LagMP=num(round(lag.iloc[0])), LagMR=num(round(lag.iloc[-1])),
         HFitIn=num(H_FIT[0]), HFitOut=num(H_FIT[1]),
         HMeasLo=f1(drift.h.min()), HMeasHi=f1(drift.h.max()), HFreeLo=f1(hfree.h.min()), HFreeHi=f1(hfree.h.max()),
         NPar=num(len(CURVE) * (len(free) - 1)))

# %% [markdown]
# ### Extensions of the model
# Log-likelihood differences and circular speeds of the variants.

# %%
e = pd.read_csv(R / 'kin_vc_drift_ext.csv')   # one row per (variant, population)
dl = e.groupby('variant', sort=False).dlnL.first()   # log-likelihood difference of each variant (same in its rows)
vc = e.groupby('variant', sort=False).vc4   # km/s, circular speed of each variant's rows
# Rounded log-likelihood differences of the variants (signs as the text quotes them), size of the matched sample, the gap between
# 'free + stream' and 'drift + stream', and the v_c range of 'drift + beta'.
M.update(DlnL=num(round(-dl['drift'])), DlnLHLo=num(round(-dl['drift, h p16'])), DlnLHHi=num(round(-dl['drift, h p84'])),
         DlnLMatched=num(round(-dl['matched: drift'])), NMatched=f"{int(e[e.variant == 'matched: drift'].n.sum()):,}".replace(',', '\\,'),
         DlnLStreamFree=num(round(dl['free + stream'])),
         GapStream=num(round(dl['free + stream'] - dl['drift + stream'])), DlnLBeta=num(round(-dl['drift + beta'])),
         VcBetaLo=f1(vc.min()['drift + beta']), VcBetaHi=f1(vc.max()['drift + beta']),
         DlnLStreamBeta=num(round(-dl['drift + stream + beta'])))

# %% [markdown]
# ### Deficit per bin
# The log-likelihood deficit of the drift model per giant bin.

# %%
de = pd.read_csv(R / 'kin_vc_drift_deficit.csv')
# Deficit per giant bin: dlnL summed over the bins in R (coordinate 'R'); range quoted with the sign reversed.
per_bin = de[de.coordinate == 'R'].groupby('population', sort=False).dlnL.sum()
M.update(DefLo=num(round(-per_bin.max())), DefHi=num(round(-per_bin.min())))

# %% [markdown]
# ### Mock test and RR Lyrae
# Recovery of the circular speed in mocks, and the RR Lyrae rotation predicted and measured.

# %%
m = pd.read_csv(R / 'kin_vc_drift_mock.csv')   # fits of the mock realisations
true_vc4 = MOCK_VC['v0'] * np.tanh(R_SIG / MOCK_VC['R0'])   # the mock curve at R_SIG (its linear term vanishes there; lmcold.kinfit.rotation)
rec = m[m.model == 'drift'].groupby('realisation').vc4.first() - true_vc4   # km/s, recovered minus true v_c per realisation
# True v_c and the largest recovery error.
M.update(MockVc=f1(true_vc4), MockDvc=f1(rec.abs().max()))
r = pd.read_csv(R / 'kin_vc_drift_rrl.csv', index_col=0)   # RR Lyrae rotation: columns predicted, measured; rows median, p16, p84
M.update(RrlPred=f1(r.predicted['median']), RrlPredLo=f1(r.predicted.p16), RrlPredHi=f1(r.predicted.p84),
         RrlMeas=f1(r.measured['median']), RrlMeasLo=f1(r.measured.p16), RrlMeasHi=f1(r.measured.p84))

# %% [markdown]
# ### Profiles
# The measured and model v_phi of the most metal-poor bin in the radial ranges the text quotes.

# %%
p = pd.read_csv(R / 'kin_vc_drift_profiles.csv')   # binned v_phi profiles: data_vphi, model_vphi [km/s] per bin lo-hi [kpc]
# The [Fe/H] < -1.5 giants, binned in R.
mp = p[(p.population == 'giants [Fe/H] < -1.5') & (p.coordinate == 'R')]
MID, OUT_FROM, INNER = (3, 5), 7, (2, 3)   # kpc: radial ranges of the comparison quoted (bins of kin_vc_drift_profiles.csv)
# Bins inside MID and bins beyond OUT_FROM; their data and drift-model v_phi ranges [km/s] are quoted as 'a--b'.
mid, out = mp[(mp.lo >= MID[0]) & (mp.hi <= MID[1])], mp[mp.lo >= OUT_FROM]
M.update(MidLo=num(MID[0]), MidHi=num(MID[1]), OutFrom=num(OUT_FROM), InLo=num(INNER[0]), InHi=num(INNER[1]),
         MidData=f'{mid[mid.model == "drift"].data_vphi.min():.0f}--{mid[mid.model == "drift"].data_vphi.max():.0f}',
         OutData=f'{out[out.model == "drift"].data_vphi.min():.0f}--{out[out.model == "drift"].data_vphi.max():.0f}',
         OutDrift=f'{out[out.model == "drift"].model_vphi.min():.0f}--{out[out.model == "drift"].model_vphi.max():.0f}')

# %% [markdown]
# ### Write the macros
# One \newcommand per entry of M, prefix \vd, into results/macros_vcdrift.tex.

# %%
# Header comment, then one \newcommand line per macro.
lines = ['% Numbers of the KB section Rotation and the circular speed (scripts/paper_text_vc_drift.py); do not edit.']
lines += [rf'\newcommand{{\vd{k}}}{{{x}}}' for k, x in M.items()]
(R / 'macros_vcdrift.tex').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
