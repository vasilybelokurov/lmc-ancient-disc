"""Letter text: every number of the Letter's account of how its five figures were made, as LaTeX macros (prefix \\lt), so that none is
typed by hand.

Two kinds of number.
- Settings: read from the code that uses them, never retyped here. Constants of the scripts are read from their source with ast
  (module-level assignments of literals; the scripts are not run), constants of lmcold by import, defaults of lmcold functions with
  inspect. Each macro names its source below.
- Results: computed from the results tables the figures are drawn from (and, for the Concerto particle mass, from the snapshot header).
Inputs: scripts/{paper_letter_signal, fit_kinematics_mcmc, sims_observe, paper_sims_kinematics, fetch_tng50, build_sims_birth,
paper_sims_birth, fetch_tng50_progenitors, sims_scramble, build_sims_vcirc, paper_letter_scramble, sims_halo_settle, fetch_concerto,
sims_sidm_shapes, paper_sims_sidm, kin_feh_lowest, kin_vc_drift}.py; results/{letter_signal_counts, kin_support_ellipsoid, kin_feh_lowest, kin_feh_lowest_null,
kin_vc_drift_h, giant_mass, sims_observed, sims_birth,
sims_scramble, sims_heating_gas, sims_vcirc, sims_halo_settle, sims_sidm_shapes}.csv; the z = 0 Concerto snapshot header.
Writes results/macros_letter.tex (copied to the paper's tables/; \\input in the Letter's preamble).
Run: .venv/bin/python scripts/manifest.py run paper_text_letter.py"""

# %% [markdown]
# ### Settings and helpers
# Imports, the dictionary M of macros (macro name without its prefix: the text it stands for) and the functions that format
# numbers as the text writes them. The inputs listed in the header are read in the step that uses them.

# %%
import re

import numpy as np
import pandas as pd

from fit_kinematics_mcmc import ring
from lmcold import samples, sims
from fetch_tng50_progenitors import FULL, FULL_SNAPS, TRACK_SNAPS
from lmcold.constants import LMC, LMC_MASS_LITERATURE
from lmcold.footprint import RHO_MAX
from lmcold.kinfit import R_SIG
from lmcold.mass_samples import FEH_EDGES
from lmcold.samples import FEH_BINS
from lmcold.sources import ROOT
from textnum import const, default, listing, num, sigfig, tex_sci


M = {}   # macro name (without the \lt prefix) -> the text it stands for
# Literal module-level constants of each script, read from its source with textnum.const (dicts name -> value)
sig, kin, obs, ksim = const('paper_letter_signal'), const('fit_kinematics_mcmc'), const('sims_observe'), const('paper_sims_kinematics')
tng, birth, pbirth = const('fetch_tng50'), const('build_sims_birth'), const('paper_sims_birth')
scr, vc, lsc, settle = const('sims_scramble'), const('build_sims_vcirc'), const('paper_letter_scramble'), const('sims_halo_settle')
conc, shp, psidm = const('fetch_concerto'), const('sims_sidm_shapes'), const('paper_sims_sidm')

# %% [markdown]
# ### Fig. 1: the signal in the data
# Settings and sample sizes of the Letter's Fig. 1.

# %%
# Fig. 1 (scripts/paper_letter_signal.py)
# Numbers of BOSS and APOGEE giants in Fig. 1 (one-row table)
counts = pd.read_csv(ROOT / 'results/letter_signal_counts.csv').iloc[0]
edges = [hi for lo, hi in FEH_BINS[:-1]]                              # lmcold.samples.FEH_BINS: -1.5, -1.1, -0.8, -0.5
# Settings of Fig. 1 (footprint radius, [Fe/H] edges, ring, pixel sizes, clipping, reference grid) and the
# sample sizes with thin spaces as thousands separators
M.update(RrlRho=num(RHO_MAX), FehA=num(edges[0]), FehB=num(edges[1]), FehC=num(edges[2]), FehD=num(edges[3]),
         RingIn=num(kin['RING'][0]), RingOut=num(kin['RING'][1]), RrlFitIn=num(kin['OUTER_INNER']), Pix=num(sig['PIX']), PixAlpha=num(100 * sig['PIX_ALPHA']),
         NBoss=f"{int(counts.boss):,}".replace(',', r'\,'), NApogee=f"{int(counts.apogee):,}".replace(',', r'\,'),
         RrPix=num(sig['RR_PIX']), RrSmooth=num(sig['RR_SMOOTH']), RrLevLo=num(100 * sig['RR_LEVELS'][0]),
         RrLevHi=num(100 * sig['RR_LEVELS'][1]), NLit=num(len(LMC_MASS_LITERATURE)),
         MassBin=num(np.diff(FEH_EDGES[1:-1])[0]), Clip=num(sig['CLIP']), ClipErr=num(sig['CLIP_ERR']),
         ProfIn=num(sig['PROFILE_RING'][0]), ProfOut=num(sig['PROFILE_RING'][1]), Sector=num(sig['SECTOR']),
         MedianSE=f"{sig['MEDIAN_SE']:.2f}", NMinPA=num(sig['NMIN']), RefDpa=num(sig['REF_DPA']),
         RefRhoA=num(sig['REF_RHO'][0]), RefRhoB=num(sig['REF_RHO'][1]), RefRhoStep=num(sig['REF_RHO'][2]))

# %% [markdown]
# ### Fig. 2: rotational support, LMC and simulations
# Settings of the mock observation, the fitted centre, the rotational support of the LMC bins against the simulated analogues,
# the early formed mass and the recovery test.

# %%
# Fig. 2 (scripts/sims_observe.py, paper_sims_kinematics.py, fetch_tng50.py)
# Per-bin fits; apo: True for the giant bins with an APOGEE-only v_rot (vrot4_apogee not NaN)
ks = pd.read_csv(ROOT / 'results/kin_support_ellipsoid.csv')
apo = ks[ks.population.str.startswith('giants')].reset_index(drop=True).vrot4_apogee.notna().to_numpy()
mf = pd.read_csv(ROOT / 'results/giant_mass.csv')                       # LMC cumulative formed mass, as sims_observe.lmc_knots
# Fraction of the LMC's formed mass below the upper edge of the most metal-poor bin
first = np.interp(edges[0], mf.feh_hi, mf.cumulative) / mf.cumulative.iloc[-1]
# Number of simulated analogues per suite
so = pd.read_csv(ROOT / 'results/sims_observed.csv', dtype={'name': str}).groupby('suite').name.nunique()
# Settings of Fig. 2: the radius of sigma, the analogue samples and their selection (stellar and total mass
# [1e9 Msun, Msun]), the mock-observation grid and windows
M.update(RSig=num(R_SIG), ApoFeh=num(FEH_BINS[int(np.argmax(apo))][0]), NTng=num(so['TNG50']), NAu=num(so['Auriga']),
         NFire=num(so['FIRE-2']), MstarLo=num(tng['MSTAR'][0] / 1e9), MstarHi=num(tng['MSTAR'][1] / 1e9),
         MtotLo=tex_sci(tng['MTOT'][0]), MtotHi=tex_sci(tng['MTOT'][1]),
         AuNames=listing([k.replace('Au', '') for k in sims.AURIGA]), FireNames=listing(list(sims.FIRE)),
         RMax=num(sims.R_MAX), AlignR=num(sims.ALIGN_R), Dist=num(LMC['D0']), FirstFrac=num(round(100 * first)),
         CellR=num(obs['CELL_R']), CellPhi=num(obs['CELL_PHI']), VlosLo=num(obs['VLOS_WINDOW'][0]), VlosHi=num(obs['VLOS_WINDOW'][1]),
         NMaxFit=num(obs['NMAX']), NMinSim=num(ksim['NMIN_SIM']))
# The fitted centre: its distance [deg] from the adopted (van der Marel & Kallivayalil 2014) centre, from the stage-2 shared dx, dy
# (tangent-plane offsets) of the joint ellipsoid fit (results/kin_mcmc_data_ellipsoid.csv), to one decimal.
kc = pd.read_csv(ROOT / 'results/kin_mcmc_data_ellipsoid.csv').query("stage == 2 and population == 'shared'").set_index('param')['median']
M.update(CentreOffset=f"{np.hypot(kc['dx'], kc['dy']):.1f}")
# Fig. 2 results. LMC (results/kin_support_ellipsoid.csv, stage-1 ellipsoid fits, posterior medians at R_SIG): the most metal-poor
# (MP) and most metal-rich (MR) giant bins, the MR bin also for APOGEE alone, and the RR Lyrae (sigma_R: proper motions alone do not
# measure sigma_z). Simulations (results/sims_kinematics_summary.csv, written by paper_sims_kinematics.py): medians, 16th-84th
# percentiles and maxima over the analogues in the MP bin; per cent of TNG50 analogues at or above the lower edge of the LMC band in
# v_rot/sigma_z per bin (as printed in Fig. 2c); analogues at or above it in the second bin.
g = ks[ks.population.str.startswith('giants')].reset_index(drop=True)
rl = ks[ks.population == 'RR Lyrae'].iloc[0]
sk = pd.read_csv(ROOT / 'results/sims_kinematics_summary.csv').set_index(['suite', 'population', 'quantity'])
# pct: per cent of TNG50 analogues at or above the LMC band, one value per giant bin
pct = [100 * sk.loc[('TNG50', k, 'ratio_z'), 'n_above_lmc'] / sk.loc[('TNG50', k, 'ratio_z'), 'n_analogues'] for k in range(len(g))]
M.update(KinVrotMP=num(round(g.vrot4.iloc[0])), KinVrotMR=num(round(g.vrot4.iloc[-1])),
         KinSigMP=num(round(g.sigma_z.iloc[0])), KinSigMR=num(round(g.sigma_z.iloc[-1])),
         KinRatioMP=num(round(g.ratio_z.iloc[0], 1)), KinRatioMR=num(round(g.ratio_z.iloc[-1], 1)),
         KinRatioMRApo=num(round(g.ratio_z_apogee.iloc[-1], 1)), KinSigRMP=num(round(g.sigma_R.iloc[0])),
         KinRatioRMP=num(round(g.ratio_R.iloc[0], 1)), KinRrlVrot=num(round(rl.vrot4)), KinRrlSigR=num(round(rl.sigma_R)),
         KinRrlRatioR=num(round(rl.ratio_R, 1)),
         TngVrotMP=num(round(sk.loc[('TNG50', 0, 'vrot4'), 'median'])), TngSigMP=num(round(sk.loc[('TNG50', 0, 'sigma_z'), 'median'])),
         TngSigMinMP=num(round(sk.loc[('TNG50', 0, 'sigma_z'), 'p16'])),
         TngRatioMP=f"{sk.loc[('TNG50', 0, 'ratio_z'), 'median']:.1f}",
         TngRatioMPLo=f"{sk.loc[('TNG50', 0, 'ratio_z'), 'p16']:.1f}",
         TngRatioMPHi=f"{sk.loc[('TNG50', 0, 'ratio_z'), 'p84']:.1f}",
         TngRatioMPMax=f"{sk.loc[('TNG50', 0, 'ratio_z'), 'max']:.1f}",
         AuRatioMP=f"{sk.loc[('Auriga', 0, 'ratio_z'), 'median']:.1f}",
         FireRatioMP=f"{sk.loc[('FIRE-2', 0, 'ratio_z'), 'median']:.1f}",
         AuRatioMRMax=f"{sk.loc[('Auriga', len(g) - 1, 'ratio_z'), 'max']:.1f}",
         FireRatioMRMax=f"{sk.loc[('FIRE-2', len(g) - 1, 'ratio_z'), 'max']:.1f}",
         TngNAboveB=num(sk.loc[('TNG50', 1, 'ratio_z'), 'n_above_lmc']),
         TngPctC=num(round(pct[2])), TngPctD=num(round(pct[3])), TngPctE=num(round(pct[4])))
# Early formed mass: the LMC's cumulative formed mass below the upper edge of the most metal-poor bin (results/giant_present_mass.csv,
# the table of Fig. 1b; bootstrap 16th-84th percentiles), an upper bound on the mass formed when those stars were born; and its ratio
# to the stellar mass at which a simulated Milky Way progenitor forms its disc, M_DISC_SEMENOV (Semenov et al. 2025a, ApJ, sections
# 3.3 and 4: 'when the galaxy stellar mass reaches ~2e9 Msun', in TNG50 and in their cold-ISM resimulation alike).
M_DISC_SEMENOV = 2e9
M_MW_SPINUP = 5e8   # Msun formed when the Milky Way spun up its disc (Belokurov & Kravtsov 2023; Kurbatov et al. 2024), as in the Introduction
gp = pd.read_csv(ROOT / 'results/giant_present_mass.csv')
# Cumulative formed mass [Msun] (median, p16, p84) at the upper edge of the most metal-poor bin;
# e7: its power of ten, used to write the mass as m x 10^e
em = {c: np.interp(edges[0], gp.feh_hi, gp[c]) for c in ('cumulative', 'cumulative_p16', 'cumulative_p84')}
e7 = 10 ** np.floor(np.log10(em['cumulative']))
M.update(EarlyMass=rf"{em['cumulative'] / e7:.0f}\times10^{{{int(np.log10(e7))}}}",
         EarlyMassLo=f"{em['cumulative_p16'] / e7:.0f}", EarlyMassHi=f"{em['cumulative_p84'] / e7:.0f}",
         EarlyExp=num(np.log10(e7)), EarlyRatio=num(10 * round(M_DISC_SEMENOV / em['cumulative'] / 10)),
         EarlyRatioMW=num(round(M_MW_SPINUP / em['cumulative'])))
# The most metal-poor bin's metallicities: its median [Fe/H] (results/kin_support_ellipsoid.csv) and the per cent of its fitted giants
# (lmcold.samples.giants at RING) below FEH_LOW, to the nearest 5.
FEH_LOW = -2.0
# [Fe/H] [dex] of the most metal-poor bin's giants within the fitted ring
mp = np.asarray(ring(samples.giants()[0][1])['feh'])
M.update(MPFehMed=f'{g.feh_med.iloc[0]:.1f}', MPFehLow=num(FEH_LOW), MPPctLow=num(5 * round(100 * (mp < FEH_LOW).mean() / 5)))
# Giants with measured [Fe/H] < FEH_LOW (scripts/kin_feh_lowest.py, results/kin_feh_lowest.csv and kin_feh_lowest_null.csv): their
# number, the formed mass below FEH_LOW as a power of ten (giant_present_mass.csv), v_rot(4 kpc) / sigma_z of the narrow bins below
# FEH_LOW (lowest and highest, 1 decimal), the per cent of BOSS-APOGEE overlap stars with calibrated BOSS [Fe/H] < FEH_LOW that have
# APOGEE [Fe/H] < FEH_LOW (1 significant figure), and the lowest purity of the null test at which every non-rotating mock (true
# ratio 0) falls below the 16th percentile of the measured value, and its complement, the largest interloper fraction.
kl = pd.read_csv(ROOT / 'results/kin_feh_lowest.csv').set_index('sample')
kn = pd.read_csv(ROOT / 'results/kin_feh_lowest_null.csv')
# low: fit of the whole [Fe/H] < FEH_LOW sample; narrow_low: v_rot/sigma_z of its two narrow bins
low = kl.loc[f'[Fe/H] < {FEH_LOW}']
narrow_low = kl.loc[['[Fe/H] < -2.2', '-2.2 < [Fe/H] < -2.0'], 'ratio_z']
# BOSS-APOGEE overlap stars (feh_a: APOGEE, feh_b: calibrated BOSS [Fe/H]); pur: fraction confirmed by APOGEE
ov = samples.overlap()
pur = (ov.feh_a[ov.feh_b < FEH_LOW] < FEH_LOW).mean()
# Null mocks with true ratio 0: the largest fitted v_rot/sigma_z at each purity
n0 = kn[(kn.bin == f'[Fe/H] < {FEH_LOW}') & (kn.ratio == 0)].groupby('purity').fit_ratio.max()
M.update(LowFeh=num(FEH_LOW), LowN=num(int(low.n)), LowMass=rf"10^{{{round(np.log10(np.interp(FEH_LOW, gp.feh_hi, gp.cumulative)))}}}",
         LowRatioLo=f'{narrow_low.min():.1f}', LowRatioHi=f'{narrow_low.max():.1f}', LowPurity=sigfig(100 * pur),
         LowPurityMin=num(round(100 * n0[n0 < low.ratio_z_p16].index.min())))
M.update(LowInterMax=num(100 - int(M['LowPurityMin'])))   # per cent of interlopers the conclusions allow: 100 - LowPurityMin
# The interlopers' APOGEE [Fe/H] (lowest and highest, 1 decimal) among those overlap stars, and settings of kin_feh_lowest.py read
# from its source: the lowest narrow-bin edge, the S/N of the best subset, the interloper bin, the hot fraction and dispersion.
inter = ov.feh_a[(ov.feh_b < FEH_LOW) & (ov.feh_a >= FEH_LOW)]
kfl = const('kin_feh_lowest')
M.update(LowInterLo=f'{inter.min():.1f}', LowInterHi=f'{inter.max():.1f}', LowNarrowFeh=num(kfl['BINS'][0][1]),
         LowSnr=num(kfl['SNR_BEST']), LowInterBinLo=num(kfl['INTERLOPER_BIN'][0]), LowInterBinHi=num(kfl['INTERLOPER_BIN'][1]),
         LowHotPct=num(round(100 * max(kfl['HOT']))), LowHotSigma=num(kfl['HOT_SIGMA']))
# Sizes: exponential scale lengths h [kpc] of the selection-weighted surface densities over H_FIT (scripts/kin_vc_drift.py,
# results/kin_vc_drift_h.csv) for the most metal-rich and most metal-poor giant bins and the RR Lyrae, and the half-mass radii
# R_e = RE_PER_H h that the exponential would give if it continued inside H_FIT[0] (projected exponential disc: R_e / h = 1.678).
RE_PER_H = 1.678
# Scale length h [kpc] per population, and the fitted radial range H_FIT [kpc]
hs = pd.read_csv(ROOT / 'results/kin_vc_drift_h.csv').set_index('population').h
h_fit = const('kin_vc_drift')['H_FIT']
# h and R_e [kpc] of the most metal-poor and most metal-rich giant bins and the RR Lyrae, 1 decimal
for key, pop in (('MP', f'giants [Fe/H] < {num(edges[0])}'), ('MR', f'giants [Fe/H] > {num(edges[3])}'), ('Rrl', 'RR Lyrae')):
    M.update({f'H{key}': f'{hs[pop]:.1f}', f'Re{key}': f'{RE_PER_H * hs[pop]:.1f}'})
M.update(HFitLo=num(h_fit[0]), HFitHi=num(h_fit[1]))
# Recovery (paper_sims_kinematics.py, figures/paper_sims_recovery.png): fitted against the particle values of the same mock giants,
# all analogues and bins of results/sims_observed.csv; median |v_rot - truth| [km/s, integer] and |sigma_z / truth - 1| [per cent].
so_all = pd.read_csv(ROOT / 'results/sims_observed.csv', dtype={'name': str})
M.update(RecVrot=num(max(1, round(abs(np.nanmedian(so_all.vrot4 - so_all.t_vphi))))),
         RecSig=num(max(1, round(100 * abs(np.nanmedian(so_all.sigma_z / so_all.t_sigma_z) - 1)))))
# TNG50 analogues as cold as the LMC in the MP bin (sigma_z at or below the upper edge of the LMC band, results/sims_observed.csv),
# and the sigma_z and v_rot of the coldest.
t0 = pd.read_csv(ROOT / 'results/sims_observed.csv', dtype={'name': str}).query("suite == 'TNG50' and population == 0")
# The TNG50 analogue with the lowest sigma_z [km/s] in the most metal-poor bin
cold = t0.loc[t0.sigma_z.idxmin()]
M.update(TngNColdMP=num((t0.sigma_z <= g.sigma_z_p84.iloc[0]).sum()), TngColdSig=num(round(cold.sigma_z)),
         TngColdVrot=num(round(cold.vrot4)))

# %% [markdown]
# ### Fig. 3: the simulated stars at birth
# Settings and results for the old cohorts at birth and today.

# %%
# Fig. 3 (scripts/build_sims_birth.py, paper_sims_birth.py, lmcold.sims)
# Birth cohorts of TNG50 and Auriga: v_phi and sigma_z [km/s] at birth and today, per cohort age [Gyr]
sb = pd.read_csv(ROOT / 'results/sims_birth.csv', dtype={'halo': str})
# Fig. 3 results: cohorts of OLD_AGES Gyr (the ages of the two most metal-poor LMC giant bins in Fig. 3), at birth and today. Each
# analogue's median over its cohorts in that range, then the median and 16th-84th percentiles over the analogues of each suite.
OLD_AGES = (10.0, 12.0)
# Cohorts of 10-12 Gyr; rb, rt: v_phi/sigma_z at birth and today (same stars)
ob = sb[(sb.age >= OLD_AGES[0]) & (sb.age < OLD_AGES[1])].assign(rb=lambda d: d.vphi_birth / d.sz_birth,
                                                                     rt=lambda d: d.vphi_same / d.sz_same)
# Per-analogue medians, then the 16th, 50th and 84th percentiles over the analogues of each suite
og = ob.groupby(['suite', 'halo'])[['rb', 'rt', 'vphi_birth', 'sz_birth', 'vphi_same', 'sz_same']].median()
oq = {s: og.loc[s].quantile([0.16, 0.5, 0.84]) for s in ('TNG50', 'Auriga')}
M.update(OldAgeLo=num(OLD_AGES[0]), OldAgeHi=num(OLD_AGES[1]),
         TngRbMed=num(round(oq['TNG50'].rb[0.5], 1)), TngRbLo=num(round(oq['TNG50'].rb[0.16], 1)),
         TngRbHi=num(round(oq['TNG50'].rb[0.84], 1)), TngRtMed=f"{oq['TNG50'].rt[0.5]:.1f}",
         TngRtLo=f"{oq['TNG50'].rt[0.16]:.1f}", TngRtHi=f"{oq['TNG50'].rt[0.84]:.1f}",
         TngVbMed=num(round(oq['TNG50'].vphi_birth[0.5])), TngVtMed=num(round(oq['TNG50'].vphi_same[0.5])),
         TngSbMed=num(round(oq['TNG50'].sz_birth[0.5])), TngStMed=num(round(oq['TNG50'].sz_same[0.5])),
         AuRbMed=num(round(oq['Auriga'].rb[0.5], 1)), AuRtMed=num(round(oq['Auriga'].rt[0.5], 1)),
         KinRatioB=num(round(g.ratio_z.iloc[1], 1)))
# Numbers of analogues and the settings of build_sims_birth.py, lmcold.sims and paper_sims_birth.py
M.update(NTngBirth=num(sb[(sb.suite == 'TNG50')].dropna(subset=['vphi_birth']).halo.nunique()),
         NAuBirth=num(sb[sb.suite == 'Auriga'].halo.nunique()), RToday=num(birth['R_TODAY']), Cohort=num(birth['COHORT']),
         MinCohort=num(birth['MIN_COHORT']), TrackGroup=num(default(sims.progenitor_track, 'group')),
         TrackSmooth=num(default(sims.progenitor_track, 'smooth')), RBirth=num(birth['R_BIRTH']),
         RMinBirth=num(default(sims.birth_phase_space, 'r_min')), MinIn=num(birth['MIN_IN']), PlotBin=num(pbirth['PLOT_BIN']),
         AgeMax=num(pbirth['AGE_MAX']))

# %% [markdown]
# ### Fig. 4: scrambling and the settling halo
# Settings and results of the three panels and the settling timescale.

# %%
# Fig. 4 (scripts/sims_scramble.py, build_sims_vcirc.py, sims_halo_settle.py, paper_letter_scramble.py, fetch_tng50_progenitors.py)
# Halo state per (progenitor, snapshot); zsnap: median redshift of each snapshot
hs = pd.read_csv(ROOT / 'results/sims_halo_settle.csv', dtype={'halo': str})
zsnap = hs.groupby('snap').z.median()
# Cohorts of the real stars (variant A), the star-forming gas at z >= Z_GAS, the orbital times, the time bins [Gyr]
sc = pd.read_csv(ROOT / 'results/sims_scramble.csv', dtype={'halo': str})
sc = sc[sc.variant == 'A']
gas = pd.read_csv(ROOT / 'results/sims_heating_gas.csv', dtype={'halo': str}).query(f"z >= {lsc['Z_GAS']}")
v = pd.read_csv(ROOT / 'results/sims_vcirc.csv', dtype={'halo': str})
te = lsc['TIME_EDGES']
# Settings of the scramble, orbital-time, settling and plotting scripts, and the sample sizes
M.update(NFull=num(len(FULL)), SnapA=num(FULL_SNAPS[0]), SnapB=num(FULL_SNAPS[-1]),
         ZSnapA=num(round(zsnap[FULL_SNAPS[0]], 1)), ZSnapB=num(round(zsnap[FULL_SNAPS[-1]], 1)),
         Window=num(1000 * scr['WINDOW']), RCohort=num(scr['R_COHORT']), MinStars=num(scr['MIN_STARS']),
         BirthEvery={2: 'second'}.get(scr['BIRTH_EVERY'], f"{scr['BIRTH_EVERY']}th"), DtMax=num(scr['DTMAX']),
         NCohorts=num(sc.groupby('halo').s_birth.nunique().sum()), NCohortGal=num(sc.halo.nunique()), RVel=num(scr['R_VEL']),
         RSupIn=num(scr['R_SUP'][0]), RSupOut=num(scr['R_SUP'][1]), Lmax=num(scr['LMAX']), Grid=num(scr['GRID']), RMinPot=num(scr['RMIN']),
         RMaxPot=num(scr['RMAX']), TimeEnd=num(te[-1]), TimeA=num(te[1]), TimeB=num(te[2]), TimeC=num(te[3]),
         ZGas=num(lsc['Z_GAS']), NGas=num(gas.halo.nunique()), ROrb=num(vc['R_ORB']), NOrbGal=num(v.halo.nunique()),
         TOrb=sigfig(v.t_orb.median()), TAfterA=num(lsc['T_AFTER'][0]), TAfterB=num(lsc['T_AFTER'][1]),
         NEps={4: 'quartiles'}.get(lsc['N_EPS'], f"{lsc['N_EPS']} quantiles"), SetLmax=num(settle['LMAX']), SetGrid=num(settle['GRID']),
         SetRMin=num(settle['RMIN']), SetRPot=num(settle['R_POT']), SetRZero=num(settle['R0']), SetRIn=num(settle['R_IN']),
         NSample=num(hs.halo.nunique()), ZSetHi=num(round(hs.z.max())), ZSetLo=num(round(hs.z.min())),
         TrackStep={3: 'third'}[TRACK_SNAPS[1] - TRACK_SNAPS[0]], NSettle=num(len(hs)), Epoch=num(lsc['EPOCH']))
# Fig. 4a results, as plotted by paper_letter_scramble.py: v_phi/sigma_z about the birth spin in the TIME_EDGES bins (per galaxy the
# median, then the median over galaxies) for the real stars (variant A, measured), frozen gravity (E), dark matter non-round only
# (G_DM) and axisymmetric (D0) test particles; the bins' median times; the star-forming gas (median over galaxies).
# All variants, plus the real stars as their own variant 'real' (measured v_phi/sigma_z); tb: median time
# [Gyr] of the TIME_EDGES bin of each row; va: median over galaxies of the per-galaxy medians per bin, for each variant
sa = pd.read_csv(ROOT / 'results/sims_scramble.csv', dtype={'halo': str}).replace([np.inf, -np.inf], np.nan)
sa = pd.concat([sa, sa[sa.variant == 'A'].assign(variant='real', vs_sim_birth=lambda d: d.vs_meas_birth)])
sa['tb'] = sa.groupby(pd.cut(sa.t, np.r_[-1e-9, te[1:-1], te[-1] + 1e-6])).t.transform('median')
va = {k: sa[sa.variant == k].groupby(['tb', 'halo']).vs_sim_birth.median().groupby(level=0).median()
      for k in ('real', 'E', 'G_DM', 'D0')}
# Values at the first, second and last time bins; ScrOrbEnd: the last time in units of the median orbital time
tb = va['real'].index
M.update(ScrStart=num(round(va['real'].iloc[0], 1)), ScrSecond=f"{va['real'].iloc[1]:.1f}",
         ScrEnd=num(round(va['real'].iloc[-1], 1)), ScrTSecond=f'{tb[1]:.2f}', ScrTEnd=sigfig(tb[-1]),
         ScrOrbEnd=num(round(tb[-1] / v.t_orb.median(), 1)), FrozenEnd=num(round(va['E'].iloc[-1], 1)),
         DmEnd=num(round(va['G_DM'].iloc[-1], 1)), AxiEnd=num(round(va['D0'].iloc[-1], 1)),
         GasRatio=num(round((gas.vphi / gas.sz).groupby(gas.halo).median().median(), 1)))
# Fig. 4b results, as paper_letter_scramble.py: each cohort's median v_phi/sigma_z T_AFTER Gyr after birth against eps2 at its birth
# snapshot (results/sims_halo_settle.csv); medians of eps2 and v_phi/sigma_z in the lowest and highest N_EPS-quantiles; Spearman rank
# correlation.
# af: median v_phi/sigma_z of each real cohort between T_AFTER[0] and T_AFTER[1] Gyr after birth;
# kb: joined to the halo state at its birth snapshot; qb: medians in N_EPS quantiles of eps2
re_ = sa[sa.variant == 'real']
af = re_[(re_.t > lsc['T_AFTER'][0]) & (re_.t <= lsc['T_AFTER'][1])].groupby(['halo', 's_birth']).vs_sim_birth.median().rename('vs')
kb = af.reset_index().merge(hs, left_on=['halo', 's_birth'], right_on=['halo', 'snap'], how='inner')
qb = kb.groupby(pd.qcut(kb.eps2, lsc['N_EPS']), observed=True).agg(x=('eps2', 'median'), m=('vs', 'median'))
M.update(EpsLow=sigfig(qb.x.iloc[0]), EpsHigh=sigfig(qb.x.iloc[-1]), VsEpsLow=num(round(qb.m.iloc[0], 1)),
         VsEpsHigh=sigfig(qb.m.iloc[-1]), EpsSpearman=sigfig(kb.vs.corr(kb.eps2, method='spearman')),
         NCohortsB=num(len(kb)))
# Fig. 4c results, as paper_letter_scramble.by_epoch: per galaxy the median in EPOCH-Gyr bins of lookback time, then the median over
# the progenitors; the earliest and latest bins (angles to the nearest 5 deg, the precision of a median over galaxies).
# Median over progenitors of the per-progenitor medians of psi [deg], eps2 and c/a in EPOCH-Gyr lookback bins
ep = hs.assign(tb=(hs.lookback / lsc['EPOCH']).round() * lsc['EPOCH']).groupby(['tb', 'halo'])[['misalign', 'eps2', 'c_a']].median()
ep = ep.groupby(level=0).median()
# Settling timescale (Fig. 4c medians): from the peak of psi until psi falls below half of that peak (alignment), and from the earliest
# bin until eps2 is within 25 per cent of its value in the latest bin (quadrupole); lookback bins are descending in time.
lbs = ep.sort_index(ascending=False)
# t_psi, t_eps [Gyr]: the alignment and quadrupole settling times defined above
pk = lbs.misalign.idxmax()
t_psi = pk - lbs[(lbs.index < pk) & (lbs.misalign < 0.5 * lbs.misalign.max())].index.max()
t_eps = lbs.index.max() - lbs[lbs.eps2 < 1.25 * lbs.eps2.iloc[-1]].index.max()
M.update(SettleLo=num(round(t_psi)), SettleHi=num(round(t_eps)))
# Earliest (largest lookback) and latest bins: psi to the nearest 5 deg, eps2 to 1 significant figure, c/a
M.update(LbEarly=num(ep.index.max()), LbLate=num(ep.index.min()), PsiEarly=num(5 * round(ep.misalign.iloc[-1] / 5)),
         PsiLate=num(5 * round(ep.misalign.iloc[0] / 5)), EpsEarly=sigfig(ep.eps2.iloc[-1]), EpsLate=sigfig(ep.eps2.iloc[0]),
         CaEarly=f'{ep.c_a.iloc[-1]:.2f}', CaLate=f'{ep.c_a.iloc[0]:.2f}')

# %% [markdown]
# ### Fig. 5: SIDM
# Settings and results of the CDM and SIDM Concerto runs, and the birth radii of the simulated early discs.

# %%
# Fig. 5 (scripts/fetch_concerto.py, sims_sidm_shapes.py, paper_sims_sidm.py)
# Shape profiles of the CDM and SIDM runs per redshift; zs: the redshifts in descending order, 1 decimal
ss = pd.read_csv(ROOT / 'results/sims_sidm_shapes.csv')
zs = [num(round(z, 1)) for z in sorted(ss.z.unique(), reverse=True)]
mpart = pd.read_csv(ROOT / 'results/sims_sidm_particle_mass.csv').set_index('run').m_part['CDM']   # from the snapshot header
# Settings of the Concerto runs (cross-section, velocity scale) and of the shape and plotting scripts
M.update(MPart=tex_sci(mpart), SigmaZero=num(conc['SIDM_SIGMA0_M']), WSidm=num(conc['SIDM_W']), ZList=listing(zs),
         CoreBox=num(shp['CORE_BOX']), NPeaks={5: 'five'}.get(shp['N_PEAKS'], num(shp['N_PEAKS'])), ShrinkA=num(shp['CORE_SHRINK'][0]),
         ShrinkB=num(shp['CORE_SHRINK'][1]), CoreR=num(shp['CORE_R']), RShellMin=num(shp['R_SHELL_MIN']),
         NShellMin=f"10^{{{int(np.log10(psidm['NMIN']))}}}", RKeep=num(shp['R_KEEP']), SLmax=num(shp['LMAX']), SGrid=num(shp['GRID']),
         SRMin=num(shp['RMIN']), FrameA=num(shp['FRAME_SHELL'][0]), FrameB=num(shp['FRAME_SHELL'][1]), REps=num(shp['R_EPS']),
         ZBin=num(psidm['Z_BIN']), ZMin=num(psidm['Z_MIN']), MinGal=num(psidm['MIN_GAL']))
# Fig. 5 results (results/sims_sidm_shapes.csv): eps2,all and eps2 (non-axisymmetric about the minor axis) at R_EPS for each run and
# redshift; c/a at R_EPS interpolated over the shells of at least NMIN particles (as plotted).
# e2: the rows at R_EPS per (run, z); ca: c/a at R_EPS per (run, z) from the shells with >= NMIN particles
e2 = ss[np.isclose(ss.r, shp['R_EPS'])].set_index(['run', 'z'])
ca = {k: np.interp(shp['R_EPS'], g.r, g.s) for k, g in ss[ss.N >= psidm['NMIN']].groupby(['run', 'z'])}
ca = pd.Series(ca)
zhi, zlate = ss.z.max(), 1.6   # the first redshift; 'late': z <= 1.5 (the 1.5 and 0 outputs)
# Ranges of eps2 over redshift for CDM; SIDM at the first redshift and late; CDM/SIDM ratio late (nearest 5); c/a
M.update(SiCdmEpsLo=sigfig(e2.loc['CDM'].eps2.min()), SiCdmEpsHi=sigfig(e2.loc['CDM'].eps2.max()),
         SiCdmNaLo=sigfig(e2.loc['CDM'].eps2_na.min()), SiCdmNaHi=sigfig(e2.loc['CDM'].eps2_na.max()),
         SiEpsFirst=sigfig(e2.loc[('SIDM', zhi)].eps2), SiNaFirst=sigfig(e2.loc[('SIDM', zhi)].eps2_na),
         SiEpsLate=sigfig(e2.loc['SIDM'].query(f'z < {zlate}').eps2.max()),
         SiRatioLate=num(5 * round(e2.loc['CDM'].query(f'z < {zlate}').eps2.min() / e2.loc['SIDM'].query(f'z < {zlate}').eps2.max() / 5)),
         SiCaCdmLo=f"{ca['CDM'].min():.1f}", SiCaCdmHi=f"{ca['CDM'].max():.1f}", SiCaSidmMin=f"{ca['SIDM'].min():.1f}",
         SiCaSidmMed=f"{ca['SIDM'].median():.1f}", SiZFirst=num(round(zhi)))

assert all(k.isalpha() for k in M), [k for k in M if not k.isalpha()]   # TeX command names: letters only
# Why the simulated early discs are measured at 0.5-3 kpc (results/sims_birth_radii.csv, scripts/sims_birth_radii.py; TNG50):
# star particles of the old cohorts at 3-5 kpc relative to 0.5-3 kpc; per cent of the cohorts with enough particles at 0.5-3 kpc
# that still have enough at 3-5 kpc (nearest 5); of the old stars at 3-5 kpc today, the median per cent born at 0.5-3 kpc and at
# 3-5 kpc (nearest 5).
# One row per TNG50 analogue: particle and cohort counts at 0.5-3 and 3-5 kpc, birth-radius fractions
br = pd.read_csv(ROOT / 'results/sims_birth_radii.csv', dtype={'halo': str}).query("suite == 'TNG50'")
M.update(RadPartRatio=f'{br.n_outer.sum() / br.n_inner.sum():.1f}',
         RadCohKept=num(5 * round(100 * br.coh_outer.sum() / br.coh_inner.sum() / 5)),
         RadBornInner=num(5 * round(100 * br.f_born_inner.median() / 5)),
         RadBornOuter=num(5 * round(100 * br.f_born_outer.median() / 5)))

# %% [markdown]
# ### Auriga and FIRE-2
# The Auriga and FIRE-2 numbers of the Letter, rounded from results/macros_sims_suites.tex, and FIRE-2 at birth.

# %%
# Section 3 summary: Auriga and FIRE-2 (results/macros_sims_suites.tex, written by scripts/paper_text_sims_suites.py from
# results/sims_birth.csv and the research tables; definitions there), rounded for the Letter.
# Macro name (without the \ss prefix) -> value, parsed from the \newcommand lines
ssm = dict(re.findall(r'\\newcommand\{\\ss(\w+)\}\{([^}]*)\}', (ROOT / 'results/macros_sims_suites.tex').read_text()))
M.update({f'Su{k}': f'{float(ssm[k]):.1f}' for k in ('OldAuAfter', 'OldFireAfter')})   # v_phi/sigma_z: 1 decimal
# Settings of paper_text_sims_suites.py: the age limit [Gyr] of the oldest stars and the time after birth [Gyr]
sus = const('paper_text_sims_suites')
M.update(SuOldLb=num(sus['OLD_LB']), SuTAfter=num(sus['T_AFTER']))
# FIRE-2 at birth with the definition of Fig. 3 (OLD_AGES cohorts; per galaxy the median, then the median over the galaxies):
# results/research/fire_early_birth.csv, cohorts of 0.1 Gyr (lb: lookback time at birth).
fb = pd.read_csv(ROOT / 'results/research/fire_early_birth.csv').query('width == 0.1')
# Cohorts born 10-12 Gyr ago; per-galaxy median v_phi/sigma_z, then the median over galaxies
fb = fb[(fb.lb >= OLD_AGES[0]) & (fb.lb < OLD_AGES[1])]
M.update(FireRbMed=f'{(fb.vphi / fb.sz).groupby(fb.halo).median().median():.1f}')
M.update({f'Su{k}': sigfig(float(ssm[k])) for k in ('AuRound', 'AuAxi')})   # fractions of the loss
# Number of Auriga cohorts in the halo tests
M.update(SuAuN=ssm['AuN'])

# %% [markdown]
# ### Write the macros
# One \newcommand per entry of M, with the prefix \lt, into results/macros_letter.tex; the text shows them as macros, so no
# number is typed by hand.

# %%
# Header line, then one \newcommand per macro; written to results/macros_letter.tex and printed
lines = ['% Numbers of the Letter (scripts/paper_text_letter.py); do not edit.']
lines += [rf'\newcommand{{\lt{k}}}{{{v}}}' for k, v in M.items()]
(ROOT / 'results/macros_letter.tex').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
