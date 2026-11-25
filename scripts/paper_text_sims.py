"""Knowledge-base text, chapters 'Simulations: analogues and mock observation' and 'Simulations: birth, scrambling, halo settling,
SIDM' (extended draft Section 6 and its appendix): the numbers of the text that no other generator writes, as LaTeX macros (prefix
\\sm), so that none is typed by hand. The other numbers of these pages are the macros of scripts/paper_text_letter.py (\\lt),
paper_text_sims_birth.py (\\sb, \\sh) and paper_text_sims_suites.py (\\ss).

As scripts/paper_text_shape.py: settings are read from the code (scripts/textnum.const, lmcold by import), results from the tables
the figures are drawn from, with the statistics of the figure scripts. Where a figure script fixes a value inside its code (a bin
edge, an age cut), the value is repeated below with the script named. Literature values quoted with their citation stay in the text.
Inputs: results/{kin_support_ellipsoid, giant_mass, sims_observed, sims_formed_mass, sims_kinematics_summary, sims_mass_counts,
sims_definitions, sims_populations, sims_scramble, sims_halo_cohorts, sims_halo_settle, sims_heating_coherence, sims_heating_gas,
sims_sidm_shapes, auriga_scramble}.csv; scripts/{sims_observe, paper_sims_kinematics, paper_sims_settle, paper_text_sims_birth,
paper_text_sims_suites, build_sims_heating, sims_sidm_shapes}.py; results/mass_survival_grid.csv. Simulation files (sources.yaml):
the Auriga and FIRE-2 snapshot headers and stars (dark-matter particle masses, formed mass), the TNG50 progenitor cutouts (late-cohort
redshifts; one full snapshot for the force check of the multipole expansion), the Concerto Rockstar catalogues (virial masses).
Checks run here: the expansion's force against a softened direct sum, and AGAMA's coefficient normalisation on a Miyamoto-Nagai disc
(AGAMA; ~10 s in all).
Writes results/macros_sims.tex.
Run: .venv/bin/python scripts/manifest.py run paper_text_sims.py"""

# %% [markdown]
# ### Settings and helpers
# Imports, the dictionary M of macros (macro name without its prefix: the text it stands for) and the functions that format
# numbers as the text writes them. The inputs listed in the header are read in the step that uses them.

# %%
import ast
import glob
import re
import tempfile

import agama
import h5py
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

import sims_scramble as scr
from fetch_tng50_progenitors import GAS_SNAPS
from lmcold import sims
from lmcold.constants import G
from lmcold.sources import ROOT
from paper_sims_kinematics import NMIN_SIM, lmc_band
from sims_observe import lmc_knots
from sims_sidm_shapes import RUNS, SNAPS as SIDM_SNAPS, halos
from textnum import const, num, sigfig

R = ROOT / 'results'  # folder of the result tables
M = {}  # macro name (without the \sm prefix) -> text
# literal module constants of the scripts whose settings the text quotes
obs, settle, sbirth, suites = const('sims_observe'), const('paper_sims_settle'), const('paper_text_sims_birth'), const('paper_text_sims_suites')
heat, sidm = const('build_sims_heating'), const('sims_sidm_shapes')
# QUIET of paper_sims_settle.py is not a pure literal (it uses EPS_QUIET): its expression is evaluated with EPS_QUIET supplied
QUIET = next(eval(compile(ast.Expression(n.value), 'QUIET', 'eval'), {'EPS_QUIET': settle['EPS_QUIET']})   # paper_sims_settle.QUIET
             for n in ast.parse((ROOT / 'scripts/paper_sims_settle.py').read_text()).body
             if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'QUIET')


def f(x, d=2):
    """x with d decimals."""
    return f'{x:.{d}f}'


def signed(x, d=2):
    """x with d decimals and an explicit '+' when not negative."""
    s = f(x, d)
    return s if s.startswith('-') else '+' + s


def sci(x, d=1):
    """x as m\\times10^e, m with d decimals (d = 0: one significant figure)."""
    e = int(np.floor(np.log10(x)))
    return rf'{x / 10 ** e:.{d}f}\times10^{{{e}}}'


def bound(x):
    """10^e with e the smallest integer such that 10^e > x (an upper bound on a small difference)."""
    return rf'10^{{{int(np.floor(np.log10(x))) + 1}}}'


# %% [markdown]
# ### LMC bins and the observed analogues
# The LMC giant bins and the mock observations of the analogues.

# %%
# LMC giant bins (results/kin_support_ellipsoid.csv: stage-1 ellipsoid fits at R = 4 kpc) and formed mass (results/giant_mass.csv)
# stage-1 ellipsoid summaries of the LMC populations; giant bins only, in [Fe/H] order
k = pd.read_csv(R / 'kin_support_ellipsoid.csv')
k = k[k.population.str.startswith('giants')].reset_index(drop=True)
# (x, y) knots of the LMC's cumulative formed-mass fraction against [Fe/H] (sims_observe.lmc_knots)
knots = lmc_knots()
frac = np.interp(k.feh_med, *knots)                                  # formed-mass fraction below each bin's median [Fe/H]
edges = np.interp([-1.5, -1.1, -0.8, -0.5], *knots)                   # ... below the bin edges (paper_sims_kinematics.py)
m_lmc = pd.read_csv(R / 'giant_mass.csv').cumulative.iloc[-1]         # formed mass within RHO_MASS
# settings of the mock observation; mass fractions and masses [Msun] below the bins; LMC v_rot/sigma_z of the most metal-poor
# bin and its half-width; second bin; anisotropy sigma_R/sigma_z; factor between the last and first bins (all, APOGEE-only)
M.update(RhoMass=num(obs['RHO_MASS']), TruthDR=num(obs['TRUTH_DR']), FirstFrac=f(100 * edges[0], 1), FracB=f(100 * edges[1], 1),
         FracC=num(round(100 * edges[2])), MLmc=sci(m_lmc), MassA=sci(frac[0] * m_lmc), MassB=sci(frac[1] * m_lmc),
         NRealMP=num(k.n[0]), LmcRatioMP=f(k.ratio_z[0]), LmcRatioMPErr=f((k.ratio_z_p84[0] - k.ratio_z_p16[0]) / 2),
         LmcRatioB=f(k.ratio_z[1]), LmcVrotB=num(round(k.vrot4[1])), LmcSzB=num(round(k.sigma_z[1])),
         AnisoLmc=f(k.sigma_R[0] / k.sigma_z[0]), ShapeLmc=f(k.sigma_z[0] / k.sigma_R[0]),
         FactorLo=sigfig(k.ratio_z.iloc[-1] / k.ratio_z[0]), FactorHi=sigfig(k.ratio_z_apogee.iloc[-1] / k.ratio_z[0]))

# The observed analogues (results/sims_observed.csv: one row per analogue and bin; bins of fewer than NMIN_SIM mock giants not used)
s = pd.read_csv(R / 'sims_observed.csv', dtype={'name': str})
ok = s[s.n >= NMIN_SIM].copy()
# ratio: measured v_rot(4 kpc) / sigma_z of each analogue's bin
ok['ratio'] = ok.vrot4 / ok.sigma_z
# recovery errors: dv [km/s], fractional sigma_z error, and ratio error, against the particle truths t_vphi, t_sigma_z
dv, dsz, dr = ok.vrot4 - ok.t_vphi, ok.sigma_z / ok.t_sigma_z - 1, ok.ratio - ok.t_vphi / ok.t_sigma_z
mp = ok.population == 0  # the most metal-poor bin (population 0)
# numbers of bins and analogues; median, 16th and 84th percentiles of the recovery errors (all bins, and the metal-poor bin)
M.update(NBins=num(len(ok)), NAll=num(s.groupby(['suite', 'name']).ngroups), NMockMP=num(round(s.n[s.population == 0].median())),
         RecDv=signed(dv.median(), 1), RecDvLo=signed(dv.quantile(.16), 1), RecDvHi=signed(dv.quantile(.84), 1),
         RecSig=f(100 * dsz.median(), 1), RecMPLo=signed(dv[mp].quantile(.16), 0), RecMPHi=signed(dv[mp].quantile(.84), 0),
         RecSzMPLo=signed(100 * dsz[mp].quantile(.16), 0), RecSzMPHi=signed(100 * dsz[mp].quantile(.84), 0),
         RecRatioMP=signed(dr[mp].median()), RecRatioMPLo=signed(dr[mp].quantile(.16)), RecRatioMPHi=signed(dr[mp].quantile(.84)))
# formed mass [Msun] of each analogue (within the same radius)
mass = pd.read_csv(R / 'sims_formed_mass.csv', dtype={'name': str}).m_formed
M.update(MSim=sci(mass.median()), MassRatio=f(mass.median() / m_lmc, 1),
         AnisoSim=f((ok.sigma_R / ok.sigma_z)[mp].median()), ShapeLo=f((ok.sigma_z / ok.sigma_R)[mp].quantile(.16)),
         ShapeHi=f((ok.sigma_z / ok.sigma_R)[mp].quantile(.84)))

# %% [markdown]
# ### Particle masses and formed mass
# Dark-matter particle masses from the snapshot headers, and the formed mass of the Auriga stars against the conversion used for
# TNG50 and FIRE-2 (needs the simulation files).

# %%
# Dark-matter particle masses [Msun], from the snapshot headers: Auriga, the high-resolution type 1 (MassTable); FIRE-2, the type-1
# Masses (all equal); TNG50, a full progenitor cutout of scripts/fetch_tng50_progenitors.py (the one of the force check below)
FORCE_HALO, FORCE_SNAP = '708636', 30   # progenitor and snapshot (z = 2.3) of the force check
# the TNG50 progenitor of the force check (particles of all types at snapshot FORCE_SNAP)
prog = scr.load(FORCE_HALO, FORCE_SNAP, sims.main_branch(FORCE_HALO))
m_au = []
for d in sims.AURIGA.values():
    # first file of the z = 0 snapshot (251): MassTable[1], 1e10 Msun/h -> Msun
    with h5py.File(sorted(glob.glob(str(d / 'snapdir_251/snapshot_251.*.hdf5')))[0]) as fh:
        m_au.append(fh['Header'].attrs['MassTable'][1] * 1e10 / fh['Header'].attrs['HubbleParam'])
m_fire = []
for d in sims.FIRE.values():
    # z = 0 snapshot (600): every type-1 mass, which must all be equal
    with h5py.File(d / 'output/snapshot_600.hdf5') as fh:
        mf = np.unique(fh['PartType1/Masses'][:])
        assert len(mf) == 1, 'FIRE-2: one dark-matter particle mass'
        m_fire.append(mf[0] * 1e10 / fh['Header'].attrs['HubbleParam'])
assert np.ptp(m_fire) / np.mean(m_fire) < 1e-3, 'the text: one FIRE-2 dark-matter particle mass'
# ranges [1e4 Msun] and the factor between the TNG50 and the median Auriga/FIRE-2 dark-matter particle mass
M.update(DmAuLo=f(min(m_au) / 1e4, 1), DmAuHi=f(max(m_au) / 1e4, 1), DmFire=sci(m_fire[0]),
         DmFactor=num(round(prog['dm']['m'][0] / np.median(m_au + m_fire))))
# The formed mass that the Kroupa conversion (present mass / present_fraction, results/mass_survival_grid.csv of
# scripts/build_present_mass.py, as lmcold.sims.load for TNG50 and FIRE-2) gives for the Auriga stars, against their GFM_InitialMass
grid = pd.read_csv(R / 'mass_survival_grid.csv')
au_below = []
for name in sims.AURIGA:
    # stars of each Auriga analogue, with present mass m and GFM_InitialMass (m_formed)
    st = sims.load('Auriga', name, grid)
    # per cent by which the Kroupa conversion falls below the true initial mass
    au_below.append(100 * (1 - (st['m'] / sims.on_grid(grid, 'present_fraction', st['feh'], st['age'])).sum() / st['m_formed'].sum()))
M.update(AuFormedLo=num(round(min(au_below))), AuFormedHi=num(round(max(au_below))))

# %% [markdown]
# ### Rotation and dispersion against metallicity
# Medians and ranges over the analogues per bin, and the values at the LMC's formed mass.

# %%
# Rotation and dispersion against metallicity (results/sims_kinematics_summary.csv, written by paper_sims_kinematics.py)
# per suite, bin and quantity: percentiles over the analogues and counts above the LMC band
sk = pd.read_csv(R / 'sims_kinematics_summary.csv').set_index(['suite', 'population', 'quantity'])
# the LMC band of v_rot(4 kpc) per bin (paper_sims_kinematics.lmc_band): (lo, hi) arrays
lo, hi = lmc_band(k, 'vrot4')
# Auriga analogues in the most metal-rich bin (population 4) that reach the LMC band
au_mr = ok[(ok.suite == 'Auriga') & (ok.population == 4)].vrot4
fast = np.sort(au_mr[au_mr >= lo[4]].to_numpy())
# LMC sigma_z band per bin; Auriga and FIRE-2 analogues must all lie above it
lo_sz, hi_sz = lmc_band(k, 'sigma_z')
hot = ok[ok.suite.isin(['Auriga', 'FIRE-2'])]
assert (hot.sigma_z.to_numpy() > hi_sz[hot.population.to_numpy()]).all(), 'the text: all Auriga and FIRE-2 analogues are hotter'
M.update(VrotMRLo=num(round(sk.loc[('TNG50', 4, 'vrot4'), 'p16'])), VrotMRHi=num(round(sk.loc[('TNG50', 4, 'vrot4'), 'p84'])),
         NAuFast={2: 'two'}.get(len(fast), num(len(fast))), AuFast=' and '.join(num(round(x)) for x in fast))
# per suite, metal-poor bin: v_rot range (TNG50: 16th-84th percentiles; Auriga, FIRE-2: min-max), median sigma_z and ratio
for tag, suite in (('Tng', 'TNG50'), ('Au', 'Auriga'), ('Fire', 'FIRE-2')):
    g = ok[mp & (ok.suite == suite)]
    M.update({f'{tag}VrotMPLo': num(round(g.vrot4.quantile(.16) if tag == 'Tng' else g.vrot4.min())),
              f'{tag}VrotMPHi': num(round(g.vrot4.quantile(.84) if tag == 'Tng' else g.vrot4.max())),
              f'{tag}SzMP': num(round(g.sigma_z.median())), f'{tag}RatioMP': f(g.ratio.median())})
g = ok[mp & (ok.suite == 'TNG50')]
# TNG50 metal-poor medians and percentiles; coldest sigma_z and largest ratios over all analogues; bins above ratio 1
M.update(TngVrotMP=num(round(g.vrot4.median())), TngSzMPLo=num(round(g.sigma_z.quantile(.16))),
         TngSzMPHi=num(round(g.sigma_z.quantile(.84))), TngRatioMPLo=f(g.ratio.quantile(.16)), TngRatioMPHi=f(g.ratio.quantile(.84)),
         ColdSz=num(round(ok[mp].sigma_z.min())), RatioMaxMP=f(ok[mp].ratio.max()), RatioMaxMPOne=f(ok[mp].ratio.max(), 1),
         PartMaxMP=f((ok.t_vphi / ok.t_sigma_z)[mp].max()),
         NAboveOne=num((ok[ok.population == 1].ratio > 1).sum()), RatioMaxB=f(ok[ok.population == 1].ratio.max()))
# At the LMC's own formed mass below the second bin: each TNG50 analogue's values joined between its bins linearly in log mass (as
# paper_sims_kinematics.counts_at_mass), read there; median over the analogues whose bins span it.
# formed mass of each analogue, indexed by (suite, name)
X = pd.Series(mass.to_numpy(), index=pd.read_csv(R / 'sims_formed_mass.csv', dtype={'name': str}).set_index(['suite', 'name']).index)
at = []
for (suite, name), d in ok[ok.suite == 'TNG50'].groupby(['suite', 'name']):
    d = d.sort_values('population')
    # log10 formed mass below each bin of this analogue [Msun]
    x = np.log10(frac[d.population] * X[suite, name])
    # analogues whose bins span the LMC's mass below the second bin: values interpolated there
    if x.min() <= np.log10(frac[1] * m_lmc) <= x.max():
        at.append([np.interp(np.log10(frac[1] * m_lmc), x, d[q]) for q in ('vrot4', 'sigma_z', 'ratio')])
at = np.median(at, axis=0)
M.update(TngVrotAtB=num(round(at[0])), TngSzAtB=num(round(at[1])), TngRatioAtB=f(at[2]))
# counts of analogues covering, and above, the LMC's mass in each bin
mc = pd.read_csv(R / 'sims_mass_counts.csv')
M.update(NCoverA=num(mc.n_covering[0]), NAboveA=num(mc.n_above_lmc[0]))
for j, tag in zip((2, 3, 4), 'CDE'):
    M.update({f'NMass{tag}': num(mc.n_above_lmc[j]), f'NCov{tag}': num(mc.n_covering[j]),
              f'NBin{tag}': num(sum(sk.loc[(su, j, 'ratio_z'), 'n_above_lmc'] for su in ('TNG50', 'Auriga', 'FIRE-2')))})
# other population definitions (by rank in age, native, oldest): ratios and age ranks
pp = pd.read_csv(R / 'sims_populations.csv')
M.update(RankAge=f(pp.rank_age.median(), 1), OldestInRank=num(round(100 * pp.oldest_in_rank.median())),
         PartRank=f(pp.ratio_rank.median()), PartNative=f(pp.ratio_native.median()), PartOldest=f(pp.ratio_oldest.median()),
         PartRankMax=f(pp.ratio_rank.max()), PartNativeMax=f(pp.ratio_native.max()), PartOldestMax=f(pp.ratio_oldest.max()))
# alternative definitions (columns z, R, RMS and their particle versions; rows 1 and 2): counts of analogues
de = pd.read_csv(R / 'sims_definitions.csv')
six = ['z', 'z particles', 'R', 'R particles', 'RMS', 'RMS particles']
M.update(DefAny=num(de['any'][1]), DefLo=num(de.loc[1, six].min()), DefHi=num(de.loc[1, six].max()),
         DefCLo=num(de.loc[2, six].min()), DefCHi=num(de.loc[2, six].max()))

# %% [markdown]
# ### Birth, heating and scrambling
# Cuts repeated from the birth and heating scripts, and the statistics of what turns the orbits.

# %%
# Birth and heating (Figs sims_birth, sims_heating): cuts repeated from the scripts named
OLD_AGE = (9, 12)        # Gyr, 'the stars now 9-12 Gyr old' (paper_text_sims_birth.py)
YOUNG_AGE = 1.5          # Gyr, young-star check (paper_text_sims_birth.py)
AGE_CUTS = (10, 8, 4)    # Gyr, cohorts born more than 10, 8-10 and less than 4 Gyr ago (paper_sims_heating.py, panel b)
LATER = (2, 3)           # Gyr after birth of the later values of the oldest and youngest cohorts (paper_text_sims_birth.py)
COH_BIN = 0.1            # Gyr, time bins of panel c (paper_sims_heating.py)
COH_END = 0.5            # Gyr, cohorts followed up to 0.5 Gyr (build_sims_heating.py)
# gas and heating per snapshot of the TNG50 progenitors
gas = pd.read_csv(R / 'sims_heating_gas.csv', dtype={'halo': str})
dex = sbirth['DEX']  # [Fe/H] edges of the birth bins (paper_text_sims_birth.DEX) [dex]
M.update(OldAgeLo=num(OLD_AGE[0]), OldAgeHi=num(OLD_AGE[1]), YoungAge=num(YOUNG_AGE), AgeOld=num(AGE_CUTS[0]), AgeMid=num(AGE_CUTS[1]),
         AgeYoung=num(AGE_CUTS[2]), LaterOld=num(LATER[0]), LaterYoung=num(LATER[1]), CohBin=num(COH_BIN), CohEnd=num(COH_END),
         WindowHeat=num(1000 * heat['WINDOW']), NGasSnaps=num(len(GAS_SNAPS)), GasZFirst=num(round(gas.groupby('snap').z.median().iloc[0])),
         DexA=num(dex[0]), DexB=num(dex[1]), DexC=num(dex[-2]), DexD=num(dex[-1]))
# Auriga and FIRE-2 (macros_sims_suites.tex): cuts of paper_text_sims_suites.py
Z_EARLY = 1.5        # cohorts born at z > 1.5 (lookback > 9.4 Gyr there)
FIRE_LATE_LB = 9     # Gyr, FIRE-2 cohorts born less than 9 Gyr ago
M.update(BirthLbLo=num(suites['BIRTH_LB'][0]), BirthLbHi=num(suites['BIRTH_LB'][1]), ZEarly=num(Z_EARLY), FireLateLb=num(FIRE_LATE_LB))

# What turns the orbits (results/sims_scramble.csv; the statistics of paper_sims_scramble.py)
# scrambling experiments: coherence C per cohort, variant (which forces kept) and time t after birth [Gyr]
a = pd.read_csv(R / 'sims_scramble.csv', dtype={'halo': str})
# variant A at t > 0: simulated against measured coherence
pa = a[(a.variant == 'A') & (a.t > 0)]
d = pa.C_sim - pa.C_meas
# median cohort size; median and 16th-84th percentiles of C_sim - C_meas; correlation of C_sim with C_meas
M.update(ScrN=num(round(a.groupby(['halo', 's_birth']).n0.first().median())), ScrDC=signed(d.median(), 3),
         ScrDCLo=signed(d.quantile(.16)), ScrDCHi=signed(d.quantile(.84)), ScrR=f(np.corrcoef(pa.C_sim, pa.C_meas)[0, 1]))
# add the measured coherence as variant 'real'; C0: each cohort's coherence at birth (t = 0); loss = C0 - C
u = pd.concat([a, a[a.variant == 'A'].assign(variant='real', C_sim=lambda t: t.C_meas)])
u = u.join(u[u.t == 0].set_index(['variant', 'halo', 's_birth']).C_sim.rename('C0'), on=['variant', 'halo', 's_birth'])
u['loss'] = u.C0 - u.C_sim
# losses at 0.2 < t <= 0.35 Gyr, summed per galaxy and variant
W = u[(u.t > 0.2) & (u.t <= 0.35)].dropna(subset=['loss'])
S = W.groupby(['variant', 'halo']).loss.sum().unstack(0)
gal = S.index.values
# 1000 bootstrap resamples of the galaxies: each variant's summed loss over that of 'real', with 16th-84th percentiles
rng = np.random.default_rng(1)   # as paper_sims_scramble.py: 1000 bootstrap resamples of the galaxies
boot = [S.loc[gal[rng.integers(0, len(gal), len(gal))]].sum() for _ in range(1000)]
for v, tag in (('E', 'E'), ('D0', 'DZero'), ('B', 'B'), ('G_DM', 'GDM'), ('G_bar', 'GBar'), ('C', 'C'), ('H2', 'HTwo'),
               ('H1', 'HOne'), ('D0_fr', 'DZeroFr')):
    fr = np.array([b[v] / b['real'] for b in boot])
    M.update({f'Scr{tag}': f(S[v].sum() / S.real.sum()), f'Scr{tag}Lo': f(np.percentile(fr, 16)), f'Scr{tag}Hi': f(np.percentile(fr, 84))})
# radius of the halo-axes measurement: the default rmax of sims_scramble.halo_axes, read from its source
M.update(HaloR=num(float(re.search(r'def halo_axes\(x, rmax=([0-9.]+)', (ROOT / 'scripts/sims_scramble.py').read_text()).group(1))))
MIS = 45   # deg, the misalignment bins of panel b are 0-15-30-45-90 (paper_sims_scramble.py)
# mean loss per variant in misalignment bins [deg]; al: bins aligned to within MIS
mb = pd.cut(W.misalign, [0, 15, 30, 45, 90])
mean = W.groupby(['variant', mb], observed=True).loss.mean()
al = [iv for iv in mean.loc['real'].index if iv.right <= MIS]
# loss of E_split and H1 in the aligned bins, and in the first bin beyond MIS
M.update(Mis=num(MIS), AlELo=f(mean.loc['E_split'][al].min()), AlEHi=f(mean.loc['E_split'][al].max()),
         AlHOneLo=f(mean.loc['H1'][al].min()), AlHOneHi=f(mean.loc['H1'][al].max()),
         MisHOne=f(mean.loc['H1'][[iv for iv in mean.loc['H1'].index if iv.left >= MIS]].iloc[0]),
         MisE=f(mean.loc['E_split'][[iv for iv in mean.loc['E_split'].index if iv.left >= MIS]].iloc[0]))
L = pd.read_csv(R / 'sims_halo_cohorts.csv', dtype={'halo': str})
coh = pd.read_csv(R / 'sims_heating_coherence.csv', dtype={'halo': str})
# coherence of randomly oriented orbits: 1 / sqrt(median cohort size)
C_RANDOM = 1 / np.sqrt(coh[coh.set == 'full'].groupby(['halo', 's_birth']).n0.first().median())
z_late = []   # redshifts of the birth snapshots of the late control cohorts (set 'track'), from the cutout headers
for s_ in sorted(coh[coh.set == 'track'].s_birth.unique()):
    # scale factor a of each birth snapshot -> redshift z = 1/a - 1
    with h5py.File(sims.TNG_TRACK_DIR / coh.halo.iloc[0] / f'snap_{s_:03d}.hdf5') as fh:
        z_late.append(1 / fh['Header'].attrs['Time'] - 1)
M.update(LateZHi=f(max(z_late)))
# frac: fraction of the available coherence lost, (C0 - C1) / (C0 - C_RANDOM)
L['frac'] = L.loss / (L.C0 - C_RANDOM)
eb = L.groupby(pd.cut(L.eps2, [0, 0.003, 0.006, 0.01, 0.015, 0.1]), observed=True)   # bins of panel c (paper_sims_scramble.py)
# per eps2 bin: median eps2, median fraction lost, median lookback time [Gyr]
c = pd.DataFrame({'x': eb.eps2.median(), 'm': eb.frac.median(), 'lb': eb.lookback.median()}).reset_index(drop=True)
rk = pd.DataFrame({q: L[q].rank() for q in ('eps2', 'frac', 'z')}).corr()   # Spearman partial correlation at fixed redshift
dm = L.groupby('halo')[['eps2', 'frac']].transform(lambda t: t - t.mean())     # within galaxies: each galaxy's means subtracted
# cohort counts and redshift range; C_RANDOM; rank correlation of eps2 and frac: all, at fixed z (partial), within galaxies
M.update(NCohHalo=num(len(L)), ZCohHi=num(round(L.z.max())), ZCohLo=f(L.z.min(), 1), StepLo=f(L.step.min(), 1), StepHi=f(L.step.max(), 1),
         CRandom=f(C_RANDOM, 3),
         RankAll=f(spearmanr(L.eps2, L.frac)[0]),
         RankZ=f((rk.eps2.frac - rk.eps2.z * rk.frac.z) / np.sqrt((1 - rk.eps2.z ** 2) * (1 - rk.frac.z ** 2))),
         RankGal=f(spearmanr(dm.eps2, dm.frac)[0]))
# one set of macros per eps2 bin
for i, tag in enumerate(('One', 'Two', 'Three', 'Four', 'Five')):
    M.update({f'EpsBin{tag}': sigfig(c.x[i], 2), f'FracBin{tag}': f(c.m[i]), f'LbBin{tag}': f(c.lb[i], 1)})
# fraction lost in bins two and three, in per cent rounded to tens
M.update(PctBinTwo=num(10 * round(10 * c.m[1])), PctBinThree=num(10 * round(10 * c.m[2])))

# %% [markdown]
# ### Force check of the multipole expansion
# The expansion's force against a softened direct sum on a TNG50 progenitor (needs the TNG50 progenitor cutouts).

# %%
# The force of the expansion (as sims_scramble.py: all particles within RMAX, lmax LMAX, no softening) against a direct sum with the
# softening of TNG50 (cubic-spline kernel, Newtonian beyond 2.8 eps; eps = EPS_TNG comoving kpc/h at z > 1, Pillepich et al. 2019,
# Table 1), at FORCE_N random star positions within FORCE_R kpc of the centre of the progenitor read above
EPS_TNG, FORCE_N, FORCE_R = 0.39, 300, (0.3, 0.6)  # comoving kpc/h; stars; inner and outer radius [kpc]
# positions and masses of all particles (dark matter, gas, stars) in the progenitor's frame
xp, mp_ = scr.particles(prog, ('dm', 'gas', 'st'), np.eye(3))
# FORCE_N random stars between FORCE_R[0] and FORCE_R[1] kpc (fixed seed 0)
r_st = np.linalg.norm(prog['st']['x'], axis=1)
pts = prog['st']['x'][np.random.default_rng(0).choice(np.flatnonzero((r_st > FORCE_R[0]) & (r_st < FORCE_R[1])), FORCE_N, replace=False)]
# the expansion's force at those points (AGAMA, configuration written to a temporary file)
with tempfile.TemporaryDirectory() as tmp:
    f_exp = agama.Potential(scr.expansion(xp, mp_, f'{tmp}/force.ini')).force(pts)
# kernel radius 2.8 eps in physical kpc (comoving kpc/h x a / h)
h = 2.8 * EPS_TNG * prog['a'] / sims.TNG_H
# direct sum over the particles in blocks of 20000 (memory)
f_sim = np.zeros_like(pts)
for j in range(0, len(xp), 20000):
    dx = xp[j:j + 20000][None] - pts[:, None]
    rr = np.linalg.norm(dx, axis=2)
    u = rr / h
    # cubic-spline kernel: force factor g(r) per unit G m dx [kpc^-3], Newtonian r^-3 beyond h
    g = np.where(u < 0.5, (32 / 3 + u ** 2 * (32 * u - 38.4)) / h ** 3, np.where(
        u < 1, (64 / 3 - 48 * u + 38.4 * u ** 2 - 32 / 3 * u ** 3 - 1 / (15 * np.maximum(u, 1e-9) ** 3)) / h ** 3, np.maximum(rr, h) ** -3.))
    f_sim += G * (mp_[j:j + 20000][None, :, None] * dx * g[..., None]).sum(1)
# force radii and number, redshift, and median per-cent difference of the expansion's force from the direct sum
M.update(ForceRLo=num(FORCE_R[0]), ForceR=num(FORCE_R[1]), ForceN=num(FORCE_N), ForceZ=f(1 / prog['a'] - 1, 1),
         ForcePct=num(round(100 * (np.median(np.linalg.norm(f_exp, axis=1) / np.linalg.norm(f_sim, axis=1)) - 1))))
# Auriga scrambling table: redshift range and number of cohorts
au = pd.read_csv(R / 'auriga_scramble.csv')
M.update(AuZHi=f(au.z.max(), 1), AuZLo=f(au.z.min(), 2), AuN=num(au.groupby(['halo', 's_birth']).ngroups))

# %% [markdown]
# ### Auriga scrambling and the settling halo
# The Auriga version of Fig. 4(a) and how the halo settles.

# %%
# The Auriga version of the Letter's Fig. 4(a) (results/auriga_scramble.csv): v_phi/sigma_z about the birth spin in the TIME_EDGES bins
# of scripts/paper_letter_scramble.py, per galaxy the median, then the median over the galaxies; the bins' median times.
te = const('paper_letter_scramble')['TIME_EDGES']
# v_phi/sigma_z: measured for 'real', simulated for the variants
au['vs'] = np.where(au.variant == 'real', au.vs_meas, au.vs_sim)
# time bins (TIME_EDGES of paper_letter_scramble.py) [Gyr after birth]
au['tb'] = pd.cut(au.t, np.r_[-1e-9, te[1:-1], te[-1] + 1e-6])
# per galaxy the median in each bin, then the median over the galaxies; tm: median time of each bin
va = au.groupby(['variant', 'tb', 'halo'], observed=True).vs.median().groupby(level=[0, 1], observed=True).median()
tm = au.groupby('tb', observed=True).t.median()
# values per variant and bin; times of bins two to four
for v, tag in (('real', 'Real'), ('E_split', 'E'), ('G_DM_fr', 'GDM'), ('D0_fr', 'DZero')):
    M.update({f'AuVs{tag}{i}': f(x) for i, x in zip(('One', 'Two', 'Three', 'Four'), va.loc[v].to_numpy())})
M.update({f'AuT{i}': f(x) for i, x in zip(('Two', 'Three', 'Four'), tm.to_numpy()[1:])})

# How the halo settles (results/sims_halo_settle.csv; per galaxy the median in 0.5-Gyr lookback bins, then over the galaxies, as
# paper_sims_settle.py, bins with at least MIN_GAL galaxies)
MIN_GAL = 20
LB_EARLY_END, LB_MID = 11.5, 10.0   # Gyr: psi is quoted up to 11.5 Gyr ago and at 10 Gyr ago
# halo settling table (per progenitor and snapshot); 0.5-Gyr lookback bins
T = pd.read_csv(R / 'sims_halo_settle.csv', dtype={'halo': str})
T['tb'] = (T.lookback * 2).round() / 2
# per bin: each galaxy's medians, then median and 16th/84th percentiles over galaxies, in bins with >= MIN_GAL galaxies
gb = T.groupby(['tb', 'halo']).median(numeric_only=True).groupby(level=0)
n = T.groupby('tb').halo.nunique()
med, p16, p84 = (x[n >= MIN_GAL] for x in (gb.median(), gb.quantile(.16), gb.quantile(.84)))
# e, l_: earliest and latest lookback bins [Gyr]; pe: misalignment medians before LB_EARLY_END
e, l_ = med.index.max(), med.index.min()
pe = med.misalign[med.index >= LB_EARLY_END]
# psi early and late [rad]; fs, fc: expected change of eps_tilt from sin 2 psi and from 1 - c/a
psi_e, psi_l = np.deg2rad(med.misalign[e]), np.deg2rad(med.misalign[l_])
fs, fc = abs(np.sin(2 * psi_l) / np.sin(2 * psi_e)), (1 - med.c_a[l_]) / (1 - med.c_a[e])
# eps_tilt from early to late; lookback at which it crosses EPS_QUIET
tilt = med.eps_tilt.sort_index(ascending=False)
below = np.flatnonzero(tilt.to_numpy() < settle['EPS_QUIET'])[0]   # first bin below the line; crossing interpolated linearly
t_cross = np.interp(settle['EPS_QUIET'], tilt.to_numpy()[below - 1:below + 1][::-1], tilt.index[below - 1:below + 1][::-1])
# counts per galaxy; psi, c/a, b/a, eps2 and its tilt and elongation parts early and late; predicted against measured change;
# rank correlations of eps_tilt with (1 - c/a)|sin 2 psi| and of eps_elong with 1 - b/a; lookback of the crossings
M.update(NPerGalLo=num(T.groupby('halo').size().min()), NPerGalHi=num(T.groupby('halo').size().max()), MinGal=num(MIN_GAL),
         PsiEarlyLo=num(round(pe.min())), PsiEarlyHi=num(round(pe.max())), LbEarlyEnd=num(LB_EARLY_END), LbMid=num(LB_MID),
         PsiMid=num(round(med.misalign[LB_MID])), PsiLate=num(round(med.misalign[l_])),
         PsiEarlyLoP=num(round(p16.misalign[e])), PsiEarlyHiP=num(round(p84.misalign[e])),
         PsiLateLoP=num(round(p16.misalign[l_])), PsiLateHiP=num(round(p84.misalign[l_])), Span=num(e - l_),
         CaEarly=f(med.c_a[e]), CaLate=f(med.c_a[l_]), BaEarly=f(med.b_a[e]), BaLate=f(med.b_a[l_]),
         EpsFactor={3: 'three'}.get(round(med.eps2[e] / med.eps2[l_]), num(round(med.eps2[e] / med.eps2[l_]))),
         EpsEarly=sigfig(med.eps2[e], 2), EpsLate=sigfig(med.eps2[l_]), TiltEarly=sigfig(med.eps_tilt[e], 2),
         TiltLate=sigfig(med.eps_tilt[l_], 2), ElongEarly=sigfig(med.eps_elong[e], 2), ElongLate=sigfig(med.eps_elong[l_]),
         TiltEarlyThree=f(med.eps_tilt[e], 4), TiltLateFour=f(med.eps_tilt[l_], 4),
         TwoPsiEarly=num(round(2 * med.misalign[e])), TwoPsiLate=num(round(2 * med.misalign[l_])),
         FacSin=f(fs), FacCa=f(fc), FacPred=f(fs * fc), FacMeas=f(med.eps_tilt[l_] / med.eps_tilt[e]),
         RankTilt=f(spearmanr(T.eps_tilt, (1 - T.c_a) * np.abs(np.sin(2 * np.deg2rad(T.misalign))))[0]),
         RankElong=f(spearmanr(T.eps_elong, 1 - T.b_a)[0]), SinSq=sigfig((np.sin(psi_l) / np.sin(psi_e)) ** 2),
         TiltFactorPct=num(round(med.eps_tilt[e] / med.eps_tilt[l_])), ElongFactor=f(med.eps_elong[e] / med.eps_elong[l_], 1),
         LbTiltQuiet=num(round(t_cross)), EpsQuiet=num(settle['EPS_QUIET']),
         LbAlign=num(med.misalign.sort_index(ascending=False).lt(QUIET['misalign'][1]).idxmax()))
q = QUIET
other = []
# median fraction lost by cohorts born on the quiet side of each QUIET threshold, and by the others
for key, (a0, a1) in q.items():
    kq = (L[key] > a0) & (L[key] <= a1)
    M[{'misalign': 'QuietPsi', 'c_a': 'QuietCa', 'b_a': 'QuietBa', 'eps2': 'QuietEps'}[key]] = f(L.frac[kq].median())
    other.append(L.frac[~kq].median())
RULE = 0.1   # the threshold rule: the edge of the first bin of cohorts whose median loss is below 10 per cent
# the first eps2 bin (below 0.003) must lose less than RULE and the second at least RULE
assert c.m[0] < RULE <= c.m[1], 'the eps2 threshold no longer follows the rule'
M.update(OtherLo=f(min(other)), OtherHi=f(max(other)), Rule=num(100 * RULE), PsiQuiet=num(q['misalign'][1]), CaQuiet=num(q['c_a'][0]),
         BaQuiet=num(q['b_a'][0]))

# %% [markdown]
# ### SIDM Concerto
# The radius where the SIDM halo is round, and the host's virial mass from the Rockstar catalogue.

# %%
# A round halo: SIDM Concerto (results/sims_sidm_shapes.csv; shells with at least NMIN particles, as paper_sims_sidm.py)
NMIN = const('paper_sims_sidm')['NMIN']  # fewest particles per shell (paper_sims_sidm.py)
R_CDM = 5.0      # kpc, the CDM centre quoted
ROUND = 0.85     # 'round': every shell out to that major-axis radius has c/a above ROUND
sh = pd.read_csv(R / 'sims_sidm_shapes.csv')
# shells with enough particles; eps2 at radius R_EPS per run and redshift
shells = sh[sh.N >= NMIN]
e2 = sh[np.isclose(sh.r, sidm['R_EPS'])].set_index(['run', 'z']).eps2
# redshifts of the snapshots, earliest first (z4 > z3 > z2 ...)
zs = sorted(sh.z.unique(), reverse=True)
z4, z3, z2 = zs[0], zs[1], zs[2]


def round_radius(run, z):
    """(radius [kpc] out to which every shell has c/a > ROUND, min c/a, max c/a there) for one run and redshift; 0 if none."""
    w = shells[(shells.run == run) & np.isclose(shells.z, z)].sort_values('r')
    # shells inside the first one with c/a <= ROUND
    bad = np.flatnonzero(w.s.to_numpy() <= ROUND)
    w = w.iloc[:bad[0] if len(bad) else len(w)]
    return (w.r.max(), w.s.min(), w.s.max()) if len(w) else (0, np.nan, np.nan)


# round radius of SIDM from the third snapshot on; CDM / SIDM eps2 ratio (z3 excluded)
late = [round_radius('SIDM', z) for z in zs[2:]]
ratio = (e2['CDM'] / e2['SIDM']).drop(z3)
# TNG50 median eps2 per 0.5 redshift bin, and the CDM run's eps2 over it at matching redshifts
tng = T.assign(zb=(T.z / 0.5).round() * 0.5).groupby(['zb', 'halo']).eps2.median().groupby(level=0).median()
cdm_tng = [e2['CDM'][z] / tng[round(z * 2) / 2] for z in zs if round(z * 2) / 2 in tng.index]
# SIDM shells at z3; CDM shells within R_CDM
w3 = shells[(shells.run == 'SIDM') & np.isclose(shells.z, z3)]
cdm = shells[(shells.run == 'CDM') & (shells.r <= R_CDM)]
# CDM shape and eps2 range; SIDM round radius and c/a at z4 and later; eps2 at each redshift; CDM-to-TNG50 eps2 ratio
M.update(SiRCdm=num(R_CDM), SiCaCdmLo=f(cdm.s.min()), SiCaCdmHi=f(cdm.s.max()), SiCdmEpsLo=sigfig(e2['CDM'].min(), 2),
         SiCdmEpsHi=sigfig(e2['CDM'].max(), 2), SiRound=f(ROUND), SiRoundZFour=num(round(round_radius('SIDM', z4)[0])),
         SiCaZFourLo=f(round_radius('SIDM', z4)[1]), SiCaZFourHi=f(round_radius('SIDM', z4)[2]),
         SiRoundLo=num(round(min(r[0] for r in late))), SiRoundHi=num(round(max(r[0] for r in late))),
         SiCaRoundLo=f(min(r[1] for r in late)), SiCaRoundHi=f(max(r[2] for r in late)),
         SiRatioLo=num(round(ratio.min())), SiRatioHi=num(5 * round(ratio.max() / 5)),
         SiEpsZFour=f(e2['SIDM'][z4], 4), SiEpsZTwo=f(e2['SIDM'][z2], 4), SiEpsLate=f(e2['SIDM'][zs[3]], 4),
         SiCaZThree=sigfig(np.interp(sidm['R_EPS'], w3.r, w3.s)), SiEpsZThree=f(e2['SIDM'][z3], 4),
         SiTngRatio=sigfig(max(cdm_tng)))
assert np.isclose(e2['SIDM'][zs[3]], e2['SIDM'][zs[4]], atol=5e-5), 'the text: the same SIDM eps2 at z = 1.5 and today'
assert round_radius('SIDM', z3)[0] == 0, 'the text: the SIDM centre is not round at z = 3'
# The host's virial mass: the Rockstar halo (sims_sidm_shapes.halos) whose Rvir is the one of the shapes table (r / r_rvir)
# Rvir of the shapes table [kpc] per redshift and run
rvir = (sh.r / sh.r_rvir).groupby([sh.z, sh.run]).median()
for z_, tag in ((z4, 'ZFour'), (zs[3], 'ZOneHalf'), (zs[4], 'ZZero')):
    mv = []
    for run, name in zip(RUNS, ('CDM', 'SIDM')):
        # snapshot and scale factor of this redshift; central halos (upid = -1) of the Rockstar catalogue
        snap, a_ = next((k_, a_) for k_, a_ in SIDM_SNAPS.items() if np.isclose(1 / a_ - 1, z_, atol=1e-4))
        t = halos(run, snap, a_).query('upid == -1')
        # the halo whose Rvir matches the shapes table to 0.01; its Mvir to one significant figure
        j = (t.Rvir - rvir[z_, name]).abs().argmin()
        assert abs(t.Rvir.iloc[j] - rvir[z_, name]) < 0.01, 'the host is found again'
        mv.append(sci(t.Mvir.iloc[j], 0))
    assert mv[0] == mv[1], 'the text: the same virial mass in both runs'
    M[f'SiMvir{tag}'] = mv[0]

# %% [markdown]
# ### Normalisation of the multipole coefficients
# A check of AGAMA's coefficient normalisation on a sampled Miyamoto-Nagai disc.

# %%
# Normalisation of AGAMA's multipole coefficients (KB section 'Normalisation of the multipole coefficients'), checked on the
# Miyamoto-Nagai disc P3 of lmcold.virial sampled with MN_N particles, expanded as in sims_scramble.py, at the grid radius nearest
# MN_R: Phi_00 against the spherical average of the expansion's potential, Phi_20 against sqrt(5) <Phi P_2>, and sum_m Phi_2m^2
# before and after a rotation of the particles (Euler angles MN_ROT)
MN_N, MN_R, MN_ROT = 500000, 2.0, (30, 50, 20)
# fixed seed for the sampling
agama.setRandomSeed(1)
# MN_N particles of a Miyamoto-Nagai disc (2.7e9 Msun, a = 1.6 kpc, b = 0.3 kpc: P3 of lmcold.virial)
xm, mm = agama.Density(agama.Potential(type='MiyamotoNagai', mass=2.7e9, scaleRadius=1.6, scaleHeight=0.3)).sample(MN_N)


def coefficients(x, tmp):
    """Multipole of the particles x (masses mm): the potential, the grid radii and {'l=..,m=..': Phi_lm at the radii}."""
    # the expansion's text file: after the '#Phi' line, a header of 'l=..,m=..' names and scr.GRID rows (radius, coefficients)
    lines = open(scr.expansion(x, mm, f'{tmp}/mn.ini')).read().split('\n')
    k = lines.index('#Phi') + 1
    tab = np.array([[float(v) for v in ln.split('\t')] for ln in lines[k + 1:k + 1 + scr.GRID]])
    return agama.Potential(f'{tmp}/mn.ini'), tab[:, 0], dict(zip(lines[k].split('\t')[1:], tab[:, 1:].T))


# rotation matrix of the Euler angles MN_ROT [deg]
rot = Rotation.from_euler('zyx', MN_ROT, degrees=True).as_matrix()
with tempfile.TemporaryDirectory() as tmp:
    # coefficients of the disc and of the rotated disc; i: grid radius nearest MN_R
    pot, rg, P0 = coefficients(xm, tmp)
    P1 = coefficients(xm @ rot.T, tmp)[2]
    i = np.abs(rg - MN_R).argmin()
    # Gauss-Legendre nodes in mu = cos(theta) and uniform phi: the quadrature of the mean over a sphere
    mu, w = np.polynomial.legendre.leggauss(64)
    MU, PH = np.meshgrid(mu, np.linspace(0, 2 * np.pi, 128, endpoint=False))
    wt = np.broadcast_to(w, MU.shape) / (2 * MU.shape[0])          # weights of the mean over the sphere (sum 1)
    # the expansion's potential on the sphere of radius rg[i]
    phi = pot.potential(rg[i] * np.c_[np.sqrt(1 - MU.ravel() ** 2) * np.cos(PH.ravel()), np.sqrt(1 - MU.ravel() ** 2) * np.sin(PH.ravel()),
                                      MU.ravel()]).reshape(MU.shape)
    # largest relative deviation of Phi_00 from <Phi> and of Phi_20 from sqrt(5) <Phi P_2>
    dev_norm = max(abs(P0['l=0,m=0'][i] / (wt * phi).sum() - 1),
                   abs(P0['l=2,m=0'][i] / (np.sqrt(5) * (wt * phi * (1.5 * MU ** 2 - 0.5)).sum()) - 1))
    # sum_m Phi_2m^2 before and after the rotation (invariant if the normalisation is right)
    q2 = [sum(P[f'l=2,m={m_}'][i] ** 2 for m_ in range(-2, 3)) for P in (P0, P1)]
M.update(MNN=sci(MN_N, 0), MNR=f(rg[i], 1), MNNorm=bound(dev_norm), MNRot=bound(abs(q2[1] / q2[0] - 1)))

# macro names must be letters only (LaTeX command names)
assert all(k_.isalpha() for k_ in M), [k_ for k_ in M if not k_.isalpha()]

# %% [markdown]
# ### Write the macros
# One \newcommand per entry of M, with the prefix \sm, into results/macros_sims.tex; the text shows them as macros, so no number
# is typed by hand.

# %%
# header line, then one \newcommand per macro
lines = ['% Numbers of the KB chapters Simulations (scripts/paper_text_sims.py); do not edit.']
lines += [rf'\newcommand{{\sm{k_}}}{{{x}}}' for k_, x in M.items()]
(R / 'macros_sims.tex').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
