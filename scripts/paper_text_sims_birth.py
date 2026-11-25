"""Paper text: the numbers of Section 6 on the simulated stars at birth and soon after (the paragraphs of Figs sims_birth and
sims_heating and their captions), as LaTeX macros, so that no number there is typed by hand.

Inputs: results/sims_birth.csv (scripts/build_sims_birth.py: birth cohorts of COHORT Gyr), results/sims_heating_history.csv,
sims_heating_gas.csv, sims_heating_coherence.csv (scripts/paper_sims_heating.py), results/kin_support_ellipsoid.csv (LMC giant bins,
stage-1 ellipsoid fits), results/giant_present_mass.csv.
Definitions:
- 'old stars': cohorts aged 9-12 Gyr today; per analogue the median over those cohorts of v_phi, sigma_z and v_phi/sigma_z (each
  cohort's own ratio), at birth and today; then the median and 16th/84th percentiles over the analogues (TNG50) or the median over the
  five Auriga analogues. 'Reach' counts the analogues with v_phi/sigma_z >= the LMC's [Fe/H] < -1.5 bin.
- TNG50 at birth at the LMC bins' ages: the median over analogues of each cohort's v_phi/sigma_z at birth, against age, interpolated
  at the formed-mass-weighted mean age of the three metal-poor giant bins (paper_sims_birth.lmc_ages); the script stops unless each
  LMC bin lies between the TNG50 16th percentile and median there (the text's wording).
- Heating: sigma_z^2 today minus at birth; the LMC's from the mean sigma_z^2 of its two most metal-poor bins; rounded to 100.
- Young stars: TNG50 cohorts younger than 1.5 Gyr, median over all cohorts; Auriga: the range over its analogues of each analogue's
  median sigma_z at birth.
- Fig. sims_heating (a): galaxy-weighted medians (paper_sims_heating.gal_median) in 0.5-dex bins of the progenitor's stellar mass,
  10^7-10^7.5 and 10^9-10^9.5 Msun, and the range over the 0.5-dex bins 10^7-10^9.5 for sigma_z; the gas likewise, its effective
  sound speed the median over the six snapshots, its v_phi/sigma_z at the first and last snapshot; the LMC's two most metal-poor
  bins: v_rot(4 kpc) and the present mass in stars more metal-poor than each bin's median [Fe/H]; TNG50 stars at that mass: the
  range of the 0.25-dex bins whose centres lie in it.
- Fig. sims_heating (b): the median time from birth to the first tracked snapshot; v_phi/sigma_z at birth, at the first step (0.5)
  and later, galaxy-weighted, as panel b. (c): C at birth and at 0.5 Gyr for z > 1.5, the late cohorts at birth, the median cohort
  size and 1/sqrt of it.
Writes results/macros_sims_birth.tex (copied to the paper's tables/; \\input in the preamble).
Run: .venv/bin/python scripts/manifest.py run paper_text_sims_birth.py"""

# %% [markdown]
# ### Settings
# Imports and the settings read from other scripts.

# %%
import numpy as np
import pandas as pd

from lmcold.sources import ROOT
from build_sims_birth import COHORT
from paper_sims_birth import lmc_ages
from paper_sims_heating import gal_median

# %% [markdown]
# ### Load the data
# The birth cohorts of TNG50 and Auriga and the LMC giant bins; the tables of Fig. sims_heating are read in their own step
# below.

# %%
# One row per cohort (suite, halo, age [Gyr]); v_phi, sigma_z [km/s] at birth and of the same stars today; their ratios vs_*.
t = pd.read_csv(ROOT / 'results/sims_birth.csv', dtype={'halo': str}).dropna(subset=['vphi_birth'])
t['vs_birth'], t['vs_same'] = t.vphi_birth / t.sz_birth, t.vphi_same / t.sz_same
# The five LMC giant bins, metal-poor first (ratio_z = v_rot/sigma_z, sigma_z and vrot4 [km/s], feh_med [dex]).
k = pd.read_csv(ROOT / 'results/kin_support_ellipsoid.csv')
k = k[k.population.str.startswith('giants')].reset_index(drop=True)

# %% [markdown]
# ### Macro store
# M collects the macros; put() writes a value with d decimals.

# %%
M = {}   # macro name (without backslash): text


def put(name, value, d=0):
    """Store value as macro name, with d decimals."""
    M[name] = f'{value:.{d}f}'   # fixed-point text with d decimals


# %% [markdown]
# ### Old stars at birth and today
# Cohorts aged 9-12 Gyr: medians per analogue, then over the analogues; the script stops if a result contradicts the wording of
# the text. Then the LMC bins against TNG50 at their ages, the heating, and the young stars.

# %%
Q = ['vphi_birth', 'sz_birth', 'vs_birth', 'vphi_same', 'sz_same', 'vs_same']
# Old cohorts (9-12 Gyr): median per analogue for each suite; lmc: v_rot/sigma_z of the [Fe/H] < -1.5 bin.
old = t[(t.age > 9) & (t.age < 12)]
tng, au = (old[old.suite == s].groupby('halo')[Q].median() for s in ('TNG50', 'Auriga'))
lmc = k.ratio_z[0]
put('sbCohort', COHORT, 1)
# TNG50 medians and 16th/84th percentiles over the analogues at birth, Auriga median; analogues reaching the LMC; then today.
put('sbVphiBirth', tng.vphi_birth.median()); put('sbSzBirth', tng.sz_birth.median()); put('sbVsBirth', tng.vs_birth.median(), 2)
put('sbVsBirthLo', tng.vs_birth.quantile(.16), 2); put('sbVsBirthHi', tng.vs_birth.quantile(.84), 2)
put('sbAuVsBirth', au.vs_birth.median(), 2)
put('sbNReach', (tng.vs_birth >= lmc).sum()); put('sbNGal', len(tng)); put('sbLmcVs', lmc, 2)
put('sbVphiToday', tng.vphi_same.median()); put('sbSzToday', tng.sz_same.median()); put('sbVsToday', tng.vs_same.median(), 2)
put('sbVsTodayHi', tng.vs_same.quantile(.84), 2); put('sbAuVsToday', au.vs_same.median(), 2)
if (tng.vs_same >= lmc).any():
    raise SystemExit('the text says no analogue reaches the LMC today; some do')
# TNG50 v_phi/sigma_z at birth against cohort age: median and 16th percentile over the analogues.
g = t[t.suite == 'TNG50'].groupby('age').vs_birth
curve, low = g.median(), g.quantile(.16)
for j, (name, (age, _, _)) in enumerate(zip(('One', 'Two', 'Three'), lmc_ages()[:3])):
    # Both curves interpolated at the mean age [Gyr] of LMC bin j (lmc_ages: mean, p16, p84).
    med, lo = np.interp(age, curve.index, curve.values), np.interp(age, low.index, low.values)
    if not lo <= k.ratio_z[j] <= med:   # the text: below the TNG50 median at birth, within its 16th-84th percentiles
        raise SystemExit(f'LMC bin {j}: {k.ratio_z[j]:.2f} not between the TNG50 16th percentile {lo:.2f} and median {med:.2f} at birth')
    put(f'sbLmcVs{name}', k.ratio_z[j], 2); put(f'sbTngVs{name}', med, 2); put(f'sbLmcAge{name}', age, 1)
# LMC sigma_z [km/s]: rms of its two most metal-poor bins; heating = sigma_z^2 now minus TNG50 at birth [(km/s)^2], rounded to 100.
sz_lmc = np.sqrt(np.mean(k.sigma_z[:2] ** 2))
put('sbLmcSzLo', k.sigma_z[:2].min()); put('sbLmcSzHi', k.sigma_z[:2].max())
put('sbHeatLmc', round(sz_lmc ** 2 - tng.sz_birth.median() ** 2, -2)); put('sbHeatSims', round(tng.sz_same.median() ** 2 - tng.sz_birth.median() ** 2, -2))
# Young TNG50 cohorts (< 1.5 Gyr): medians over all cohorts [km/s]; Auriga: range of each analogue's median sigma_z at birth.
young = t[(t.suite == 'TNG50') & (t.age < 1.5)]
put('sbYoungSz', young.sz_birth.median()); put('sbYoungVphiBirth', young.vphi_birth.median()); put('sbYoungVphiToday', young.vphi_same.median())
a = t[t.suite == 'Auriga'].groupby('halo').sz_birth.median()
put('sbAuSzLo', a.min()); put('sbAuSzHi', a.max())

# %% [markdown]
# ### Fig. sims_heating
# The tables of scripts/paper_sims_heating.py: kinematics at birth against the progenitor's stellar mass (a), against time since
# birth (b), and the coherence of the cohorts (c).

# %%
h = pd.read_csv(ROOT / 'results/sims_heating_history.csv', dtype={'halo': str})
gas = pd.read_csv(ROOT / 'results/sims_heating_gas.csv', dtype={'halo': str})
coh = pd.read_csv(ROOT / 'results/sims_heating_coherence.csv', dtype={'halo': str})
# v_phi/sigma_z of the cohorts and of the gas; birth: the dt = 0 rows of the history table.
h['vs'], gas['vs'] = h.vphi / h.sz, gas.vphi / gas.sz
birth = h[h.dt == 0].copy()
# Edges [log10 Msun] of the 0.5-dex bins of the progenitor's stellar mass; galaxy-weighted medians per bin of stars and gas.
DEX = [7, 7.5, 8, 8.5, 9, 9.5]
birth['mb'], gas['mb'] = pd.cut(birth.logMs, DEX), pd.cut(gas.logMs, DEX)
sv, ss, gv, gs = (gal_median(d, 'mb', q).m for d, q in ((birth, 'vphi'), (birth, 'sz'), (gas, 'vphi'), (gas, 'sz')))
put('shVphiLow', sv.iloc[0]); put('shVphiHigh', sv.iloc[-1]); put('shSzLo', ss.min()); put('shSzHi', ss.max())
put('shGasVphiLow', gv.iloc[0]); put('shGasVphiHigh', gv.iloc[-1]); put('shGasSzLo', gs.min()); put('shGasSzHi', gs.max())
# Gas medians per snapshot: sound speed cs [km/s] (median over snapshots), v_phi/sigma_z at the first and last snapshot, last z.
g = gas.groupby('snap')[['z', 'cs', 'vs']].median()
put('shGasCs', g.cs.median()); put('shGasVsFirst', g.vs.iloc[0], 2); put('shGasVsLast', g.vs.iloc[-1], 1); put('shGasZLast', g.z.iloc[-1], 1)
# Present cumulative mass [Msun] below each bin edge (last row, the total, dropped), interpolated at the median [Fe/H] of LMC bins
# A, B.
cm = pd.read_csv(ROOT / 'results/giant_present_mass.csv').iloc[:-1]
logm = np.log10(np.interp(k.feh_med[:2], cm.feh_hi, cm.present_cumulative))
put('shLmcVrotLo', k.vrot4[:2].min()); put('shLmcVrotHi', k.vrot4[:2].max()); put('shLmcLogMLo', logm.min(), 1); put('shLmcLogMHi', logm.max(), 1)
# 0.25-dex mass bins; TNG50 v_phi at birth in the bins whose centres lie within the LMC's mass range.
birth['mq'] = pd.cut(birth.logMs, np.arange(6.5, 10.01, 0.25))
s = gal_median(birth, 'mq', 'vphi').m
s = s[[logm.min() <= i.mid <= logm.max() for i in s.index]]
put('shTngVphiLo', s.min()); put('shTngVphiHi', s.max())
# Time since birth [Gyr] rounded to 0.5 (at least 0.5 after birth); shFirstStep: median time to the first tracked snapshot.
h['dtb'] = np.where(h.dt == 0, 0, np.clip((h.dt * 2).round() / 2, 0.5, None))
put('shFirstStep', h[h.dt > 0].sort_values('dt').groupby(['halo', 'age']).dt.first().median(), 2)
# Age groups [Gyr]: galaxy-weighted v_phi/sigma_z at birth, at 0.5 Gyr, and later (2 Gyr for the old, 3 Gyr for the young).
for name, sel in (('Old', h.age > 10), ('Mid', (h.age > 8) & (h.age <= 10)), ('Young', h.age < 4)):
    m = gal_median(h[sel], 'dtb', 'vs').m
    put(f'sh{name}Birth', m[0], 2); put(f'sh{name}First', m[0.5], 2)
    if name == 'Old':
        put('shOldTwo', m[2.0], 2)
    if name == 'Young':
        put('shYoungThree', m[3.0], 1)
# Coherence C of the early cohorts in 0.1-Gyr steps; late control cohorts: at birth (dt < 0.2 Gyr) or the next snapshot (0.45).
early = coh[coh.set == 'full'].assign(dtb=lambda d: (d.dt / 0.1).round() * 0.1)
c = gal_median(early, 'dtb', 'C').m
late = coh[(coh.set == 'track') & (coh.dt < 0.75)].assign(dtb=lambda d: np.where(d.dt < 0.2, 0.0, 0.45))
# Median early-cohort size, rounded to 10, and 1/sqrt of it.
n_med = early.groupby(['halo', 's_birth']).n0.first().median()
put('shCohBirth', c.iloc[0], 2); put('shCohEnd', c.iloc[-1], 2); put('shCohLate', gal_median(late, 'dtb', 'C').m[0.0], 2)
put('shCohN', round(n_med, -1)); put('shCohRandom', 1 / np.sqrt(n_med), 2)

# %% [markdown]
# ### Write the macros
# One \newcommand per entry of M into results/macros_sims_birth.tex.

# %%
# One \newcommand per macro, written and printed.
text = ''.join(f'\\newcommand{{\\{n}}}{{{v}}}\n' for n, v in M.items())
(ROOT / 'results/macros_sims_birth.tex').write_text(text)
print(text)
