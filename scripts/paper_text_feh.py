"""Knowledge-base text, chapter 'Metallicity structure' (extended draft Section 3 and its appendix on published gradients and
distributions): every number of the text as a LaTeX macro (prefix \\fe), so that none is typed by hand.

As scripts/paper_text_shape.py: settings are read from the code (scripts/textnum.const, lmcold by import, or the source line where a
setting is written inline), results from the tables the figures are drawn from. Published values are read from
lmcold.constants (LMC_FEH_GRADIENT_LITERATURE, LMC_FEH_GRADIENT_DEG_LITERATURE, LMC_FEH_LEVEL_LITERATURE); other literature values
quoted with their citation stay in the text. The number of fields of Cole et al. (2005) is counted from their catalogue; the APOGEE
weight share at bright G and the BOSS giants in the 3-deg field of Carrera et al. (2008) are counted from the mass samples.
Inputs: results/{feh_profile, feh_gradients, mass_conversion_factors, kin_mcmc_data_ellipsoid, rrab_independence}.csv and
{feh_profile, boss_response, boss_selection, boss_mock, snr_threshold_sensitivity, mdf_literature}.json; data/{mdf_cole05,
mdf_carrera08}.csv and the BOSS and APOGEE giants of lmcold.mass_samples.samples() (with BOSS parent photometry); settings of
scripts/{paper_feh_profile, paper_feh_literature, paper_mdf_literature}.py, lmcold.{constants, footprint, mass_samples, rrlyrae,
selection}.
Writes results/macros_feh.tex.
Run: .venv/bin/python scripts/manifest.py run paper_text_feh.py"""

# %% [markdown]
# ### Settings and helpers
# Imports, the dictionary M of macros (macro name without its prefix: the text it stands for) and the functions that format
# numbers as the text writes them. The inputs listed in the header are read in the step that uses them.

# %%
import json
import re

import healpy as hp
import numpy as np
import pandas as pd

from lmcold import footprint, mass_samples, selection
from lmcold.constants import LMC, LMC_FEH_GRADIENT_DEG_LITERATURE, LMC_FEH_GRADIENT_LITERATURE, LMC_FEH_LEVEL_LITERATURE
from lmcold.mass_samples import FEH_EDGES
from lmcold.sources import DATA, ROOT
from textnum import const, num, sigfig

R = ROOT / 'results'   # directory of the input tables and of the output macros
M = {}                 # macro name (without the prefix \fe) -> text


def f1(x):
    """x with one decimal."""
    return f'{x:.1f}'


def f2(x):
    """x with two decimals."""
    return f'{x:.2f}'


def f3(x):
    """x with three decimals."""
    return f'{x:.3f}'


def src(path, pattern):
    """The number captured by `pattern` in the source file `path` (a setting written inline)."""
    return float(re.search(pattern, (ROOT / path).read_text()).group(1))


# Literal module constants of the three figure scripts (read without running them).
fp, fl, mdf = const('paper_feh_profile'), const('paper_feh_literature'), const('paper_mdf_literature')
t = pd.read_csv(R / 'feh_profile.csv')                          # radial profiles: sample, R_lo, R_hi [kpc], median [dex], sigma_*
js = json.loads((R / 'feh_profile.json').read_text())          # fit settings, scale lengths, bootstrap counts
br = json.loads((R / 'boss_response.json').read_text())        # BOSS [Fe/H] response: bias table, overlap count
# Median [Fe/H] [dex] per radial bin, indexed by the bin's inner radius R_lo [kpc], one Series per sample.
med = {s: d.set_index('R_lo')['median'] for s, d in t.groupby('sample')}
BRIGHT = fl['BRIGHT']   # sample name of the BOSS giants with G < 16.5

# %% [markdown]
# ### Settings of the profiles
# The disc plane, the radial bins and the bootstrap of scripts/paper_feh_profile.py.

# %%
# Disc plane, bins and bootstrap (lmcold.constants.LMC; scripts/paper_feh_profile.py)
km = pd.read_csv(R / 'kin_mcmc_data_ellipsoid.csv')   # posterior summaries of the ellipsoid kinematic fits
inc_kin = km[(km.stage == 2) & (km.param == 'inc')]['median'].iloc[0]   # the joint (shared-geometry) fit
bias = pd.DataFrame(br['bias'])   # BOSS minus APOGEE [Fe/H] bias per [Fe/H] bin (columns feh, median)
M.update(Inc=num(LMC['inc']), Theta=num(LMC['theta']), DZero=num(LMC['D0']), IncKin=num(round(inc_kin)),
         MinorPct=num(round(100 * (1 - np.cos(np.radians(LMC['inc'])) / np.cos(np.radians(inc_kin))))),
         RBin=num(t.R_hi[0] - t.R_lo[0]), RMax=num(t.R_hi.max()), Rad=num(json.loads((R / 'giant_mass.json').read_text())['full_radius_deg']),RhoRr=num(footprint.RHO_MAX),
         TrueA=num(mass_samples.TRUE_EDGES[2]), TrueB=num(mass_samples.TRUE_EDGES[3]),
         MeasW=num(mass_samples.MEASURED_EDGES[2] - mass_samples.MEASURED_EDGES[1]),
         BiasRich=f2(np.ceil(100 * bias['median'][bias.feh > FEH_EDGES[3]].abs().max()) / 100),
         NBoot=num(js['nboot']), Nside=num(src('scripts/paper_feh_profile.py', r"ang2pix\((\d+), s\.ra")),
         NOverlap=f"{br['overlap']:,}".replace(',', '\\,'), NSkip=num(js['skipped_bins']), NFits=f"{js['nboot'] * len(med['BOSS']):,}".replace(',', '\\,'))
M['Pix'] = f1(hp.nside2resol(int(M['Nside']), arcmin=True) / 60)   # HEALPix pixel size [deg] at Nside

# %% [markdown]
# ### Profiles of the giants
# Median [Fe/H] against radius, the fitted gradients, surface densities and scale lengths, and the same quantities in the mock
# of the BOSS selection.

# %%
# Medians (results/feh_profile.csv)
boss, apo, bright, rr = med['BOSS'], med['APOGEE'], med[BRIGHT], med['RRab']
BREAK, AGREE = 6., 8.   # kpc: the radius of the break, and the radius out to which the bright BOSS and APOGEE medians are compared
inner = boss.index < BREAK
M.update(BossIn=f2(boss.iloc[0]), BossMid=f2(boss[BREAK - 1]), BossOut=f2(boss.iloc[-1]), BossMidLo=num(BREAK - 1), Break=num(BREAK),
         ApoOffLo=f2((boss - apo).min()), ApoOffHi=f2((boss - apo).max()), GBright=num(fp['G_BRIGHT']), GLim=num(selection.PG_EDGES[-1]),
         BrightDropLo=f2((boss - bright)[inner].min()), BrightDropHi=f2((boss - bright)[inner].max()),
         BrightAgree=f3((bright - apo)[bright.index < AGREE].abs().max()), Agree=num(AGREE))
# Gradients, ours (results/feh_gradients.csv, rows 'this work')
g = pd.read_csv(R / 'feh_gradients.csv')
ours = g[g.label == 'this work'].set_index(['sample', 'r_lo', 'r_hi'])
for k, s in (('Bright', BRIGHT), ('Apo', 'APOGEE'), ('Boss', 'BOSS')):
    for part, rng in (('In', (0., BREAK)), ('Out', (BREAK, t.R_hi.max()))):
        r = ours.loc[(s, *rng)]
        M.update({f'Sl{k}{part}': f3(r.slope), f'Sl{k}{part}Err': f3(r.err)})
r = ours.loc[('RRab', 0., BREAK)]
M.update(SlRrIn=f3(r.slope), SlRrInErr=f3(r.err))
# Surface densities and scale lengths (results/feh_profile.csv, feh_profile.json)
q = t[t['sample'] == 'BOSS'].reset_index(drop=True)
M.update(FitLo=num(js['fit_range_kpc'][0]), FitHi=num(js['fit_range_kpc'][1]))
for k, key in (('Poor', 'lt15'), ('Mid', '15to10'), ('Rich', 'gt10')):
    h = js[key]
    M.update({f'H{k}': f2(h['h']), f'H{k}Up': f2(h['h_p84'] - h['h']), f'H{k}Dn': f2(h['h'] - h['h_p16']),
              f'H{k}Zero': f2(h['h_foreground_x0']), f'H{k}Double': f2(h['h_foreground_x2']), f'H{k}One': f1(h['h'])})
mid_r = .5 * (q.R_lo + q.R_hi)
fall = (q.sigma_gt10 / q.sigma_gt10.shift(-1))[mid_r > BREAK].dropna()   # factor by which the density falls from each bin to the next
M.update(FallLo=f1(fall.min()), FallHi=f1(fall.max()), FallFit=f1(np.exp(1 / js['gt10']['h'])),
         SigIn=num(round(q.sigma_lt15[0])), SigInTwo=num(round(q.sigma_lt15[1])), SigRef=num(round(q.sigma_lt15[2])),
         SigRefHi=num(q.R_hi[2]), OutLo=num(t.R_lo.max()))
sel = json.loads((R / 'boss_selection.json').read_text())['R_ring_closure']['final']
ring = next(iter(sel))   # innermost ring, e.g. '0-1.5'
M.update(RecIn=f2(sel[ring]['<-1.5']), RecRho=ring.split('-')[1])
# The same, per ring in the mock of the BOSS selection (results/boss_mock.json: central failure model, standard coverage, ring fill;
# ring0 = within the first edge of lmcold.selection.RING_EDGES)
mk = json.loads((R / 'boss_mock.json').read_text())['fill']['central']['standard/ring']['ring0']
M.update(MockIn=f2(mk['mean'][0]), MockInErr=f2(mk['sd'][0]), MockRho=num(selection.RING_EDGES[0]))

# %% [markdown]
# ### Completeness and weights
# The lowest completeness of the RR Lyrae union and the share of the APOGEE selection weight in bright stars.

# %%
# Completeness of the RR Lyrae union (OGLE-IV + Gaia): its minimum over the radial bins of results/rrab_independence.csv
M['UnionMin'] = num(round(100 * pd.read_csv(R / 'rrab_independence.csv').C_union_by_mag.min()))
# Share of the APOGEE selection weight at G < G_BRIGHT (Gaia G from the BOSS parent photometry, lmcold.mass_samples.with_colour)
bg, ag, _ = mass_samples.samples()
ag = mass_samples.with_colour(ag)
M['ApoBright'] = num(round(100 * ag.w[ag.g < fp['G_BRIGHT']].sum() / ag.w.sum()))
sn = json.loads((R / 'snr_threshold_sensitivity.json').read_text())
lo_snr, hi_snr = sorted((k for k in sn if k.isdigit()), key=int)   # S/N thresholds: adopted, alternative
dh = max(abs(sn[hi_snr]['h'][k] - sn[lo_snr]['h'][k]) for k in sn[lo_snr]['h'])
dm = np.abs(np.subtract(sn[f'median_feh_by_R_{hi_snr}'], sn[f'median_feh_by_R_{lo_snr}']))[1:].max()   # outside the first 1-kpc bin
M.update(SnrLo=lo_snr, SnrHi=hi_snr, SnrDh=f1(np.ceil(10 * dh) / 10), SnrDm=f3(dm))

# %% [markdown]
# ### Profiles of the RR Lyrae
# Median [Fe/H] and surface density of the RR Lyrae per ring.

# %%
# RR Lyrae: medians and surface densities (counts per 1-kpc ring in results/feh_profile.json)
FLAT = (4., 7.)   # kpc, the stretch of the RRab profile quoted as flat
flat = rr[(rr.index >= FLAT[0]) & (rr.index < FLAT[1])]
M.update(RrIn=f2(rr.iloc[0]), RrOut=f2(rr.iloc[-1]), FlatLo=num(FLAT[0]), FlatHi=num(FLAT[1]), FlatMedLo=f2(flat.max()),
         FlatMedHi=f2(flat.min()))
rrp = js['rr_lt15']
edges = np.arange(len(rrp['n']) + 1) * (t.R_hi[0] - t.R_lo[0])
dens = np.array(rrp['n']) / (np.pi * np.diff(edges ** 2))   # RRab below -1.5 per kpc^2
ratio = dens / q.sigma_lt15[:len(dens)].to_numpy()
fit = (.5 * (edges[1:] + edges[:-1]) > rrp['fit_range_kpc'][0]) & (.5 * (edges[1:] + edges[:-1]) < rrp['fit_range_kpc'][1])
M.update(RrRMax=num(rrp['fit_range_kpc'][1]), RrFitLo=num(rrp['fit_range_kpc'][0]), RatioMean=f2(ratio[fit].mean()),
         RatioLo=f2(ratio[fit].min()), RatioHi=f2(ratio[fit].max()), RatioSd=f2(ratio[fit].std()), HRrPoor=f2(rrp['h']),
         HRrRich=f2(js['rr_gt15']['h']), RrCentre=num(round(dens[0])), SigInLo=num(round(q.sigma_lt15_lo[0])),
         SigInHi=num(round(q.sigma_lt15_hi[0])), RatioIn=f1(ratio[0]), RatioInLo=f1(dens[0] / q.sigma_lt15_hi[0]),
         RatioInHi=num(round(dens[0] / q.sigma_lt15_lo[0])), RatioTwo=f1(ratio[1]), RatioTwoLo=f1(dens[1] / q.sigma_lt15_hi[1]),
         RatioTwoHi=f1(dens[1] / q.sigma_lt15_lo[1]))

# %% [markdown]
# ### Giants per unit mass
# The number of BOSS giants per unit formed mass (eta) in the metal-poor and metal-rich bins.

# %%
# Giants per unit mass (results/mass_conversion_factors.csv, BOSS, whole parent, reference weighting, measured alpha)
cv = pd.read_csv(R / 'mass_conversion_factors.csv').query("field == 'full' and prior == 'AMR' and alpha_model == 'measured'")
cv = cv[cv.survey == 'BOSS'].set_index('feh').eta
poor, rich = cv[(cv.index > -2) & (cv.index < -1.5)], cv[(cv.index > -.75) & (cv.index < -.25)]
M.update(EtaPoor=f1(poor.mean() * 1e4), EtaRichLo=f1(rich.min() * 1e4), EtaRichHi=f1(rich.max() * 1e4))

# %% [markdown]
# ### Published gradients
# The published gradients (lmcold.constants) against ours over the same radial ranges.

# %%
# Published gradients (lmcold.constants) against ours over the same ranges (results/feh_gradients.csv)
lit = {L['label']: L for L in LMC_FEH_GRADIENT_LITERATURE}   # published gradient [dex/kpc] by label
row = {(r.label, r['sample']): r for _, r in g.iterrows()}   # our fits over each published range, by (label, sample)


# Published slopes and errors as given (format g).
for k, label in (('Povick', 'Povick+25'), ('Grady', 'Grady+21'), ('Cioni', 'Cioni 09'), ('ChTwo', 'Choudhury+21'), ('Sk', 'Skowron+16 inner')):
    L = lit[label]
    M.update({f'Lit{k}': f'{L["slope"]:g}', f'Lit{k}Err': f'{L["err"]:g}'})
M.update(LitChTwoR=num(lit['Choudhury+21']['r_hi']), LitSkR=num(lit['Skowron+16 inner']['r_hi']))
# Choudhury et al. (2016): a range of slopes (slope to slope_hi).
c16 = lit['Choudhury+16']
M.update(LitChOneLo=f3(c16['slope']), LitChOneHi=f3(c16['slope_hi']), LitChOneR=num(c16['r_hi']))
# Our slopes [dex/kpc] over the same ranges, for the bright BOSS and the APOGEE giants.
for k, label in (('Povick', 'Povick+25'), ('ChOne', 'Choudhury+16'), ('ChTwo', 'Choudhury+21')):
    for s, sk in ((BRIGHT, 'Bright'), ('APOGEE', 'Apo')):
        r = row[label, s]
        M.update({f'Our{k}{sk}': f3(r.slope), f'Our{k}{sk}Err': f3(r.err)})
# Grady+21 and Cioni 09: range of the ratio published / our slope, and range of our slopes over the same ranges.
steep = [row[l, s] for l in ('Grady+21', 'Cioni 09') for s in (BRIGHT, 'APOGEE')]
M.update(SteepLo=f1(min(r.slope_lit / r.slope for r in steep)), SteepHi=f1(max(r.slope_lit / r.slope for r in steep)),
         OurSteepLo=f3(max(r.slope for r in steep)), OurSteepHi=f3(min(r.slope for r in steep)))
# Munoz et al. (2023) gradient per degree and our fits over its range (weighted, and unweighted for the bright giants).
mu = LMC_FEH_GRADIENT_DEG_LITERATURE[0]
mrow = {s: row[next(l for l, x in row if l.startswith('Muñoz')), s] for s in (BRIGHT, 'APOGEE')}
M.update(LitMunoz=f'{mu["slope"]:g}', LitMunozR=num(mu['rho_hi']), OurMunozApo=f3(mrow['APOGEE'].slope),
         OurMunozApoErr=f3(mrow['APOGEE'].err), OurMunozBright=f3(mrow[BRIGHT].slope), OurMunozBrightErr=f3(mrow[BRIGHT].err),
         OurMunozUnw=f3(mrow[BRIGHT].slope_unweighted))
# Carrera et al. (2008) [Fe/H] levels: inner field and the outer field at the mean of its radial range [deg].
car = [L for L in LMC_FEH_LEVEL_LITERATURE if L['label'] == 'Carrera+08']
M.update(CarIn=f1(car[0]['feh']), CarInR=num(car[0]['rho_hi']), CarOut=f1(car[1]['feh']), CarOutR=num(np.mean([car[1]['rho_lo'], car[1]['rho_hi']])))
# Skowron et al. (2016) RR Lyrae ranges: our RRab slopes over their middle and inner ranges.
skm = row['Skowron+16 middle', 'RRab']
M.update(SkMidLo=num(skm.r_lo), SkMidHi=num(skm.r_hi), OurSkMid=f3(skm.slope), OurSkMidErr=f3(skm.err),
         OurSkIn=f3(row['Skowron+16 inner', 'RRab'].slope), OurSkInErr=f3(row['Skowron+16 inner', 'RRab'].err),
         SkOutLo=num(lit['Skowron+16 outer']['r_lo']))

# %% [markdown]
# ### Appendix: fit settings and RR Lyrae metallicity scale
# Settings of the fits and of the RR Lyrae metallicities.

# %%
# Appendix: fit settings; scale of the RR Lyrae metallicities
SK16_ZW = (0.88, 1.431)   # Skowron et al. (2016): [Fe/H]_ZW = ([Fe/H]_J95 - 0.88) / 1.431
zw_c09 = src('src/lmcold/rrlyrae.py', r'return ([.0-9]+) \* feh_zw')   # [Fe/H]_C09 = 1.105 [Fe/H]_ZW + 0.160 (Carretta et al. 2009)
M.update(Floor=f3(src('scripts/paper_feh_literature.py', r"\.clip\(([.0-9]+)\)")),
         MinBins=num(src('scripts/paper_feh_literature.py', r'if len\(q\) < (\d+)')),
         KpcDeg=f3(LMC['D0'] * np.pi / 180), CrestOff=f2(src('src/lmcold/rrlyrae.py', r'feh_zw \+ [.0-9]+ \+ ([.0-9]+)')),
         ZwSlope=f2(zw_c09 / SK16_ZW[1]), ZwInner=f3(lit['Skowron+16 inner']['slope'] * zw_c09 / SK16_ZW[1]))

# %% [markdown]
# ### Appendix: published metallicity distributions
# Comparison with the field-star distributions of Cole et al. (2005), Carrera et al. (2008) and Omkumar et al. (2025).

# %%
# Appendix: metallicity distributions (results/mdf_literature.json)
md = json.loads((R / 'mdf_literature.json').read_text())
pa, pb, pc, pd_ = (next(v for k, v in md.items() if k.startswith(f'({x})')) for x in 'abcd')
cole = pd.read_csv(DATA / 'mdf_cole05.csv').Field.str.split('/').str[0]
sc = md['scale_check']
# BOSS giants within FIELD_R of the 3-deg field of Carrera et al. (2008) (their radial-velocity members within 4 deg of the centre, as
# scripts/paper_mdf_literature.py, which leaves the field out)
car = pd.read_csv(DATA / 'mdf_carrera08.csv').query('170 <= rv <= 380')
car = car[footprint.angular_distance(car.ra.to_numpy(), car.dec.to_numpy(), LMC['ra0'], LMC['dec0']) < 4]
M['NCarIn'] = num((footprint.angular_distance(bg.ra.to_numpy(), bg.dec.to_numpy(), car.ra.mean(), car.dec.mean()) < mdf['FIELD_R']).sum())


def stat(d, key, f='median'):
    """Statistic f of sample key in panel dict d: the median [dex] with two decimals, a fraction as a rounded percentage."""
    x = d[key][f]
    return f2(x) if f == 'median' else num(round(100 * x))


M.update(FieldR=num(mdf['FIELD_R']), NColeF=num(cole.nunique()), NColeMatch=num(sc['BOSS - Cole+05']['n']),
         ColeMed=stat(pa, 'Cole+05, Ca II triplet'), BarBoss=stat(pa, 'BOSS, as measured'), BarApo=stat(pa, 'APOGEE'),
         ColeRich=stat(pa, 'Cole+05, Ca II triplet', 'frac_above_m05'),
         CarMed=stat(pb, 'Carrera+08, Ca II triplet'), DiscBoss=stat(pb, 'BOSS, as measured'), DiscApo=stat(pb, 'APOGEE'),
         CarPoor=stat(pb, 'Carrera+08, Ca II triplet', 'frac_below_m1'), DiscBossPoor=stat(pb, 'BOSS, as measured', 'frac_below_m1'),
         DiscApoPoor=stat(pb, 'APOGEE', 'frac_below_m1'),
         ScCar=f2(sc['BOSS - Carrera+08']['median']), NScCar=num(sc['BOSS - Carrera+08']['n']), SdCar=f2(sc['BOSS - Carrera+08']['robust_sd']),
         ScApo=f2(sc['APOGEE - Carrera+08']['median']), NScApo=num(sc['APOGEE - Carrera+08']['n']),
         SdApo=f2(sc['APOGEE - Carrera+08']['robust_sd']), ScCole=f2(sc['BOSS - Cole+05']['median']),
         ApoTgt=stat(pc, 'APOGEE, as targeted'), ApoW=stat(pc, 'APOGEE, weighted'),
         ApoTgtRich=stat(pc, 'APOGEE, as targeted', 'frac_above_m05'), ApoWRich=stat(pc, 'APOGEE, weighted', 'frac_above_m05'),
         ApoTgtPoor=stat(pc, 'APOGEE, as targeted', 'frac_below_m1'), ApoWPoor=stat(pc, 'APOGEE, weighted', 'frac_below_m1'),
         BossAll=stat(pc, 'BOSS, weighted, as measured'), BossAllRich=stat(pc, 'BOSS, weighted, as measured', 'frac_above_m05'),
         NXp=f"{pd_['BOSS, as measured']['n']:,}".replace(',', '\\,'), OmkMed=stat(pd_, 'Omkumar+25, XP Strömgren'),
         XpBossMed=stat(pd_, 'BOSS, as measured'), OmkRich=stat(pd_, 'Omkumar+25, XP Strömgren', 'frac_above_m05'),
         OmkPoor=stat(pd_, 'Omkumar+25, XP Strömgren', 'frac_below_m1'), XpBossRich=stat(pd_, 'BOSS, as measured', 'frac_above_m05'),
         XpBossPoor=stat(pd_, 'BOSS, as measured', 'frac_below_m1'), AndMed=stat(pd_, 'Andrae+23, XP'),
         AndShift=sigfig(pd_['BOSS, as measured']['median'] - pd_['Andrae+23, XP']['median']),
         ScaleCaT=sigfig(-np.mean([sc['BOSS - Carrera+08']['median'], sc['BOSS - Cole+05']['median']])))

# %% [markdown]
# ### Write the macros
# One \newcommand per entry of M, with the prefix \fe, into results/macros_feh.tex; the page text shows them through
# {glue:text}, so no number is typed by hand.

# %%
# Header line, then one \newcommand per macro; written to results/macros_feh.tex and printed.
lines = ['% Numbers of the KB chapter Metallicity structure (scripts/paper_text_feh.py); do not edit.']
lines += [rf'\newcommand{{\fe{k}}}{{{x}}}' for k, x in M.items()]
(R / 'macros_feh.tex').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
