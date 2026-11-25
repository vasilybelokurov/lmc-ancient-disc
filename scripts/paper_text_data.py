"""Knowledge-base text, the three data chapters ('Data: the samples', 'Data: selection functions', 'Data: RR Lyrae and BOSS
calibration'; extended draft Sections 2.1-2.2 and the data appendices): every number of their text as a LaTeX macro (prefix \\da),
so that none is typed by hand.

As scripts/paper_text_shape.py: settings are read from the code (module constants by import or scripts/textnum.const, function
defaults with textnum.default, and literals inside functions with grab() from the source), results from the tables and catalogues
the figures are drawn from. Literature values quoted with their citation stay in the text.
Inputs: results/{boss_selection, apogee_selection, boss_calibration, boss_hot_bias, boss_mock, selection_gaps, snr_cut_check,
snr_threshold_sensitivity, boss_response, giant_counts}.json; results/{rrab_membership, rrab_independence, rrab_pw_fit,
rrab_feh_spec, rrab_feh_transform, parent_extinction, kin_mcmc_data_ellipsoid}.csv; data/{rrab, giants_apogee, giants_boss,
boss_weights, boss_cartons, apogee_parent, apogee_weights, spec_giants}.parquet; scripts/{build_rrab, build_spec_giants, build_giants,
boss_selection, apogee_selection, boss_mock, build_xp_andrae, fig_rrab_membership, paper_letter_signal, paper_maps, paper_selection,
paper_selection_checks, paper_selection_gaps, paper_boss_calibration, paper_boss_hot_bias, fit_kinematics_mcmc}.py; lmcold.{constants,
footprint, rrlyrae, giants, magcloud, selection, apogee_target, mass_samples, maps, samples, kinfit}.
Writes results/macros_data.tex.
Run: .venv/bin/python scripts/manifest.py run paper_text_data.py"""

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

from fit_kinematics_mcmc import ring as kin_ring
from lmcold import apogee_target, giants, kinfit, ogle, rrlyrae, samples, selection
from lmcold.constants import KAPPA, LMC
from lmcold.footprint import DMU_WINDOW, RHO_MAX, SMC_MASK_RADIUS, angular_distance
from lmcold.maps import pixel_map
from lmcold.mass_samples import TRUE_EDGES
from lmcold.sources import DATA, ROOT
from lmcold.tex import count
from textnum import const, default, num, sigfig

R = ROOT / 'results'  # folder of the result tables read here and of the macro file written
M = {}  # macro name (without the \da prefix): text it stands for


def js(name):
    """results/<name>.json as a dict."""
    return json.loads((R / f'{name}.json').read_text())  # parsed JSON


def grab(path, pattern, k=1):
    """Number in group k of the regular expression `pattern` in the source file `path` (relative to the repository root): a
    literal inside a function, read from the code rather than retyped."""
    # First match of the pattern in the file text; group k converted to float.
    return float(re.search(pattern, (ROOT / path).read_text()).group(k))


def f2(x):
    """x with two decimals."""
    return f'{x:.2f}'


def signed(x, fmt=f2):
    """x formatted by fmt with an explicit sign ('+' added to non-negative values)."""
    s = fmt(x)
    return s if s.startswith('-') else '+' + s


def pct(x, d=0):
    """Fraction x in per cent with d decimals."""
    return f'{100 * x:.{d}f}'


def pm_err(x, e):
    """Value and error, the error to one significant figure (two if its first digit is 1) and the value to the same decimals."""
    # Error string: one significant figure, or two if the error's leading digit is 1.
    s = sigfig(e)
    s = sigfig(e, 2) if f'{e:e}'.startswith('1') else s
    d = len(s.split('.')[1]) if '.' in s else 0  # decimals of the error string
    return f'{x:.{d}f}', s  # (value, error) strings


def edges(xs):
    """A list of bin edges as written in the text, e.g. '12, 15.5, 16.25'."""
    return ', '.join(num(x) for x in xs)


# %% [markdown]
# ### Chapter 1: the RR Lyrae sample
# Query radius and cross-match of the RRab catalogues, the counts of the union and its members, the metallicity scales, the
# period-Wesenheit fit and the foreground contamination (data/rrab.parquet and the RR Lyrae result tables).

# %%
# ------------------------------------------------------------------------------------------------------------------------------
# Chapter 1: the samples
# ------------------------------------------------------------------------------------------------------------------------------
M.update(RaZero=num(LMC['ra0']), DecZero=num(LMC['dec0']), RrlQuery=num(grab('scripts/build_rrab.py', r't\.rho < (\d+)\]')),
         Match=num(grab('scripts/build_rrab.py', r'sep\.arcsec < ([\d.]+)')), FirstAdded=num(rrlyrae.FIRST_GAIA_ADDED))
# RRab union (data/rrab.parquet) and its LMC members: counts by catalogue, proper motions and [Fe/H] source.
t = pd.read_parquet(DATA / 'rrab.parquet')
mem = t[t.member]
M.update(NUnion=count(len(t)), NBoth=count((t.in_ogle & t.in_gaia).sum()), NOwn=count(t.ogle_own.sum()),
         NMembers=count(len(mem)), NMembersPM=count((mem.pmra.notna() & mem.pmra_pmdec_corr.notna()).sum()),
         NFehI=count((mem.feh_src == 'I').sum()), NFehG=count((mem.feh_src == 'G').sum()), NFehNone=count((mem.feh_src == '').sum()))
# Photometric [Fe/H] transformation (shift and Deming slope) and the offsets from three spectroscopic samples (G04, B04, B06).
tr = pd.read_csv(R / 'rrab_feh_transform.csv').iloc[0]
M.update(FehShift=f2(tr['shift']), NShift=count(tr.N), Deming=f'{tr.deming_b:.3f}', DemingErr=f'{tr.deming_b_err:.3f}')
sp = pd.read_csv(R / 'rrab_feh_spec.csv').query("estimator == 'feh_I'").set_index('spec')
M.update(GrattonOff=f2(sp.d_median.G04), GrattonErr=f2(sp.d_median_err.G04), NGratton=num(sp.N.G04),
         BorA=f2(sp.d_median.B04), BorAErr=f2(sp.d_median_err.B04), NBorA=num(sp.N.B04),
         BorB=f2(sp.d_median.B06), BorBErr=f2(sp.d_median_err.B06), NBorB=num(sp.N.B06))
# Period-Wesenheit fit: core radius and Wesenheit colour coefficients read from build_rrab.py, scatter per Wesenheit index,
# membership window in distance modulus.
pw = pd.read_csv(R / 'rrab_pw_fit.csv').set_index('wesenheit')
M.update(PwRho=num(grab('scripts/build_rrab.py', r'core = \(\(t\.rho < (\d+)\)')),
         WICoef=f2(grab('scripts/build_rrab.py', r't\.I - ([\d.]+) \* \(t\.V - t\.I\)')),
         WGCoef=f2(grab('scripts/build_rrab.py', r't\.int_average_g - ([\d.]+) \*')),
         ScatI=f2(pw.rsd.I), NPwI=count(pw.N.I), ScatG=f2(pw.rsd.G),
         RhoMax=num(RHO_MAX), SmcMask=num(SMC_MASK_RADIUS), DmuWin=num(DMU_WINDOW), DmuScat=sigfig(pw.rsd.G),
         DmuRatio=num(round(DMU_WINDOW / float(sigfig(pw.rsd.G)))))
# Foreground contamination per rho bin (results/rrab_membership.csv): inner, 2-8 deg, outer and beyond RHO_MAX.
side = const('fig_rrab_membership')['SIDE']
cm = pd.read_csv(R / 'rrab_membership.csv')
mid = cm[(cm.rho_lo >= 2) & (cm.rho_hi <= 8)]
M.update(SideLo=num(side[0]), SideHi=num(side[1]), ContIn=pct(cm.contam.iloc[0], 1), RhoContIn=num(cm.rho_hi.iloc[0]),
         ContMidLo=pct(mid.contam.min(), 1), ContMidHi=pct(mid.contam.max(), 1), RhoContMid=num(mid.rho_hi.max()),
         ContOut=pct(cm.contam[cm.rho_hi == RHO_MAX].iloc[0], 1),
         ContMax=pct(cm.contam[cm.rho_hi <= RHO_MAX].max(), 1), ContBeyond=num(round(100 * cm.contam[cm.rho_lo == RHO_MAX].iloc[0])),
         RhoBeyond=num(cm.rho_hi[cm.rho_lo == RHO_MAX].iloc[0]))

# %% [markdown]
# ### Chapter 1: the giant samples
# The cuts of the giant samples and the velocity clip of the Letter's Fig. 1 with the number of stars it drops in each
# population.

# %%
# Giants: sample definition (scripts/build_spec_giants.py, build_giants.py; lmcold.giants)
win = re.search(r'< (\d+)\) & \(t\.v > (\d+)\) & \(t\.v < (\d+)\)', (ROOT / 'scripts/build_spec_giants.py').read_text())
sig = const('paper_letter_signal')   # the velocity clip of the Letter's Fig. 1 (c)
M.update(Clip=num(sig['CLIP']), ClipErr=num(sig['CLIP_ERR']))
# Stars the clip drops per population, as internal() of scripts/paper_letter_signal.py (which prints them): residuals from the
# population's stage-2 disc model of the joint ellipsoid fit, proper-motion components against CLIP + CLIP_ERR x the star's error
# (KAPPA x disc distance x error [km/s]), line of sight against CLIP.
km = pd.read_csv(R / 'kin_mcmc_data_ellipsoid.csv')  # posterior summaries of the joint ellipsoid fit
rr = samples.rrab()
nan = np.full(len(rr['ra']), np.nan)
clip = []   # (stars, dropped): RR Lyrae, then the giant bins from metal-poor to metal-rich
for name, c in [('RR Lyrae', kin_ring(rr | dict(vlos=nan, evlos=nan)))] + [(f'giants {k}', kin_ring(c)) for k, c in samples.giants()]:
    # Parameter vector of the population: stage-2 medians of the shared and of its own parameters.
    p = kinfit.start()
    for _, r in km[(km.stage == 2) & km.population.isin(['shared', name])].iterrows():
        p[kinfit.PARAMS.index(r.param)] = r['median']
    # Residuals of the proper-motion velocity components [km/s] from the disc model.
    _, rt, rr_ = (np.asarray(x) for x in kinfit.residual_polar(p, c))
    d = kinfit.data(*(c[k] for k in sig['COLS']))
    mod = kinfit.model(p, d['rhat'], d['ea'], d['ed'])
    # Clip limit per star [km/s]: CLIP + CLIP_ERR x proper-motion error converted at the disc distance mod[3] [kpc].
    lim = sig['CLIP'] + sig['CLIP_ERR'] * KAPPA * np.asarray(mod[3]) * np.hypot(c['pmra_error'], c['pmdec_error'])
    ok = (np.abs(rt) < lim) & (np.abs(rr_) < lim) & ~(np.abs(np.nan_to_num(np.asarray(c['vlos']) - np.asarray(mod[2]))) > sig['CLIP'])
    clip.append((len(ok), (~ok).sum()))
drop = [x / n for n, x in clip[1:]]  # fraction dropped in each giant bin
M.update(ClipNRr=count(clip[0][0]), ClipRr=num(clip[0][1]), ClipRrPct=sigfig(100 * clip[0][1] / clip[0][0]),
         ClipGLo=sigfig(100 * min(drop)), ClipGHi=sigfig(100 * max(drop)), ClipNMp=count(clip[1][0]), ClipMp=num(clip[1][1]))
# Giant-sample cuts: from lmcold constants and function defaults, and from the build scripts' sources.
M.update(NFields=num(len(apogee_target.FIELDS)), GiantRho=win.group(1), VlosLo=win.group(2), VlosHi=win.group(3),
         GLim=num(grab('src/lmcold/magcloud.py', r'ok & \(g <= ([\d.]+)\)')),
         SnrApoKin=num(default(giants.apogee_spectrum_ok, 'snr_min')), SnrApoSel=num(const('apogee_selection')['SNR_MIN']),
         SnrBoss=num(giants.BOSS_SNR_MIN), EvBoss=num(default(giants.boss_spectrum_ok, 'ev_max')),
         TeffMax=num(default(giants.is_giant, 'teff_max')), LoggMax=num(default(giants.is_giant, 'logg_max')),
         EvApo=num(default(giants.rv_ok, 'ev_max')), VscatApo=num(default(giants.rv_ok, 'vscatter_max')),
         Ruwe=num(default(giants.gaia_ok, 'ruwe_max')),
         PlxZero=num(grab('scripts/build_giants.py', r'\(s\.parallax \+ ([\d.]+)\) / s\.parallax_error < (\d+)')),
         PlxSig=num(grab('scripts/build_giants.py', r'\(s\.parallax \+ ([\d.]+)\) / s\.parallax_error < (\d+)', 2)))
# BOSS [Fe/H] slope against APOGEE per S/N bin (results/snr_cut_check.json): bins below the BOSS S/N cut, and the top bin.
sn = pd.DataFrame(js('snr_cut_check')['by_snr'])
lo, top = sn[sn.snr_hi <= giants.BOSS_SNR_MIN], sn.iloc[-1]
M.update(SnrA=num(lo.snr_lo.iloc[0]), SnrB=num(lo.snr_hi.iloc[0]), SnrC=num(lo.snr_hi.iloc[1]), SnrTop=num(top.snr_lo),
         SlopeA=f2(lo.slope.iloc[0]), SlopeAErr=f2(lo.slope_err.iloc[0]), NSlopeA=num(lo.n.iloc[0]),
         SlopeB=f2(lo.slope.iloc[1]), SlopeBErr=f2(lo.slope_err.iloc[1]), NSlopeB=num(lo.n.iloc[1]),
         SlopeTop=f2(top.slope), SlopeTopErr=f2(top.slope_err),
         CalHighA=f2(lo.calibrated_minus_apogee.iloc[0]), CalHighB=f2(lo.calibrated_minus_apogee.iloc[1]))
# Bins above the cut: the first, the second, and those at S/N >= SNR_HIGH quoted as a range.
above = sn[sn.snr_lo >= giants.BOSS_SNR_MIN]
SNR_HIGH = 30   # BOSS S/N above which the slopes are quoted together
high = above[above.snr_lo >= SNR_HIGH]
r16 = above.iloc[0]
M.update(SlopeC=f2(r16.slope), SlopeCErr=f2(r16.slope_err), NSlopeC=num(r16.n), SnrD=num(r16.snr_hi),
         SlopeD=f2(above.slope.iloc[1]), SlopeDErr=f2(above.slope_err.iloc[1]), SnrE=num(above.snr_hi.iloc[1]),
         SnrF=num(SNR_HIGH), SlopeFLo=f2(high.slope.min()), SlopeFHi=f2(high.slope.max()), SlopeFErr=f2(high.slope_err.max()),
         CalAbove=f2(above.calibrated_minus_apogee.abs().max()))

# %% [markdown]
# ### Chapter 1: BOSS calibration
# The calibration of the BOSS [Fe/H] and velocities to APOGEE on the stars observed by both.

# %%
# BOSS calibration (results/boss_calibration.json; lmcold.giants.regressors)
cal, ho = js('boss_calibration')['calibration'], js('boss_calibration')['held_out_test']
# Calibration coefficients (velocity d0, d1; [Fe/H] c0-c3), pivots read from lmcold.giants, and the held-out test.
M.update(NOverlap=count(cal['N']), TPivot=num(giants.T_PIVOT), TScale=num(grab('src/lmcold/giants.py', r'T_PIVOT\) / ([\d.]+)')),
         SPivot=num(grab('src/lmcold/giants.py', r'([\d.]+) / np\.asarray\(snr, float\) - 1')),
         VcalA=f2(cal['d0']), VcalB=f2(-cal['d1']), FcalA=f'{cal["c0"]:.3f}', FcalB=f'{cal["c1"]:.3f}', FcalC=f'{-cal["c2"]:.3f}',
         FcalD=f'{cal["c3"]:.3f}', Compress=f2(cal['c1']),
         HoMed=f'{ho["dfeh_median"]:.3f}', HoRsd=f'{ho["dfeh_rsd"]:.3f}', HoDv=signed(ho['dv_median']),
         HoBin=sigfig(max(abs(v) for k in ('dfeh_median_by_snr', 'dfeh_median_by_teff') for v in ho[k].values())))
# Residual bias of the calibrated BOSS [Fe/H] in the APOGEE range BIAS_FEH.
bias = js('boss_response')['bias']
BIAS_FEH = (-2.0, -1.5)   # APOGEE [Fe/H] range [dex] of the residual bias quoted (results/boss_response.json, 0.25-dex bins)
bmid = [m for f, m in zip(bias['feh'], bias['median']) if BIAS_FEH[0] < f < BIAS_FEH[1]]
M.update(BiasLo=f2(-max(bmid)), BiasHi=f2(-min(bmid)), BiasFehLo=num(BIAS_FEH[0]), BiasFehHi=num(BIAS_FEH[1]))
# Sizes of the two kinematic giant catalogues and their overlap.
ga, gb = pd.read_parquet(DATA / 'giants_apogee.parquet'), pd.read_parquet(DATA / 'giants_boss.parquet')
M.update(NKinApo=count(len(ga)), NKinBoss=count(len(gb)), NKinBoth=count(ga.in_both.sum()))

# %% [markdown]
# ### Chapter 2: the RR Lyrae and BOSS selection functions
# Completeness of the RR Lyrae union, the BOSS parent and its groups, the pass factors, the coverage and the tests of the XP
# bins.

# %%
# ------------------------------------------------------------------------------------------------------------------------------
# Chapter 2: selection functions
# ------------------------------------------------------------------------------------------------------------------------------
# Carton cuts read from the source of lmcold.magcloud: proper-motion window, parallax cut, SFD extinction rule, A_V scaling.
mc = (ROOT / 'src/lmcold/magcloud.py').read_text()
pmw = re.search(r'np\.hypot\(pmL - ([\d.]+), pmB - ([\d.]+)\) < ([\d.]+)', mc)
plx = re.search(r'\(parallax \+ ([\d.]+)\) / parallax_error > (\d+)', mc)
sfd = re.search(r'\(lmcrad <= ([\d.]+)\).*?\(ebv >= ([\d.]+)\), ([\d.]+) \* ebv', mc)
av = re.search(r'av = ([\d.]+) \* ([\d.]+) \* rescaled_ebv', mc)
# Carton colour and proper-motion cuts; APOGEE targeting (H limits, field radius, bright neighbours); velocity window of the
# mass samples; coverage threshold NMIN_GOOD.
M.update(BpRp=num(grab('src/lmcold/magcloud.py', r'\(bp - rp >= ([\d.]+)\)')), PmBox=num(grab('src/lmcold/magcloud.py', r'np\.abs\(pmra\) <= ([\d.]+)')),
         PmL=pmw.group(1), PmB=pmw.group(2), PmWin=pmw.group(3), PlxOff=plx.group(1), PlxSigC=plx.group(2),
         SfdRad=sfd.group(1), SfdEbv=sfd.group(2), SfdScale=sfd.group(3), AvA=av.group(1), AvB=av.group(2),
         HBright=num(grab('src/lmcold/apogee_target.py', r'\(h >= ([\d.]+) - tol\)')),
         HFaint=num(grab('src/lmcold/apogee_target.py', r'np\.where\(h < ([\d.]+), 1, 2\)')),
         HlimLo=num(min(apogee_target.MAGLIM.values())), HlimMid=num(sorted(set(apogee_target.MAGLIM.values()))[1]),
         HlimHi=num(max(apogee_target.MAGLIM.values())), FieldR=num(apogee_target.RADIUS),
         NbrMag=num(default(apogee_target.bright_neighbour, 'dmag')), NbrSep=num(default(apogee_target.bright_neighbour, 'radius')),
         VMassLo=num(grab('src/lmcold/mass_samples.py', r'\(b\.vlos > (\d+)\) & \(b\.vlos < (\d+)\)')),
         VMassHi=num(grab('src/lmcold/mass_samples.py', r'\(b\.vlos > (\d+)\) & \(b\.vlos < (\d+)\)', 2)),
         NminGood=num(selection.NMIN_GOOD))

# RR Lyrae completeness (results/rrab_independence.csv; data/rrab.parquet as scripts/paper_selection.py, 1-deg bins, >= 20 stars)
ri = pd.read_csv(R / 'rrab_independence.csv')
M.update(SplitMax=pct(max((ri.C_union_by_P - ri.C_union_by_mag).abs().max(), (ri.C_union_by_amp - ri.C_union_by_mag).abs().max()), 1),
         UnionIn=pct(ri.C_union_by_mag.iloc[0]), UnionRhoA=num(ri.rho_hi.iloc[0]), UnionRhoB=num(ri.rho_hi.iloc[2]),
         UnionMidLo=pct(ri.C_union_by_mag.iloc[1:3].min(), 1), UnionMidHi=pct(ri.C_union_by_mag.iloc[1:3].max(), 1),
         UnionOutLo=pct(ri.C_union_by_mag.iloc[3:].min(), 1), UnionOutHi=pct(ri.C_union_by_mag.iloc[3:].max(), 1))
t['in_fp'] = ogle.in_footprint(t.ra.to_numpy(), t.dec.to_numpy())
RHO_BINS = np.arange(0, 16.1, 1.0)


def complete(s, flag):
    """Fraction of s with `flag` per 1-deg bin of rho (NaN below 20 stars), as panel (a) of figures/paper_selection.png."""
    k = np.digitize(s.rho, RHO_BINS) - 1  # 1-deg rho bin index per star
    return np.array([s[flag][k == i].mean() if (k == i).sum() >= 20 else np.nan for i in range(len(RHO_BINS) - 1)])


# Completeness of each catalogue measured on the other: Gaia on OGLE's own RRab, OGLE on Gaia RRab inside the OGLE footprint.
cg, co = complete(t[t.ogle_own], 'in_gaia'), complete(t[t.in_gaia & t.in_fp], 'ogle_own')
GAIA_FULL, OGLE_IN = 9, 6   # deg: Gaia completeness quoted beyond GAIA_FULL; OGLE's own search quoted inside OGLE_IN
M.update(GaiaMissIn=pct(1 - cg[0]), GaiaRhoOut=num(GAIA_FULL), GaiaMissOut=pct(1 - np.nanmin(cg[GAIA_FULL:])),
         OgleRho=num(OGLE_IN), OgleLo=pct(co[:OGLE_IN].min()), OgleHi=pct(co[:OGLE_IN].max()))

# BOSS parent and groups (results/boss_selection.json; data/boss_weights.parquet; scripts/boss_selection.py)
bs = js('boss_selection')
summ = bs['summary']
bw = pd.read_parquet(DATA / 'boss_weights.parquet')
cart = pd.read_parquet(DATA / 'boss_cartons.parquet')
# Parent stars in the mwm_magcloud_rgb_boss carton.
in_cart = bw[['source_id']].merge(cart, on='source_id').cartons.str.contains('mwm_magcloud_rgb_boss')
sel = const('boss_selection')
ngood_g = bw.group.map(bw.groupby('group').good.sum())  # good spectra in each star's selection group
# Parent, carton share, observed and good counts; selection-group settings of boss_selection.py; groups; bootstrap nside.
M.update(NParent=count(summ['eligible']), InCarton=pct(in_cart.sum() / summ['eligible']), NObs=count(bw.observed.sum()),
         NGood=count(summ['good']), Nside=num(sel['LEVELS'][0][0]), PixArea=f2(hp.nside2pixarea(sel['LEVELS'][0][0], degrees=True)),
         GEdges=edges(sel['GEDGES']), NminGroup=num(sel['NMIN']), NsideCoarse=', '.join(num(n) for n, _ in sel['LEVELS'][2:-1]),
         NsideLast=num(sel['LEVELS'][-1][0]), NGroups=count(bw.group.nunique()), FinestShare=pct(bw.group.str.startswith('0:').mean()),
         NsideBoot=num(grab('scripts/boss_selection.py', r'pix8, pix16 = hp\.ang2pix\(\d+, theta, phi, nest=True\), hp\.ang2pix\((\d+),')))
# Members lost to the astrometric cuts (results/selection_gaps.json).
sg = js('selection_gaps')
al = sg['astrometric_loss_of_members']
M.update(NAstroB=count(al['LMC18-35']['n']), LossB=pct(al['LMC18-35']['fail'], 1), NAstroA=count(al['LMC1-17']['n']),
         LossA=pct(al['LMC1-17']['fail'], 1))
pe = pd.read_csv(R / 'parent_extinction.csv').query("model == 'region'")
DUSTY = 0.15   # E(V-I) [mag] above which the central parent loses stars (results/parent_extinction.csv, region model)
dz = pe[pe.evi_lo >= DUSTY]  # E(V-I) bins above DUSTY; q is the retained fraction, so n_g (1/q - 1) stars are missing
M.update(DustMissing=pct((dz.n_g * (1 / dz.q - 1)).sum() / pe.n_g.sum(), 1))

# Pass factors (results/boss_selection.json P_pass_factor; lmcold.selection; scripts/build_xp_andrae.py)
pf = bs['P_pass_factor']
xa = const('build_xp_andrae')
nx, nr = len(pf['xp_edges']), len(pf['ring_edges']) + 1
i0 = ((len(pf['g_edges']) - 2) * nx + 0) * nr   # faintest G bin, lowest XP bin, ring 0
# Pass factors with errors in the three rings, for the faintest G bin and lowest XP bin.
pa = [pm_err(pf['factor'][i0 + r], pf['err'][i0 + r]) for r in range(nr)]
M.update(XpTeff=num(xa['TEFF_MAX']), XpLogg=num(xa['LOGG_MAX']), XpShare=pct(summ['with_xp'] / summ['eligible']),
         PgEdges=edges(selection.PG_EDGES), XpEdges=edges(selection.XP_EDGES[1:-1]), XpLow=num(selection.XP_EDGES[1]),
         RingA=num(selection.RING_EDGES[0]), RingB=num(selection.RING_EDGES[1]), RingOut=M['GiantRho'],
         PassG=num(selection.PG_EDGES[1]), PassXp=num(selection.XP_EDGES[3]),
         GFaintLo=num(pf['g_edges'][-2]), GFaintHi=num(pf['g_edges'][-1]),
         PassA=pa[0][0], PassAErr=pa[0][1], PassB=pa[1][0], PassBErr=pa[1][1], PassC=pa[2][0], PassCErr=pa[2][1])
# Ring closure ([M/H] < -1.5): pass factor common to all rings ('common') against the final weights.
rc = bs['R_ring_closure']
M.update(CommonIn=f2(rc['common']['0-1.5']['<-1.5']), CommonOut=f2(rc['common']['6-12']['<-1.5']), RingIn=list(rc['final'])[0].split('-')[1],
         RingFinA=f2(rc['final']['0-1.5']['<-1.5']), RingFinB=f2(rc['final']['1.5-3']['<-1.5']),
         RingFinC=f2(rc['final']['3-6']['<-1.5']), RingFinD=f2(rc['final']['6-12']['<-1.5']))

# Coverage (results/selection_gaps.json)
cf, sparse = sg['coverage_factor'], sg['sparse_groups']
M.update(NCovGroups=count(bw[ngood_g >= selection.NMIN_GOOD].group.nunique()), NCovParent=count(summ['covered']),
         CovShare=pct(summ['covered'] / summ['eligible'], 1), NUncov=count(sg['uncovered']), UncovShare=pct(sg['uncovered_share'], 1),
         UncovIn=pct(sg['uncovered_by_ring']['0'] / sg['uncovered']),
         CovRingLo=f2(min(cf['per_ring'])), CovRingHi=f2(max(cf['per_ring'])), CovMax=f2(cf['ring0'][0]), CovMaxG=num(sel['GEDGES'][1]),
         SparseFrac=num(100 * grab('scripts/paper_selection_gaps.py', r'gs\.good / gs\.parent < ([\d.]+)')), SparseN=num(sparse['groups']),
         SparseShare=pct(sparse['parent_share'], 1), SparseGood=num(sparse['good']), SparseRho=f'{sparse["median_rho"]:.1f}',
         MaxW=count(round(sparse['max_w'])))
# Weighted non-member fraction of BOSS good spectra, overall and per ring; share of good spectra that are not giants.
nm = sg['boss_nonmember']
gd = bw[bw.good]
M.update(NonMem=pct(nm['all'], 1), NonMemLo=pct(min(nm['by_ring'].values()), 1), NonMemHi=pct(max(nm['by_ring'].values()), 1),
         Dwarfs=pct((~gd.giant).mean(), 1))

# Tests: the share of each XP bin in the good spectra over that in the covered parent (as figures/paper_selection.png b)
XP_BINS = np.array([-2.5, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, 0.0])   # edges of panel (b), scripts/paper_selection.py
cov = bw[(ngood_g >= selection.NMIN_GOOD) & bw.mh_xp.notna()]  # covered parent with XP [M/H]
# XP bin per star; par: covered parent per bin; unw: share of good spectra over share of parent (unweighted, 1 = no bias).
k = np.digitize(cov.mh_xp, XP_BINS) - 1
ok = (k >= 0) & (k < len(XP_BINS) - 1)
par = np.bincount(k[ok], minlength=len(XP_BINS) - 1)
s_ = ok & cov.good.to_numpy()
unw = np.bincount(k[s_], minlength=len(XP_BINS) - 1) / s_.sum() / (par / par.sum())
OVER_XP = -1.25   # [dex]: the unweighted over-representation is quoted for the XP bins below this edge
poor = XP_BINS[1:] <= OVER_XP
# Over-representation range, closure (final and group weights), held-out sky, and the parent fraction below -1.5.
cl = bs['C1_closure']
M.update(OverLo=pct(unw[poor].min() - 1), OverHi=pct(unw[poor].max() - 1), OverXp=num(OVER_XP), Under=pct(1 - unw[-1]),
         FinA=f'{cl["final"]["<-1.0"]["diff"]:.4f}', FinAErr=f'{cl["final"]["<-1.0"]["err"]:.4f}',
         FinB=f'{cl["final"]["<-1.5"]["diff"]:.4f}', FinBErr=f'{cl["final"]["<-1.5"]["err"]:.4f}',
         FinC=f'{cl["final"]["<-2.0"]["diff"]:.4f}', FinCErr=f'{cl["final"]["<-2.0"]["err"]:.4f}',
         GrpA=f'{cl["group"]["<-1.0"]["diff"]:.4f}', GrpB=f'{cl["group"]["<-1.5"]["diff"]:.4f}', GrpC=f'{cl["group"]["<-2.0"]["diff"]:.4f}',
         HeldShort=f'{-bs["H_held_out"]["final"][1]:.4f}', HeldNoPass=f'{-bs["H_held_out"]["group"][1]:.4f}',
         ParentFrac=f'{sg["parent_xp_fraction_covered"]["<-1.5"]:.3f}',
         HeldRatio=pct(-bs['H_held_out']['final'][1] / sg['parent_xp_fraction_covered']['<-1.5']), NsideHeld=num(grab('scripts/boss_selection.py', r'pix8, pix16 = hp\.ang2pix\((\d+),')))
# Mock of the BOSS selection (results/boss_mock.json; scripts/boss_mock.py): cumulative recovery and coverage fill per ring.
mk = js('boss_mock')
bm = const('boss_mock')
c0 = mk['obs_bias_1.0']['final']['cumulative']
ring = mk['fill']['central']['standard/ring']
M.update(MockA=f2(c0['mean'][0]), MockAErr=f2(c0['sd'][0]), MockB=f2(c0['mean'][1]), MockBErr=f2(c0['sd'][1]),
         FillA=f2(ring['ring0']['mean'][0]), FillAErr=f2(ring['ring0']['sd'][0]), FillB=f2(ring['ring1']['mean'][0]),
         FillBErr=f2(ring['ring1']['sd'][0]), FillC=f2(ring['ring2']['mean'][0]), FillCErr=f2(ring['ring2']['sd'][0]))
xg = {round(r['below'], 3): r for r in sg['xp_gaps']}  # XP gap rows keyed by threshold [dex]


def rel(x):
    """Relative difference of the uncovered and covered parent fractions below XP [M/H] = x."""
    return abs(1 - xg[x]['uncovered'] / xg[x]['covered_same_mix'])


# Gap differences, sensitivity to the coverage threshold (NMIN_GOOD = 2, 5), and the S/N cut check.
ct = sg['coverage_threshold']
sc = js('snr_cut_check')
M.update(GapA=sigfig(100 * rel(-1.0)), GapB=sigfig(100 * rel(-1.5)), CovTwo=f'{100 * (ct["nmin=2"]["all"] - 1):.1f}',
         CovFive=f'{100 * (ct["nmin=5"]["all"] - 1):.1f}', SnrDiff=f'{sc["diff"]:.3f}', SnrDiffHi=f'{sc["diff_p84"] - sc["diff"]:.3f}',
         SnrDiffLo=f'{sc["diff"] - sc["diff_p16"]:.3f}', SnrN=num(sc['n_rejected']))

# %% [markdown]
# ### Chapter 2: the APOGEE selection function
# Candidates, observed stars, ASPCAP failures and the field-bootstrap error of the APOGEE weights.

# %%
# APOGEE (results/apogee_selection.json; data/apogee_parent.parquet, apogee_weights.parquet; scripts/apogee_selection.py)
asel = js('apogee_selection')
ap = pd.read_parquet(DATA / 'apogee_parent.parquet').sort_values(['apogee_id', 'observed'], ascending=[True, False]).drop_duplicates('apogee_id')
late = ap[(ap.field.str[3:].astype(int) >= 18) & ~ap.nbr]  # LMC18-35 candidates without a bright neighbour
aw = pd.read_parquet(DATA / 'apogee_weights.parquet')
# ASPCAP failures: bright (BrtRGB) observed candidates with S/N above the selection threshold, so that a failure is STAR_BAD or no
# [Fe/H] (lmcold.giants.apogee_spectrum_ok), not low S/N
snr_sel = const('apogee_selection')['SNR_MIN']
bright = aw[aw.observed & (aw.cls == 1)].merge(ap[['apogee_id', 'snr', 'extratarg', 'aspcapflag', 'feh']], on='apogee_id', suffixes=('_w', ''))
bright = bright[bright.snr >= snr_sel]
bright['fail'] = ~giants.apogee_spectrum_ok(bright, snr_sel)
ASP_XP = (-1.5, -1.0)   # XP [M/H] [dex] below and above which the ASPCAP failure rates of bright observed stars are quoted
# Coverage rule of lmcold.apogee_target, read from its source (minimum counts and observed fractions).
cov_rule = re.search(r'n_q >= (\d+), \(n_q >= (\d+)\) & \(f_obs >= ([\d.]+)\)\) \| \(\(f_q >= ([\d.]+)\)',
                     (ROOT / 'src/lmcold/apogee_target.py').read_text())
asum = asel['summary']
# Candidate and observed counts, astrometric failures, targeting zones, coverage rule, ASPCAP failure rates and the tests T5-T7.
M.update(NPoly=count(len(ap)), NNbr=count(ap.nbr.sum()), InPoly=pct(asel['T1']['wd_in_parent'], 2), InPolyB=pct(asel['T1']['lmc18_35_in_parent']),
         NCand=count(asum['candidates']), AstroFail=pct(1 - late.astro_ok.mean()), AstroGiant=pct(al['LMC18-35']['fail'], 1),
         ZoneR=num(grab('scripts/apogee_selection.py', r'\(c\.rfield >= ([\d.]+)\)')),
         HStep=num(grab('scripts/apogee_selection.py', r'\(c\.h - 12\.8\) / ([\d.]+)\)')), NColour=num(len(const('apogee_selection')['XEDGES']) + 1),
         NApoObs=count(asum['observed']), NApoGood=count(asum['quality']), CovBright=cov_rule.group(1), CovFaint=cov_rule.group(2),
         CovFobs=cov_rule.group(3), CovHalf=cov_rule.group(4), NApoCov=count(asum['covered']),
         PgHEdges=edges(const('apogee_selection')['PG_H_EDGES']), AspXpLo=num(ASP_XP[0]), AspXpHi=num(ASP_XP[1]),
         AspMP=pct(bright.fail[bright.mh_xp < ASP_XP[0]].mean()), AspMR=pct(bright.fail[bright.mh_xp > ASP_XP[1]].mean(), 1),
         VelA=pct(sg['apogee_nonmember']['LMC1-17']['weighted'], 1), VelB=pct(sg['apogee_nonmember']['LMC18-35']['weighted'], 1),
         Jk=f'{max(abs(v) for k_, v in asel["T5_dJK"].items() if k_ != "pass"):.4f}',
         Fdiff=f'{np.ptp([v["f_below_m15"] for k_, v in asel["T6_snr_neutrality"].items() if k_ != "pass"]):.4f}',
         ApoLateFin=f2(asel['T7_xp_closure']['LMC18-35 final']['<-1.5']), ApoLateNo=f2(asel['T7_xp_closure']['LMC18-35 no_pass']['<-1.5']),
         ApoEarlyFin=f2(asel['T7_xp_closure']['LMC1-17 final']['<-1.5']), ApoEarlyNo=f2(asel['T7_xp_closure']['LMC1-17 no_pass']['<-1.5']))
# Field-bootstrap error of the LMC1-17 ratio before the pass factor (T7 of scripts/apogee_selection.py): weights without the pass
# factor, 1 / (quality fraction of the group), for quality stars in covered groups; fields drawn with replacement, weights fixed.
NBOOT_T7 = 1000
x = aw[aw.mh_xp.notna() & (aw.field.str[3:].astype(int) <= 17)]  # LMC1-17 candidates with XP [M/H]
w0 = np.where(x.covered & x.quality, 1 / x.groupby('group')['quality'].transform('mean'), 0.)
low = x.mh_xp < -1.5
# Per field: weighted and covered counts, all and with XP [M/H] < -1.5; the ratio is (weighted fraction) / (covered fraction).
fs = pd.DataFrame(dict(field=x.field, a=w0 * low, b=w0, c=x.covered & low, d=x.covered)).groupby('field').sum()
rng = np.random.default_rng(1)  # fixed seed
t7 = [(s.a.sum() / s.b.sum()) / (s.c.sum() / s.d.sum()) for s in (fs.iloc[rng.integers(0, len(fs), len(fs))] for _ in range(NBOOT_T7))]
M.update(ApoEarlyNoErr=f2(np.std(t7)))
# Members of the two mass samples.
gc = js('giant_counts')
M.update(NMassBoss=count(gc['boss_members']), NMassApo=count(gc['apogee_members']))

# %% [markdown]
# ### Chapter 2: appendix checks
# The mock test of the BOSS selection function, the coverage gaps and the non-members among the weighted good BOSS spectra.

# %%
# Appendix checks: mock (results/boss_mock.json; scripts/boss_mock.py) and gaps (results/selection_gaps.json)
grp = mk['obs_bias_1.0']['group']['binned']['mean']  # recovery per [Fe/H] bin with group weights only
fe = mk['feh_edges']  # [Fe/H] bin edges of the mock
b13 = mk['obs_bias_1.3']['final']['cumulative']['mean'][0]
xp_ = mk['fill']['central']['standard/xp']
# Inner-ring fill for the sparse-gap variants, and the total recovery per ring.
spg = [mk['fill']['central'][k]['ring0']['mean'][0] for k in ('sparse_gap/ring', 'sparse_gap/xp')]
tot = [ring[f'ring{r}']['mean'][2] for r in range(3)]
M.update(GaiaBrightRho=re.search(r'\(2, (\d+)\), \(4,', (ROOT / 'scripts/paper_selection_checks.py').read_text()).group(1), NReal=num(bm['NREAL']), NFill=num(bm['NFILL']),
         GroupLoss=pct(1 - min(m for m, hi in zip(grp, fe[1:]) if hi <= -2)), BiasFac=[k.split('_')[-1] for k in mk if k.startswith('obs_bias_')][-1], BiasHigh=pct(b13 - 1),
         XpFillA=f2(xp_['ring0']['mean'][0]), XpFillAErr=f2(xp_['ring0']['sd'][0]), XpFillB=f2(xp_['ring1']['mean'][0]),
         XpFillBErr=f2(xp_['ring1']['sd'][0]), XpFillC=f2(xp_['ring2']['mean'][0]), XpFillCErr=f2(xp_['ring2']['sd'][0]),
         SparseLo=f2(min(spg)), SparseHi=f2(max(spg)), TotA=f2(tot[0]), TotB=f2(tot[1]), TotC=f2(tot[2]),
         PixCoarse=num(round(np.sqrt(hp.nside2pixarea(sel['LEVELS'][-1][0], degrees=True)))))
# XP gap fractions (uncovered, covered same mix, bootstrap half-width) below -1, -1.5 and -1.75.
for name, x, d in (('One', -1.0, 3), ('Two', -1.5, 4), ('Three', -1.75, 4)):
    r = xg[x]
    M.update({f'GapU{name}': f'{r["uncovered"]:.{d}f}', f'GapC{name}': f'{r["covered_same_mix"]:.{d}f}',
              f'GapE{name}': f'{(r["uncovered_p84"] - r["uncovered_p16"]) / 2:.{d}f}'})
M.update(GapXpThree=num(-1.75), Deficit=pct(rel(-1.75) * sg['uncovered_share'], 1))
# Non-members of the weighted good BOSS spectra per ring x G bin, as panel (c) of figures/paper_selection_gaps.png
v = pd.read_parquet(DATA / 'spec_giants.parquet')
vb = v[v.survey == 'boss'][['source_id', 'v', 'teff', 'logg']].drop_duplicates('source_id')
gg = bw[bw.good & (bw.w > 0)].merge(vb, on='source_id', how='left')
# Ring (RING_EDGES) and G bin per star; non-member = fails the giant cuts or the mass-sample velocity window.
gg['ring'] = np.digitize(angular_distance(gg.ra.to_numpy(), gg.dec.to_numpy(), LMC['ra0'], LMC['dec0']), selection.RING_EDGES)
gg['gb'] = np.digitize(gg.g, const('paper_selection_gaps')['G_EDGES'])
gg['nm'] = ~((gg.teff < default(giants.is_giant, 'teff_max')) & (gg.logg < default(giants.is_giant, 'logg_max'))
             & (gg.v > float(M['VMassLo'])) & (gg.v < float(M['VMassHi'])))
per = gg.groupby(['ring', 'gb']).apply(lambda s: np.average(s.nm, weights=s.w), include_groups=False)
M.update(NonMemCellLo=pct(per.min(), 1), NonMemCellHi=pct(per.max(), 1),
         ApoNonMemMax=pct(max(x['weighted'] for x in sg['apogee_nonmember'].values()), 1))

# %% [markdown]
# ### Chapter 3: RR Lyrae membership and metallicity; BOSS calibration
# The numbers of the calibration chapter.

# %%
# ------------------------------------------------------------------------------------------------------------------------------
# Chapter 3: RR Lyrae membership and metallicity; BOSS calibration
# ------------------------------------------------------------------------------------------------------------------------------
# BOSS S/N threshold 16 (adopted) against 20: change of formed masses [per cent], scale lengths h and the [Fe/H] profile.
st = js('snr_threshold_sensitivity')
a16, a20 = st['16'], st['20']  # results with S/N >= 16 and S/N >= 20
prof = np.abs(np.array(st['median_feh_by_R_20']) - np.array(st['median_feh_by_R_16']))[1:]   # profile bins beyond the first kpc
# Alternative threshold (second numeric key), relative changes of total, [Fe/H] < -1.5 and < -2 formed mass, largest change
# of h, largest change of the median [Fe/H] profile beyond the first kpc.
M.update(SnrAlt=[k for k in st if k.isdigit()][1], DTot=f'{100 * (a20["formed_total"] / a16["formed_total"] - 1):.1f}',
         DMP=num(round(100 * (a20['formed_below_m15'] / a16['formed_below_m15'] - 1))),
         DVMP=signed(round(100 * (a20['formed_below_m2'] / a16['formed_below_m2'] - 1)), num),
         DH=f2(max(abs(a20['h'][k] - a16['h'][k]) for k in a16['h'])), DProf=f'{prof.max():.3f}', ProfR=num(1))
# BOSS [Fe/H] bias of hot giants (results/boss_hot_bias.json; scripts/paper_boss_hot_bias.py).
hb = js('boss_hot_bias')
te = hb['t_edges']  # Teff bin edges [K]
T_HOT = grab('scripts/paper_boss_hot_bias.py', r'hot = \(b\.teff > (\d+)\)')   # K: 'hot' giants of the figure script
cool = [j for j in range(len(te) - 1) if te[j + 1] <= T_HOT]  # Teff bins entirely below T_HOT
warm = te.index(T_HOT)  # index of the first bin above T_HOT; the next one (warm + 1) is the 'hot' bin


def offs(j):
    """BOSS - APOGEE [Fe/H] offsets of Teff bin j at low and high S/N."""
    return [hb['offset_low'][j], hb['offset_high'][j]]  # [dex]


red = max(j for j, f in enumerate(hb['frac_boss_low']) if np.isfinite(f))   # reddest dereddened BP-RP bin of panel (b) with stars
# S/N split, Teff edges, offset ranges per Teff group, fractions of BOSS stars in the bluest and reddest colour bins at low and
# high S/N, APOGEE range, and the cuts of the figure script read from its source (G range, rho, minimum stars, metal-rich bin).
M.update(SnrSplit=num(hb['snr_split']), TCool=num(T_HOT), TWarm=num(te[warm + 1]), THot=num(te[warm + 2]),
         CoolMax=f2(max(abs(x) for j in cool for x in offs(j))), WarmLo=f2(min(offs(warm))), WarmHi=f2(max(offs(warm))),
         HotLo=f2(min(offs(warm + 1))), HotHi=f2(max(offs(warm + 1))),
         BlueLow=pct(hb['frac_boss_low'][0]), BlueHigh=pct(hb['frac_boss_high'][0]), BlueEdge=num(hb['c_edges'][1]),
         RedLow=pct(hb['frac_boss_low'][red]), RedHigh=pct(hb['frac_boss_high'][red]), RedLo=num(hb['c_edges'][red]),
         RedHi=num(hb['c_edges'][red + 1]),
         ApoRichLo=pct(np.nanmin(hb['frac_apogee'])), ApoRichHi=pct(np.nanmax(hb['frac_apogee'])),
         HotGLo=num(grab('scripts/paper_boss_hot_bias.py', r't\.g\.between\(([\d.]+), ([\d.]+)\)')),
         HotGHi=num(grab('scripts/paper_boss_hot_bias.py', r't\.g\.between\(([\d.]+), ([\d.]+)\)', 2)),
         HotRho=num(grab('scripts/paper_boss_hot_bias.py', r't\.rho >= (\d+)\)')),
         HotNMin=num(grab('scripts/paper_boss_hot_bias.py', r'T_EDGES, mid, (\d+)\)')),
         TBin=num(np.diff(te)[0]),
         RichLo=num(grab('scripts/paper_boss_hot_bias.py', r'rich = o\[\(o\.feh_a >= ([-.\d]+)\) & \(o\.feh_a < ([-.\d]+)\)\]')),
         RichHi=num(grab('scripts/paper_boss_hot_bias.py', r'rich = o\[\(o\.feh_a >= ([-.\d]+)\) & \(o\.feh_a < ([-.\d]+)\)\]', 2)),
         SplitFeh=num(TRUE_EDGES[-2]), CalNMin=num(grab('scripts/paper_boss_calibration.py', r'def running\(ax, x, y, edges, nmin=(\d+)\)')))

# %% [markdown]
# ### Settings quoted in captions and tests
# Values read from the figure scripts and lmcold that the captions and tests quote.

# %%
# Settings quoted in the captions and tests (figure scripts, lmcold.maps, scripts/boss_selection.py, boss_mock.py, apogee_selection.py)
# The three (inner, outer) rings [deg] of scripts/paper_selection_checks.py, read from its source.
rings = re.search(r'zip\(\(\((\d+), (\d+)\), \((\d+), (\d+)\), \((\d+), (\d+)\)\)', (ROOT / 'scripts/paper_selection_checks.py').read_text())
# Map pixel size and minimum stars; closure thresholds of boss_selection.py and boss_mock.py; APOGEE S/N thresholds of T6;
# bin size, minimum stars and HEALPix nside of paper_selection.py; magnitude step, rings, minimum stars and colour limit of
# paper_selection_checks.py.
M.update(MapPix=num(grab('scripts/paper_maps.py', r'np\.arange\(-L, L \+ 0\.01, ([\d.]+)\)')), MapNMin=num(default(pixel_map, 'nmin')),
         CutA=num(sel['CUTS'][0]), CutB=num(sel['CUTS'][1]), CutC=num(sel['CUTS'][2]),
         MockCutA=num(bm['CUTS'][0]), MockCutB=num(bm['CUTS'][1]),
         SnrT=', '.join(k.split('>=')[1] for k in asel['T6_snr_neutrality'] if k != 'pass'),
         SelBin=num(grab('scripts/paper_selection.py', r'np\.arange\(0, 16\.1, ([\d.]+)\)\)')),
         SelNBin=num(grab('scripts/paper_selection.py', r'if n\[i\] >= (\d+) else')), SelNMin=num(const('paper_selection')['NMIN']),
         SelNside=num(grab('scripts/paper_selection.py', r'hp\.ang2pix\((\d+),')),
         ChkMag=num(grab('scripts/paper_selection_checks.py', r'np\.arange\(17\.75, 20\.6, ([\d.]+)\)')),
         ChkRA=rings.group(2), ChkRB=rings.group(4), ChkRC=rings.group(6), ChkNB=num(grab('scripts/paper_selection_checks.py', r'if nB >= (\d+):')),
         ChkVmin=num(grab('scripts/paper_selection_checks.py', r'vmin=([\d.]+), vmax')))

# %% [markdown]
# ### Write the macros
# One \newcommand per entry of M, with the prefix \da, into results/macros_data.tex; the page text shows them through
# {glue:text}, so no number is typed by hand.

# %%
# Header line, then one \newcommand per macro; written to results/macros_data.tex and printed.
lines = ['% Numbers of the KB data chapters 1-3 (scripts/paper_text_data.py); do not edit.']
lines += [rf'\newcommand{{\da{k}}}{{{x}}}' for k, x in M.items()]
(R / 'macros_data.tex').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
