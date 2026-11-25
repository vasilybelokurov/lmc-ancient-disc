"""Knowledge-base text, page 'Rotational support, centre and sensitivity' of the kinematics chapter (extended draft Sections 4.4-4.8
and the further checks of its kinematics appendix): every number of the text as a LaTeX macro (prefix \\ks), so that none is typed by
hand.

As scripts/paper_text_shape.py: settings are read from the code (scripts/textnum.const, lmcold by import), results from the tables
the figures are drawn from; the stage-1 posterior draws (data/kin_mcmc_data_ellipsoid*_samples.npz, read only) give the quantities
the tables do not hold (v_rot(4 kpc) of the outer run, the RR Lyrae correlations, both stages of the joint fit); the release
samples give the Gaia proper-motion errors in km/s and, on the stars in both surveys, the [Fe/H] leakage and the BOSS velocity-error
test. No fit is made.
Inputs: results/{kin_support_ellipsoid, kin_support, kin_geometry_feh, kin_mcmc_data_ellipsoid, kin_mcmc_data_ellipsoid_outer,
kin_mcmc_data_stream, kin_mcmc_data_centre, density_centre, kin_feh_contamination, kin_feh_forward,
kin_tilted_ring, kin_los_scale, kin_turning, kin_ellipsoid_checks, kin_dispersion_cut, kin_data_split, kin_mock_physical, kin_footprint,
kin_residual_profiles_ellipsoid}.csv, data/giants_{apogee,boss}.parquet and data/rrab.parquet (lmcold.samples),
results/boss_calibration.json; scripts/{fit_kinematics_mcmc, paper_kin_support_ellipsoid, paper_kin_feh_mixing,
paper_kin_dispersion_cut, paper_kin_centre, fig_boss_velocity_errors}.py (settings); lmcold.constants (LMC distance), lmcold.kinfit (R_SIG, RHO_TILT, centre).
Writes results/macros_kinsupport.tex.
Run: .venv/bin/python scripts/manifest.py run paper_text_kin_support.py"""

# %% [markdown]
# ### Settings and helpers
# Imports, the dictionary M of macros (macro name without its prefix: the text it stands for) and the functions that format
# numbers as the text writes them. The inputs listed in the header are read in the step that uses them.

# %%
import ast
import json

import numpy as np
import pandas as pd

from fit_kinematics_mcmc import ring
from lmcold import kinfit, samples
from lmcold.constants import KAPPA, LMC
from lmcold.footprint import lmc_xy
from lmcold.samples import FEH_BINS
from lmcold.sources import DATA, ROOT, path
from textnum import const, num, sigfig

R = ROOT / 'results'   # folder of the result tables
M = {}   # macro name (without the \ks prefix): text it stands for
BINS = 'ABCDE'   # giant [Fe/H] bins, metal-poor first (macro suffixes)
SIGZ_BAND = (20, 35)   # km/s: sigma_z range of the giants, for the RR Lyrae sigma_phi quoted where their sigma_z is giant-like


def f1(x):
    """x with one decimal."""
    return f'{x:.1f}'


def f2(x):
    """x with two decimals."""
    return f'{x:.2f}'


def pm(med, lo, hi, fmt=f1):
    """'med ± e' if the two half-widths agree when rounded, else 'med (+u/−l)'."""
    # Upper and lower half-widths, formatted as the value is.
    u, d = fmt(hi - med), fmt(med - lo)
    return f'{fmt(med)} ± {u}' if u == d else f'{fmt(med)} (+{u}/−{d})'


def rng(x, fmt=f1):
    """(min, max) of x as formatted strings."""
    return fmt(min(x)), fmt(max(x))


def put(name, pair):
    """Store a (low, high) pair as the macros <name>Lo and <name>Hi."""
    M[name + 'Lo'], M[name + 'Hi'] = pair


def pct_per_kpc(gamma):
    """Per cent decline of the dispersions per kpc for the logarithmic gradient gamma [1/kpc]."""
    return 100 * (1 - np.exp(gamma))


# %% [markdown]
# ### Settings
# The radial ranges, [Fe/H] bin edges and other settings, read from the fit and figure scripts.

# %%
# Settings
# Literal constants of the fit and figure scripts, read from their source (textnum.const) without running them.
fk = const('fit_kinematics_mcmc')
mix, cut, cen = const('paper_kin_feh_mixing'), const('paper_kin_dispersion_cut'), const('paper_kin_centre')
# Radial ranges [deg], R_SIG [kpc], the hot-giant Teff limit [K], the HEALPix cell size [deg] (sqrt of 4 pi / (12 NSIDE^2) sr),
# and the inner edges and mock values of scripts/paper_kin_dispersion_cut.py.
M.update(RhoIn=num(fk['RING'][0]), RhoOut=num(fk['RING'][1]), RhoOuter=num(fk['OUTER_INNER']), RSig=num(kinfit.R_SIG),
         THot=num(mix['T_HOT']), NsideDeg=f1(np.rad2deg(np.sqrt(4 * np.pi / (12 * mix['NSIDE'] ** 2)))),
         MinApogee=num(mix['MIN_APOGEE']), RhoTilt=num(kinfit.RHO_TILT), InnerLo=num(min(cut['INNER'])),
         InnerHi=num(max(cut['INNER'])), MockSigma=num(cut['MOCK_SIGMA']), MockKIn=f2(cut['MOCK_K'][0]),
         MockKOut=f2(cut['MOCK_K'][1]), SigzBandLo=num(SIGZ_BAND[0]), SigzBandHi=num(SIGZ_BAND[1]))
M.update({f'Feh{BINS[i]}': num(FEH_BINS[i][1]) for i in range(4)})   # bin edges -1.5, -1.1, -0.8, -0.5
M.update(BinWidth=f1(max(hi - lo for lo, hi in FEH_BINS[2:4])), InnerTwo=num(cut['INNER'][1]))   # closed bins above -1.1
M['ApogeeFrom'] = num(const('paper_kin_support_ellipsoid')['APOGEE_FROM'])

# %% [markdown]
# ### Rotational support
# v_rot/sigma of each group from the stage-1 fits at 2-9 deg and 4-9 deg, the joint fit, and the RR Lyrae.

# %%
# Rotational support (results/kin_support_ellipsoid.csv: stage-1 posteriors of the velocity-ellipsoid model)
# One row per group: g the five giant bins (metal-poor first), rr the RR Lyrae; v_rot(4 kpc) and dispersions [km/s], ratios.
S = pd.read_csv(R / 'kin_support_ellipsoid.csv').set_index('population')
g, rr = S.iloc[:5], S.loc['RR Lyrae']
# Values of selected bins (indices into BINS) quoted in the text: column q, macro stem n, format fmt.
for q, n, fmt, bins in (('vrot4', 'Vrot', f1, (0, 4)), ('sigma_R', 'SigR', f1, (0, 2, 3, 4)), ('sigma_phi', 'SigPhi', f1, (0, 4)),
                        ('sigma_z', 'Sigz', f1, (0, 2, 3, 4)), ('ratio_R', 'RatR', f2, (0, 4)), ('ratio_z', 'Ratz', f2, (0, 4))):
    M.update({f'{n}{BINS[i]}': fmt(g[q].iloc[i]) for i in bins})
SIG = ['sigma_R', 'sigma_phi', 'sigma_z']
# Differences between bins [km/s]: A-B (rotation and largest dispersion difference), C-E and A-E (rounded to integers).
M.update(DVrotAB=f1(g.vrot4.iloc[1] - g.vrot4.iloc[0]), DSigAB=f1((g[SIG].iloc[0] - g[SIG].iloc[1]).abs().max()),
         DVrotCE=num(round(g.vrot4.iloc[4] - g.vrot4.iloc[2])), DSigRCE=num(round(g.sigma_R.iloc[2] - g.sigma_R.iloc[4])),
         DSigzCE=num(round(g.sigma_z.iloc[2] - g.sigma_z.iloc[4])),
         DVrotAE=num(round(g.vrot4.iloc[4] - g.vrot4.iloc[0])), DSigRAE=num(round(g.sigma_R.iloc[0] - g.sigma_R.iloc[4])),
         DSigzAE=num(round(g.sigma_z.iloc[0] - g.sigma_z.iloc[4])))
# Range over the bins of how much larger v_rot/sigma_z is than v_rot/sigma_R [per cent].
put('SuppEx', rng(100 * (g.ratio_z / g.ratio_R - 1), lambda x: num(round(x))))
# Geometry and support ratios per giant bin: g1 from stage 1 (own fits), g2 from stage 2 (joint fit), with p16 and p84.
geo = pd.read_csv(R / 'kin_geometry_feh.csv')
g1, g2 = (geo[(geo.stage == s) & (geo.population != 'RR Lyrae')].reset_index(drop=True) for s in (1, 2))
put('RatPhi', rng(g1.ratio_phi, f2))
put('RatZ', rng(g1.ratio_z, f2))
half = lambda t, q: (t[f'{q}_p84'] - t[f'{q}_p16']) / 2  # noqa: E731  half-width of the 16th-84th percentile range
# Bins D and E: v_rot/sigma_z ± half the 16th-84th range; bin E from the APOGEE giants alone: sigma_z / sigma_R.
M.update(RatZD=f'{f2(g1.ratio_z[3])} ± {f2(half(g1, "ratio_z")[3])}', RatZE=f'{f2(g1.ratio_z[4])} ± {f2(half(g1, "ratio_z")[4])}',
         RatZApoE=f2(S.sigma_z_apogee.iloc[4] / S.sigma_R_apogee.iloc[4]))
# Posterior summaries of the ellipsoid model at 2-9 deg (E) and 4-9 deg (O); e1, o1: stage-1 medians, one row per group.
E = pd.read_csv(R / 'kin_mcmc_data_ellipsoid.csv')
O = pd.read_csv(R / 'kin_mcmc_data_ellipsoid_outer.csv')
e1 = E[E.stage == 1].pivot_table(index='population', columns='param', values='median').loc[g.index]
o1 = O[O.stage == 1].pivot_table(index='population', columns='param', values='median')
# Decline of the dispersions [per cent per kpc]: the two metal-poor bins, then the three metal-rich ones.
put('GamPctMP', rng(pct_per_kpc(e1.dlnsig_dR.iloc[:2])))
put('GamPctMR', rng(pct_per_kpc(e1.dlnsig_dR.iloc[2:])))
# Outer slopes vslope [1/kpc] of the giant bins (2-9 deg) and the RR Lyrae (4-9 deg); Lo is the largest, Hi the most negative.
slopes = list(e1.vslope) + [o1.vslope['RR Lyrae']]
M.update(SlopeLo=f2(max(slopes)), SlopeHi=f2(min(slopes)))
Rg = np.linspace(0.5, 9, 1701)   # kpc: peak of the stage-1 median rotation curve v0 tanh(R/R0) [1 + vslope (R - R_SIG)]
peaks = [Rg[np.argmax(r.v0 * np.tanh(Rg / r.R0) * (1 + r.vslope * (Rg - kinfit.R_SIG)))] for _, r in e1.iterrows()]
put('Peak', rng(peaks))
# Same fits at 4 < rho < 9 deg (run 'ellipsoid_outer'), from the paired posterior draws


def draws(run, i, stage=1):
    """Posterior draws (flattened) of group i of run `run` (stage 1: its own fit; stage 2: the joint fit): v0, R0 [kpc], the three
    dispersions [km/s], k, and v_rot(4 kpc) = v0 tanh(R_SIG / R0) [km/s]."""
    z = np.load(DATA / f'kin_mcmc_data_{run}_samples.npz')
    # Keys of the npz: '1|<group>|<param>_0' in stage 1, '2|all|<param>_<group>' in stage 2.
    k = (lambda n: f'1|{i}|{n}_0') if stage == 1 else (lambda n: f'2|all|{n}_{i}')
    d = {n: z[k(n)].reshape(-1) for n in ('v0', 'R0', 'sigma_R', 'sigma_phi', 'sigma_z', 'k')}
    d['vrot4'] = d['v0'] * np.tanh(kinfit.R_SIG / d['R0'])
    return d


# The five giant bins at 4-9 deg: median v_rot(4 kpc) [km/s] and the changes from 2-9 deg of rotation, dispersions and both ratios.
out = [draws('ellipsoid_outer', i) for i in range(5)]
vout = np.array([np.median(d['vrot4']) for d in out])
dvo = vout - g.vrot4.to_numpy()
dsig = {s: o1.loc[g.index, s].to_numpy() - g[s].to_numpy() for s in SIG}
drat = [max(abs(np.median(d['vrot4'] / d['sigma_R']) - g.ratio_R.iloc[i]), abs(np.median(d['vrot4'] / d['sigma_z']) - g.ratio_z.iloc[i]))
        for i, d in enumerate(out)]
put('DVrotOut', rng(dvo))
put('DSigPhiOut', rng(-dsig['sigma_phi']))
# Largest change of sigma_R or sigma_z [km/s], of the ratios, and v_rot(4 kpc) at 4-9 deg of bins A and B.
M.update(DSigRzOut=f1(max(np.abs(dsig['sigma_R']).max(), np.abs(dsig['sigma_z']).max())), DRatOut=f1(max(drat)),
         VrotOutA=f1(vout[0]), VrotOutB=f1(vout[1]))
# Stage 1 against stage 2 (joint fit) of the giants: v_rot(4 kpc) and dispersions
st2 = [draws('ellipsoid', i, 2) for i in range(5)]
# Largest difference [km/s] between the joint-fit median and the stage-1 value of v_rot(4 kpc) or a dispersion.
M['DStage'] = f1(max(max(abs(np.median(st2[i][q]) - g[q].iloc[i]) for q in ['vrot4'] + SIG) for i in range(5)))
# RR Lyrae
# RR Lyrae stage-1 summaries at 2-9 deg (rin) and 4-9 deg (rou), indexed by parameter.
rin = E[(E.stage == 1) & (E.population == 'RR Lyrae')].set_index('param')
rou = O[(O.stage == 1) & (O.population == 'RR Lyrae')].set_index('param')
# RR Lyrae: Gaia error scale k and dispersion gradient [1/kpc] in both runs; v_rot(4 kpc), dispersions and ratio with their errors;
# [Fe/H] median and spread [dex]; DVrotRrlOut: largest difference from bins A and B at 4-9 deg [km/s], rounded up.
M.update(KRrlIn=f2(rin.loc['k', 'median']), GamRrlIn=f2(rin.loc['dlnsig_dR', 'median']),
         GamPctRrlIn=num(round(pct_per_kpc(rin.loc['dlnsig_dR', 'median']))), KRrlOut=f2(rou.loc['k', 'median']),
         GamRrlOut=f2(rou.loc['dlnsig_dR', 'median']),
         VrotRrl=pm(rr.vrot4, rr.vrot4_p16, rr.vrot4_p84), SigRRrl=pm(rr.sigma_R, rr.sigma_R_p16, rr.sigma_R_p84),
         SigPhiRrl=pm(rr.sigma_phi, rr.sigma_phi_p16, rr.sigma_phi_p84), RatRRrl=pm(rr.ratio_R, rr.ratio_R_p16, rr.ratio_R_p84, f2),
         FehRrl=f2(rr.feh_med), FehRrlLo=f2(rr.feh_p16), FehRrlHi=f2(rr.feh_p84),
         DVrotRrlOut=num(np.ceil(np.abs(rr.vrot4 - vout[:2]).max())))
# Ranges over the two metal-poor bins.
put('SigRMP', rng(g.sigma_R.iloc[:2]))
put('RatRMP', rng(g.ratio_R.iloc[:2], f2))
put('SigPhiMP', rng(g.sigma_phi.iloc[:2]))
# RR Lyrae draws at 4-9 deg (group 5): correlation of sigma_phi with k, and median sigma_phi of the draws with sigma_z in SIGZ_BAND.
d = draws('ellipsoid_outer', 5)
band = (d['sigma_z'] > SIGZ_BAND[0]) & (d['sigma_z'] < SIGZ_BAND[1])
M.update(CorrRrl=sigfig(np.corrcoef(d['sigma_phi'], d['k'])[0, 1]), SigPhiRrlBand=num(round(np.median(d['sigma_phi'][band]))))

# %% [markdown]
# ### Rotation-curve parameters
# Shape parameters of the rotation curves quoted in the table caption.

# %%
# Table caption: shape parameters of the rotation curves
# Turnover radius R0 [kpc] range of the giants, outer slope [1/kpc]; RR Lyrae R0 and slope from the 4-9 deg run.
put('RNullG', rng(e1.R0))
M.update(SlopeGLo=f2(e1.vslope.max()), SlopeGHi=f2(e1.vslope.min()),
         RNullRrl=pm(rou.loc['R0', 'median'], rou.loc['R0', 'p16'], rou.loc['R0', 'p84']),
         SlopeRrl=pm(rou.loc['vslope', 'median'], rou.loc['vslope', 'p16'], rou.loc['vslope', 'p84'], f2))

# %% [markdown]
# ### Metallicity errors and the two surveys
# Leakage of stars between [Fe/H] bins and the BOSS velocity errors, on the stars in both surveys.

# %%
# Metallicity errors and the two surveys (results/kin_feh_*.csv; boss_calibration.json)
# cal: BOSS [Fe/H] calibration (held-out scatter); ct: sigma_los per bin and subset; fw: forward-modelled sigma_los; labels: bin
# names.
cal = json.loads((R / 'boss_calibration.json').read_text())
ct = pd.read_csv(R / 'kin_feh_contamination.csv')
fw = pd.read_csv(R / 'kin_feh_forward.csv')
labels = list(ct.bin.unique())
# Stars in both release catalogues (as scripts/paper_kin_feh_mixing.py and fig_boss_velocity_errors.py): of the cool BOSS giants
# measured above FEH_BINS[4][0], the fraction whose APOGEE [Fe/H] lies in the neighbouring bin FEH_BINS[3]
m = pd.read_parquet(path('release_giants_apogee')).merge(pd.read_parquet(path('release_giants_boss')), on='source_id', suffixes=('_a', '_b'))
# _a: APOGEE columns, _b: BOSS columns of the stars in both catalogues; top: cool BOSS giants measured in the most metal-rich bin.
top = m[(m.teff_b < mix['T_HOT']) & (m.feh_b >= FEH_BINS[4][0])]
M.update(BossScatter=sigfig(cal['held_out_test']['dfeh_rsd']),
         LeakE=num(round(100 * np.mean((top.feh_a >= FEH_BINS[3][0]) & (top.feh_a < FEH_BINS[3][1])))),
         NApoA=num(ct[(ct.bin == labels[0]) & (ct.subset == 'APOGEE')].n.iloc[0]))
# Bins C, D, E: APOGEE-alone dispersions [km/s] and ratios, and the predicted and measured sigma_los [km/s] of the cool BOSS giants.
for i in (2, 3, 4):
    b = BINS[i]
    M.update({f'SigzApo{b}': f1(S.sigma_z_apogee.iloc[i]), f'SigRApo{b}': f1(S.sigma_R_apogee.iloc[i]),
              f'RatRApo{b}': f2(S.ratio_R_apogee.iloc[i]), f'RatzApo{b}': f2(S.ratio_z_apogee.iloc[i]),
              f'Pred{b}': f1(fw[(fw.bin == labels[i]) & (fw.cls == 'cool')].sigma_los_pred.iloc[0]),
              f'Meas{b}': f1(ct[(ct.bin == labels[i]) & (ct.subset == 'BOSS cool')].sigma_los.iloc[0])})
# Measured minus predicted in bin C [km/s], rounded; bin E: sigma_los of the hot and of the cool BOSS giants.
M['ShortC'] = num(round(float(M['MeasC']) - float(M['PredC'])))
M.update(HotE=f1(ct[(ct.bin == labels[4]) & (ct.subset == 'BOSS hot')].sigma_los.iloc[0]), CoolE=M['MeasE'])
# BOSS velocity errors on the same stars: robust standard deviation (1.4826 MAD) of z = (v_BOSS - v_APOGEE) / sqrt(e_BOSS^2 + e_APOGEE^2)
# in the bins of S/N, G and Teff of scripts/fig_boss_velocity_errors.py (its BINS, read from the source; bins with >= 20 stars)
# vbins: {column: bin edges} from the BINS = dict(...) assignment of that script.
tree = ast.parse((ROOT / 'scripts/fig_boss_velocity_errors.py').read_text())
vbins = next({k.arg: ast.literal_eval(k.value) for k in n.value.keywords} for n in tree.body
             if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'BINS')
z = (m.vlos_b - m.vlos_a) / np.hypot(m.evlos_a, m.evlos_b)
# Robust standard deviation of z in every bin of every column; the range is quoted; NBoth: stars in both, with a LaTeX thin space.
rsd = [1.4826 * np.median(np.abs(x - np.median(x))) for col, e in vbins.items() for lo, hi in zip(e[:-1], e[1:])
       for x in [z[(m[col] >= lo) & (m[col] < hi)]] if len(x) >= 20]
put('Zsd', rng(rsd, f2))
M['NBoth'] = f'{len(m):,}'.replace(',', '\\,')

# %% [markdown]
# ### Centre and geometry
# The fitted centre, its shift with metallicity and with the sky coverage.

# %%
# Centre and geometry (results/kin_mcmc_data_*.csv stage 2 'shared'; density_centre.csv; kin_data_split.csv; kin_tilted_ring.csv)
shared = lambda t: t[(t.stage == 2) & (t.population == 'shared')].set_index('param')  # noqa: E731
# Shared (stage-2) parameters: j the ellipsoid model, j1 the Level-1 'centre' run, js the sky-aligned 'stream' run.
j, j1, js = shared(E), shared(pd.read_csv(R / 'kin_mcmc_data_centre.csv')), shared(pd.read_csv(R / 'kin_mcmc_data_stream.csv'))
# Fitted centre [deg]; offsets dx (east) and dy (north) [deg] on the tangent plane (dy negated where the text says 'south').
ra0, dec0 = (float(np.rad2deg(a)) for a in kinfit.centre(j.loc['dx', 'median'], j.loc['dy', 'median']))
M.update(CenDx=f2(j.loc['dx', 'median']), CenRa=f2(ra0), CenDec=f2(dec0), LoneDx=f2(j1.loc['dx', 'median']),
         LoneDy=f2(-j1.loc['dy', 'median']), SkyDDx=f2(js.loc['dx', 'median'] - j.loc['dx', 'median']),
         SkyDx=f2(js.loc['dx', 'median']), SkyDy=f2(-js.loc['dy', 'median']), SkyTheta=f1(js.loc['theta', 'median']),
         SkyInc=f1(js.loc['inc', 'median']), SkyMtwo=f2(js.loc['m2c', 'median']),
         MoveL=f2(j1.loc['dx', 'median'] - j.loc['dx', 'median']))   # east offset between the Level-1 and Level-2 joint centres
# ... and with the sky coverage: east shift of the two metal-rich bins resampled onto the positions of the most metal-poor bin
# (results/kin_footprint.csv, scripts/fig_kin_footprint.py; symmetric disc, PM + LOS)
fp = pd.read_csv(R / 'kin_footprint.csv')
fp = fp[fp.fit == 'PM + LOS, centre']
M['MoveSky'] = f2((fp[fp.footprint == 'matched'].groupby('sample').dx.mean() - fp[fp.footprint == 'own'].set_index('sample').dx)
                  .loc[list(g.index[3:])].max())
# Each group's own stage-1 centre: east offsets range, largest southward offset [deg]; RR Lyrae offsets at 4-9 deg and 2-9 deg.
put('OwnDx', rng(e1.dx, f2))
M['OwnDySouth'] = f2(-e1.dy.min())
M.update(RrlOutDx=f'{f2(rou.loc["dx", "median"])} ± {f2((rou.loc["dx", "p84"] - rou.loc["dx", "p16"]) / 2)}',
         RrlInDx=f2(rin.loc['dx', 'median']))
# Density centres (dx, dy [deg]) of the RR Lyrae and the red giants in several apertures [deg].
dens = pd.read_csv(R / 'density_centre.csv')
GIANTS = next(p for p in dens.population.unique() if p.startswith('red giants'))   # 'red giants, Gaia G<...'
dens = dens[dens.population.isin(['RR Lyrae', GIANTS])]
M['GLim'] = GIANTS.split('<')[1]
put('DensDx', rng(dens.dx, f2))
M['Apertures'] = ', '.join(num(a) for a in sorted(dens.aperture.unique())[:-1]) + ' and ' + num(dens.aperture.max())
# Distance of the density centres from the joint kinematic centre [deg], and in kpc at D0; then from each group's own centre.
dist = np.hypot(dens.dx - j.loc['dx', 'median'], dens.dy - j.loc['dy', 'median'])
put('DistDens', rng(dist, f2))
put('DistDensKpc', rng(np.deg2rad(dist) * LMC['D0'], f2))
own_dist = [np.hypot(dens.dx - r.dx, dens.dy - r.dy) for _, r in e1.iterrows()]
put('OwnDistDens', rng(np.concatenate(own_dist)))
M.update(OwnDxRoundLo=f1(e1.dx.min()), OwnDxRoundHi=f1(e1.dx.max()))
pub = {k: v for k, v in cen['PUBLISHED'].items() if 'PM' in k}   # published proper-motion centres (Gaia, Gaia/VMC)
# Published centres projected on the tangent plane, x (east), y (north) [deg].
xy = np.array([lmc_xy(ra, dec) for v in pub.values() for ra, dec in v[0]])
put('PubDx', rng(xy[:, 0]))
put('PubDy', rng(xy[:, 1]))
# Largest centre offset [deg] of the PM + LOS fits of the symmetric mocks, rounded up to 0.1 deg.
mp = pd.read_csv(R / 'kin_mock_physical.csv')
mp = mp[mp.data.isin(['PM + LOS', 'PM + LOS stream'])]
M['MockOffMax'] = f1(np.ceil(10 * np.hypot(mp.dx, mp.dy).max()) / 10)   # symmetric (centred) mocks
# Joint inclination and line of nodes [deg] with half the 16th-84th range (stage-2 row of bin A; the geometry is shared in the joint
# fit).
M.update(IncJoint=f1(j.loc['inc', 'median']), ThetaJoint=f1(j.loc['theta', 'median']),
         IncJointErr=f1(half(g2.iloc[[0]], 'inc').iloc[0]), ThetaJointErr=f1(half(g2.iloc[[0]], 'theta').iloc[0]))
# Inclination [deg] of the giant fits on proper motions alone and with line-of-sight velocities.
sp = pd.read_csv(R / 'kin_data_split.csv')
sp = sp[sp['sample'].str.startswith('giants')]
put('IncPM', rng(sp[sp.data == 'PM only'].inc, lambda x: num(round(x))))
put('IncBoth', rng(sp[sp.data == 'PM + LOS'].inc, lambda x: num(round(x))))
tr = pd.read_csv(R / 'kin_tilted_ring.csv')
# Inclination and line of nodes [deg] from the RR Lyrae proper motions and distances.
dd = tr[(tr['sample'] == 'RR Lyrae PM + distances') & (tr.model == 'stream')].iloc[0]
M.update(IncDist=num(round(dd.inc)), ThetaDist=num(round(dd.theta)))
# Each bin's own line of nodes and inclination [deg]; NSig: shift of bins A, B from stage 1 to stage 2 in units of their errors.
put('ThetaOwn', rng(g1.theta))
M.update(ThetaOwnIntLo=num(int(np.floor(g1.theta.min()))), ThetaOwnIntHi=num(int(np.ceil(g1.theta.max()))))
M.update(IncA=f'{f1(g1.inc[0])} ± {f1(half(g1, "inc")[0])}', IncB=f'{f1(g1.inc[1])} ± {f1(half(g1, "inc")[1])}',
         IncAInt=f1(g1.inc[0]), IncBInt=f1(g1.inc[1]))
put('IncRich', rng(g1.inc[2:]))
nsig = (g2.inc[:2] - g1.inc[:2]) / half(g1, 'inc')[:2]
put('NSig', rng(nsig))
# RR Lyrae geometry and v_rot/sigma_phi in stage 1 (r1) and in the joint fit (r2); largest stage-1 - stage-2 change of the ratios.
r1 = geo[(geo.stage == 1) & (geo.population == 'RR Lyrae')].iloc[0]
r2 = geo[(geo.stage == 2) & (geo.population == 'RR Lyrae')].iloc[0]
M.update(IncRrl=pm(r1.inc, r1.inc_p16, r1.inc_p84), ThetaRrl=pm(r1.theta, r1.theta_p16, r1.theta_p84, lambda x: num(round(x))),
         RatPhiRrl=pm(r1.ratio_phi, r1.ratio_phi_p16, r1.ratio_phi_p84, f2),
         RatPhiRrlJ=pm(r2.ratio_phi, r2.ratio_phi_p16, r2.ratio_phi_p84, f2),
         DRatStage=f'{max((g1.ratio_phi - g2.ratio_phi).abs().max(), (g1.ratio_z - g2.ratio_z).abs().max()):.3f}')
put('RatErr', rng(pd.concat([half(g1, 'ratio_phi'), half(g1, 'ratio_z')]), lambda x: f'{x:.3f}'))

# %% [markdown]
# ### Sensitivity to the model
# Tilted rings, a free scale of the line-of-sight velocities, a turning disc and the checks of the velocity-ellipsoid model.

# %%
# Sensitivity to the model, sky-aligned dispersions (maximum likelihood; results/kin_tilted_ring, kin_los_scale, kin_turning.csv)
# lnl: -ln L of each giant bin for model m of table t; col: column c of the same rows.
pops = list(g.index)
lnl = lambda t, m: t[t.model == m].set_index('sample').loc[pops].nll.to_numpy()  # noqa: E731  -ln L of each giant bin
col = lambda t, m, c: t[t.model == m].set_index('sample').loc[pops][c].to_numpy()  # noqa: E731
ls = pd.read_csv(R / 'kin_los_scale.csv')
ls = ls[ls.velocities == 'all']
tu = pd.read_csv(R / 'kin_turning.csv')
# Gains in ln L over the 'stream' model of a tilted disc and of a free line-of-sight scale qlos.
ref = lnl(tr, 'stream')
dl_tilt, dl_q = ref - lnl(tr, 'tilt + stream'), ref - lnl(ls, 'qlos + stream')
put('DIncTilt', rng(-col(tr, 'tilt + stream', 'dinc'), lambda x: num(round(x))))
# qlos of each bin; text quotes bins D, E and the largest of the others, with their ln L gains.
q = col(ls, 'qlos + stream', 'qlos')
M.update(DlnLTiltD=num(round(dl_tilt[3])), DlnLTiltE=num(round(dl_tilt[4])), DlnLTiltOther=num(round(dl_tilt[:3].max())),
         DVTilt=f1(np.abs(col(tr, 'tilt + stream', 'v0') - col(tr, 'stream', 'v0')).max()),
         QlosD=f2(q[3]), QlosE=f2(q[4]), DlnLQD=num(round(dl_q[3])), DlnLQE=num(round(dl_q[4])),
         QlosOther=f1(np.abs(q[:3]).max()), DlnLQOther=num(round(dl_q[:3].max())))
put('IncQ', rng(col(ls, 'qlos + stream', 'inc')[3:], lambda x: num(round(x))))
# Turning disc: ln L gains (negative: worse) and the turning rates dthdt; the rotation period converts them to degrees per turn.
dl_turn, dl_turns = ref - lnl(tu, 'turn'), ref - lnl(tu, 'turn + stream')
dth = col(tu, 'turn + stream', 'dthdt')
KPC_KMS_GYR = 977.79   # kpc / (km/s) in Myr x 1e3: 1 kpc / (1 km/s) = 977.79 Myr
period = round(2 * np.pi * kinfit.R_SIG / g.vrot4.max() * KPC_KMS_GYR / 1e3, 1)   # Gyr, rotation period at R_SIG of the fastest giants, as quoted
M.update(TurnWorseLo=num(round(dl_turn.max())), TurnWorseHi=num(round(dl_turn.min())), TurnBetterLo=num(round(dl_turns.min())),
         TurnBetterHi=num(round(dl_turns.max())), DthdtLo=num(round(dth.min())), DthdtHi=num(round(dth.max())), Period=f1(period),
         TurnLo=num(10 * round(dth.min() * period / 10)), TurnHi=num(10 * round(dth.max() * period / 10)))
# sigma_los and v0 [km/s] of every bin in every model, shape (6 models, 5 bins); their ratio v0/sigma_los.
models = [(tr, 'sym'), (tr, 'stream'), (tr, 'tilt + stream'), (ls, 'qlos + stream'), (tu, 'turn'), (tu, 'turn + stream')]
slos = np.array([col(t, m, 'sigma_los') for t, m in models])
v0s = np.array([col(t, m, 'v0') for t, m in models])
ratios = v0s / slos
M.update(DSigLos=f1(np.ptp(slos, axis=0).max()), DVAlt=num(round(np.abs(v0s[2:4] - v0s[1]).max())))
put('DVTurn', rng(v0s[5] - v0s[1], lambda x: num(round(x))))
# Whether v0/sigma_los rises from bin A to E in each model; the asserts stop the script if the text's statement no longer holds.
mono = [bool(np.all(np.diff(r) > 0)) for r in ratios]
assert mono == [True, True, True, True, False, True], mono   # only the turning disc alone breaks the ordering
assert np.all(np.diff(ratios[4][1:]) > 0) and ratios[4][0] > ratios[4][1]   # ... by swapping the two most metal-poor bins
M.update(TurnSwapA=f2(ratios[4][0]), TurnSwapB=f2(ratios[4][1]))
# v0/sigma_los of bins A and E (all giants) and of bin E from APOGEE alone (scripts/paper_kin_support.py).
ks = pd.read_csv(R / 'kin_support.csv').set_index('population')
M.update(RatLosA=f2(ks.ratio.iloc[0]), RatLosE=f2(ks.ratio.iloc[4]), RatLosApoE=f2(ks.ratio_apogee.iloc[4]))
# ... and with the velocity ellipsoid (results/kin_ellipsoid_checks.csv)
ec = pd.read_csv(R / 'kin_ellipsoid_checks.csv')
ecv = lambda m, c: ec[ec.model == m].set_index('population').loc[pops][c].to_numpy()  # noqa: E731
QUANT = ['vrot4', 'sigma_R', 'sigma_phi', 'sigma_z']
# Changes [km/s] of rotation and dispersions from the reference ellipsoid fit with a tilt (dt) or a free qlos (dq); their ln L
# gains.
dt = {c: ecv('tilt', c) - ecv('reference', c) for c in QUANT}
dq = {c: ecv('qlos', c) - ecv('reference', c) for c in QUANT}
et, eq = ecv('tilt', 'dlnL'), ecv('qlos', 'dlnL')
M.update(DlnLETiltD=num(round(et[3])), DlnLETiltE=num(round(et[4])), DlnLETiltOtherLo=f1(et[:3].min()),
         DlnLETiltOtherHi=num(round(et[:3].max())), DVETilt=f1(np.abs(dt['vrot4']).max()),
         DSigETilt=f1(max(np.abs(dt[c]).max() for c in SIG)))
put('DIncETilt', rng(-ecv('tilt', 'dinc')[3:], lambda x: num(round(x))))
# qlos of the ellipsoid fits; the two metal-poor bins sit at the upper bound (checked by the assert).
qe = ecv('qlos', 'qlos')
assert np.all(qe[:2] == qe[:2].max())   # the two metal-poor bins at the bound
M.update(QlosED=f2(qe[3]), DlnLQED=f1(eq[3]), QlosEE=f2(qe[4]), DlnLQEE=num(round(eq[4])), IncQEE=num(round(ecv('qlos', 'inc')[4])),
         QBound=f1(qe[0]), DlnLQEALo=f1(eq[:2].min()), DlnLQEAHi=f1(eq[:2].max()),
         DSigEQ=f1(max(np.abs(dq[c]).max() for c in SIG)), DVEQ=num(round(np.abs(dq['vrot4']).max())))
# Error scale k for the bright and faint halves, and the change of the dispersions [km/s] between them (bins A-C, then D-E).
put('KBright', rng(ecv('bright', 'k'), f2))
put('KFaint', rng(ecv('faint', 'k'), f2))
dhalf = {c: ecv('faint', c) - ecv('bright', c) for c in SIG}
M['DSigHalf'] = f1(max(np.abs(dhalf[c][:3]).max() for c in SIG))
put('DSigHalfMR', rng(np.concatenate([dhalf[c][3:] for c in SIG]), lambda x: num(round(x))))

# %% [markdown]
# ### Further checks
# The inner edge of the sample, the m = 2 residual and the proper-motion errors.

# %%
# Further checks: the inner edge of the sample (results/kin_dispersion_cut.csv)
dc = pd.read_csv(R / 'kin_dispersion_cut.csv')
# Spread of each giant bin's sigma [km/s] over the inner edges; rd, md: the RR Lyrae data and mock, indexed by inner edge [deg].
gd = dc[(dc['sample'] == 'data') & dc.population.str.startswith('giants')]
put('DSigGiant', rng(gd.groupby('population').sigma.agg(np.ptp)))
rd = dc[(dc['sample'] == 'data') & (dc.population == 'RR Lyrae')].set_index('inner')
md = dc[(dc['sample'] == 'mock')].set_index('inner')
# RR Lyrae k and sigma [km/s] at the 2-deg edge (where sigma sits at its bound) and at 3 deg; the mock sigma at 2 deg.
M.update(KRrlCutHi=f2(rd.k.iloc[0]), KRrlCutLo=f2(rd.k.min()), SigBound=num(rd.sigma.iloc[0]), SigRrlThree=num(round(rd.sigma[3.0])),
         MockSigTwo=num(round(md.sigma.iloc[0])))
# Ranges for inner edges of 4 deg and beyond (OUTER_INNER).
put('SigRrlOut', rng(rd.sigma[rd.index >= fk['OUTER_INNER']], lambda x: num(round(x))))
put('MockSigOut', rng(md.sigma[md.index >= fk['OUTER_INNER']], lambda x: num(round(x))))
put('MockKOutFit', rng(md.k[md.index >= fk['OUTER_INNER']], f2))

# m=2 residual about a symmetric disc: amplitude of the m=2 term of v_t about the Level-1 fit, all groups
# (results/kin_residual_profiles_ellipsoid.csv, scripts/paper_kin_residuals.py)
harm = pd.read_csv(R / 'kin_residual_profiles_ellipsoid.csv')
put('MTwoAmp', rng(harm[(harm.model == 'L1') & (harm.component == 'v_t')].A2, lambda x: sigfig(x, 2)))

# Gaia proper-motion errors in km/s (release samples): per star the mean of the two components' errors times kappa D0; giants and RR
# Lyrae at 2-9 deg (medians), and the change of the RR Lyrae sigma_PM at 4-9 deg for a change DK of k: the likelihood constrains
# sigma^2 + k^2 e^2, so sigma -> sqrt(sigma^2 -+ 2 k e^2 DK), with sigma and k of the fit at 4-9 deg (kin_dispersion_cut.csv) and e
# the median error there


def err_kms(c):
    """Mean of the two Gaia proper-motion errors of each star of catalogue c, in km/s at the LMC distance D0."""
    return 0.5 * (c['pmra_error'] + c['pmdec_error']) * KAPPA * LMC['D0']


# Median proper-motion error [km/s] of each giant bin at 2-9 deg; DK: change of k considered (dimensionless).
put('ErrMedG', rng([np.median(err_kms(ring(c))) for _, c in samples.giants()], lambda x: num(round(x))))
DK = 0.1
# RR Lyrae at 4-9 deg: median error e [km/s]; sigma and k of the 4-deg fit; change of sigma for k +- DK.
e_out = np.median(err_kms(ring(samples.rrab(), fk['OUTER_INNER'])))
s_out, k_out = rd.sigma[fk['OUTER_INNER']], rd.k[fk['OUTER_INNER']]
dsk = [abs(np.sqrt(s_out ** 2 + sgn * 2 * k_out * e_out ** 2 * DK) - s_out) for sgn in (1, -1)]
M.update(ErrMedRrl=num(round(np.median(err_kms(ring(samples.rrab()))))), ErrMedRrlOut=num(round(e_out)), DK=num(DK))
put('DSigK', rng(dsk, sigfig))

# %% [markdown]
# ### Write the macros
# One \newcommand per entry of M, with the prefix \ks, into results/macros_kinsupport.tex; the page text shows them through
# {glue:text}, so no number is typed by hand.

# %%
# Header line, one \newcommand per macro, written and printed.
lines = ['% Numbers of the KB page Rotational support, centre and sensitivity (scripts/paper_text_kin_support.py); do not edit.']
lines += [rf'\newcommand{{\ks{k}}}{{{x}}}' for k, x in M.items()]
(R / 'macros_kinsupport.tex').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
