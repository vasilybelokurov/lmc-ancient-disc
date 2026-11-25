"""Knowledge-base text, chapter 'Kinematics: velocity field and disc model' (extended draft Sections 4.1-4.3 and the appendix on the
ellipsoid mocks and the sky-aligned model): every number of the text as a LaTeX macro (prefix \\km), so that none is typed by hand.

As scripts/paper_text_shape.py: settings are read from the code (scripts/textnum.const, lmcold by import), results from the tables
the figures and tables are made from, and the posterior samples of the velocity-ellipsoid fits (for the rotation speed at R_sigma of
the joint fit, which no table holds), and the release samples (for the Gaia proper-motion errors in km/s). Literature values
quoted with their citation stay in the text.
Inputs: results/{kin_mcmc_data_ellipsoid, kin_mcmc_data_ellipsoid_outer, kin_mcmc_data_stream, kin_mcmc_data_stream_outer,
kin_support_ellipsoid, kin_support, kin_model_comparison, kin_ellipsoid_mock, kin_mock_physical, kin_tilted_ring, kin_footprint,
kin_residual_profiles_ellipsoid, kin_residual_cells_ellipsoid, kin_residual_radial_ellipsoid, kin_residual_radial}.csv;
data/kin_mcmc_data_ellipsoid{,_outer}_samples.npz; scripts/{fit_kinematics_mcmc, paper_kin_residuals, paper_kin_ellipsoid_mock,
fig_kin_mock_physical}.py; lmcold.kinfit (bounds, R_SIG, S_OUT), lmcold.constants, lmcold.samples (FEH_BINS, giants, rrab).
Writes results/macros_kinmodel.tex.
Run: .venv/bin/python scripts/manifest.py run paper_text_kin_model.py"""

# %% [markdown]
# ### Settings and helpers
# Imports, the dictionary M of macros (macro name without its prefix: the text it stands for) and the functions that format
# numbers as the text writes them. The inputs listed in the header are read in the step that uses them.

# %%
import numpy as np
import pandas as pd
from scipy.stats import norm

from fig_kin_mock_physical import COLD, E_VLOS, H_THICK, HOT, N_STARS, RD, TRUTH
from fit_kinematics_mcmc import CHAINS, DRAWS, OUTER_INNER, RING, WARMUP, WARMUP_ELLIPSOID, ring
from lmcold import kinfit, samples
from lmcold.constants import KAPPA, LMC, LMC_DISTANCE_MODULUS
from lmcold.samples import FEH_BINS
from lmcold.sources import DATA, ROOT
from textnum import const, num, sigfig

R = ROOT / 'results'  # folder of the result tables
M = {}  # macro name (without the \km prefix) -> text
B = kinfit.BOUNDS  # prior bounds of every kinfit parameter: name -> (lo, hi)
res = const('paper_kin_residuals')  # literal module constants of scripts/paper_kin_residuals.py
mock = const('paper_kin_ellipsoid_mock')  # literal module constants of scripts/paper_kin_ellipsoid_mock.py
DISC_MOCKS = ['cold thin', 'hot', 'hot, no drift', 'hot thick']   # the four disc mocks; the 'hot spheroid' is quoted on its own


def f1(x):
    """x with one decimal."""
    return f'{x:.1f}'


def f2(x):
    """x with two decimals."""
    return f'{x:.2f}'


def f3(x):
    """x with three decimals."""
    return f'{x:.3f}'


def two(x):
    """A velocity or amplitude to two significant figures (11, 4.9, 0.48)."""
    return sigfig(x, 2)


def signed(x, fmt=f3):
    """x formatted by fmt, with an explicit '+' when not negative."""
    s = fmt(x)
    return s if s.startswith('-') else '+' + s


def count(n):
    """Integer n with thin spaces between thousands (LaTeX '\\,')."""
    return f'{int(n):,}'.replace(',', '\\,')


def rng(s, fmt=two, key=''):
    """Macros <key>Lo and <key>Hi: the minimum and maximum of s."""
    M.update({key + 'Lo': fmt(min(s)), key + 'Hi': fmt(max(s))})


def stage(t, st, pop, p):
    """Posterior median of parameter p of population pop at stage st (1: each group alone; 2: joint) in table t."""
    r = t[(t.stage == st) & (t.population == pop) & (t.param == p)]
    return r['median'].iloc[0]


# %% [markdown]
# ### Settings of the fits
# The [Fe/H] bins, radial ranges, distance, model constants and priors.

# %%
# Settings: [Fe/H] bins, radial ranges, distance, model constants and priors
# FehA-FehD: upper edges of the first four [Fe/H] bins [dex]; RhoIn/RhoOut: the fitted ring [deg]; RhoOuter: inner edge of the
# RR Lyrae fit [deg]; profile ring, clipping, cell size and minimum count of the residual maps (paper_kin_residuals.py)
M.update(FehA=num(FEH_BINS[0][1]), FehB=num(FEH_BINS[1][1]), FehC=num(FEH_BINS[2][1]), FehD=num(FEH_BINS[3][1]),
         RhoIn=num(RING[0]), RhoOut=num(RING[1]), RhoOuter=num(OUTER_INNER), ProfIn=num(res['PROFILE_RING'][0]),
         ProfOut=num(res['PROFILE_RING'][1]), Clip=num(res['CLIP']), Cell=num(res['CELL']), NMin=num(res['NMIN']),
         PaBin=num(30), Field=num(2 * RING[1]))   # 30 deg: the position-angle bins of paper_kin_residuals.py (PA_EDGES)
# distance [kpc] implied by the adopted distance modulus: 10^(mu/5 + 1) pc
d_mod = 10 ** (LMC_DISTANCE_MODULUS / 5 + 1) / 1e3   # kpc
# adopted distance D0 [kpc], modulus, its distance and their per-cent difference; KAPPA (km/s per mas/yr per kpc), R_sigma [kpc],
# outlier width S_OUT [mas/yr] and in km/s at D0; the vdMK14 systemic velocity and centre; prior bounds (P...) of the fits
M.update(DZero=num(LMC['D0']), DistMod=f'{LMC_DISTANCE_MODULUS:.3f}', DMod=f1(d_mod), DPct=num(round(100 * (1 - d_mod / LMC['D0']))),
         Kappa=f'{KAPPA:.5f}', RSig=num(kinfit.R_SIG), SOut=num(kinfit.S_OUT), SOutKms=sigfig(KAPPA * LMC['D0'] * kinfit.S_OUT, 2),
         VlosVdm=num(LMC['vlos0']), RaVdm=f2(LMC['ra0']), DecVdm=f2(LMC['dec0']),
         PDx=num(B['dx'][1]), PIncLo=num(B['inc'][0]), PIncHi=num(B['inc'][1]), PPm=num(B['pmra0'][1] - LMC['pmra0']),
         PVlos=num(B['vlos0'][1] - LMC['vlos0']), PVZero=num(B['v0'][1]), PVZeroLo=num(B['v0'][0]), PThetaLo=num(B['theta'][0]), PThetaHi=num(B['theta'][1]), PRZeroLo=num(B['R0'][0]), PRZeroHi=num(B['R0'][1]),
         PSlope=num(B['vslope'][1]), PStream=num(B['m2c'][1]), PSigLo=num(B['sigma_R'][0]), PSigHi=num(B['sigma_R'][1]),
         PGam=num(B['dlnsig_dR'][1]), PKLo=num(B['k'][0]), PKHi=num(B['k'][1]), PLogitLo=num(B['logit_fout'][0]),
         PLogitHi=num(B['logit_fout'][1]))

# %% [markdown]
# ### Velocity-ellipsoid fits
# Posterior values of the stage-1 (each group alone) and stage-2 (joint) fits, the Gaia proper-motion errors in km/s and their
# effect on sigma_phi.

# %%
# Velocity-ellipsoid fits (results/kin_mcmc_data_ellipsoid*.csv): stage 1 (each group alone) and stage 2 (joint)
# E: posterior summaries of the 2-9 deg fits; EO: of the 4-9 deg fits (one row per stage, population, parameter)
E = pd.read_csv(R / 'kin_mcmc_data_ellipsoid.csv')
EO = pd.read_csv(R / 'kin_mcmc_data_ellipsoid_outer.csv')
# populations other than the shared parameters of stage 2; giants: the five giant bins
pops = [p for p in E.population.unique() if p != 'shared']
giants = [p for p in pops if p.startswith('giants')]
# stage-1 posterior medians: one row per population, one column per parameter
s1 = E[E.stage == 1].pivot_table(index='population', columns='param', values='median')
g1 = s1.loc[giants]
# stage-2 shared parameters (centre, orientation, systemic motion, streaming), indexed by parameter name
sh = E[(E.stage == 2) & (E.population == 'shared')].set_index('param')
# numbers of parameters of one stage-1 fit (giants, RR Lyrae) and of the shared set of stage 2
M.update(NPar=num(len(E[(E.stage == 1) & (E.population == giants[0])])), NParRrl=num(len(E[(E.stage == 1) & (E.population == 'RR Lyrae')])),
         NShared=num(len(sh)))
# largest offset [deg] of a giant bin's stage-1 centre from the joint centre, and largest inclination difference [deg]
dcen = np.hypot(g1.dx - sh.loc['dx', 'median'], g1.dy - sh.loc['dy', 'median'])
M.update(DCen=f1(dcen.max()), DInc=f1((g1.inc - sh.loc['inc', 'median']).abs().max()))
# ranges over the giant bins of v0 [km/s] and R0 [kpc]
rng(g1.v0, lambda x: num(round(x)), 'VZero')
rng(g1.R0, f1, 'RZero')
Rg = np.linspace(2, 10, 8001)   # kpc: radius at which each giant group's posterior-median curve peaks
peak = [Rg[np.argmax(r.v0 * np.tanh(Rg / r.R0) * (1 + r.vslope * (Rg - kinfit.R_SIG)))] for _, r in g1.iterrows()]
rng(peak, f1, 'RPeak')
# v_rot(4 kpc) posterior percentiles per population (results/kin_support_ellipsoid.csv); range of its half-width over the giant bins
ks = pd.read_csv(R / 'kin_support_ellipsoid.csv').set_index('population')
rng(((ks.vrot4_p84 - ks.vrot4_p16) / 2).loc[giants], f1, 'VrotErr')
# error scale factor k of the Gaia proper-motion errors: range over the giant bins, and the RR Lyrae at 4-9 deg and 2-9 deg
rng(g1.k, f2, 'KG')
M.update(KRrlOut=f2(stage(EO, 1, 'RR Lyrae', 'k')), KRrl=f2(s1.loc['RR Lyrae', 'k']))
# joint inclination [deg] and its cos^2, sin^2
inc = sh.loc['inc', 'median']
M.update(IncJ=num(round(inc)), CosSq=f2(np.cos(np.deg2rad(inc)) ** 2), SinSq=f2(np.sin(np.deg2rad(inc)) ** 2))
# stars per giant bin (thin-space thousands)
M.update(NG=count(ks.n.iloc[0]), NGb=count(ks.n.iloc[1]), NGc=count(ks.n.iloc[2]), NGd=count(ks.n.iloc[3]), NGe=count(ks.n.iloc[4]))

# Gaia proper-motion errors in km/s (release samples, lmcold.samples): per star the mean of the two components' errors times
# kappa D0; giants at RING, RR Lyrae at OUTER_INNER < rho < RING[1] (5th-95th percentiles) and at RING (median)


def err_kms(c):
    """Gaia proper-motion error [km/s] per star of catalogue c: mean of the two components [mas/yr] x KAPPA x D0."""
    return 0.5 * (c['pmra_error'] + c['pmdec_error']) * KAPPA * LMC['D0']


# errors per giant bin at 2-9 deg; RR Lyrae at 4-9 deg (er_out) and 2-9 deg (er)
eg = {f'giants {lab}': err_kms(ring(c)) for lab, c in samples.giants()}
er_out, er = err_kms(ring(samples.rrab(), OUTER_INNER)), err_kms(ring(samples.rrab()))
pg = np.array([np.percentile(eg[p], [5, 50, 95]) for p in giants])   # rows: giant bins in the order of giants
# 5th and 95th percentiles of the RR Lyrae errors at 4-9 deg [km/s]
pr = np.percentile(er_out, [5, 95])
# extremes over the giant bins and the RR Lyrae spread and median [km/s]
M.update(ErrGLo=num(round(pg[:, 0].min())), ErrGHi=num(round(pg[:, 2].max())), ErrRrlLo=num(round(pr[0])), ErrRrlHi=num(round(pr[1])),
         ErrFacRrl=f1(pr[1] / pr[0]), ErrMedRrl=num(round(np.median(er))))
# ratio of 95th to 5th percentile per giant bin; medians; medians times the fitted k [km/s]
rng(pg[:, 2] / pg[:, 0], f1, 'ErrFacG')
rng(pg[:, 1], lambda x: num(round(x)), 'ErrMedG')
rng(pg[:, 1] * g1.k.loc[giants].to_numpy(), lambda x: num(round(x)), 'ErrKG')
# A change DK of k moves sigma_phi^2 of the most metal-rich bin by about 2 k e^2 DK (e: its median error), and sigma_phi by that
# divided by 2 sigma_phi
DK = 0.1  # change of k considered
# 2 k e^2 DK [km^2/s^2] for the most metal-rich bin (e: its median error)
kterm = 2 * g1.k[giants[-1]] * pg[-1, 1] ** 2 * DK
# its sigma_phi [km/s] (results/kin_support_ellipsoid.csv)
sphi = ks.sigma_phi[giants[-1]]
M.update(DK=num(DK), KTerm=sigfig(kterm, 2), SigPhiSq=sigfig(sphi ** 2), DSigPhiK=sigfig(kterm / (2 * sphi)))
# model comparison table: stars per population at the inner radius of each fit (2 deg and OUTER_INNER)
mc = pd.read_csv(R / 'kin_model_comparison.csv')
mc_in, mc_out = mc[mc.rho_inner == RING[0]].set_index('population'), mc[mc.rho_inner == OUTER_INNER].set_index('population')
M.update(NRrl=count(mc_in.n['RR Lyrae']), NRrlOut=count(mc_out.n['RR Lyrae']))

# %% [markdown]
# ### Stage 1 against stage 2
# The rotation speed at R_sigma from the posterior draws of both stages.

# %%
# Stage 1 against stage 2: rotation speed at R_sigma from the posterior draws (data/kin_mcmc_data_ellipsoid*_samples.npz)


def vrot4(npz):
    """{(stage, population): median of v0 tanh(R_sigma / R0) over the draws}."""
    s = np.load(npz)
    names = list(s['names'])
    out = {}
    for i, n in enumerate(names):
        # sample keys: stage 1 '1|<i>|v0_0' (population i alone), stage 2 '2|all|v0_<i>' (population i of the joint fit)
        for st, key in ((1, f'1|{i}|{{}}_0'), (2, f'2|all|{{}}_{i}')):
            out[st, n] = np.median(s[key.format('v0')] * np.tanh(kinfit.R_SIG / s[key.format('R0')]))
    return out


# both fits: 2-9 deg (giants) and 4-9 deg (RR Lyrae)
v4 = vrot4(DATA / 'kin_mcmc_data_ellipsoid_samples.npz')
v4o = vrot4(DATA / 'kin_mcmc_data_ellipsoid_outer_samples.npz')
# stage-2 posterior medians per population, and the three dispersions sigma_R, sigma_phi, sigma_z
s2 = E[E.stage == 2].pivot_table(index='population', columns='param', values='median')
disp = list(kinfit.ELLIPSOID[:3])
# largest stage-1 minus stage-2 differences over the giant bins: v_rot(R_sigma) and dispersions [km/s]
M.update(DVrotStage=f1(max(abs(v4[1, p] - v4[2, p]) for p in giants)),
         DSigStage=f1((s1.loc[giants, disp] - s2.loc[giants, disp]).abs().values.max()))
# the 4-9 deg fit: RR Lyrae stage 1, all populations stage 2, and the shared stage-2 parameters
o1 = EO[EO.stage == 1].pivot_table(index='population', columns='param', values='median').loc['RR Lyrae']
o2 = EO[EO.stage == 2].pivot_table(index='population', columns='param', values='median')
osh = EO[(EO.stage == 2) & (EO.population == 'shared')].set_index('param')['median']
# joint orientation of the 4-9 deg fit [deg]; RR Lyrae stage-1 against stage-2 differences [km/s]
M.update(IncOut=f1(osh.inc), ThetaOut=f1(osh.theta), DVrotRrl=f1(abs(v4o[1, 'RR Lyrae'] - v4o[2, 'RR Lyrae'])),
         DSigRRrl=f1(abs(o2.loc['RR Lyrae', 'sigma_R'] - o1.sigma_R)), DSigPhiRrl=f1(abs(o2.loc['RR Lyrae', 'sigma_phi'] - o1.sigma_phi)))
half = EO[(EO.stage == 1) & (EO.population == 'RR Lyrae')].set_index('param')
half = (half.p84 - half.p16) / 2   # stage-1 posterior half-widths of the RR Lyrae at 4-9 deg
M.update(SigRRrlErr=f1(half.sigma_R), SigPhiRrlErr=f1(half.sigma_phi))
# spread of the stage-1 line of nodes, inclination and centre offset over the giant bins [deg]; joint streaming m2c
M.update(DTheta=f1(g1.theta.max() - g1.theta.min()), IncGLo=f1(g1.inc.min()), IncGHi=f1(g1.inc.max()), DxGLo=f2(g1.dx.min()),
         DxGHi=f2(g1.dx.max()), MTwoC=f3(sh.loc['m2c', 'median']))

# %% [markdown]
# ### Sampling diagnostics
# Chains, draws and the convergence diagnostics of the fits.

# %%
# NUTS sampling (scripts/fit_kinematics_mcmc.py settings; diagnostics of the result tables, as scripts/paper_text_mcmc.py)


def diagnostics(*runs):
    """Worst convergence over the given runs: max r_hat (rounded up to 3 decimals), min n_eff and the fits with divergences."""
    # all stages and populations of the runs; divergences per fit
    t = pd.concat([pd.read_csv(R / f'kin_mcmc_data_{r}.csv').assign(run=r) for r in runs])
    d = t.groupby(['run', 'stage', 'population']).divergences.max()
    return np.ceil(t.r_hat.max() * 1000) / 1000, np.floor(t.n_eff.min()), d[d > 0]


# velocity-ellipsoid fits (2-9 and 4-9 deg)
r_hat, n_eff, div = diagnostics('ellipsoid', 'ellipsoid_outer')
assert len(div) == 1, div   # the text names one fit with divergent transitions
# stage and population of the one fit with divergent transitions, and their number
(_, div_stage, div_pop), div_n = next(iter(div.items()))
# chains, draws, warm-up lengths, worst r_hat and n_eff; the divergent fit's stage, count and [Fe/H] bin
M.update(NChain=num(CHAINS), NDraw=count(DRAWS), NWarm=count(WARMUP_ELLIPSOID), NWarmOuter=count(WARMUP), RHat=f3(r_hat),
         NEff=count(n_eff), DivStage=num(div_stage), DivN=num(div_n), NTot=count(CHAINS * DRAWS),
         DivLo=num(FEH_BINS[[p for p in giants].index(div_pop)][0]), DivHi=num(FEH_BINS[[p for p in giants].index(div_pop)][1]))
# the sky-aligned (stream) fits
r_hat, n_eff, div = diagnostics('stream', 'stream_outer')
assert len(div) == 1 and div.index[0][2] == 'RR Lyrae', div   # the text names the RR Lyrae fit at 4-9 deg
M.update(RHatS=f3(r_hat), NEffS=count(n_eff), DivNS=num(div.iloc[0]))

# %% [markdown]
# ### The measured velocity field and residuals
# Harmonic fits to the tangential velocity against position angle and the residuals about the models.

# %%
# The measured velocity field and the residuals (results/kin_residual_*_ellipsoid.csv, written by paper_kin_residuals.py ellipsoid)
# harmonic fits of the residual velocity profiles against position angle, one row per population, model and component
h = pd.read_csv(R / 'kin_residual_profiles_ellipsoid.csv')
hg = h[h.population.str.startswith('giants')]


def harm(model, comp, col, who=hg):
    """Column col of the harmonic-fit rows of `model` and velocity component `comp` (default: giant bins only)."""
    return who[(who.model == model) & (who.component == comp)][col]


# Range of v_t over position angle: minimum and maximum over psi of each group's harmonic fit c0 + A1 cos(psi - psi1) +
# A2 cos 2(psi - psi2) to the stars at PROFILE_RING, extremes over the groups (giant bins and RR Lyrae)
psi = np.deg2rad(np.arange(360))
vt = [r['mean'] + r.A1 * np.cos(psi - np.deg2rad(r.phase1)) + r.A2 * np.cos(2 * (psi - np.deg2rad(r.phase2)))
      for _, r in h[(h.model == 'data') & (h.component == 'v_t')].iterrows()]
# slowest and fastest rotation [km/s] over position angle: v_t is negative, so slowest = -max(v_t), fastest = -min(v_t)
M.update(VtSlow=num(round(-max(v.max() for v in vt))), VtFast=num(round(-min(v.min() for v in vt))))
# ranges of the data's m = 2 and m = 1 amplitudes [km/s]
rng(pd.concat([harm('data', 'v_t', 'A2', h), harm('data', 'v_r', 'A2', h)]), lambda x: num(round(x)), 'DataATwo')
rng(harm('data', 'v_t', 'A1', h), lambda x: num(round(x)), 'DataAOne')
# m = 1 amplitudes of the vdMK14 model in each component
for c, k in (('v_t', 'Vt'), ('v_r', 'Vr'), ('v_los', 'Vl')):
    rng(harm('vdMK14', c, 'A1'), two, f'Vdm{k}')
# Level-1 model (L1): m = 1 and m = 2 amplitudes, and the m = 2 phase [deg] (giants, RR Lyrae)
rng(pd.concat([harm('L1', c, 'A1') for c in ('v_t', 'v_r', 'v_los')]), lambda x: num(round(x)), 'LOneAOne')
rng(harm('L1', 'v_t', 'A2'), two, 'LOneVt')
rng(harm('L1', 'v_r', 'A2'), two, 'LOneVr')
rng(harm('L1', 'v_t', 'phase2'), lambda x: num(round(x)), 'PhaseG')
M.update(PhaseRrl=num(round(h[(h.model == 'L1') & (h.component == 'v_t') & (h.population == 'RR Lyrae')].phase2.iloc[0])))
# Level-2 model (L2): m = 2 and m = 1 amplitudes [km/s]
rng(harm('L2', 'v_t', 'A2'), two, 'LTwoVt')
rng(harm('L2', 'v_r', 'A2'), two, 'LTwoVr')
rng(harm('L2', 'v_t', 'A1'), two, 'LTwoAOne')
# residual maps in sky cells: z-score spread (z_sd) and fraction with |z| > 3, for metal-poor (first three bins + RR Lyrae) and rich
cells = pd.read_csv(R / 'kin_residual_cells_ellipsoid.csv')
poor = cells[cells.population.isin(giants[:3] + ['RR Lyrae'])]
rich = cells[cells.population.isin(giants[3:])]
rng(poor.z_sd, f2, 'ZsdPoor')
rng(rich.z_sd, f1, 'ZsdRich')
rng(100 * rich.z_gt3, lambda x: num(round(x)), 'ZgtRich')
# ZgtNorm: per cent of a unit Gaussian beyond |z| = 3
M.update(ZgtPoor=num(round(100 * poor.z_gt3.max())), ZgtNorm=sigfig(100 * 2 * norm.sf(3)))
# radial profiles of the residuals (median per ring); rt: tangential component
rad = pd.read_csv(R / 'kin_residual_radial_ellipsoid.csv')
rt = rad[rad.component == 'v_t']
four = giants[:4]   # the four giant bins below [Fe/H] = -0.5
RINGS = dict(Fast=(3, 5), Agree=5, Out=6, Tanh=7)   # deg: ring boundaries of the statements on the radial residuals
M.update({f'Ring{k}': num(v) for k, v in RINGS.items() if k != 'Fast'}, RingFastLo=num(RINGS['Fast'][0]),
         RingFastHi=num(RINGS['Fast'][1]))
# tangential residuals of the four metal-poor bins between 3 and 5 deg
mid = rt[rt.population.isin(four) & (rt.rho_lo >= RINGS['Fast'][0]) & (rt.rho_hi <= RINGS['Fast'][1])]['median']
assert (mid < 0).all()   # faster rotation than the model in every ring and bin
rng(-mid, lambda x: num(round(x)), 'RadFast')
# largest tangential residual inside 3 deg; largest |residual| of any giant bin beyond 5 deg [km/s]
inner = rt[rt.population.isin(four) & (rt.rho_hi <= RINGS['Fast'][0])]['median']
M.update(RadSlow=num(round(inner.max())), RadOuter=f1(rt[rt.population.isin(giants) & (rt.rho_lo >= RINGS['Agree'])]['median'].abs().max()))
# radial component: inward motion inside OUTER_INNER (giants from the third bin up); largest residual beyond 6 deg (from the second)
rr = rad[rad.component == 'v_r']
rin = rr[rr.population.isin(giants[2:]) & (rr.rho_hi <= OUTER_INNER)]['median']
assert (rin < 0).all()   # inward motion in every ring and bin
rng(-rin, lambda x: num(round(x)), 'RadIn')
rout = rr[rr.population.isin(giants[1:]) & (rr.rho_lo >= RINGS['Out'])].groupby('population')['median'].max()
rng(rout, lambda x: num(round(x)), 'RadOut')
# RR Lyrae rings beyond OUTER_INNER with positive tangential residual
rrl = rt[(rt.population == 'RR Lyrae') & (rt.rho_lo >= OUTER_INNER) & (rt['median'] > 0)]['median']
rng(rrl, lambda x: num(round(x)), 'RadRrl')
M.update(NRadRrl=num(len(rrl)), NRingRrl=num(len(rt[(rt.population == 'RR Lyrae') & (rt.rho_lo >= OUTER_INNER)])))

# %% [markdown]
# ### Mocks and footprint
# Fits to mocks drawn from symmetric discs, and the effect of the footprint.

# %%
# Mocks drawn from symmetric discs (results/kin_mock_physical.csv, scripts/fig_kin_mock_physical.py)
# fits to mocks drawn from symmetric discs: with proper motions and line-of-sight velocities (pl), with distances (pdist)
mp = pd.read_csv(R / 'kin_mock_physical.csv')
pl = mp[mp.data.isin(['PM + LOS', 'PM + LOS stream'])]
pdist = mp[mp.data.isin(['PM + distances', 'PM + distances stream']) & mp.mock.str.contains('thick')]
pls = mp[mp.data == 'PM + LOS stream']
# largest centre offset [deg] and streaming amplitude in the mocks; sign of m2s; number of realisations
M.update(MockCen=f2(np.hypot(pl.dx, pl.dy).max()), MockCenDist=f2(np.hypot(pdist.dx, pdist.dy).max()),
         MockMTwo=f2(np.hypot(pls.m2c, pls.m2s).max()), MockMsLo=f3(pls.m2s.min()), MockMsHi=signed(pls.m2s.max()),
         MockNPos=num((pls.m2s > 0).sum()), MockNReal=num(len(pls)), NReal=num(mp[mp.mock == 'correlated PM'].realisation.nunique()))
# streaming amplitude |m2| of the data: tilted-ring 'stream' fits and stage-1 ellipsoid fits
tr = pd.read_csv(R / 'kin_tilted_ring.csv')
trg = tr[(tr.model == 'stream') & tr['sample'].str.startswith('giants')]
rng(np.hypot(trg.m2c, trg.m2s), f2, 'DataMTwo')
rng(np.hypot(g1.m2c, g1.m2s), f2, 'DataMTwoE')

# Footprint (results/kin_footprint.csv, scripts/fig_kin_footprint.py): symmetric disc, sky-aligned dispersions, PM + LOS
# footprint test: centre offset east dx [deg] with each bin's own footprint and the matched (common) footprint
fp = pd.read_csv(R / 'kin_footprint.csv')
fp = fp[fp.fit == 'PM + LOS, centre']
own = fp[fp.footprint == 'own'].set_index('sample').dx
mat = fp[fp.footprint == 'matched'].groupby('sample').dx.mean()
rng(own.loc[giants], f2, 'FpOwn')
rng(mat.loc[giants], f2, 'FpMat')
# shift of the two most metal-rich bins between footprints, mean error, and the shift in the mock
rng((mat - own).loc[giants[3:]], f2, 'FpShift')
M.update(FpErr=sigfig(fp[(fp.footprint == 'matched') & fp['sample'].isin(giants[3:])].dx_err.mean()),
         FpMock=f2(mat['mock'] - own['mock']))

# %% [markdown]
# ### Appendix: mocks of the velocity ellipsoid
# Recovery of the input parameters from mock catalogues.

# %%
# Appendix: recovery of the velocity ellipsoid from mocks (results/kin_ellipsoid_mock.csv, scripts/paper_kin_ellipsoid_mock.py)
# ellipsoid mocks: fitted against true dispersions with PM + LOS, PM + LOS with a tanh curve, and proper motions only
em = pd.read_csv(R / 'kin_ellipsoid_mock.csv')
C3 = ('sigma_R', 'sigma_phi', 'sigma_z')
both = em[em.data == 'PM + LOS'].set_index('mock')
tanh = em[em.data == 'PM + LOS, tanh'].set_index('mock')
pm = em[em.data == 'PM'].set_index('mock')
# fitted minus true dispersion per mock [km/s]
dev = {c: both[c] - both[f'truth_{c}'] for c in C3}
# mock settings (stars, line-of-sight error, thickness, scale length, spheroid flattening) and true dispersions; recovery
# errors and posterior errors of the PM + LOS fits; largest change with the tanh curve
M.update(NMock=count(N_STARS), EVlos=num(E_VLOS), HThick=num(H_THICK), RD=num(RD), QSph=num(mock['Q_SPH']), RmsSph=f1(mock['Q_SPH'] * np.sqrt(3) * RD),
         SigHotR=num(HOT[0]), SigHotPhi=num(HOT[1]), SigHotZ=num(HOT[2]), SigCold=num(COLD[0]),
         DSigRLo=f1(dev['sigma_R'].min()), DSigRHi=signed(dev['sigma_R'].max(), f1), DSigPhi=f1(dev['sigma_phi'].abs().max()),
         DSigZ=f1(dev['sigma_z'].abs().max()), DSigRZ=num(np.ceil(max(dev['sigma_R'].abs().max(), dev['sigma_z'].abs().max()))),
         ErrLo=f1(both[[c + '_err' for c in C3]].values.min()), ErrHi=f1(both[[c + '_err' for c in C3]].values.max()),
         DTanh=f1(np.ceil(10 * (tanh[list(C3)] - both[list(C3)]).abs().values.max()) / 10))
# proper motions only, disc mocks: in-plane errors, sigma_phi of the thick mock, and sigma_z
pmd = pm.loc[DISC_MOCKS]
inplane = pd.concat([pmd.sigma_R - pmd.truth_sigma_R, pmd.sigma_phi - pmd.truth_sigma_phi], axis=1)
thick_phi = pmd.sigma_phi['hot thick'] - pmd.truth_sigma_phi['hot thick']
M.update(PmInplane=f1(inplane.abs().values[inplane.abs().values < abs(thick_phi)].max()), PmThickPhi=f1(thick_phi),
         PmSzLo=num(round(pmd.sigma_z.min())), PmSzHi=num(round(pmd.sigma_z.max())), PmTzLo=num(pmd.truth_sigma_z.min()),
         PmTzHi=num(pmd.truth_sigma_z.max()), PmDzLo=num(round((pmd.sigma_z - pmd.truth_sigma_z).min())),
         PmDzHi=num(round((pmd.sigma_z - pmd.truth_sigma_z).max())), PmEzLo=f1(pmd.sigma_z_err.min()), PmEzHi=f1(pmd.sigma_z_err.max()))
# the hot-spheroid mock: dispersion errors and orientation with PM + LOS, and sigma_z, inc with proper motions only
sph, sphp = both.loc['hot spheroid'], pm.loc['hot spheroid']
M.update(SphDSig=f1(np.ceil(10 * max(abs(sph[c] - sph[f'truth_{c}']) for c in C3)) / 10), SphInc=f1(sph.inc),
         TruthInc=num(TRUTH['inc']), SphTheta=num(round(sph.theta)), TruthTheta=num(TRUTH['theta']),
         SphPmSz=num(round(sphp.sigma_z)), SphPmSzErr=num(round(sphp.sigma_z_err)), SphPmInc=num(round(sphp.inc)))
# model comparison: likelihood gain of the velocity ellipsoid at 2-9 deg (giants) and 4-9 deg (RR Lyrae), and its extra k
M.update(DlnLLo=num(round(mc_in.dlnL_max.loc[giants].min())), DlnLHi=count(round(mc_in.dlnL_max.loc[giants].max())),
         DlnLRrlOut=num(round(mc_out.dlnL_max['RR Lyrae'])), DkG=num(mc_in.dk.loc[giants].iloc[0]), DkRrl=num(mc_in.dk['RR Lyrae']))
# likelihood gain per population (letters A-F in the order of pops) for the inner and outer fits
for t, k in ((mc_in, 'In'), (mc_out, 'Out')):
    M.update({f'DlnL{k}{"ABCDEF"[i]}': count(round(v)) for i, v in enumerate(t.dlnL_max.loc[pops])})
M.update(NDraws=num(300))   # posterior draws per model in scripts/kin_model_comparison.py

# %% [markdown]
# ### Appendix: the sky-aligned dispersion model
# Numbers of the appendix on the sky-aligned model.

# %%
# Appendix: the sky-aligned dispersion model
# the hot mock fitted with the sky-aligned model: its sigma (proper motions) and sigma_los
hot = mp[(mp.mock == 'hot') & (mp.data == 'PM + LOS')].iloc[0]
# true inclination [rad] and dispersions (sigma_R, sigma_phi, sigma_z) [km/s] of the hot mock
i0 = np.deg2rad(TRUTH['inc'])
sR, sP, sZ = HOT
# AvgSig, AvgSigLos: proper-motion (per component) and line-of-sight dispersions of the hot ellipsoid at inclination i0,
# averaged over azimuth
M.update(HotSig=f1(hot.sigma), HotSigLos=f1(hot.sigma_los),
         AvgSig=f1(np.sqrt(0.5 * (1 - 0.5 * np.sin(i0) ** 2) * (sR ** 2 + sP ** 2) + 0.5 * sZ ** 2 * np.sin(i0) ** 2)),
         AvgSigLos=f1(np.sqrt(sZ ** 2 * np.cos(i0) ** 2 + 0.5 * np.sin(i0) ** 2 * (sR ** 2 + sP ** 2))))
# the sky-aligned (stream) joint fit at 2-9 deg: centre (SkyDy with the sign flipped), orientation, m2c, and differences
# from the velocity-ellipsoid joint fit
S = pd.read_csv(R / 'kin_mcmc_data_stream.csv')
ssh = S[(S.stage == 2) & (S.population == 'shared')].set_index('param')['median']
M.update(SkyDx=f3(ssh.dx), SkyDy=f3(-ssh.dy), SkyInc=f1(ssh.inc), SkyTheta=f1(ssh.theta), SkyMTwoC=f3(ssh.m2c),
         SkyDDx=f2(abs(ssh.dx - sh.loc['dx', 'median'])), SkyDTheta=f1(abs(ssh.theta - sh.loc['theta', 'median'])))
# v_rot / sigma_los of the first and last giant bins, and the largest APOGEE-only ratio (results/kin_support.csv)
ksup = pd.read_csv(R / 'kin_support.csv').set_index('population')
M.update(RatioLo=f2(ksup.ratio.loc[giants].iloc[0]), RatioHi=f2(ksup.ratio.loc[giants].iloc[-1]), RatioApo=f2(ksup.ratio_apogee.max()))
# tangential residuals of the sky-aligned model beyond 7 deg: metal-rich bins + RR Lyrae, and metal-poor bins
rs = pd.read_csv(R / 'kin_residual_radial.csv')
rs = rs[(rs.component == 'v_t') & (rs.rho_lo >= RINGS['Tanh'])]
rng(rs[rs.population.isin(giants[2:] + ['RR Lyrae'])]['median'], lambda x: num(round(x)), 'TanhRich')
rng(rs[rs.population.isin(giants[:2])]['median'], f1, 'TanhPoor')
# v0 [km/s] of each population in the 4-9 deg sky-aligned fit, and stars per giant bin at 4-9 deg
SO = pd.read_csv(R / 'kin_mcmc_data_stream_outer.csv')
M.update({f'VZeroOut{"ABCDEF"[i]}': f1(stage(SO, 1, p, 'v0')) for i, p in enumerate(pops)})
M.update({f'NOut{"ABCDE"[i]}': count(mc_out.n[p]) for i, p in enumerate(giants)})

# %% [markdown]
# ### Captions
# The centres of the joint fits and the contour levels of the corner plot.

# %%
# Centres of the joint fits [deg] (map captions)
# joint-fit centre (RA, Dec) [deg] of the ellipsoid (E) and sky-aligned (S) models
for t, k in ((sh['median'], 'E'), (ssh, 'S')):
    ra, dec = (float(np.rad2deg(v)) for v in kinfit.centre(t['dx'], t['dy']))
    M.update({f'Ra{k}': f2(ra), f'Dec{k}': f2(dec)})

# Corner-plot contour levels: mass inside 1 and 2 sigma of a 2D Gaussian (paper_kin_corner.py LEVELS)
M.update(LevOne=num(round(100 * (1 - np.exp(-0.5)))), LevTwo=num(round(100 * (1 - np.exp(-2)))))

# %% [markdown]
# ### Write the macros
# One \newcommand per entry of M, with the prefix \km, into results/macros_kinmodel.tex; the page text shows them through
# {glue:text}, so no number is typed by hand.

# %%
# header line, then one \newcommand per macro
lines = ['% Numbers of the KB chapter Kinematics: velocity field and disc model (scripts/paper_text_kin_model.py); do not edit.']
lines += [rf'\newcommand{{\km{k}}}{{{x}}}' for k, x in M.items()]
(R / 'macros_kinmodel.tex').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
