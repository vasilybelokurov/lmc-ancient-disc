"""Knowledge-base text, chapter 'Shape of the old population' (extended draft Section 5 and its appendix): every number of the text as
a LaTeX macro (prefix \\sh), so that none is typed by hand.

As scripts/paper_text_letter.py: settings are read from the code (scripts/textnum.const, lmcold by import), results from the tables
the figures are drawn from; the effect of the m=2 streaming on q is computed here from the posterior medians of the ellipsoid fits
(data/kin_mcmc_data_ellipsoid_samples.npz) with lmcold.virial (about 10 s). Literature values quoted with their citation stay in the text.
Inputs: results/{shape_virial, shape_virial_sims, kin_vc_drift_h, kin_support_ellipsoid, kin_mcmc_data_ellipsoid,
kin_mcmc_data_ellipsoid_outer, shape_rrl_cells, shape_rrl_fit_stats, shape_rrl_harmonics, shape_rrl_triaxial, shape_rrl_depth,
kin_ellipsoid_mock}.csv; scripts/{shape_virial, shape_rrl_cells, shape_rrl_triaxial, paper_kin_ellipsoid_mock, fig_kin_mock_physical}.py;
lmcold.virial (potentials), lmcold.shape (sky cells); tests/test_virial.py (accuracy of the grid potential).
Writes results/macros_shape.tex.
Run: .venv/bin/python scripts/manifest.py run paper_text_shape.py"""

# %% [markdown]
# ### Settings and helpers
# Imports, the dictionary M of macros (macro name without its prefix: the text it stands for) and the functions that format
# numbers as the text writes them. The inputs listed in the header are read in the step that uses them.

# %%
import ast
import re

import numpy as np
import pandas as pd

from lmcold import kinfit, shape, sims, spheroid, virial
from lmcold.samples import FEH_BINS
from lmcold.sources import ROOT
from fig_kin_mock_physical import H_THICK, HOT, RD, TRUTH
from shape_rrl_triaxial import C_ALL, CASES
from shape_virial import draws
from textnum import const, num

R = ROOT / 'results'   # directory of the input tables and of the output macros
M = {}                 # macro name (without the prefix \sh) -> text


def f2(x):
    """x with two decimals."""
    return f'{x:.2f}'


def f1(x):
    """x with one decimal."""
    return f'{x:.1f}'


def signed(x, fmt=f2):
    """x formatted by fmt with an explicit sign ('+' added if not negative)."""
    s = fmt(x)
    return s if s.startswith('-') else '+' + s


# %% [markdown]
# ### Settings
# Settings read from the shape and mock scripts, and the parameters of the potentials P1-P3 from the source of lmcold.virial.

# %%
# Settings
# Literal module constants of the shape and mock scripts (const reads them without running the script); the triaxial-mock grid
# and the physical-mock settings are imported.
sv, cells = const('shape_virial'), const('shape_rrl_cells')
tri = dict(C_ALL=C_ALL, CASES=CASES)
mock = const('paper_kin_ellipsoid_mock')
phys = dict(H_THICK=H_THICK, HOT=HOT, RD=RD, TRUTH=TRUTH)
pot = {}   # numbers of lmcold.virial.POTENTIALS, from its source: P1 (V, a), P2 (V, a, q_Phi), P3 (M, a, b, V, a)
# Walk the module syntax tree of lmcold.virial: the literal arguments of the calls inside the POTENTIALS dict, per potential.
for n in ast.parse((ROOT / 'src/lmcold/virial.py').read_text()).body:
    if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'POTENTIALS':
        for k, v in zip(n.value.keys, n.value.values):
            pot[k.value] = [a.value for call in ast.walk(v) if isinstance(call, ast.Call) for a in call.args if isinstance(a, ast.Constant)]
M.update(VOne=num(pot['P1'][0]), AOne=num(pot['P1'][1]), QPhi=num(pot['P2'][2]), MMN=num(pot['P3'][0] / 1e9), AMN=num(pot['P3'][1]),
         BMN=num(pot['P3'][2]), VThree=num(pot['P3'][3]), AThree=num(pot['P3'][4]), NDraw=num(sv['NDRAW']), RFloor=num(sv['R_FLOOR']))
VC_R = (2.0, 8.0)   # kpc, radii over which the circular speeds of the three potentials are quoted
Rv = np.linspace(*VC_R, 13)   # radii [kpc] in the disc plane
vc = [np.sqrt(Rv * g(np.column_stack([Rv, 0 * Rv, 0 * Rv]))[:, 0]) for g in virial.POTENTIALS.values()]   # V_c = sqrt(R dPhi/dR)
M.update(VcLo=num(5 * round(min(v.min() for v in vc) / 5)), VcHi=num(5 * round(max(v.max() for v in vc) / 5)))   # nearest 5 km/s
# Radii, simulation and cylinder settings, [Fe/H] bin edges; Hernquist test flattening, tolerance [%] from tests/test_virial.py.
M.update(VcRIn=num(VC_R[0]), VcROut=num(VC_R[1]), RMaxSim=num(sims.R_MAX), CylR=num(spheroid.CYL[0]), CylZ=num(spheroid.CYL[1]),
         FehMP=num(FEH_BINS[1][1]), FehMR=num(FEH_BINS[-1][0]),
         HernQ=num(float(re.search(r'spheroid\.spheroid\([0-9.]+, ([0-9.]+),', (ROOT / 'tests/test_virial.py').read_text()).group(1))),
         NSrc=f'{virial.NSRC:,}'.replace(',', '\\,'),
         HernTol=num(100 * float(re.search(r'hernquist\(y\)\.T\) - 1\) < ([0-9.]+)', (ROOT / 'tests/test_virial.py').read_text()).group(1))))

# %% [markdown]
# ### Flattening implied by the motions
# The axis ratio q of each bin from the tensor virial theorem, and the change of q when the m = 2 streaming is included.

# %%
# Flattening implied by the motions (results/shape_virial.csv; bins 0-4 = giants by [Fe/H], RR Lyrae)
v = pd.read_csv(R / 'shape_virial.csv')
pops = list(v.population.unique())   # population names in file order
g0 = v[(v.r_floor == 0) & v.population.str.startswith('giants')]   # giants, no central floor
# Giant bin i -> its rows indexed by potential (P1, P2, P3); 0 is the most metal-poor, 4 the most metal-rich.
b = {i: g0[g0.population == p].set_index('potential') for i, p in enumerate(pops[:5])}
half = ((b[0].q_p84 - b[0].q_p16) / 2).mean()   # mean half-width of the 16th-84th interval of q, most metal-poor bin
M.update(QMPOne=f2(b[0].q.P1), QMPTwo=f2(b[0].q.P2), QMPThree=f2(b[0].q.P3), QHalf=f2(half),
         QMRLo=f2(b[4].q.min()), QMRHi=f2(b[4].q.max()), DqPotMP=f2(b[0].q.max() - b[0].q.min()),
         DqRatio=num(round((b[0].q.max() - b[0].q.min()) / half)), DqPotMR=f2(b[4].q.max() - b[4].q.min()),
         QOldLo=f2(min(b[0].q.min(), b[1].q.min())), QOldHi=f2(max(b[0].q.max(), b[1].q.max())))
# 'balance' column of shape_virial.csv: range over bins 0-3 in P1, P2; bin 4 mean in P1, P2; range over all bins in P3.
low = pd.concat([b[i] for i in range(4)])
M.update(BalLo=f2(low.loc[['P1', 'P2']].balance.min()), BalHi=f2(low.loc[['P1', 'P2']].balance.max()),
         BalMR=f2(b[4].loc[['P1', 'P2']].balance.mean()),
         BalThreeLo=f2(pd.concat([b[i] for i in range(5)]).loc['P3'].balance.min()),
         BalThreeHi=f2(pd.concat([b[i] for i in range(5)]).loc['P3'].balance.max()))
# q with and without the central floor R_FLOOR: largest change over giant bins and potentials.
fl = v[v.population.str.startswith('giants')].pivot_table(index=['population', 'potential'], columns='r_floor', values='q')
M.update(DqFloor=f'{(fl[sv["R_FLOOR"]] - fl[0.0]).abs().max():.3f}')
h = pd.read_csv(R / 'kin_vc_drift_h.csv').set_index('population')   # scale length h [kpc] per population
M.update(HMP=f1(h.h.iloc[0]))
# The m=2 streaming of the fits (m2c, m2s of v_phi; r2c, r2s of v_R, in units of the rotation speed; lmcold.kinfit), which the
# realisation (lmcold.spheroid.velocities) omits: averaged over azimuth it adds v_rot^2 (m2c^2 + m2s^2 + r2c^2 + r2s^2) / 2 to the
# in-plane v^2. Fraction f of the in-plane kinetic energy and change of q when Omega_kin is raised by (1 + f), for the
# posterior-median model of each giant bin (as scripts/shape_virial.py, no floor) in each potential (same positions for both q).
f_stream, dq_stream = [], []
for i in range(5):
    # Population name and its posterior-median parameter vector (shape_virial.draws).
    pop, med, _ = draws('kin_mcmc_data_ellipsoid_samples.npz', i, np.random.default_rng(0))
    # Realisation: NREAL positions X [kpc] of a spheroid with scale length h and axis ratio 1.0, velocities V [km/s], shape (N, 3).
    rg = np.random.default_rng(1)
    X = spheroid.spheroid(h.h[pop], 1.0, virial.NREAL, rg)
    V = spheroid.velocities(med, X, rg)
    vrot = np.asarray(kinfit.rotation(med, np.hypot(X[:, 0], X[:, 1])))   # mean rotation [km/s] at each in-plane radius
    # Azimuthal average of the squared m = 2 amplitudes / 2 (units of v_rot^2); f = added / existing in-plane kinetic energy.
    s2 = sum(med[kinfit.PARAMS.index(n)] ** 2 for n in ('m2c', 'm2s', 'r2c', 'r2s')) / 2
    f = np.sum(vrot ** 2) * s2 / np.sum(V[:, 0] ** 2 + V[:, 1] ** 2)
    om = virial.omega_kin(V)   # Omega_kin = in-plane / (2 x vertical) kinetic energy of the realisation (lmcold.virial)
    f_stream.append(f)
    # q solved with Omega_kin and with Omega_kin (1 + f), in each potential; change dq.
    for grad in virial.POTENTIALS.values():
        q0, q1 = (virial.solve_q(t, h.h[pop], grad, np.random.default_rng(2)) for t in (om, om * (1 + f)))
        dq_stream.append(q1 - q0)
M.update(StreamKE=f1(100 * max(f_stream)), DqStream=f'{max(np.abs(dq_stream)):.3f}')

# %% [markdown]
# ### Test on the simulated analogues
# The virial axis ratio against the true one in the simulations.

# %%
# Test on the simulated analogues (results/shape_virial_sims.csv)
s = pd.read_csv(R / 'shape_virial_sims.csv')
d = s.q_model_sim - s.q_true   # virial q (simulation's own potential) minus true q
M.update(NSimGal=num(s.name.nunique()), NSimBin=num(len(s)), QTrueLo=f2(s.q_true.min()), QTrueHi=f2(s.q_true.max()),
         DqSim=f2(d.median()), DqSimLo=f2(d.quantile(.16)), DqSimHi=f2(d.quantile(.84)), DqSimMin=f2(d.min()), DqSimMax=signed(d.max()))
# The same with each model potential P1-P3, signed.
for k, P in zip(('One', 'Two', 'Three'), ('P1', 'P2', 'P3')):
    e = s[f'q_model_{P}'] - s.q_true
    M.update({f'DqSim{k}': signed(e.median()), f'DqSim{k}Lo': signed(e.quantile(.16)), f'DqSim{k}Hi': signed(e.quantile(.84))})

# %% [markdown]
# ### Kinematic geometry
# Inclinations and other geometry of the stage-1 fits quoted in the text.

# %%
# Kinematic geometry quoted (results/kin_support_ellipsoid.csv, kin_mcmc_data_ellipsoid*.csv: stage-1 fits)
ks = pd.read_csv(R / 'kin_support_ellipsoid.csv').set_index('population')
km = pd.read_csv(R / 'kin_mcmc_data_ellipsoid.csv')
km = km[km.stage == 1].pivot_table(index='population', columns='param', values='median')   # population x parameter medians
old = list(cells['OLD'])   # the two metal-poor giant bins (OLD of scripts/shape_rrl_cells.py)
inc_old, theta_old = km.loc[old, 'inc'].mean(), km.loc[old, 'theta'].mean()
inc_rich = km.loc[[p for p in pops[2:5]], 'inc'].mean()
# RR Lyrae inclination from the 4-9 deg ('_outer') ellipsoid fit, stage 1.
ko = pd.read_csv(R / 'kin_mcmc_data_ellipsoid_outer.csv')
inc_rrl = ko[(ko.stage == 1) & (ko.population == 'RR Lyrae') & (ko.param == 'inc')]['median'].iloc[0]
M.update(FehRrl=f2(ks.feh_med['RR Lyrae']), SigzMP=num(round(ks.sigma_z.iloc[0])), IncOld=f1(inc_old), IncOldInt=num(round(inc_old)), ThetaOld=num(round(theta_old)),
         IncRich=num(round(inc_rich)), IncRrlPM=f1(inc_rrl), CosIncOld=f2(np.cos(np.deg2rad(inc_old))))

# %% [markdown]
# ### RR Lyrae sky cells
# Fits of the RR Lyrae distances in sky cells, the triaxial mocks and the depth gradient.

# %%
# RR Lyrae sky cells (results/shape_rrl_*.csv; lmcold.shape)
st = pd.read_csv(R / 'shape_rrl_fit_stats.csv').set_index('fit')            # fit statistics (star and cell counts, chi^2, ...)
fr = pd.read_csv(R / 'shape_rrl_cells.csv').query("fit == 'free'").iloc[0]  # free oblate fit: inc [deg], q, theta [deg]
t = pd.read_csv(R / 'shape_rrl_triaxial.csv')                               # oblate fits of the mocks; the data row has c NaN
data = t[t.c.isna()].iloc[0]                                                # observed q_sky and depth gradient g
M.update(NRrl=f'{int(st.n_stars.free):,}'.replace(',', '\\,'), NCells=num(st.n_cells.free), NRings=num(len(shape.CELL_RHO) - 1),
         NSect=num(shape.CELL_NPA), RhoIn=num(shape.CELL_RHO[0]), RhoOut=num(shape.CELL_RHO[-1]),
         QSky=f2(data.q_sky), G=f2(data.g), QSkyThree=f'{data.q_sky:.3f}', GThree=f'{data.g:.3f}',
         IncFree=num(round(fr.inc)), IncFreeOne=f1(fr.inc), QFree=f2(fr.q), ThetaFree=num(round(fr.theta)),
         ChiCounts=num(round(st.chi2_counts.free)), ChiMed=num(round(st.chi2_medians.free)),
         PearsonMin=f1(st.pearson_min.free), PearsonMax=signed(st.pearson_max.free, f1),
         Scatter=f'{st.rms_modulus.free:.3f}', MedErr=f'{st.median_se.free:.3f}')
hm = pd.read_csv(R / 'shape_rrl_harmonics.csv')   # m = 1 harmonic a1, pa1 [deg] and coverage per ring rho_lo-rho_hi [deg]
out, inner = hm[hm.rho_lo >= 5], hm.iloc[0]       # rings from 5 deg out, and the innermost ring
M.update(AOneOutLo=num(round(100 * out.a1.min())), AOneOutHi=num(round(100 * out.a1.max())),
         AOneOutLoF=f2(out.a1.min()), AOneOutHiF=f2(out.a1.max()), AOneIn=f2(inner.a1), RhoInHi=num(inner.rho_hi),
         RhoOutLo=num(out.rho_lo.min()), PAOneLo=num(round(out.pa1.min())), PAOneHi=num(round(out.pa1.max())),
         Coverage=f'{100 * hm.coverage.min():.1f}')
# The elongated mock shown in figure paper_shape_rrl (c/a 0.41, b/a 0.85, angle 150 deg: ARROW of scripts/paper_shape_rrl.py).
mk = t[(t.c == 0.41) & (t.ba == 0.85) & (t.phi == 150)].iloc[0]
M.update(MockC=num(mk.c), MockBA=num(mk.ba), MockPhi=num(mk.phi), MockInc=num(round(mk.inc)), MockIncOne=f1(mk.inc), MockQ=f2(mk.q),
         MockTheta=num(round(mk.theta)), MockQSky=f2(mk.q_sky), MockQSkyThree=f'{mk.q_sky:.3f}', MockG=f2(mk.g), MockGThree=f'{mk.g:.3f}',
         CsOne=num(tri['C_ALL'][0]), CsTwo=num(tri['C_ALL'][1]), BAOne=num(tri['CASES'][1][0]), BATwo=num(tri['CASES'][-1][0]),
         PhiStep=num(tri['CASES'][2][1] - tri['CASES'][1][1]), PhiMin=num(tri['CASES'][1][1]), PhiMax=num(tri['CASES'][-1][1]))
el = t[t.c.notna() & (t.ba < 1)]   # elongated (triaxial) mocks
NEAR, FAR = (0, 30, 150), (60, 90, 120)   # deg: major axis within 30 deg of the line of nodes, and 60-120 deg from it
near = el[el.phi.isin(NEAR)]
# FAR excludes the mocks with c = C_ALL[0], b/a = CASES[1][0] at 60 and 120 deg (the code gives no reason).
far = el[el.phi.isin(FAR) & ~((el.c == tri['C_ALL'][0]) & (el.ba == tri['CASES'][1][0]) & el.phi.isin([60, 120]))]
ob = t[t.c.notna() & (t.ba == 1)].set_index('c')   # oblate mocks (b/a = 1), indexed by c/a
M.update(NearIncLo=num(round(near.inc.min())), NearIncHi=num(round(near.inc.max())), NearQLo=f2(near.q.min()), NearQHi=f2(near.q.max()),
         NearQSkyLo=f2(near.q_sky.min()), NearQSkyHi=f2(near.q_sky.max()), FarIncLo=num(round(far.inc.min())),
         FarIncHi=num(round(far.inc.max())), FarQLo=f2(far.q.min()), FarQHi=f2(far.q.max()),
         ObIncOne=num(round(ob.inc[tri['C_ALL'][0]])), ObQOne=f2(ob.q[tri['C_ALL'][0]]), ObIncTwo=num(round(ob.inc[tri['C_ALL'][1]])),
         ObQTwo=f2(ob.q[tri['C_ALL'][1]]), FarPhiLo=num(min(FAR)), FarPhiHi=num(max(FAR)))
dp = pd.read_csv(R / 'shape_rrl_depth.csv')   # q from the depth gradient: case 'data', 'bootstrap' draws, 'dipole*' variants
bs = dp[dp.case == 'bootstrap'].q
M.update(QDepth=f2(dp[dp.case == 'data'].q.iloc[0]), QDepthLo=f2(bs.quantile(.16)), QDepthHi=f2(bs.quantile(.84)),
         QDipLo=f2(dp[dp.case.str.startswith('dipole')].q.min()), QDipHi=f2(dp[dp.case.str.startswith('dipole')].q.max()),
         Dipole=num(cells['DIPOLE']))

# %% [markdown]
# ### The shape and the kinematic model
# Mocks of the velocity-ellipsoid model.

# %%
# The shape and the kinematic model (results/kin_ellipsoid_mock.csv; scripts/paper_kin_ellipsoid_mock.py, fig_kin_mock_physical.py)
mo = pd.read_csv(R / 'kin_ellipsoid_mock.csv')
sph = mo[mo.mock == 'hot spheroid'].set_index('data')   # fits to the hot-spheroid mock, by data used
both, pm = sph.loc['PM + LOS'], sph.loc['PM']          # with proper motions and velocities, and with proper motions only
# Largest error [km/s] of the three dispersions in the PM + LOS fit.
dsig = max(abs(both[c] - both[f'truth_{c}']) for c in ('sigma_R', 'sigma_phi', 'sigma_z'))
M.update(QSph=num(mock['Q_SPH']), RD=num(phys['RD']), HThick=num(phys['H_THICK']), RmsSph=f1(mock['Q_SPH'] * np.sqrt(3) * phys['RD']),
         RmsMP=f1(0.5 * np.sqrt(3) * h.h.iloc[0]), SigR=num(phys['HOT'][0]), SigPhi=num(phys['HOT'][1]), SigZ=num(phys['HOT'][2]),
         IncMock=num(phys['TRUTH']['inc']), DSigMock=f1(np.ceil(10 * dsig) / 10), DIncMock=f1(abs(both.inc - phys['TRUTH']['inc'])),
         DThetaMock=num(round(phys['TRUTH']['theta'] - both.theta)), SigzPM=num(round(pm.sigma_z)), SigzPMErr=num(round(pm.sigma_z_err)),
         IncPM=num(round(pm.inc)))

# %% [markdown]
# ### Write the macros
# One \newcommand per entry of M, with the prefix \sh, into results/macros_shape.tex; the text shows them as macros, so no
# number is typed by hand.

# %%
# Header line, then one \newcommand per macro; written to results/macros_shape.tex and printed.
lines = ['% Numbers of the KB chapter Shape (scripts/paper_text_shape.py); do not edit.']
lines += [rf'\newcommand{{\sh{k}}}{{{x}}}' for k, x in M.items()]
(R / 'macros_shape.tex').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
