"""The shared-v_c + drift model: one fitted circular-speed curve v_c(R) for all giant groups, each group's mean rotation lagging it
by its asymmetric drift, computed from the group's own dispersions and density scale length with the radial Jeans equation
(lmcold.kinfit.rotation with h > 0). v_c(R) is a parametrised curve fitted to the kinematics, not derived from a mass model.

Steps
1. Density scale lengths h (fixed in the fits): for each giant [Fe/H] bin (measured [Fe/H], lmcold.samples.FEH_BINS) the
   selection-weighted surface density of the BOSS giants (weight w x coverage fill x (1 - Milky Way fraction), as in
   scripts/paper_feh_profile.py) in 1-kpc annuli of in-plane radius R, with R in the frame of the joint ellipsoid fit
   (results/kin_mcmc_data_ellipsoid.csv, stage 2); an exponential (straight line in ln Sigma, unweighted) over H_FIT kpc; errors from
   NBOOT bootstraps of nside-64 sky pixels. The RR Lyrae members the same way (weight 1). The Jeans equation holds for any tracer
   selected independently of velocity, so each measured bin uses its own density, with no deconvolution of [Fe/H] errors.
2. Mocks (MOCK_GROUPS, sharing MOCK_VC): thin exponential discs of scale length h, in the joint-fit orientation, stars at
   2 < rho < 9 deg, velocities from the aligned ellipsoid with sigma_i exp(gamma (R - 4 kpc)) about the exact drift mean
   (no floor: every star must have vbar^2 > 0), proper-motion errors of the nearest real giant (Gaia covariance), line-of-sight
   errors E_VLOS; NREAL realisations. Joint maximum-likelihood fits (geometry, systemic motion and streaming shared and free):
   'drift' (one curve v0, R0, vslope shared; h fixed at the truth), 'drift, h free' and 'free' (own v0, R0, vslope per group).
3. Data: the five giant bins at 2 < rho < 9 deg (as fit_kinematics_mcmc.py), joint maximum likelihood of 'free' and 'drift' (h from
   step 1) on the same stars, starting from the stage-2 ellipsoid medians; the number of stars with vbar^2 <= 0 is reported.
4. RR Lyrae (not fitted): their mean rotation predicted from the data v_c(R) and their own ellipsoid (stage 1 of the 4-9 deg run,
   draw by draw) and h, against their measured v_rot(4 kpc).

Joint fits are restarted from their own optimum (RESTARTS) until -lnL improves by less than 0.01.

Writes results/kin_vc_drift_h.csv (population, h, h_p16, h_p84 [kpc]), results/kin_vc_drift_mock.csv (realisation, model, nll, v_c(4),
V_c, R_c, s_c, and per group vrot4 and sigma_R, sigma_phi, gamma), results/kin_vc_drift_data.csv (model, nll, n_neg, per group
vrot4 free and drift-predicted, sigma_R, sigma_phi, sigma_z, gamma, k, and the shared v_c parameters) , results/kin_vc_drift_rrl.csv (RR Lyrae v_rot(4 kpc) predicted and measured: p16, median, p84); the figures
are made by scripts/fig_kin_vc_drift.py and the numbers of the KB page 'Rotation and the circular speed' by
scripts/paper_text_vc_drift.py. Diagnostic (not in the Letter). Its joint_params() is also imported by scripts/sims_observe.py,
and its fit helpers by scripts/kin_vc_drift_extensions.py.

Inputs: the release giants (lmcold.samples.giants(), Zenodo bundle 1); the BOSS mass sample (lmcold.mass_samples.samples) with
data/giant_foreground.parquet (coverage fill and Milky Way fraction); data/rrab.parquet (RR Lyrae members); the joint ellipsoid
fit results/kin_mcmc_data_ellipsoid.csv and the posterior draws data/kin_mcmc_data_ellipsoid_outer_samples.npz
(scripts/fit_kinematics_mcmc.py).

Run: .venv/bin/python scripts/kin_vc_drift.py   (run time not recorded)
"""
# %% [markdown]
# ### Settings
# The annuli and radial range of the scale-length fits, the parameters shared or fitted per group in the joint fits, and the
# circular speed and groups of the mocks.

# %%
import healpy as hp
import jax
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from fit_kinematics_mcmc import COLS, ring
from lmcold import geometry as geo
from lmcold import kinfit, samples
from lmcold.constants import KAPPA, LMC
from lmcold.footprint import angular_distance
from lmcold.mass_samples import samples as mass_samples
from lmcold.sources import DATA, ROOT

P = kinfit.PARAMS.index  # P(name): position of a parameter in a kinfit.PARAMS vector
R_EDGES = np.arange(0., 11., 1.)  # kpc, annuli of the surface densities
H_FIT = (2., 8.)  # kpc, range of the exponential fits (inside 2 kpc the bar; RR Lyrae cover all azimuths to 8 kpc)
NBOOT = 100  # bootstraps of the sky pixels for the errors of h
SHARED_FREE = ('dx', 'dy', 'pmra0', 'pmdec0', 'inc', 'theta', 'vlos0') + kinfit.STREAM  # free and shared in every joint fit
CURVE = ('v0', 'R0', 'vslope')  # the rotation curve: shared (v_c) in 'drift', own in 'free'
OWN = kinfit.ELLIPSOID + ('k', 'logit_fout')  # per group always: sigma_R, sigma_phi, sigma_z [km/s], dlnsig_dR [1/kpc], error scale, outliers
MOCK_VC = dict(v0=85.0, R0=2.0, vslope=-0.02)  # mock circular speed: v0 [km/s], R0 [kpc], vslope [1/kpc]
# Mock groups, hot to cold: ellipsoid at R_SIG [km/s], its radial gradient dlnsig_dR [1/kpc], density scale length h [kpc].
MOCK_GROUPS = [dict(sigma_R=38., sigma_phi=31., sigma_z=32., dlnsig_dR=-0.05, h=2.0),
               dict(sigma_R=30., sigma_phi=25., sigma_z=24., dlnsig_dR=-0.07, h=1.6),
               dict(sigma_R=21., sigma_phi=17., sigma_z=16., dlnsig_dR=-0.08, h=1.2)]
N_MOCK, NREAL, E_VLOS = 4000, 3, 4.0  # stars per mock group, realisations, line-of-sight velocity error [km/s]
RESTARTS = 5  # restarts of a joint fit from its own optimum, until -lnL changes by < 0.01
rng = np.random.default_rng(7)  # used by the bootstraps of scale_length


# %% [markdown]
# ### Helpers
# Reading the joint-fit frame, radii in the disc plane, the scale-length fit, the joint maximum-likelihood fit of several groups,
# and small functions for the drift model and the mocks.

# %%
def joint_params(csv='kin_mcmc_data_ellipsoid.csv'):
    """Stage-2 posterior medians of results/<csv>: dict population -> kinfit.PARAMS vector (shared plus own)."""
    # stage 2 = the joint fit: rows of the shared parameters (population 'shared') and of each population's own parameters
    t = pd.read_csv(ROOT / 'results' / csv)
    t = t[t.stage == 2]
    out = {}
    # one vector per population: the shared values plus its own, other parameters at kinfit.start()
    for pop in t.population.unique():
        if pop == 'shared':
            continue
        p = kinfit.start()
        for _, r in t[t.population.isin(['shared', pop])].iterrows():
            p[P(r.param)] = r['median']
        out[pop] = p
    return out


def radius(p, ra, dec):
    """In-plane radius R [kpc] of sky positions in the frame of parameter vector p.

    p : kinfit.PARAMS vector (centre, inc, theta used); ra, dec : arrays of shape (N,) [deg]. Returns R, shape (N,).
    """
    # kinfit.data needs proper motions and errors; dummy values (zeros, unit errors) suffice for the geometry
    n = len(ra)
    d = kinfit.data(ra, dec, np.zeros(n), np.zeros(n), np.ones(n), np.ones(n), np.zeros(n))
    # element 4 of kinfit._model's output: in-plane radius R [kpc] of each star
    return np.asarray(kinfit._model(p, d['rhat'], d['ea'], d['ed'])[4])


def scale_length(R, w, pix):
    """Exponential scale length [kpc] of the weighted surface density over H_FIT, and its 16th, 84th bootstrap percentiles.

    R : in-plane radii [kpc], w : weights (selection-weighted counts), pix : HEALPix pixel of each star (the bootstrap resamples
    whole pixels); all shape (N,). Returns (h, h_p16, h_p84) [kpc]: h = -1 / slope of ln Sigma against R in the annuli R_EDGES.
    """
    # annulus centres [kpc] and areas [kpc^2]; m: annuli inside H_FIT
    mid, area = .5 * (R_EDGES[1:] + R_EDGES[:-1]), np.pi * np.diff(R_EDGES ** 2)
    m = (mid > H_FIT[0]) & (mid < H_FIT[1])

    def h(weights):
        # weighted surface density per annulus [weight / kpc^2]
        sigma = np.histogram(R, R_EDGES, weights=weights)[0] / area
        k = m & (sigma > 0)  # annuli with stars (a compact simulated population can leave the outer ones empty)
        if k.sum() < 2:
            return np.nan
        # h = -1 / slope of the straight-line fit of ln Sigma against R
        return -1 / np.polyfit(mid[k], np.log(sigma[k]), 1)[0]

    # bootstrap over HEALPix pixels: weights times each pixel's multiplicity in a resampling of the pixels with replacement
    upix, inv = np.unique(pix, return_inverse=True)
    boot = [h(w * np.bincount(rng.integers(0, len(upix), len(upix)), minlength=len(upix))[inv]) for _ in range(NBOOT)]
    return h(w), *np.percentile(boot, [16, 84])


def joint_fit(ds, p0s, shared, extra=(), own=CURVE + OWN, add=None):
    """Joint maximum likelihood (ellipsoid likelihood) of populations ds: names in `shared` common (taken from p0s[0]), `own`
    (CURVE unless shared, OWN) and `extra` (e.g. ('h',)) per population, the rest held at p0s.

    add : (names, starts, bounds, adjust) or None
        Extra shared scalars outside PARAMS (e.g. streaming slopes, a thickness coefficient) and a function adjust(vectors, values)
        returning the population vectors they modify.

    Returns (vectors, errors (NaN: no Hessian), nll, info): info has 'extra' (the fitted add values), 'success' (L-BFGS-B
    converged on the last restart) and 'grad' (max |projected gradient| at the optimum).
    """
    # parameters fitted per population: `own` and `extra`, unless shared
    own = [n for n in tuple(own) + tuple(extra) if n not in shared]
    # whether each population has line-of-sight velocities; vlos0 is fitted only if one has
    has_los = [bool(np.any(np.asarray(d['los']))) for d in ds]
    sh = [n for n in shared if n != 'vlos0' or any(has_los)]
    # positions of the shared and own parameters in a kinfit.PARAMS vector
    idx_sh, idx_own = np.array([P(n) for n in sh], dtype=int), np.array([P(n) for n in own], dtype=int)
    names, starts, add_bounds, adjust = add if add else ((), (), (), None)
    n_add = len(names)
    # x0: [shared values from p0s[0], own values of each population in turn, extra shared scalars]
    x0 = np.concatenate([p0s[0][idx_sh]] + [p[idx_own] for p in p0s] + [np.asarray(starts, float)])
    # bounds in the same order; the start is moved just inside them
    bounds = [kinfit.BOUNDS[n] for n in sh] + [kinfit.BOUNDS[n] for n in own] * len(ds) + list(add_bounds)
    x0 = np.clip(x0, [b[0] + 1e-6 for b in bounds], [b[1] - 1e-6 for b in bounds])

    def vectors(x):
        # vectors(x): the full kinfit vector of each population from the packed parameter array x
        out = []
        for i, p in enumerate(p0s):
            q = kinfit.jnp.asarray(p).at[idx_sh].set(x[:len(sh)])
            out.append(q.at[idx_own].set(x[len(sh) + i * len(own): len(sh) + (i + 1) * len(own)]))
        return adjust(out, x[len(x) - n_add:]) if n_add else out

    # f: total negative log-likelihood of all populations (ellipsoid likelihood), jit-compiled; g: its gradient
    f = jax.jit(lambda x: sum(kinfit.neg_loglike(q, d, True) for q, d in zip(vectors(x), ds)))
    g = jax.jit(jax.grad(f))
    x, last = x0, np.inf
    for _ in range(RESTARTS):  # L-BFGS-B can stop at its iteration limit on these large joint fits: restart until it stalls
        r = minimize(lambda q: float(f(q)), x, jac=lambda q: np.asarray(g(q)), method='L-BFGS-B', bounds=bounds,
                     options=dict(ftol=1e-15, gtol=1e-6, maxiter=20000, maxfun=50000))
        if r.fun < last:  # keep only improvements
            x = r.x
        if last - r.fun < 0.01:
            break
        last = min(last, r.fun)
    # No Hessian: for ~57 parameters over ~57 000 stars it exhausts memory (the process aborts); errors come from NUTS.
    # projected gradient at the optimum: components at a bound pointing outward set to zero
    lo, hi = np.array(bounds).T
    gr = np.asarray(g(x))
    proj = np.where((x <= lo + 1e-9) & (gr > 0) | (x >= hi - 1e-9) & (gr < 0), 0.0, gr)  # gradient pointing out of the box ignored
    info = dict(extra=np.asarray(x[len(x) - n_add:]) if n_add else np.array([]), success=bool(r.success), grad=float(np.abs(proj).max()))
    # the population vectors at the optimum; errors returned as NaN (see above)
    ps = [np.array(q) for q in vectors(x)]
    return ps, [np.full(len(kinfit.PARAMS), np.nan)] * len(ds), float(f(x)), info


def n_negative(p, d):
    """Stars whose asymmetric-drift mean has vbar^2 <= 0 (no equilibrium) for vector p (h > 0)."""
    # in-plane radius R [kpc] of each star
    R = np.asarray(kinfit._model(p, d['rhat'], d['ea'], d['ed'])[4])
    # circular speed v_c(R) [km/s] of the parametrised curve (tanh rise times a linear slope about R_SIG)
    vc = p[P('v0')] * np.tanh(R / p[P('R0')]) * (1 + p[P('vslope')] * (R - kinfit.R_SIG))
    # g: radial gradient of ln sigma [1/kpc]
    g = p[P('dlnsig_dR')]
    # u2 = vbar^2 from the radial Jeans equation for an exponential density of scale length h and dispersions exp(g (R - R_SIG))
    u2 = vc ** 2 - np.exp(2 * g * (R - kinfit.R_SIG)) * (p[P('sigma_phi')] ** 2 + p[P('sigma_R')] ** 2 * (R / p[P('h')] - 1 - 2 * g * R))
    return int((u2 <= 0).sum())


def vrot4(p):
    """Mean rotation [km/s] at R_SIG of vector p (free curve, or drift mean when h > 0). With np.r_[p[:32], 0.0] (h set to 0),
    the circular speed v_c(R_SIG) of the vector."""
    return float(kinfit.rotation(p, np.array([kinfit.R_SIG]))[0])


def mock(group, frame, err_cat, seed):
    """Catalogue dict of one mock group (module docstring, step 2) in the frame (centre, orientation, systemic motion) of `frame`,
    and its true parameter vector."""
    # own random generator per mock group (seed passed in)
    g = np.random.default_rng(seed)
    p = frame.copy()
    # the frame's centre, orientation and systemic motion, with the mock curve and the group's ellipsoid and h
    for k, v in (MOCK_VC | group).items():
        p[P(k)] = v
    # no streaming in the mocks
    p[[P(n) for n in kinfit.STREAM]] = 0.0
    # r0: unit vector to the centre; eE0, eN0: east and north unit vectors there; line: line of nodes; n: disc normal
    r0, eE0, eN0, line, n = (np.asarray(v) for v in geo.symmetry_axis(*(float(v) for v in kinfit.centre(p[0], p[1])),
                                                                       np.deg2rad(p[P('inc')]), np.deg2rad(p[P('theta')])))
    m = 30 * N_MOCK  # stars drawn before the 2 < rho < 9 deg cut keeps N_MOCK of them
    R, ph = g.gamma(2.0, group['h'], m), g.uniform(0, 2 * np.pi, m)  # exponential surface density: R ~ Gamma(2, h)
    # 3D positions [kpc]: centre at distance D0 plus (R, phi) in the disc plane
    xyz = LMC['D0'] * r0 + R[:, None] * (np.cos(ph)[:, None] * line + np.sin(ph)[:, None] * np.cross(n, line))
    # unit vectors to the stars, then (ra, dec) [deg] and distance from the vdMK14 centre rho [deg]
    rhat = xyz / np.linalg.norm(xyz, axis=1)[:, None]
    ra, dec = np.rad2deg(np.arctan2(rhat[:, 1], rhat[:, 0])) % 360, np.rad2deg(np.arcsin(rhat[:, 2]))
    rho = angular_distance(ra, dec, LMC['ra0'], LMC['dec0'])
    keep = np.flatnonzero((rho > 2) & (rho < 9))[:N_MOCK]  # the ring of fit_kinematics_mcmc.RING about the vdMK14 centre
    ra, dec = ra[keep], dec[keep]
    # unit vectors of the real giants; each mock star takes the proper-motion errors of the nearest one
    xyzc = np.c_[np.cos(np.deg2rad(err_cat['dec'])) * np.cos(np.deg2rad(err_cat['ra'])),
                 np.cos(np.deg2rad(err_cat['dec'])) * np.sin(np.deg2rad(err_cat['ra'])), np.sin(np.deg2rad(err_cat['dec']))]
    near = cKDTree(xyzc).query(rhat[keep])[1]
    e = {k: np.asarray(err_cat[k])[near] for k in ('pmra_error', 'pmdec_error', 'pmra_pmdec_corr')}
    # the kinfit data of the mock stars (proper motions zero for now: only the geometry and errors are used)
    d = kinfit.data(ra, dec, np.zeros(len(ra)), np.zeros(len(ra)), e['pmra_error'], e['pmdec_error'], e['pmra_pmdec_corr'])
    # D: distance [kpc]; Rm: in-plane radius [kpc]; eR, ephi, nn: radial, azimuthal and normal unit vectors (N, 3)
    _, _, _, D, Rm, eR, ephi, nn = (np.asarray(v) for v in kinfit._model(p, d['rhat'], d['ea'], d['ed']))
    assert n_negative(p, d) == 0, 'mock with vbar^2 <= 0'
    # dispersion scale factor exp(dlnsig_dR (R - R_SIG)) per star
    scale = np.exp(p[P('dlnsig_dR')] * (Rm - kinfit.R_SIG))
    vbar = np.asarray(kinfit.rotation(p, Rm))  # exact drift mean (no floor reached: n_negative == 0 and u2 >> DRIFT_EPS^2)
    # 3D velocities (N, 3) [km/s]: systemic motion of the centre plus random radial, azimuthal (about vbar) and vertical parts
    V = (p[P('vlos0')] * r0 + KAPPA * LMC['D0'] * (p[P('pmra0')] * eE0 + p[P('pmdec0')] * eN0))[None] \
        + g.normal(0, group['sigma_R'] * scale)[:, None] * eR + g.normal(vbar, group['sigma_phi'] * scale)[:, None] * ephi \
        + g.normal(0, group['sigma_z'] * scale)[:, None] * nn
    # proper motions [mas/yr]: velocity projected on east/north / (KAPPA x D), plus Gaia noise (Cholesky of the covariance)
    ea, ed, rh = (np.asarray(d[k]) for k in ('ea', 'ed', 'rhat'))
    noise = np.einsum('nij,nj->ni', np.linalg.cholesky(np.asarray(d['cov'])), g.normal(size=(len(ra), 2)))
    pmra, pmdec = (V * ea).sum(1) / (KAPPA * D) + noise[:, 0], (V * ed).sum(1) / (KAPPA * D) + noise[:, 1]
    # line-of-sight velocity [km/s] plus Gaussian error E_VLOS
    vlos = (V * rh).sum(1) + g.normal(0, E_VLOS, len(ra))
    # no distance moduli in the mocks (dmu NaN)
    return dict(ra=ra, dec=dec, pmra=pmra, pmdec=pmdec, vlos=vlos, evlos=np.full(len(ra), E_VLOS),
                dmu=np.full(len(ra), np.nan), **e), p


def summary(model, ps, es, nll, names, ds, extra=None):
    """Rows for the csv: per population its curve, v_rot(4 kpc) (drift mean when h > 0), v_c(4 kpc), ellipsoid, k, f, h, and the
    number of its stars with vbar^2 <= 0."""
    rows = []
    # one row per population: rotation at 4 kpc, circular speed at 4 kpc (h set to 0), h, stars with vbar^2 <= 0
    for name, p, e, d in zip(names, ps, es, ds):
        r = dict(model=model, population=name, nll=nll, vrot4=vrot4(p), vc4=vrot4(np.r_[p[:32], 0.0]), h=p[P('h')],
                 h_err=e[P('h')], n_neg=n_negative(p, d) if p[P('h')] > 0 else 0)
        # plus the curve, ellipsoid, error scale, outlier fraction, inc and theta, and their errors
        r |= {n: p[P(n)] for n in CURVE + OWN + ('inc', 'theta')} | {f'{n}_err': e[P(n)] for n in CURVE + OWN}
        rows.append(r | (extra or {}))
    return rows


def with_h(ps, hs):
    """Copies of the vectors ps with h set to hs (0 switches the drift off)."""
    out = []
    for p, h in zip(ps, hs):  # hs [kpc]
        q = p.copy()
        q[P('h')] = h
        out.append(q)
    return out


if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data
    # The disc frame of the joint ellipsoid fit (stage-2 medians per population) and the giants of the five [Fe/H] bins at
    # 2 < $\rho$ < 9 deg (no distance moduli: NaN). The BOSS mass sample, the foreground table and the RR Lyrae catalogue are read in
    # the next step, where the scale lengths are measured, and the RR Lyrae posterior draws in the last step.

    # %%
    frames = joint_params()  # population -> kinfit.PARAMS vector (results/kin_mcmc_data_ellipsoid.csv)
    # giants of the five bins, cut to 2 < rho < 9 deg; dmu NaN (distance moduli not used)
    giants = [(f'giants {lab}', ring(cat | dict(dmu=np.full(len(cat['ra']), np.nan)))) for lab, cat in samples.giants()]
    # population labels, e.g. 'giants -0.8 < [Fe/H] < -0.5'
    names = [n for n, _ in giants]
    frame = frames[names[0]].copy()  # its shared part is the joint-fit frame

    # %% [markdown]
    # ### Density scale lengths
    # The exponential scale length $h$ of each giant bin, from the selection-weighted surface density of the BOSS giants of the mass
    # sample, and of the RR Lyrae members, both against the in-plane radius $R$ of the joint-fit frame; errors from bootstraps of
    # sky pixels. Writes `results/kin_vc_drift_h.csv`.

    # %%
    b, _, _ = mass_samples()  # BOSS mass sample: one row per giant, with selection weight w and [Fe/H] feh
    # coverage fill and Milky Way fraction per star (columns fill, f_mw)
    fg = pd.read_parquet(DATA / 'giant_foreground.parquet')
    b = b.merge(fg[['source_id', 'fill', 'f_mw']], on='source_id', validate='one_to_one')
    b['W'] = b.w * b['fill'] * (1 - b.f_mw)  # weight: selection x coverage fill x (1 - Milky Way fraction), as paper_feh_profile.py
    b['R'] = radius(frame, b.ra.to_numpy(), b.dec.to_numpy())  # kpc
    # nside-64 HEALPix pixel of each star (the bootstrap unit)
    b['pix'] = hp.ang2pix(64, b.ra.to_numpy(), b.dec.to_numpy(), lonlat=True)
    hrows = []
    # one scale length per measured [Fe/H] bin (samples.FEH_BINS) [dex]
    for (lo, hi), name in zip(samples.FEH_BINS, names):
        s = b[(b.feh >= lo) & (b.feh < hi)]
        hrows.append(dict(population=name, **dict(zip(('h', 'h_p16', 'h_p84'),
                                                       scale_length(s.R.to_numpy(), s.W.to_numpy(), s.pix.to_numpy())))))
    # the RR Lyrae members, weight 1
    rr = pd.read_parquet(DATA / 'rrab.parquet')
    rr = rr[rr.member]
    hrows.append(dict(population='RR Lyrae', **dict(zip(('h', 'h_p16', 'h_p84'), scale_length(
        radius(frame, rr.ra.to_numpy(), rr.dec.to_numpy()), np.ones(len(rr)),
        hp.ang2pix(64, rr.ra.to_numpy(), rr.dec.to_numpy(), lonlat=True))))))
    # population -> h, h_p16, h_p84 [kpc]
    H = pd.DataFrame(hrows).set_index('population')
    H.to_csv(ROOT / 'results/kin_vc_drift_h.csv', float_format='%.3f')
    print(H.round(2).to_string(), flush=True)

    # %% [markdown]
    # ### Mocks with a known circular speed
    # Three mock groups share one $v_c(R)$ and rotate at their exact drift means; each realisation is fitted with 'drift' ($h$ at
    # the truth), 'drift, h free' and 'free'. Writes `results/kin_vc_drift_mock.csv`.

    # %%
    # Positions and Gaia proper-motion errors of all giants: each mock star takes the errors of the nearest one.
    err_cat = {k: np.concatenate([c[k] for _, c in giants]) for k in ('ra', 'dec', 'pmra_error', 'pmdec_error', 'pmra_pmdec_corr')}
    mrows = []
    # NREAL realisations of the three mock groups; seeds 100 x realisation + group
    for real in range(NREAL):
        cats, truths = zip(*(mock(g, frame, err_cat, 100 * real + j) for j, g in enumerate(MOCK_GROUPS)))
        # kinfit data of each mock catalogue
        ds = [kinfit.data(*(c[k] for k in COLS)) for c in cats]
        mnames = [f'mock {j}' for j in range(len(ds))]
        starts = []
        for t in truths:  # start away from the truth: v0, R0 of a plain tanh, isotropic 30 km/s, no gradient
            q = t.copy()
            q[[P(n) for n in ('v0', 'R0', 'vslope', 'sigma_R', 'sigma_phi', 'sigma_z', 'dlnsig_dR')]] = [70., 2.5, 0., 30., 30., 30., 0.]
            starts.append(q)
        # true scale lengths of the mock groups [kpc]
        hs = [t[P('h')] for t in truths]
        # 'drift': one shared curve, h fixed at the truth
        ps, es, nll, _ = joint_fit(ds, with_h(starts, hs), SHARED_FREE + CURVE)
        mrows += summary('drift', ps, es, nll, mnames, ds, dict(realisation=real))
        # 'drift, h free': started from the 'drift' optimum, h fitted per group
        ps2, es2, nll2, _ = joint_fit(ds, ps, SHARED_FREE + CURVE, extra=('h',))
        mrows += summary('drift, h free', ps2, es2, nll2, mnames, ds, dict(realisation=real))
        # 'free': h = 0 (no drift), own curve per group
        ps3, es3, nll3, _ = joint_fit(ds, with_h(starts, [0.] * 3), SHARED_FREE)
        mrows += summary('free', ps3, es3, nll3, mnames, ds, dict(realisation=real))
        # recovered v_c(4 kpc) and drift parameters against the truth; likelihood gain of 'free' over 'drift'
        print(f'mock {real}: truth v_c(4) {vrot4(np.r_[truths[0][:32], 0.]):.1f}; drift {vrot4(np.r_[ps[0][:32], 0.]):.1f} '
              f'(V {ps[0][P("v0")]:.1f}, R0 {ps[0][P("R0")]:.2f}, s {ps[0][P("vslope")]:.3f}); h free {[round(p[P("h")], 2) for p in ps2]}; '
              f'free v_rot(4) {[round(vrot4(p), 1) for p in ps3]} vs true mean {[round(vrot4(t), 1) for t in truths]}; '
              f'dlnL drift - free {nll3 - nll:.1f}', flush=True)
    pd.DataFrame(mrows).to_csv(ROOT / 'results/kin_vc_drift_mock.csv', index=False, float_format='%.4f')

    # %% [markdown]
    # ### The giant bins
    # Joint maximum-likelihood fits of the five giant bins: free rotation curves ('free'), one circular speed with each bin's drift
    # ('drift', $h$ from the scale-length step) and the same with $h$ free. Writes `results/kin_vc_drift_data.csv`.

    # %%
    # kinfit data of the five giant bins
    ds = [kinfit.data(*(c[k] for k in COLS)) for _, c in giants]
    # 'free': own curves, h = 0, started from the stage-2 medians
    pf, ef, nllf, _ = joint_fit(ds, with_h([frames[n] for n in names], [0.] * 5), SHARED_FREE)
    # 'drift' starts: the free solution with each bin's measured h
    p0 = with_h(pf, H.loc[names, 'h'].to_numpy())
    for p in p0:  # start of the common curve: v_c a little above the fastest free curve
        p[[P(n) for n in CURVE]] = [90., 2.5, 0.]
    # The curve alone first, everything else at the free solution: from there the gradient of the full problem is large enough
    # that the first L-BFGS-B step reaches the corners of BOUNDS, where the likelihood is not finite.
    p0, _, _, _ = joint_fit(ds, p0, CURVE, own=())
    # 'drift': everything shared and own free, then 'drift, h free'
    pj, ej, nllj, _ = joint_fit(ds, p0, SHARED_FREE + CURVE)
    ph, eh, nllh, _ = joint_fit(ds, pj, SHARED_FREE + CURVE, extra=('h',))
    # one table: rows per (model, population)
    rows = (summary('free', pf, ef, nllf, names, ds) + summary('drift', pj, ej, nllj, names, ds)
            + summary('drift, h free', ph, eh, nllh, names, ds))
    T = pd.DataFrame(rows)
    T.to_csv(ROOT / 'results/kin_vc_drift_data.csv', index=False, float_format='%.4f')
    print(T[['model', 'population', 'nll', 'vrot4', 'vc4', 'h', 'h_err', 'n_neg', 'sigma_R', 'sigma_phi', 'dlnsig_dR', 'k']].round(2).to_string(index=False))
    print(f'dlnL drift - free {nllf - nllj:.1f} (12 fewer parameters: one curve for five); h free - h fixed {nllj - nllh:.1f}', flush=True)

    # %% [markdown]
    # ### The RR Lyrae prediction
    # The RR Lyrae were not fitted: their mean rotation at $R$ = 4 kpc is predicted from the fitted $v_c$, their own $h$ and their
    # ellipsoid (stage-1 posterior draws of the 4-9 deg fit), draw by draw, and compared with their measured $v_{\rm rot}(4\,{\rm kpc})$.
    # Writes `results/kin_vc_drift_rrl.csv`.

    # %%
    # stage-1 posterior samples of the 4-9 deg fit
    z = np.load(DATA / 'kin_mcmc_data_ellipsoid_outer_samples.npz')
    i = list(z['names']).index('RR Lyrae')
    g = lambda n: z[f'1|{i}|{n}_0'].reshape(-1)  # noqa: E731  (all stage-1 draws of RR Lyrae parameter n)
    # one vector per draw: data 'drift' fit of the first giant bin (indices 0-31, with the shared v_c) and the RR Lyrae h
    q = np.tile(np.r_[pj[0][:32], H.loc['RR Lyrae', 'h']], (len(g('v0')), 1))
    # RR Lyrae's own ellipsoid and gradient, draw by draw
    for n in ('sigma_R', 'sigma_phi', 'dlnsig_dR'):
        q[:, P(n)] = g(n)
    # predicted mean rotation at R_SIG = 4 kpc for each draw [km/s]
    pred = np.array([vrot4(x) for x in q])
    # measured rotation at R_SIG: v0 tanh(R_SIG / R0) (the vslope term vanishes at R_SIG)
    meas = g('v0') * np.tanh(kinfit.R_SIG / g('R0'))
    print('RR Lyrae v_rot(4 kpc): predicted', np.percentile(pred, [16, 50, 84]).round(1), 'measured', np.percentile(meas, [16, 50, 84]).round(1))
    # 16th, 50th, 84th percentiles of both [km/s]
    pd.DataFrame(dict(predicted=np.percentile(pred, [16, 50, 84]), measured=np.percentile(meas, [16, 50, 84])),
                 index=['p16', 'median', 'p84']).to_csv(ROOT / 'results/kin_vc_drift_rrl.csv', float_format='%.2f')
