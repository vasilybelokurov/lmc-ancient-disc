"""Can thickness, a hot anisotropic population with asymmetric drift, or spatially correlated Gaia proper-motion errors bias the
thin-disc fits (lmcold.kinfit) of the centre, inclination and line of nodes? Mocks that are NOT drawn from the fitting model.
Used in KB chapter 6 ('What the models leave in the data': the fitted centre and m = 2 amplitude of symmetric mock discs) and
chapter 7; the numbers are written from results/kin_mock_physical.csv by scripts/paper_text_kin_model.py and
scripts/paper_text_kin_support.py. Its draw() also makes the mocks of scripts/paper_kin_ellipsoid_mock.py.

Every mock has the adopted centre and bulk motion (lmcold.constants.LMC) and the disc orientation TRUTH (about the giants'
streaming fit, scripts/fig_kin_streaming.py). Positions are drawn in 3D: exponential surface density of scale length RD (R from a
Gamma(2, RD) distribution), uniform azimuth, height z ~ N(0, h); they are projected on the sky, and N_STARS stars at
2 < rho < 9 deg from the centre are kept (the RR Lyrae ring of scripts/fit_kinematics_mcmc.py). Drawing in 3D matters for the
distances: along one line of sight the stars above and below the plane lie at different R, so the radial gradient weights them
unequally (a fixed sky position with z of zero mean would hide that). Each star takes the Gaia proper-motion covariance of the
nearest RR Lyrae member (lmcold.samples.rrab). In the plane: radius R, unit vectors e_R and e_phi = n x e_R (the rotation sense
of kinfit). Velocities: v_R ~ N(0, sigma_R), v_phi ~ N(mean, sigma_phi), v_z ~ N(0, sigma_z), with mean = vc tanh(R/R0) for the cold
disc and, for the hot one, the asymmetric-drift (Jeans) mean
    mean^2 = vc^2 tanh^2(R/R0) - sigma_R^2 (R/RD - 1 + sigma_phi^2/sigma_R^2)
(exponential surface density of scale length RD, constant dispersions, no tilt term; floored at 0). The 3D velocity (systemic +
these) is projected on each star's sky basis at its own distance: proper motions with the Gaia covariance added as noise,
line-of-sight velocities (every star; error E_VLOS) and distance moduli 5 log10(D_star / D0) + N(0, E_MU).
Mocks (MOCKS): 'cold thin' (h 0, isotropic 20 km/s: the kinfit model up to the error model, a control); 'thick' (h H_THICK, cold
kinematics); 'hot' (h 0, sigma_R, sigma_phi, sigma_z = HOT and the drift); 'hot, no drift' (the same dispersions, mean
vc tanh(R/R0)); 'hot thick' (h H_THICK, HOT and the drift); 'correlated PM' (cold thin plus a
Gaussian random field of proper-motion offsets, rms PM_SYS per component, built from NMODES plane waves of wavelength WAVELENGTH deg,
NREAL realisations). Expectation for thickness: for a density stratified on similar oblate spheroids of axis ratio q the mean
distance along each line of sight lies on the plane conjugate to the line of sight, whose apparent tilt is below i (q = 0.5,
i = 28 deg: 20.4 deg); h = H_THICK at R = 2-8 kpc spans q ~ 0.1-0.5.

Fits (kinfit.fit, symmetric disc, i and theta free; centre fixed first, then free): 'PM' (kinfit.BASE), 'LOS' (centre, i, theta,
v0, R0, outliers, vlos0, sigma_los; bulk motion held at the 'PM' fit), 'PM + LOS', 'distances' (theta, i, dmu0, sigma_mu, outliers;
centre fixed) and 'PM + distances'; and 'PM + LOS stream', 'PM + distances stream': the same with the in-plane streaming
kinfit.STREAM free. None of the mocks has streaming, so any fitted m = 2 (|m2| = hypot(m2c, m2s); in the data 0.09-0.17 for the
giants, PM + LOS, with the v_phi minimum within 4 deg of the line of nodes in the metal-rich bins) is made by the mock's departure
from the thin, cold model (thickness, asymmetric drift, anisotropy, correlated errors).

Writes results/kin_mock_physical.csv (mock, realisation, data, nll, every parameter of kinfit.PARAMS and its error *_err) and
figures/fig_kin_mock_physical.png: four equal panels against the mock: (a) inclination, (b) line of nodes [deg], (c) centre
offset east and north [deg] (two markers); one marker per data set; truth as horizontal lines; (d) streaming amplitude |m2| of the
two streaming fits, with the range of the giants' data (grey band). Legend right of (d). Diagnostic figure (not in the paper).

Inputs: lmcold.samples (data/rrab.parquet, key release_rrab).

Run: .venv/bin/python scripts/fig_kin_mock_physical.py   (run time not recorded)
"""
# %% [markdown]
# ### Settings
# The true disc orientation and rotation curve of the mocks, their dispersions and thickness, the error model, and the parameters
# freed in each kind of fit.

# %%
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from fit_kinematics_mcmc import ring
from lmcold import geometry as geo
from lmcold import kinfit, samples
from lmcold.constants import KAPPA, LMC
from lmcold.spheroid import spheroid
from lmcold.layout import panel_grid
from lmcold.sources import ROOT

TRUTH = dict(inc=28.0, theta=143.0)  # deg   # inclination, line of nodes (module docstring)
VC, R0, RD = 85.0, 2.0, 1.5  # km/s circular-speed amplitude, kpc turnover radius, kpc exponential scale length
COLD = (20.0, 20.0, 20.0)  # km/s sigma_R, sigma_phi, sigma_z
HOT = (40.0, 28.0, 30.0)  # km/s, sigma_phi / sigma_R ~ 1/sqrt(2) as for a flat rotation curve
H_THICK = 1.0  # kpc, Gaussian height dispersion
E_VLOS, E_MU = 2.0, 0.08  # km/s, mag
PM_SYS, WAVELENGTH, NMODES, NREAL = 0.025, 1.5, 200, 3  # mas/yr per component, deg, modes, realisations
MOCKS = {'cold thin': (0.0, COLD, False), 'thick': (H_THICK, COLD, False), 'hot': (0.0, HOT, True),
         'hot, no drift': (0.0, HOT, False), 'hot thick': (H_THICK, HOT, True), 'correlated PM': (0.0, COLD, False)}  # (h, dispersions, drift)
N_STARS = 15000  # stars per mock at 2 < rho < 9 deg
C = ('ra', 'dec', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error', 'pmra_pmdec_corr')  # catalogue columns for kinfit.data
LOS, DIST = ('vlos0', 'sigma_los'), ('dmu0', 'sigma_mu')   # kinfit parameters added by velocities, distances
# Parameters freed in each kind of fit (names of kinfit.PARAMS); the rest stay at kinfit.start() (module docstring, 'Fits').
FREE = {'PM': kinfit.BASE, 'LOS': ('dx', 'dy', 'inc', 'theta', 'v0', 'R0', 'logit_fout') + LOS, 'PM + LOS': kinfit.BASE + LOS,
        'distances': ('inc', 'theta', 'logit_fout') + DIST, 'PM + distances': kinfit.BASE + DIST,
        'PM + LOS stream': kinfit.BASE + LOS + kinfit.STREAM, 'PM + distances stream': kinfit.BASE + DIST + kinfit.STREAM}
# Plot marker of each kind of fit.
MARKERS = {'PM': 's', 'LOS': '^', 'PM + LOS': 'o', 'distances': 'v', 'PM + distances': 'D', 'PM + LOS stream': 'P',
           'PM + distances stream': 'X'}
M2_DATA = (0.094, 0.171)  # |m2| range of the giants' PM + LOS streaming fits (results/kin_tilted_ring.csv, model 'stream')


# %% [markdown]
# ### Drawing a mock
# `draw` places N_STARS stars in three dimensions (an exponential disc, optionally thick, or a spheroid), gives each the Gaia errors
# of the nearest RR Lyrae, draws velocities about the mean rotation (with or without asymmetric drift) and projects them on the sky.

# %%
def draw(c, h, disp, drift, rng, pm_field=False, q=None):
    """Mock catalogue dict (C columns plus vlos, evlos, dmu): positions drawn in 3D, Gaia errors of the nearest star of c (see the
    module docstring); with q, positions from an oblate K0 spheroid of axis ratio q instead (lmcold.spheroid.spheroid, face-on
    exponential of scale length RD, its thickness growing with R as for the old population of paper Section 5; h is then unused).

    Parameters
    ----------
    c : dict
        Catalogue whose positions (ra, dec [deg]) and proper-motion errors (pmra_error, pmdec_error [mas/yr], pmra_pmdec_corr) give
        each mock star its errors (nearest star on the sky).
    h : float
        Gaussian height dispersion [kpc]; 0 for a thin disc.
    disp : tuple of float
        (sigma_R, sigma_phi, sigma_z) [km/s], constant over the disc.
    drift : bool
        Mean rotation lagging vc tanh(R/R0) by the asymmetric drift (module docstring) if True, vc tanh(R/R0) if False.
    rng : numpy.random.Generator
    pm_field : bool
        Add the correlated proper-motion field (PM_SYS, WAVELENGTH, NMODES).
    q : float or None
        Axis ratio of a K0 spheroid for the positions instead of the disc.

    Returns
    -------
    dict
        Arrays of length N_STARS: ra, dec [deg], pmra, pmdec [mas/yr], their errors and correlation, vlos, evlos [km/s], dmu [mag].
    """
    # Unit vectors (3,): line of sight to the centre, east and north there, line of nodes, and the disc normal (geo.symmetry_axis).
    r0, eE0, eN0, line, n = (np.asarray(v) for v in geo.symmetry_axis(np.deg2rad(LMC['ra0']), np.deg2rad(LMC['dec0']),
                                                                       np.deg2rad(TRUTH['inc']), np.deg2rad(TRUTH['theta'])))
    # LMC distance [kpc] (lmcold.constants).
    D0 = LMC['D0']
    m = 20 * N_STARS  # draw many, keep N_STARS in the ring
    if q is None:
        # In-plane radius [kpc] from Gamma(2, RD) (exponential surface density), azimuth [rad] uniform, height z [kpc] ~ N(0, h).
        Rd, ph = rng.gamma(2.0, RD, m), rng.uniform(0, 2 * np.pi, m)
        z = rng.normal(0, h, m) if h > 0 else np.zeros(m)
    else:
        # Spheroid positions (m, 3) [kpc] in the disc frame, converted to the same R, phi, z.
        X = spheroid(RD, q, m, rng)
        Rd, ph, z = np.hypot(X[:, 0], X[:, 1]), np.arctan2(X[:, 1], X[:, 0]), X[:, 2]
    # Position relative to the LMC centre [kpc], (m, 3): in-plane axes line and n x line, plus z along the normal.
    pos = Rd[:, None] * (np.cos(ph)[:, None] * line + np.sin(ph)[:, None] * np.cross(n, line)) + z[:, None] * n
    xyz = D0 * r0 + pos  # heliocentric position [kpc]
    Ds = np.linalg.norm(xyz, axis=1)
    rhat = xyz / Ds[:, None]
    # Angular distance from the centre [deg]; the first N_STARS stars at 2 < rho < 9 deg are kept (the ring of the module
    # docstring).
    rho = np.rad2deg(np.arccos(np.clip(rhat @ r0, -1, 1)))
    keep = np.flatnonzero((rho > 2) & (rho < 9))[:N_STARS]
    rhat, Ds, pos = rhat[keep], Ds[keep], pos[keep]
    # Sky coordinates [deg] of each mock star; then the nearest catalogue star on the unit sphere gives its proper-motion errors.
    ra, dec = np.rad2deg(np.arctan2(rhat[:, 1], rhat[:, 0])) % 360, np.rad2deg(np.arcsin(rhat[:, 2]))
    _, near = cKDTree(np.c_[np.cos(np.deg2rad(c['dec'])) * np.cos(np.deg2rad(c['ra'])),
                            np.cos(np.deg2rad(c['dec'])) * np.sin(np.deg2rad(c['ra'])), np.sin(np.deg2rad(c['dec']))]).query(rhat)
    c = {k: c[k][near] for k in ('pmra_error', 'pmdec_error', 'pmra_pmdec_corr')} | dict(ra=ra, dec=dec)
    # kinfit.data with zero proper motions, used only for its sky basis (ea, ed: east and north unit vectors, (N, 3)) and the
    # proper-motion covariance matrices cov, (N, 2, 2) [(mas/yr)^2].
    d = kinfit.data(ra, dec, np.zeros(len(ra)), np.zeros(len(ra)), c['pmra_error'], c['pmdec_error'], c['pmra_pmdec_corr'])
    ea, ed = np.asarray(d['ea']), np.asarray(d['ed'])
    D = Ds
    # In-plane radius R [kpc] and unit vectors e_R, e_phi = n x e_R (the rotation sense of kinfit).
    inplane = pos - (pos @ n)[:, None] * n
    R = np.linalg.norm(inplane, axis=1)
    eR = inplane / R[:, None]
    ephi = np.cross(n, eR)
    sR, sphi, sz = disp
    # Mean v_phi [km/s]: vc tanh(R/R0), or with drift the asymmetric-drift mean of the module docstring (floored at 0).
    vc = VC * np.tanh(R / R0)
    mean = np.sqrt(np.maximum(vc**2 - sR**2 * (R / RD - 1 + sphi**2 / sR**2), 0)) if drift else vc
    # Velocities [km/s] in the plane; V (N, 3): the systemic velocity (line-of-sight vlos0 plus the bulk proper motion at D0) plus
    # these.
    vR, vphi, vz = (rng.normal(m, s, len(D)) for m, s in ((0, sR), (mean, sphi), (0, sz)))
    V = LMC['vlos0'] * r0 + KAPPA * D0 * (LMC['pmra0'] * eE0 + LMC['pmdec0'] * eN0) + vR[:, None] * eR + vphi[:, None] * ephi \
        + vz[:, None] * n
    # Proper motions [mas/yr] at each star's own distance Ds [kpc] (mu = v_t / (KAPPA D)), plus Gaia noise drawn from its
    # covariance.
    pmra, pmdec = (V * ea).sum(1) / (KAPPA * Ds), (V * ed).sum(1) / (KAPPA * Ds)
    noise = np.einsum('nij,nj->ni', np.linalg.cholesky(np.asarray(d['cov'])), rng.normal(size=(len(D), 2)))
    pmra, pmdec = pmra + noise[:, 0], pmdec + noise[:, 1]
    if pm_field:
        # Gaussian random field: NMODES plane waves of wavelength WAVELENGTH deg in random directions and phases on the tangent
        # plane (x east, y north [deg]), scaled to rms PM_SYS per component.
        x, y = (rhat - r0) @ eE0 * 180 / np.pi, (rhat - r0) @ eN0 * 180 / np.pi
        for comp in range(2):
            ang, ph = rng.uniform(0, 2 * np.pi, NMODES), rng.uniform(0, 2 * np.pi, NMODES)
            k = 2 * np.pi / WAVELENGTH
            f = np.sqrt(2.0 / NMODES) * np.cos(k * (np.outer(x, np.cos(ang)) + np.outer(y, np.sin(ang))) + ph).sum(1)
            if comp == 0:
                pmra = pmra + PM_SYS * f
            else:
                pmdec = pmdec + PM_SYS * f
    # Line-of-sight velocity [km/s] with error E_VLOS; distance modulus relative to D0 [mag] with error E_MU.
    vlos = (V * rhat).sum(1) + rng.normal(0, E_VLOS, len(D))
    dmu = 5 * np.log10(Ds / D0) + rng.normal(0, E_MU, len(D))
    return dict(c, pmra=pmra, pmdec=pmdec, vlos=vlos, evlos=np.full(len(D), E_VLOS), dmu=dmu)


# %% [markdown]
# ### Fitting a mock
# `fit` fits one data set of a mock (proper motions, line-of-sight velocities, distances, or combinations) by maximum likelihood.

# %%
def fit(m, kind, bulk=None):
    """Fit data set `kind` (a key of FREE) of mock m, the data not used set to NaN; bulk (pmra0, pmdec0 [mas/yr]) holds the bulk
    motion for 'LOS'. Returns (p, err, nll): best-fit vector and its inverse-Hessian errors (shape (33,) each, kinfit.PARAMS order)
    and the negative log likelihood."""
    # The data not used in this kind of fit are set to NaN (pmra NaN: no proper motions for that star).
    nan = np.full(len(m['ra']), np.nan)
    pmra = nan if kind in ('LOS', 'distances') else m['pmra']
    vlos = m['vlos'] if 'LOS' in kind else nan
    dmu = m['dmu'] if 'distances' in kind else nan
    # Arrays for the likelihood (positions [deg], proper motions [mas/yr], v_los [km/s], distance moduli [mag]).
    d = kinfit.data(m['ra'], m['dec'], pmra, m['pmdec'], m['pmra_error'], m['pmdec_error'], m['pmra_pmdec_corr'], vlos,
                    m['evlos'], dmu)
    # Start vector (33,); for 'LOS', the bulk motion pmra0, pmdec0 (indices 2, 3) held at the 'PM' fit.
    p0 = kinfit.start()
    if bulk is not None:
        p0[2:4] = bulk
    free = FREE[kind]
    # When the centre (dx, dy) is free, first fit with it fixed, then start the full fit from there.
    if free[:2] == ('dx', 'dy'):
        p0, _, _ = kinfit.fit(d, free=free[2:], p0=p0)
    return kinfit.fit(d, free=free, p0=p0)


# %% [markdown]
# ### Load the data, draw and fit every mock, and draw the figure
# `main` reads the RR Lyrae members (`lmcold.samples.rrab()`, Zenodo bundle 1) for the footprint and errors, draws each mock
# (three realisations of the correlated field), fits every data set, and writes `results/kin_mock_physical.csv` and the figure.

# %%
def main():
    """Draw every mock, fit every data set, write the csv and the figure of the module docstring."""
    rr = samples.rrab()
    # The RR Lyrae at 2 < rho < 9 deg: footprint and Gaia errors of the mocks.
    c = ring({k: rr[k] for k in C})
    rows = []
    for name, (h, disp, drift) in MOCKS.items():
        for real in range(NREAL if name == 'correlated PM' else 1):
            # One mock per realisation (seed 100 + realisation); then every kind of fit, the 'PM' fit first so its bulk motion feeds
            # 'LOS'.
            m = draw(c, h, disp, drift, np.random.default_rng(100 + real), pm_field=name == 'correlated PM')
            bulk = None
            for kind in FREE:
                p, e, nll = fit(m, kind, bulk if kind == 'LOS' else None)
                if kind == 'PM':
                    bulk = p[2:4]
                rows.append(dict(mock=name, realisation=real, data=kind, nll=nll, **dict(zip(kinfit.PARAMS, p)),
                                 **{f'{n}_err': v for n, v in zip(kinfit.PARAMS, e)}))
                # Progress line; the indices follow kinfit.PARAMS (0 dx, 1 dy, 4 inc, 5 theta, 6 v0, 8 sigma, 12 sigma_los, 17-21
                # streaming).
                print(name, real, kind, f'i {p[4]:.1f}+-{e[4]:.1f} theta {p[5]:.1f} dx {p[0]:.3f}+-{e[0]:.3f} dy {p[1]:.3f}+-{e[1]:.3f}'
                      f' v0 {p[6]:.1f} sigma {p[8]:.1f} sigma_los {p[12]:.1f} nll {nll:.1f} m2c {p[17]:.3f} m2s {p[18]:.3f}'
                      f' vr0 {p[19]:.3f} r2c {p[20]:.3f} r2s {p[21]:.3f}', flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / 'results/kin_mock_physical.csv', index=False, float_format='%.5f')

    plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
    # One x position per mock (and per realisation of the correlated field); four equal panels in a row.
    labels = [f'{r.mock}' + (f' {r.realisation}' if r.mock == 'correlated PM' else '') for _, r in
              t.drop_duplicates(['mock', 'realisation']).iterrows()]
    fig, ax = panel_grid(1, 4, width=9.0, left=0.5, right=1.3, hgap=0.55, top=0.15, bottom=0.75)
    for k, ((name, real), s) in enumerate(t.groupby(['mock', 'realisation'], sort=False)):
        for j, kind in enumerate(FREE):
            r = s[s.data == kind].iloc[0]
            # The fits of one mock spread around its x position; (a) inclination, (b) line of nodes, (d) |m2| of the streaming fits,
            # (c) centre offsets east (red) and north (blue, open) of every fit with a centre.
            x = k + 0.12 * (j - 2)
            ax[0, 0].errorbar(x, r.inc, yerr=r.inc_err, fmt=MARKERS[kind], ms=3, lw=0.6, color='k',
                              label=kind if k == 0 else None)
            ax[0, 1].errorbar(x, r.theta, yerr=r.theta_err, fmt=MARKERS[kind], ms=3, lw=0.6, color='k')
            if 'stream' in kind:
                ax[0, 3].plot(x, np.hypot(r.m2c, r.m2s), MARKERS[kind], ms=3, color='k')
            if kind != 'distances':
                ax[0, 2].errorbar(x, r.dx, yerr=r.dx_err, fmt=MARKERS[kind], ms=3, lw=0.6, color='tab:red')
                ax[0, 2].errorbar(x, r.dy, yerr=r.dy_err, fmt=MARKERS[kind], ms=3, lw=0.6, color='tab:blue', mfc='white')
    # Truth as horizontal grey lines, mock names as tick labels.
    for a, v, lab in ((ax[0, 0], TRUTH['inc'], r'inclination $i$ [deg]'), (ax[0, 1], TRUTH['theta'], r'line of nodes $\theta$ [deg]'),
                      (ax[0, 2], 0.0, 'centre offset [deg]')):
        a.axhline(v, color='0.5', lw=0.6)
        a.set(ylabel=lab, xticks=range(len(labels)), xlim=(-0.5, len(labels) - 0.5))
        a.set_xticklabels(labels, rotation=60, ha='right', fontsize=5.5)
    # Panel (d): the giants' |m2| range as a grey band.
    ax[0, 3].axhspan(*M2_DATA, color='0.85', lw=0)
    ax[0, 3].set(ylabel='|m2| [fraction of vrot]', xticks=range(len(labels)), xlim=(-0.5, len(labels) - 0.5), ylim=(0, 0.2))
    ax[0, 3].set_xticklabels(labels, rotation=60, ha='right', fontsize=5.5)
    # Legend entries for the band and the two centre markers; one legend right of panel (d), outside the data.
    ax[0, 3].plot([], [], 's', color='0.85', ms=5, label='giants data, |m2|')
    ax[0, 2].plot([], [], 'o', color='tab:red', ms=3, label='east (red)')
    ax[0, 2].plot([], [], 'o', color='tab:blue', mfc='white', ms=3, label='north (blue, open)')
    h0, l0 = ax[0, 0].get_legend_handles_labels()
    h2, l2 = ax[0, 2].get_legend_handles_labels()
    h3, l3 = ax[0, 3].get_legend_handles_labels()
    ax[0, 3].legend(h0 + h2 + h3, l0 + l2 + l3, loc='upper left', bbox_to_anchor=(1.03, 1), fontsize=5.5, frameon=False)
    fig.savefig(ROOT / 'figures/fig_kin_mock_physical.png', dpi=200)


# %% [markdown]
# ### Run

# %%
if __name__ == '__main__':
    main()
