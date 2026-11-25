"""Paper figures: the measured velocity field of the old population and what the disc models leave in it (kinematics section).

Every star's proper motion and line-of-sight velocity is compared with a model velocity from lmcold.kinfit.model; the difference
is projected on the sky-polar basis about the model's centre (tangential v_t, 90 deg east of the direction away from the centre,
and radial v_r; lmcold.kinfit.residual_polar, each star at its disc distance) and, for the giants, along the line of sight (v_los).
Models, as parameter vectors of kinfit.PARAMS:
- 'data': the frame of the joint Level-2 fit without internal motion (its centre, systemic proper motion and velocity, inclination
  and line of nodes; v0 = 0, which also removes the streaming, proportional to v0). The difference is the internal velocity field
  with the bulk motion and its perspective removed, in the polar basis about the fitted centre: the data.
- 'model': the internal velocity field of Level 2 in the same frame ('data' minus 'L2' residuals, i.e. L2 model minus frame).
- 'pull': the 'L2' residual of each star divided by the scatter Level 2 predicts for it in that component: sqrt(sigma^2 +
  k^2 e^2) for v_t and v_r (sigma the population's proper-motion dispersion, k its Gaia error scale, e the star's proper-motion
  error ellipse projected on the component, in km/s at its disc distance) and sqrt(sigma_los^2 + evlos^2) for v_los, as in the
  likelihood (lmcold.kinfit.loglike_stars). N(0, 1) per star if Level 2 is right.
- 'vdMK14': the vdMK14 geometry and systemic motion (lmcold.constants.LMC) with each population's rotation curve, dispersions,
  error scale and outlier fraction fitted in that frame (maximum likelihood, kinfit.fit with VDMK14_FREE).
- 'L1': the joint symmetric-disc fit (results/kin_mcmc_data_centre.csv, stage 2: shared centre, systemic motion, i, theta; each
  population's own v0, R0); 'L2': the joint fit with streaming (results/kin_mcmc_data_stream.csv, stage 2, shared m = 2 terms).
  Posterior medians.
Stars: RR Lyrae members (proper motions only; lmcold.samples.rrab) and the giants in their five measured [Fe/H] bins
(lmcold.samples.giants), at 2 < rho < 9 deg from the vdMK14 centre (the fitted range); a star whose L2 residual exceeds CLIP km/s
(plus three times its own error for the proper-motion components) is dropped from every panel (Milky Way stars and gross errors;
2-6 per cent in the metal-poor bins, < 1 per cent above -1.1).

Writes (all panels equal; maps: x east to the left, y north [deg] on the orthographic plane about the joint Level-2 centre,
lmcold.geometry.tangent_plane_coords):
- figures/paper_kin_maps_vt.png, paper_kin_maps_vr.png, paper_kin_maps_vlos.png (Sects. 4.1, 4.3): one figure per component;
  rows 'data', 'model' and the significance of the residual: z = median pull x sqrt(n) / 1.2533 per cell of n stars (the
  median of n unit normals has standard deviation 1.2533 / sqrt(n), so z is N(0, 1) per cell if Level 2 is right; the median
  keeps the remaining Milky Way stars from dominating a cell); columns the giant bins metal-poor to metal-rich and the RR Lyrae (not for
  v_los: no velocities); median per CELL-deg cell with >= NMIN stars; colour VLIM for data and model, -ZLIM..ZLIM for z;
  vdMK14 centre (+, the origin of rho for the star selection), Level-2 line of nodes (dashed).
- figures/paper_kin_data_profiles.png (Sect. 4.1): 3 x 1 panels, v_t, v_r, v_los of 'data' against position angle (north through
  east about the Level-2 centre) at 3 < rho < 7 deg: median per 30-deg bin, one line per population; line of nodes (dotted) and
  minor axis (dash-dotted).
- figures/paper_kin_residual_profiles.png (Sect. 4.3): 3 x 3 panels; rows v_t, v_r, v_los residuals against position angle (about
  each model's centre) at 3 < rho < 7 deg; columns 'vdMK14', 'L1', 'L2'; one line per population; in each panel the range over
  the giant bins of the m = 1 and m = 2 amplitudes of a harmonic fit (lmcold.harmonics) to the stars.
- results/kin_residual_profiles.csv: population, model, component, n, mean, A1, phase1, A2, phase2 [km/s, deg] (harmonic fit at
  3 < rho < 7 deg; phase = position angle of the maximum).
- results/kin_residual_cells.csv: population, component, n_cells, z_sd (standard deviation of the cell significances z of the
  maps), z_gt3 (fraction of cells with |z| > 3).
- results/kin_residual_radial.csv and figures/paper_kin_residual_radial.png (appendix): median Level-2 residual of each component
  [km/s] in rings RADIAL_EDGES [deg] about the Level-2 centre, error 1.2533 x standard deviation / sqrt(n); one line per
  population, three equal panels (v_t, v_r, v_los).

With the argument 'ellipsoid', 'L2' is the joint fit of the disc-frame velocity-ellipsoid model (results/kin_mcmc_data_ellipsoid.csv,
stage 2: Level-2 mean field with the outer slope vslope, dispersions sigma_R, sigma_phi, sigma_z with radial gradient), the pull
uses the scatter that ellipsoid predicts for each star (its intrinsic covariance, lmcold.kinfit.intrinsic_cov, projected on t, r and
the line of sight), and every output name above gets the suffix '_ellipsoid'. The RR Lyrae own parameters (RRL_OWN) are refitted
(maximum likelihood) on their stars at OUTER_INNER < rho < 9 deg with the joint geometry and streaming held: over 2-9 deg one Gaia
error scale cannot describe the crowded inner disc and the fit drives their dispersion gradient to -0.57 per kpc, so their quoted
values come from 4-9 deg (scripts/paper_kin_support_ellipsoid.py). All RR Lyrae at 2-9 deg are then shown with these parameters.

Inputs: the giant and RR Lyrae catalogues of lmcold.samples (Zenodo bundle 1) and the joint-fit tables results/kin_mcmc_data_centre.csv,
kin_mcmc_data_stream.csv and kin_mcmc_data_ellipsoid.csv (scripts/fit_kinematics_mcmc.py). The paper and KB chapter 6 use both runs:
with 'ellipsoid' (main text) and without (the sky-aligned model of the checks).

Run: .venv/bin/python scripts/paper_kin_residuals.py [ellipsoid]
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d

from fit_kinematics_mcmc import OUTER_INNER, ring
from lmcold import geometry, harmonics, kinfit, samples
from lmcold.constants import KAPPA, LMC
from lmcold.footprint import angular_distance
from lmcold.layout import panel_grid
from lmcold.sources import ROOT

# %% [markdown]
# ### Settings and helper functions
# The catalogue columns, the clipping of outliers (`CLIP`), the map cells (`CELL`, `NMIN`), the ring of the position-angle profiles
# (`PROFILE_RING`), the colour ranges, and the model whose residuals are drawn: with the argument `ellipsoid` the joint fit of the
# velocity-ellipsoid model (`results/kin_mcmc_data_ellipsoid.csv`), otherwise that of the sky-aligned model. `params()` reads the
# stage-2 posterior medians of a group, `residuals()` gives each star's residual velocities about a model, and `predicted_sd()` the
# scatter the model predicts for each of them.

# %%
COLS = ('ra', 'dec', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error', 'pmra_pmdec_corr', 'vlos', 'evlos')   # kinfit.data() inputs
CLIP = 150.0  # km/s: residual beyond which a star is dropped (plus three times its own error for proper motions)
CELL, NMIN = 1.0, 5  # deg, minimum stars per map cell
PROFILE_RING = (3.0, 7.0)  # deg, stars used in the position-angle profiles
PA_EDGES = np.arange(0, 361, 30)  # deg: position-angle bins of the profiles
COMPS = ('v_t', 'v_r', 'v_los')   # residual components: tangential, radial, line of sight
COLOURS = ['#d7191c', '#fdae61', '#a6d96a', '#1a9641', '#2878b5', '#7b3294']  # giant bins metal-poor first, RR Lyrae (corner plots)
# colour range [km/s] and map of the data and model rows: v_t is negative everywhere (clockwise rotation), so a sequential map
VLIM = dict(v_t=(-100, 0, 'viridis'), v_r=(-50, 50, 'RdBu_r'), v_los=(-50, 50, 'RdBu_r'))
ZLIM = 4.0  # colour range of the residual significance z
LABELS = {'v_t': r'$v_t$', 'v_r': r'$v_r$', 'v_los': r'$v_{\rm los}$'}   # axis labels of the components
VDMK14_FREE = ('v0', 'R0', 'sigma', 'k', 'logit_fout')  # fitted in the vdMK14 frame (plus sigma_los for the giants)
RRL_OWN = ('v0', 'R0', 'vslope', 'k', 'logit_fout') + kinfit.ELLIPSOID  # RR Lyrae, refitted at 4-9 deg in the joint frame (ellipsoid)
RADIAL_EDGES = np.array([2., 3., 4., 5., 6., 7., 7.5, 9.5])  # deg from the Level-2 centre, radial residual profile
ELLIPSOID = 'ellipsoid' in sys.argv[1:]          # command-line argument: use the velocity-ellipsoid run as Level 2
SUFFIX = '_ellipsoid' if ELLIPSOID else ''       # appended to every output name
L2_RUN = 'ellipsoid' if ELLIPSOID else 'stream'  # results/kin_mcmc_data_<L2_RUN>.csv
# Column titles of the residual-profile figure.
MODEL_TITLES = dict(vdMK14='geometry of vdMK14', L1='Level 1 (symmetric disc)',
                    L2='Level 2, velocity ellipsoid' if ELLIPSOID else 'Level 2 (+ streaming)')


def params(csv, pop):
    """Stage-2 posterior medians of results/<csv>: shared parameters plus population pop's own, as a kinfit.PARAMS vector."""
    t = pd.read_csv(ROOT / 'results' / csv)   # columns stage, population, param, median, p16, p84
    t = t[(t.stage == 2) & t.population.isin(['shared', pop])]
    # Start from kinfit.start() (values of parameters not in the fit) and insert each fitted median.
    p = kinfit.start()
    for _, r in t.iterrows():
        p[kinfit.PARAMS.index(r.param)] = r['median']
    return p


def residuals(p, c):
    """Residuals (observed - model) of catalogue dict c for parameter vector p: position angle about p's centre [deg] and
    v_t, v_r, v_los [km/s] (v_los NaN without a velocity)."""
    pa, vt, vr = kinfit.residual_polar(p, c)   # PA [deg], v_t, v_r residuals [km/s], shape (N,) each
    # Line-of-sight residual: observed minus model v_los (third output of kinfit.model) [km/s].
    d = kinfit.data(*(c[k] for k in COLS))
    vl = np.asarray(c['vlos']) - np.asarray(kinfit.model(p, d['rhat'], d['ea'], d['ed'])[2])
    return pa, dict(v_t=vt, v_r=vr, v_los=vl)


def predicted_sd(p, c):
    """Scatter [km/s] that parameter vector p predicts for each star's v_t, v_r and v_los residuals: intrinsic dispersion plus
    the k-scaled measurement errors, as in kinfit.loglike_stars. residual_polar is linear in the proper motion, so the change of
    its output for a unit change of pmra (pmdec) gives each star's km/s per mas/yr along t and r (kappa D times the east (north)
    component of the unit vector)."""
    # Residuals as observed and with pmra or pmdec raised by 1 mas/yr: their differences are the Jacobian (km/s per mas/yr).
    _, vt, vr = kinfit.residual_polar(p, c)
    _, vt_a, vr_a = kinfit.residual_polar(p, c | dict(pmra=c['pmra'] + 1.0))
    _, vt_d, vr_d = kinfit.residual_polar(p, c | dict(pmdec=c['pmdec'] + 1.0))
    # Proper-motion errors [mas/yr] and their covariance [(mas/yr)^2].
    sa, sd = c['pmra_error'], c['pmdec_error']
    cov = c['pmra_pmdec_corr'] * sa * sd
    # Sky-aligned model: PM dispersion sigma [km/s], Gaia error scale k, line-of-sight dispersion sigma_los [km/s].
    sigma, k, sigma_los = (p[kinfit.PARAMS.index(n)] for n in ('sigma', 'k', 'sigma_los'))
    if ELLIPSOID:
        # Intrinsic variance along a sky direction: (je, jn) / (kappa D) is its unit vector in (east, north).
        d = kinfit.data(*(c[k] for k in COLS))
        C = np.asarray(kinfit.intrinsic_cov(p, d))
        kD = KAPPA * np.asarray(kinfit.model(p, d['rhat'], d['ea'], d['ed'])[3])
        intrinsic = lambda je, jn: (je ** 2 * C[:, 0, 0] + jn ** 2 * C[:, 1, 1] + 2 * je * jn * C[:, 0, 1]) / kD ** 2  # noqa: E731
        intrinsic_los = C[:, 2, 2]
    else:
        # Sky-aligned model: the same isotropic variance sigma^2 in every sky direction.
        intrinsic = lambda je, jn: sigma ** 2  # noqa: E731
        intrinsic_los = sigma_los ** 2
    # Predicted SD [km/s] of v_t and v_r: intrinsic variance + k^2 x the PM error ellipse projected with the Jacobian (je, jn).
    out = {}
    for comp, (je, jn) in (('v_t', (vt_a - vt, vt_d - vt)), ('v_r', (vr_a - vr, vr_d - vr))):
        out[comp] = np.sqrt(intrinsic(je, jn) + k ** 2 * (je ** 2 * sa ** 2 + jn ** 2 * sd ** 2 + 2 * je * jn * cov))
    out['v_los'] = np.sqrt(intrinsic_los + np.asarray(c['evlos']) ** 2)   # v_los: intrinsic plus velocity error [km/s]
    return out


# %% [markdown]
# ### Load the data
# The giant catalogues of Zenodo bundle 1 through `lmcold.samples.giants()`, merged and split into the five [Fe/H] bins, and the
# RR Lyrae members through `lmcold.samples.rrab()` (from `data/rrab.parquet`; proper motions only, so their line-of-sight velocity
# columns are set to NaN), each cut to $2<\rho<9^\circ$ by `ring`. The joint fits are read inside the loop below by `params()`:
# the stage-2 posterior medians of `results/kin_mcmc_data_centre.csv` (Level 1) and of `results/kin_mcmc_data_stream.csv` (Level 2;
# with the argument `ellipsoid`, `results/kin_mcmc_data_ellipsoid.csv`).

# %%
# Giants: (name, catalogue dict) for each [Fe/H] bin, from the release catalogues data/giants_apogee.parquet, giants_boss.parquet.
cats = [(f'giants {lab}', ring(c)) for lab, c in samples.giants()]
# RR Lyrae members from data/rrab.parquet, without line-of-sight velocities.
cats.append(('RR Lyrae', ring(samples.rrab() | dict(vlos=np.full(len(samples.rrab()['ra']), np.nan),
                                                     evlos=np.full(len(samples.rrab()['ra']), np.nan)))))

# %% [markdown]
# ### Residuals of every star about each model
# For each giant [Fe/H] bin and the RR Lyrae at 2-9 deg: the residuals about the joint frame without rotation (the data), about
# the vdMK14 frame (rotation curve and dispersions refitted by maximum likelihood), about the joint Level-1 and Level-2 fits, and the
# residual divided by its predicted scatter (the pull). With `ellipsoid`, the RR Lyrae parameters are first refitted at 4-9 deg in
# the joint frame. Stars with residuals beyond `CLIP` are dropped, and each component is fitted with an $m=1$ plus $m=2$ harmonic
# series in position angle (`lmcold.harmonics`).

# %%
vdmk14 = kinfit.start()  # vdMK14 centre, systemic motion, i, theta (lmcold.constants.LMC)
res, rows = {}, []   # per-population results for the figures; rows of the harmonic table
for name, c in cats:
    # Level-1 and Level-2 parameter vectors of this population (stage-2 posterior medians).
    l1, l2 = params('kin_mcmc_data_centre.csv', name), params(f'kin_mcmc_data_{L2_RUN}.csv', name)
    # Ellipsoid run: RR Lyrae own parameters refitted on stars at OUTER_INNER (4) to 9 deg, joint frame held (docstring).
    if ELLIPSOID and name == 'RR Lyrae':
        o = ring(c, OUTER_INNER)
        l2 = kinfit.fit(kinfit.data(*(o[k] for k in COLS)), free=RRL_OWN, p0=l2, ellipsoid=True)[0]
        print('RR Lyrae refitted at 4-9 deg:', {n: round(float(l2[kinfit.PARAMS.index(n)]), 3) for n in RRL_OWN}, flush=True)
    # 'data' frame: Level 2 with v0 = 0 (no rotation, no streaming).
    data_p = l2.copy()
    data_p[kinfit.PARAMS.index('v0')] = 0.0
    # vdMK14 frame: rotation curve, dispersions, error scale, outlier fraction (and sigma_los if velocities) fitted by ML.
    los = ('sigma_los',) if np.isfinite(c['vlos']).any() else ()
    vdmk14_p = kinfit.fit(kinfit.data(*(c[k] for k in COLS)), free=VDMK14_FREE + los, p0=vdmk14)[0]
    # Residuals (PA [deg], {component: km/s}) about each model; 'model' = data minus L2 residual = L2 model minus frame.
    out = {m: residuals(p, c) for m, p in (('data', data_p), ('vdMK14', vdmk14_p), ('L1', l1), ('L2', l2))}
    out['model'] = (out['data'][0], {k: out['data'][1][k] - out['L2'][1][k] for k in COMPS})
    # Pull: Level-2 residual / its predicted SD (dimensionless).
    sd = predicted_sd(l2, c)
    out['pull'] = (out['L2'][0], {k: out['L2'][1][k] / sd[k] for k in COMPS})
    # Proper-motion components: CLIP plus three times the star's own error [km/s] (faint RR Lyrae errors reach 100 km/s).
    # D: disc distance of each star [kpc] (fourth output of kinfit.model); epm: PM error in km/s at that distance.
    D = np.asarray(kinfit.model(l2, *(kinfit.data(*(c[k] for k in COLS))[k] for k in ('rhat', 'ea', 'ed')))[3])
    epm = KAPPA * D * np.hypot(c['pmra_error'], c['pmdec_error'])
    vv = out['L2'][1]   # Level-2 residuals [km/s]
    # Clipped stars (the same mask is applied to every model and panel).
    bad = (np.abs(vv['v_t']) > CLIP + 3 * epm) | (np.abs(vv['v_r']) > CLIP + 3 * epm) | (np.abs(np.nan_to_num(vv['v_los'])) > CLIP)
    rho = angular_distance(c['ra'], c['dec'], LMC['ra0'], LMC['dec0'])   # [deg] from the vdMK14 centre
    # Stars of the position-angle profiles: PROFILE_RING (3-7 deg), not clipped.
    prof = (rho > PROFILE_RING[0]) & (rho < PROFILE_RING[1]) & ~bad
    cra, cdec = (float(v) for v in kinfit.centre(*l2[:2]))  # joint Level-2 centre [rad], the same for every population
    # Orthographic tangent-plane coordinates about the Level-2 centre [deg]: x east, y north.
    x, y = (np.rad2deg(np.asarray(v)) for v in geometry.tangent_plane_coords(np.deg2rad(c['ra']), np.deg2rad(c['dec']), cra, cdec))
    # Kept for plotting: positions, profile mask, residuals of every model (clipped stars removed), vdMK14 centre position [deg]
    # on the same plane, and the Level-2 line of nodes theta [rad].
    res[name] = dict(x=x[~bad], y=y[~bad], prof=prof[~bad], out={m: (pa[~bad], {k: v[~bad] for k, v in vv.items()})
                                                                  for m, (pa, vv) in out.items()},
                     vdmk14=[float(np.rad2deg(v)) for v in geometry.tangent_plane_coords(np.deg2rad(LMC['ra0']), np.deg2rad(LMC['dec0']),
                                                                                         cra, cdec)],
                     theta=np.deg2rad(l2[kinfit.PARAMS.index('theta')]))
    # Harmonic fit (m <= 2) of each component of the residual models 'data', 'vdMK14', 'L1', 'L2', over >= 50 profile stars.
    for m, (pa, vv) in out.items():
        if m in ('model', 'pull'):
            continue
        for comp, v in vv.items():
            k = prof & ~bad & np.isfinite(v)
            if k.sum() < 50:
                continue
            # Coefficients [c0, a1, b1, a2, b2] [km/s]; amplitude [km/s] and PA of the maximum [deg] of m = 1 and m = 2.
            cc, _ = harmonics.fit(pa[k], v[k])
            (A1, P1), (A2, P2) = harmonics.amplitude_phase(cc, 1), harmonics.amplitude_phase(cc, 2)
            rows.append(dict(population=name, model=m, component=comp, n=int(k.sum()), mean=cc[0], A1=A1, phase1=P1, A2=A2, phase2=P2))
    print(f'{name:28s} N {len(c["ra"]):6d}, dropped {bad.sum():5d} ({100 * bad.mean():.1f}%)', flush=True)
# %% [markdown]
# ### Harmonic amplitudes and plotting functions
# The table of harmonic amplitudes and phases (`results/kin_residual_profiles*.csv`), the line of nodes of each frame, and the two
# drawing functions: `maps()` for the data, model and significance rows of one component, `profiles()` for the medians against
# position angle.

# %%
# Harmonic amplitudes and phases of every population, model and component.
harm = pd.DataFrame(rows)
harm.to_csv(ROOT / f'results/kin_residual_profiles{SUFFIX}.csv', index=False, float_format='%.3f')
names = [n for n, _ in cats]   # population names, giant bins metal-poor first, then RR Lyrae
# Line of nodes [deg] of each frame (shared by all populations in the joint fits), for the reference lines of the profiles.
THETA = dict(vdMK14=LMC['theta'], L1=l1[kinfit.PARAMS.index('theta')], L2=l2[kinfit.PARAMS.index('theta')],
             data=l2[kinfit.PARAMS.index('theta')])
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
edges = np.arange(-10.0, 10.01, CELL)  # the 9-deg selection about vdMK14 reaches 9.5 deg from the Level-2 centre
centres_pa = 0.5 * (PA_EDGES[1:] + PA_EDGES[:-1])   # PA bin centres [deg]


def maps(comp):
    """Maps of component comp: rows data, Level-2 model, residual; one column per population with this component."""
    pops = [n for n in names if np.isfinite(res[n]['out']['data'][1][comp]).any()]   # RR Lyrae dropped for v_los
    size = (7.0 - 0.55 - 0.75 - 5 * 0.05) / 6  # panel width [inch] of the six-column figures
    fig, ax = panel_grid(3, len(pops), width=0.55 + 0.75 + len(pops) * size + (len(pops) - 1) * 0.05, left=0.55, right=0.75,
                         hgap=0.05, vgap=0.05, top=0.3)
    # Rows: (model key, row label, (vmin, vmax, colour map)).
    rows = (('data', 'data', VLIM[comp]), ('model', 'Level-2 ellipsoid model' if ELLIPSOID else 'Level-2 model', VLIM[comp]),
            ('pull', 'residual significance', (-ZLIM, ZLIM, 'RdBu_r')))
    for j, name in enumerate(pops):
        r = res[name]
        for i, (m, lab, (vmin, vmax, cmap)) in enumerate(rows):
            a, v = ax[i, j], r['out'][m][1][comp]
            ok = np.isfinite(v)
            # Median and number of stars per CELL x CELL deg cell (shape: len(edges) - 1 squared).
            med = binned_statistic_2d(r['x'][ok], r['y'][ok], v[ok], 'median', bins=[edges, edges]).statistic
            n = binned_statistic_2d(r['x'][ok], r['y'][ok], v[ok], 'count', bins=[edges, edges]).statistic
            if m == 'pull':
                med = med * np.sqrt(n) / 1.2533  # significance z of the cell's median pull
                z = med[n >= NMIN]
                CELLS.append(dict(population=name, component=comp, n_cells=len(z), z_sd=np.std(z), z_gt3=np.mean(np.abs(z) > 3)))
            # Cells with fewer than NMIN stars blank; transpose because binned_statistic_2d indexes [x, y].
            im = a.pcolormesh(edges, edges, np.where(n >= NMIN, med, np.nan).T, cmap=cmap, vmin=vmin, vmax=vmax,
                              rasterized=True)
            # Level-2 line of nodes through the origin (PA theta, north through east) and the vdMK14 centre (+).
            th = r['theta']
            a.plot(10 * np.sin(th) * np.array([-1, 1]), 10 * np.cos(th) * np.array([-1, 1]), 'k--', lw=0.5)
            a.plot(*r['vdmk14'], '+', color='k', ms=5, mew=0.7)
            a.set(xlim=(10, -10), ylim=(-10, 10), aspect='equal', xticks=[5, 0, -5], yticks=[-5, 0, 5])
            a.tick_params(labelsize=5)
            # Column titles on the top row; tick labels only on the outer panels.
            if i == 0:
                a.text(0.5, 1.04, name.replace('giants ', ''), transform=a.transAxes, ha='center', fontsize=6)
            if i < 2:
                a.tick_params(labelbottom=False)
            else:
                a.set_xlabel('east [deg]', fontsize=6)
            if j:
                a.tick_params(labelleft=False)
            else:
                a.set_ylabel(f'{lab}\nnorth [deg]', fontsize=6)
            if j == len(pops) - 1 and i != 1:
                # colour bars right of the last column: data and model share one (spanning rows 1-2), the significance its own
                cax = a.inset_axes([1.08, -1.05 if i == 0 else 0.0, 0.07, 2.05 if i == 0 else 1.0])
                fig.colorbar(im, cax=cax)
                cax.tick_params(labelsize=5)
                cax.set_ylabel(LABELS[comp] + r' [km s$^{-1}$]' if i == 0 else f'$z$ ({LABELS[comp]} residual)', fontsize=6)
    fig.savefig(ROOT / f'figures/paper_kin_maps_{comp.replace("_", "")}{SUFFIX}.png')
    plt.close(fig)


def profiles(ax_col, model, letters, harmonics_text):
    """Median v_t, v_r, v_los of model per PA bin at PROFILE_RING, one line per population, in the axes column ax_col."""
    for i, comp in enumerate(COMPS):
        a = ax_col[i]
        for name, colour in zip(names, COLOURS):
            pa, vv = res[name]['out'][model]
            k = res[name]['prof'] & np.isfinite(vv[comp])
            if not k.any():
                continue
            # PA bin of each star; median per bin with >= 10 stars, else NaN.
            b = np.digitize(pa[k], PA_EDGES) - 1
            med = [np.median(vv[comp][k][b == j]) if (b == j).sum() >= 10 else np.nan for j in range(len(centres_pa))]
            a.plot(centres_pa, med, 'o-', ms=2.2, lw=0.8, color=colour, label=name.replace('giants ', 'giants, '))
        th = THETA[model]  # the line of nodes of the frame the residuals are taken in
        # Line of nodes (dotted) and minor axis (dash-dotted) at theta, theta + 180 and theta -+ 90 deg.
        for pa0, ls in ((th, ':'), (th + 180, ':'), (th - 90, '-.'), (th + 90, '-.')):
            a.axvline(pa0 % 360, color='0.6', lw=0.5, ls=ls)
        a.axhline(0, color='0.6', lw=0.5)
        a.set(xlim=(0, 360), xticks=np.arange(0, 361, 60))
        a.text(0.02, 0.92, f'({letters[i]})', transform=a.transAxes, fontsize=6)
        # Range over the giant bins of the m = 1 and m = 2 amplitudes [km/s] of this model and component.
        if harmonics_text:
            h = harm[(harm.model == model) & (harm.component == comp) & harm.population.str.startswith('giants')]
            a.text(0.98, 0.05, f'giants: $A_1$ = {h.A1.min():.0f}$-${h.A1.max():.0f}, $A_2$ = {h.A2.min():.0f}$-${h.A2.max():.0f}'
                   r' km s$^{-1}$', transform=a.transAxes, ha='right', fontsize=5)


# %% [markdown]
# ### Maps of $v_t$, $v_r$ and $v_{\rm los}$
# One figure per component; the statistics of the cell significances go to `results/kin_residual_cells*.csv`.

# %%
CELLS = []   # filled by maps(): statistics of the cell significances z per population and component
# One map figure per component (v_t, v_r, v_los).
for comp in COMPS:
    maps(comp)
# Write and print the cell statistics.
pd.DataFrame(CELLS).to_csv(ROOT / f'results/kin_residual_cells{SUFFIX}.csv', index=False, float_format='%.3f')
print(pd.DataFrame(CELLS).round(2).to_string(index=False))

# %% [markdown]
# ### Radial profile of the residuals
# Median Level-2 residual of each component in rings about the Level-2 centre (`RADIAL_EDGES`).

# %%
# Radial profile of the Level-2 residuals about the Level-2 centre.
radial = []
for name in names:
    r = res[name]
    rho = np.hypot(r['x'], r['y'])   # [deg] from the Level-2 centre on the tangent plane
    for comp in COMPS:
        v = r['out']['L2'][1][comp]   # Level-2 residuals [km/s]
        # Rings of RADIAL_EDGES with >= 20 stars: median and its error 1.2533 SD / sqrt(n) [km/s].
        for lo, hi in zip(RADIAL_EDGES[:-1], RADIAL_EDGES[1:]):
            k = (rho > lo) & (rho < hi) & np.isfinite(v)
            if k.sum() >= 20:
                radial.append(dict(population=name, component=comp, rho_lo=lo, rho_hi=hi, n=int(k.sum()), median=np.median(v[k]),
                                   err=1.2533 * np.std(v[k]) / np.sqrt(k.sum())))
# Table and figure: three equal panels (v_t, v_r, v_los), one line per population, offset by 0.04 deg in x per population.
radial = pd.DataFrame(radial)
radial.to_csv(ROOT / f'results/kin_residual_radial{SUFFIX}.csv', index=False, float_format='%.3f')
fig, ax = panel_grid(1, 3, width=7.0, left=0.55, right=0.1, hgap=0.5, top=0.6, aspect=0.8)
for a, comp, letter in zip(ax[0], COMPS, 'abc'):
    a.text(0.04, 0.92, f'({letter})', transform=a.transAxes, fontsize=7)
    for (name, g), colour in zip(radial[radial.component == comp].groupby('population', sort=False), COLOURS):
        a.errorbar(0.5 * (g.rho_lo + g.rho_hi) + 0.04 * (COLOURS.index(colour) - 2.5), g['median'], yerr=g.err, fmt='o-', ms=2.5,
                   lw=0.8, color=colour, label=name.replace('giants ', 'giants, '))
    a.axhline(0, color='0.6', lw=0.5)   # zero residual
    a.set(xlabel=r'$\rho$ from the Level-2 centre [deg]', ylabel=r'median $\Delta$' + LABELS[comp] + r' [km s$^{-1}$]', ylim=(-15, 15))
ax[0, 0].legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=3, fontsize=5, frameon=False)
fig.savefig(ROOT / f'figures/paper_kin_residual_radial{SUFFIX}.png')
plt.close(fig)

# %% [markdown]
# ### The data against position angle
# Median $v_t$, $v_r$ and $v_{\rm los}$ of every group per 30-deg bin of position angle at 3-7 deg, in the joint frame.

# %%
# Three stacked panels (v_t, v_r, v_los) of the 'data' model, no amplitude text.
fig, ax = panel_grid(3, 1, width=3.4, left=0.55, right=0.1, vgap=0.1, top=0.65, aspect=0.55)
profiles(ax[:, 0], 'data', 'abc', harmonics_text=False)
# y ranges [km/s] per component; x tick labels only on the bottom panel.
for i, (comp, lim) in enumerate(zip(COMPS, ((-110, 10), (-40, 40), (-60, 60)))):
    ax[i, 0].set(ylim=lim, ylabel=LABELS[comp] + r' [km s$^{-1}$]')
    if i < 2:
        ax[i, 0].tick_params(labelbottom=False)
ax[2, 0].set_xlabel('position angle [deg]')
# Legend above the top panel; save.
ax[0, 0].legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=2, fontsize=5, frameon=False)
fig.savefig(ROOT / f'figures/paper_kin_data_profiles{SUFFIX}.png')
plt.close(fig)

# %% [markdown]
# ### Residuals of the three models against position angle
# Columns: the vdMK14 frame, the joint Level-1 fit and the joint Level-2 fit.

# %%
# 3 x 3 panels: rows v_t, v_r, v_los; columns the three models, each with the amplitude range of the giants.
fig, ax = panel_grid(3, 3, width=7.0, left=0.55, right=0.1, hgap=0.1, vgap=0.1, top=0.75, aspect=0.7)
for j, model in enumerate(('vdMK14', 'L1', 'L2')):
    profiles(ax[:, j], model, 'abcdefghi'[3 * j:3 * j + 3], harmonics_text=True)
    ax[0, j].set_title(MODEL_TITLES[model], fontsize=7)
    for i, comp in enumerate(COMPS):
        # Common y range +-35 km/s; the 360 tick only on the last column.
        ax[i, j].set(ylim=(-35, 35), xticks=np.arange(0, 361 if j == 2 else 360, 60))
        # Outer tick labels only: x labels on the bottom row, y labels on the first column.
        if i < 2:
            ax[i, j].tick_params(labelbottom=False)
        else:
            ax[i, j].set_xlabel('position angle [deg]')
        if j:
            ax[i, j].tick_params(labelleft=False)
        else:
            ax[i, j].set_ylabel({'v_t': r'$\Delta v_t$', 'v_r': r'$\Delta v_r$', 'v_los': r'$\Delta v_{\rm los}$'}[comp]
                                + r' [km s$^{-1}$]')
# Legend above the top-left panel; save; print the range of the giants' amplitudes per model and component.
ax[0, 0].legend(loc='lower left', bbox_to_anchor=(0, 1.15), ncol=6, fontsize=5, frameon=False)
fig.savefig(ROOT / f'figures/paper_kin_residual_profiles{SUFFIX}.png')
print(harm[harm.population.str.startswith('giants')].groupby(['model', 'component'])[['A1', 'A2']].agg(['min', 'max']).round(1))
