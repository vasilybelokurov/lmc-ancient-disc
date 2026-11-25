"""Observe the simulated LMC analogues like the LMC giants and fit them with the same kinematic model (paper Section 6).

Makes results/sims_observed.csv, the table behind the Letter's Fig. 2 (figures/paper_sims_kinematics_fraction.png), the extended
paper's Fig. sims_kinematics, Fig. sims_recovery and Table sims_definitions (all drawn by scripts/paper_sims_kinematics.py), and the
'observed' column of scripts/fig_sims_populations.py.

Reads: the star particles of every analogue (scripts/fetch_tng50.py, fetch_auriga.py, fetch_fire2.py; read by lmcold.sims.load);
the real giants (lmcold.samples.giants(), lmcold.samples.overlap(): the APOGEE and BOSS catalogues of Zenodo bundle 1); the joint
ellipsoid fit of the data, results/kin_mcmc_data_ellipsoid.csv (stage 2: the frame; stage 1: the error factors k); the LMC's formed
mass per [Fe/H] bin, results/giant_mass.csv; the giants per formed mass, results/giant_conversion.csv; and the PARSEC grid
(data/sources.yaml key parsec_gaiaEDR3) for the formed masses of TNG50 and FIRE-2.

For each analogue (lmcold.sims.names(): TNG50, Auriga L3, FIRE-2):
1. Particles within R_MAX of the centre, in the frame of their spin (lmcold.sims.disc_frame), placed with the frame of the joint
   ellipsoid fit of the data (results/kin_mcmc_data_ellipsoid.csv, stage 2: centre, inclination, line of nodes, systemic proper motion
   and v_los; distance LMC['D0']): every particle at its own 3D position (lmcold.sims.place).
2. Tracer: expected BOSS-selected red giants per particle E = m_formed x n_box(age, [M/H]) (results/giant_conversion.csv, BOSS grid).
3. Metallicity on the LMC scale (lmcold.sims.lmc_scale): matched in cumulative formed mass inside 12 deg (the radius of the LMC mass
   measurement, results/giant_mass.csv: the cumulative fraction at each upper bin edge, and 0 at LMC_FEH_FLOOR).
4. Sampling where the data are: each real kinematic giant (lmcold.samples.giants(), 2 < rho < 9 deg) lies in a cell (CELL_R kpc in
   in-plane radius R x CELL_PHI deg in azimuth, both in the fit frame, from its sky position as in lmcold.kinfit). For each real
   giant one mock giant is drawn, with probability proportional to E, from the particles in the same cell (with replacement), inside
   the same ring, with the same velocity window (100 < v_los < 450 km/s). The mock giant takes the real giant's proper-motion
   covariance and v_los error: proper-motion noise = that covariance x k of the real giant's bin (stage-1 ellipsoid fit of the data,
   so the mock errors are as underestimated as the real ones), v_los noise = its error. Its measured [Fe/H] = LMC-scale [Fe/H] + an
   error: 0 for an APOGEE donor; for a BOSS donor a random BOSS - APOGEE difference of the stars in both surveys (lmcold.samples.
   overlap). Mock giants are then binned in measured [Fe/H] (lmcold.samples.FEH_BINS). A real giant whose cell holds no particle
   gets no mock (counted).
5. Fit: each bin on its own (at most NMAX mock giants, a random subset of a larger bin: only the precision of the metal-rich bins,
   which hold up to 29,000 real giants, is reduced; the time per analogue drops from ~9 to ~3 min) with the data's stage-1 maximum likelihood (scripts/fit_kinematics_mcmc.ml, model 'ellipsoid'; geometry,
   systemic motion, streaming, rotation curve, ellipsoid, k and outliers free).
6. Truth: from the particles of the mock giants (disc frame, 3D): mean v_phi and sigma_R, sigma_phi, sigma_z of those with
   |R - 4| < TRUTH_DR kpc.

Writes results/sims_observed.csv (suite, name, population, n (mock giants in the bin), n_part (distinct particles), lost (real giants
without a mock in this analogue), the fitted vrot4 [v_rot(4 kpc)], sigma_R, sigma_phi, sigma_z, dlnsig_dR, k, logit_fout, inc,
theta, nll, fit_warnings (number of 'kinfit.fit did not converge' warnings of the bin's maximum likelihood, three fits; 0 when
all converged), the truth t_vphi, t_sigma_R, t_sigma_phi, t_sigma_z, t_n, and the whole fitted vector as p_<parameter>).
Velocities in km/s, dispersions in km/s, inc and theta in deg.

Each analogue is written to its own file, PARTS/<suite>_<name>.csv (written whole and renamed into place, so a file is complete or
absent); analogues whose file exists are skipped, so workers on disjoint subsets can run in parallel and be restarted. 'merge' joins
them into results/sims_observed.csv only if every analogue of lmcold.sims.names() is present with all its [Fe/H] bins (else it exits
with an error and leaves the csv unchanged), in that order.

Run: .venv/bin/python scripts/sims_observe.py [index of this worker] [number of workers]
     .venv/bin/python scripts/sims_observe.py merge
Run time: about 13 h for all analogues on 5 parallel workers (docs/release.md), about 3 min per analogue (step 5).
"""
# %% [markdown]
# ### Settings
# The sizes of the sampling cells, the velocity window of the giant catalogues, the LMC metallicity floor and mass radius, the
# radius window of the particle truth and the largest bin fitted.

# %%
import os
import sys
import time
import warnings
import zlib

import numpy as np
import pandas as pd

from fit_kinematics_mcmc import COLS, ml, ring
from kin_vc_drift import joint_params
from lmcold import kinfit, samples, sims, surviving_mass
from lmcold.constants import LMC
from lmcold.footprint import angular_distance
from lmcold.sources import DATA, ROOT, path

CELL_R, CELL_PHI = 1.0, 45.0  # kpc, deg: cells of the spatial sampling
VLOS_WINDOW = (100.0, 450.0)  # km/s, as the release giant catalogues
LMC_FEH_FLOOR = -2.5  # dex, [Fe/H] where the LMC cumulative mass is taken to reach zero (below the lowest measured edge, -2)
RHO_MASS = 12.0  # deg, radius of the LMC mass measurement
TRUTH_DR = 0.5  # kpc
NMAX = 8000  # mock giants fitted per bin
OUT = ROOT / 'results/sims_observed.csv'
PARTS = DATA / 'sims_observe_parts'  # one csv per analogue
P = kinfit.PARAMS.index   # P('inc') etc.: position of a named parameter in the kinfit parameter vector
# Columns of the csv, fixed so that appended rows of bins without a fit (fewer than 50 mock giants) line up.
FIELDS = (['suite', 'name', 'population', 'n', 'n_part', 'lost', 't_vphi', 't_sigma_R', 't_sigma_phi', 't_sigma_z', 't_n', 'vrot4', 'nll', 'fit_warnings']
          + list(kinfit.ELLIPSOID + ('k', 'logit_fout', 'inc', 'theta')) + [f'p_{n}' for n in kinfit.PARAMS])


# %% [markdown]
# ### Helpers: sampling cells and the LMC's metallicity scale
# `sky_cell` gives the cell (in-plane radius and azimuth in the fitted disc frame) of any sky position, so that real giants and
# simulated particles are put in the same cells. `lmc_knots` is the LMC's cumulative formed-mass fraction against [Fe/H], to which
# the simulated metallicities are matched (step 3).

# %%
def sky_cell(p, ra, dec):
    """Cell index (R // CELL_R, phi // CELL_PHI) of sky positions in the frame of p, with R and phi as lmcold.kinfit computes them
    (line of sight to the disc plane).

    p: kinfit parameter vector (kinfit.PARAMS); ra, dec: (N,) [deg]. Returns (N,) int, 1000 x radial cell + azimuthal cell, with
    the in-plane radius R [kpc] and the azimuth phi [deg, 0-360, from the line of nodes] where the line of sight meets the disc plane."""
    n = len(ra)
    # Line of sight of each position meets the disc plane of frame p: R [kpc], unit radial vector eR and disc normal nn.
    d = kinfit.data(ra, dec, np.zeros(n), np.zeros(n), np.ones(n), np.ones(n), np.zeros(n))   # positions only; motions are dummies
    _, _, _, _, R, eR, _, nn = (np.asarray(v) for v in kinfit._model(p, d['rhat'], d['ea'], d['ed']))
    # Unit vector along the line of nodes of the frame (centre, inclination, position angle theta of p).
    _, _, _, line, _ = (np.asarray(v) for v in kinfit.geo.symmetry_axis(*(float(c) for c in kinfit.centre(p[0], p[1])),
                                                                        np.deg2rad(p[P('inc')]), np.deg2rad(p[P('theta')])))
    phi = np.rad2deg(np.arctan2(eR @ np.cross(nn[0], line), eR @ line)) % 360   # deg, azimuth from the line of nodes
    return (R // CELL_R).astype(int) * 1000 + (phi // CELL_PHI).astype(int)


def lmc_knots():
    """The LMC's cumulative formed-mass fraction against [Fe/H] (results/giant_mass.csv): ([Fe/H] knots from LMC_FEH_FLOOR, fraction
    of the formed mass within RHO_MASS below each).

    Returns two arrays of equal length: [Fe/H] [dex] (LMC_FEH_FLOOR, then the upper edges of the mass bins but the last) and the
    fraction (0 at the floor, rising to below 1)."""
    # One row per [Fe/H] bin: feh_hi (upper edge), cumulative (formed mass below that edge).
    m = pd.read_csv(ROOT / 'results/giant_mass.csv')
    # Knots: the floor plus every upper edge but the last; fractions normalised by the total formed mass.
    return np.r_[LMC_FEH_FLOOR, m.feh_hi.to_numpy()[:-1]], np.r_[0, m.cumulative.to_numpy()[:-1] / m.cumulative.iloc[-1]]


# %% [markdown]
# ### Inputs shared by all analogues
# `setup` reads everything that does not depend on the analogue: the frame of the joint fit, the real giants (their positions, errors,
# survey and cell, and the error factor $k$ of their [Fe/H] bin), the BOSS − APOGEE [Fe/H] differences, the LMC metallicity knots, and
# the grids of giants per formed mass and of surviving mass.

# %%
def setup():
    """Inputs shared by all analogues: frame vector, real giants (one table, with cell, bin, k), BOSS - APOGEE differences, LMC
    cumulative-mass knots, n_box and survival grids.

    Returns (frame: kinfit parameter vector of the joint stage-2 fit; real: DataFrame, one row per real giant at 2-9 deg with ra, dec
    [deg], pmra_error, pmdec_error [mas/yr], pmra_pmdec_corr, evlos [km/s], survey, k, cell; dfeh: (M,) BOSS - APOGEE [Fe/H] [dex] of
    the stars in both surveys; knots: lmc_knots(); nbox: BOSS rows of results/giant_conversion.csv; survival: the surviving-mass
    grid of lmcold.surviving_mass)."""
    frame = next(iter(joint_params().values()))   # the shared parameters (centre, orientation, systemic motion) are the same in every population's vector
    t = pd.read_csv(ROOT / 'results/kin_mcmc_data_ellipsoid.csv')
    ks = t[(t.stage == 1) & (t.param == 'k')].set_index('population')['median']   # proper-motion error factor of each real bin
    real = []
    for j, (lab, c) in enumerate(samples.giants()):
        c = ring(c)   # 2 < rho < 9 deg, as the kinematic fits
        real.append(pd.DataFrame({k: c[k] for k in ('ra', 'dec', 'pmra_error', 'pmdec_error', 'pmra_pmdec_corr', 'evlos', 'survey')})
                    .assign(k=ks[f'giants {lab}']))
    real = pd.concat(real, ignore_index=True)   # all real kinematic giants, every [Fe/H] bin
    real['cell'] = sky_cell(frame, real.ra.to_numpy(), real.dec.to_numpy())
    o = samples.overlap()   # stars in both APOGEE and BOSS
    dfeh = (o.feh_b - o.feh_a).to_numpy()   # (M,) BOSS - APOGEE [Fe/H] [dex]
    knots = lmc_knots()
    g = pd.read_csv(ROOT / 'results/giant_conversion.csv')
    nbox = g[g.survey == 'BOSS']   # giants in the BOSS selection box per unit formed mass, on an ([M/H], log age) grid
    # Surviving fraction of the formed mass on the PARSEC grid, used by lmcold.sims.load for the formed masses.
    survival = surviving_mass.survival_grid(pd.read_parquet(path('parsec_gaiaEDR3')))
    return frame, real, dfeh, knots, nbox, survival


# %% [markdown]
# ### One analogue: placing it and weighting its particles (steps 1-3)
# The particles are rotated into the frame of their spin, placed at the LMC with the fitted frame, given their LMC-scale [Fe/H]
# (only inside the 12-deg cone of the LMC's mass measurement) and weighted by their expected number of BOSS giants $E$.

# %%
def particles(suite, name, frame, knots, nbox, survival):
    """Particles of one analogue (steps 1-3): disc-frame X, V (N, 3), the sky observables of lmcold.sims.place, rho [deg] from the
    vdMK14 centre, LMC-scale [Fe/H] (NaN outside RHO_MASS), the expected BOSS giants per particle E, and the loaded particles
    (lmcold.sims.load: native feh, age, m_formed).

    X [kpc], V [km/s]; the sky observables are a dict of (N,) arrays ra, dec [deg], dist [kpc], pmra, pmdec [mas/yr], vlos [km/s];
    E (N,) is in giants (m_formed [Msun] times giants per Msun)."""
    st = sims.load(suite, name, survival)   # dict of particle arrays: x, v, m, m_formed, feh, age
    X, V, _ = sims.disc_frame(st['x'], st['v'], st['m'])   # (N, 3) [kpc], [km/s] in the frame of the spin
    s = sims.place(X, V, frame)   # sky observables with the LMC's fitted frame and distance
    rho = angular_distance(s['ra'], s['dec'], LMC['ra0'], LMC['dec0'])   # deg from the reference centre (lmcold.constants)
    cone = rho < RHO_MASS   # (N,) bool, inside the 12-deg radius of the LMC mass measurement
    feh = np.full(len(rho), np.nan)
    feh[cone] = sims.lmc_scale(st['feh'][cone], st['m_formed'][cone], *knots)   # matched to the LMC's mass inside the same cone
    # E = formed mass [Msun] x BOSS giants per Msun interpolated at each particle's native [M/H] and age.
    return X, V, s, rho, feh, st['m_formed'] * sims.on_grid(nbox, 'n_box', st['feh'], st['age']), st


# %% [markdown]
# ### Drawing the mock giants (step 4)
# For each real giant, one particle of the same cell, drawn with probability proportional to $E$; the mock giant then gets the real
# giant's errors (proper motions scaled by $k$) and, for a BOSS giant, a BOSS − APOGEE [Fe/H] difference.

# %%
def observe(suite, name, frame, real, dfeh, knots, nbox, survival, rng):
    """Mock giant catalogue of one analogue (module docstring, steps 1-4): dict of arrays with the kinematic columns (COLS), feh
    (measured, LMC scale), particle index, and the disc-frame R, v_R, v_phi, v_z of the particle; and the number of real giants
    without a mock.

    The inputs after suite, name are those of setup(); rng is a numpy Generator. Units: ra, dec [deg], pmra, pmdec and their errors
    [mas/yr], vlos, evlos, vR, vphi, vz [km/s], R [kpc], feh [dex]; one entry per real giant that has a mock."""
    X, V, s, rho, feh, E, _ = particles(suite, name, frame, knots, nbox, survival)
    # Particles that could be giants of the sample: LMC-scale [Fe/H] defined, 2 < rho < 9 deg, inside the velocity window.
    ok = np.isfinite(feh) & (rho > 2) & (rho < 9) & (s['vlos'] > VLOS_WINDOW[0]) & (s['vlos'] < VLOS_WINDOW[1])
    E = np.where(ok, E, 0.0)
    idx = np.flatnonzero(E > 0)   # indices of the candidate particles
    cell = sky_cell(frame, s['ra'][idx], s['dec'][idx])
    pick = np.full(len(real), -1)   # particle drawn for each real giant (-1: none in its cell)
    # Particles sorted by cell, so that the members of each cell are one slice.
    order = np.argsort(cell, kind='stable')
    cells, start = np.unique(cell[order], return_index=True)
    # Cell -> (first, last + 1) position of its members in `order`.
    bounds = dict(zip(cells, zip(start, np.r_[start[1:], len(order)])))
    # For each cell holding real giants, draw as many particles as there are real giants in it.
    for c, rows in real.groupby('cell').indices.items():
        if c in bounds:
            members = idx[order[bounds[c][0]:bounds[c][1]]]
            pick[rows] = rng.choice(members, size=len(rows), p=E[members] / E[members].sum())   # with replacement, weight E
    have = pick >= 0
    r, i = real[have], pick[have]   # the real giants with a mock, and their particles
    # Proper-motion noise: the real giant's 2 x 2 covariance [(mas/yr)^2], drawn through its Cholesky factor, times k of its bin.
    cov = np.stack([np.stack([r.pmra_error ** 2, r.pmra_pmdec_corr * r.pmra_error * r.pmdec_error], -1),
                    np.stack([r.pmra_pmdec_corr * r.pmra_error * r.pmdec_error, r.pmdec_error ** 2], -1)], -2)
    noise = r.k.to_numpy()[:, None] * np.einsum('nij,nj->ni', np.linalg.cholesky(cov), rng.normal(size=(len(r), 2)))
    err = np.where(r.survey == 'boss', rng.choice(dfeh, len(r)), 0.0)   # [Fe/H] error [dex]: none for an APOGEE donor
    R = np.hypot(X[i, 0], X[i, 1])   # kpc, in-plane radius of the particle in its own disc frame
    # Mock catalogue: true sky observables plus noise, the donor's errors, measured [Fe/H], and the particle's disc-frame
    # velocities (v_R, v_phi from the in-plane position and velocity; v_z) for the truth.
    # dmu (RR Lyrae distance modulus) is NaN: no distance term for giants.
    cat = dict(ra=s['ra'][i], dec=s['dec'][i], pmra=s['pmra'][i] + noise[:, 0], pmdec=s['pmdec'][i] + noise[:, 1],
               pmra_error=r.pmra_error.to_numpy(), pmdec_error=r.pmdec_error.to_numpy(), pmra_pmdec_corr=r.pmra_pmdec_corr.to_numpy(),
               vlos=s['vlos'][i] + r.evlos.to_numpy() * rng.normal(size=len(r)), evlos=r.evlos.to_numpy(),
               dmu=np.full(len(r), np.nan), feh=feh[i] + err, particle=i, R=R,
               vR=(X[i, 0] * V[i, 0] + X[i, 1] * V[i, 1]) / R, vphi=(X[i, 0] * V[i, 1] - X[i, 1] * V[i, 0]) / R, vz=V[i, 2])
    return cat, int((~have).sum())


# %% [markdown]
# ### Output files
# Each analogue goes to its own file; `merge` joins them into `results/sims_observed.csv` once all are present.

# %%
def part(suite, name):
    """Output file of one analogue (PARTS/<suite>_<name>.csv)."""
    return PARTS / f'{suite}_{name}.csv'


def merge():
    """Join the per-analogue files into OUT, in the order of sims.names(); exit with an error, leaving OUT unchanged, if an analogue
    is missing or lacks one of its [Fe/H] bins."""
    names = sims.names()   # list of (suite, name) of every analogue
    # Stop if any analogue has no file yet.
    missing = [f'{s} {n}' for s, n in names if not part(s, n).exists()]
    if missing:
        sys.exit(f'merge: {len(missing)} of {len(names)} analogues have no output (e.g. {", ".join(missing[:3])})')
    # Join in the order of sims.names(); names read as strings (TNG50 names are subhalo ids).
    t = pd.concat([pd.read_csv(part(s, n), dtype={'name': str}) for s, n in names], ignore_index=True)
    # Every analogue must have exactly one row per [Fe/H] bin of lmcold.samples.FEH_BINS.
    rows = t.groupby(['suite', 'name']).population.nunique()
    if (rows != len(samples.FEH_BINS)).any() or t.duplicated(['suite', 'name', 'population']).any():
        sys.exit(f'merge: incomplete or duplicated bins in {list(rows[rows != len(samples.FEH_BINS)].index)[:3]}')
    # Write to a temporary file and rename, so that OUT is never half-written.
    tmp = OUT.with_suffix('.tmp')
    t.to_csv(tmp, index=False, float_format='%.4f')
    os.replace(tmp, OUT)
    print(f'merge: {len(names)} analogues, {len(t)} rows; bins with a non-converged fit: {int((t.fit_warnings > 0).sum())}')


if __name__ == '__main__':
    # %% [markdown]
    # ### Merge, or choose the analogues of this worker
    # With the argument `merge` the script only joins the per-analogue files. Otherwise worker `i` of `n` takes every `n`-th analogue
    # of `lmcold.sims.names()`, starting at `i`.

    # %%
    # 'merge': join the per-analogue files and stop.
    if sys.argv[1:2] == ['merge']:
        merge()
        sys.exit()
    # Worker index and number of workers from the command line; one worker doing everything by default.
    worker, nworkers = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (0, 1)

    # %% [markdown]
    # ### Load the data
    # `setup` reads the joint fit `results/kin_mcmc_data_ellipsoid.csv`, the real giants and the APOGEE–BOSS overlap (Zenodo bundle 1),
    # `results/giant_mass.csv`, `results/giant_conversion.csv` and the PARSEC grid. The particles of each analogue are read inside the
    # loop below (`particles`, through `lmcold.sims.load`).

    # %%
    inputs = setup()   # (frame, real, dfeh, knots, nbox, survival), see setup()
    todo = sims.names()[worker::nworkers]   # every nworkers-th analogue, starting at this worker's index
    PARTS.mkdir(exist_ok=True)   # folder of the per-analogue files

    # %% [markdown]
    # ### Observe and fit each analogue (steps 4-6)
    # For each analogue: draw the mock giants (seeded by the analogue's name, so a rerun gives the same sample), split them into the
    # five measured-[Fe/H] bins, take the particle truth of each bin near $R$ = 4 kpc, and fit each bin of at least 50 mock giants
    # (at most `NMAX` of them) by maximum likelihood with the ellipsoid model. One file per analogue.

    # %%
    for suite, name in todo:
        # Skip an analogue already done (by this or another worker).
        if part(suite, name).exists():
            continue
        t0 = time.time()   # s, for the run-time printout
        # Random generator seeded by the CRC32 checksum of 'suite|name': the same mock sample on every rerun.
        rng = np.random.default_rng(zlib.crc32(f'{suite}|{name}'.encode()))
        cat, lost = observe(suite, name, *inputs, rng)
        rows = []   # one dict per [Fe/H] bin
        for j, (lo, hi) in enumerate(samples.FEH_BINS):
            b = (cat['feh'] >= lo) & (cat['feh'] < hi)   # mock giants in bin j of measured [Fe/H]
            c = {k: v[b] for k, v in cat.items()}   # the mock catalogue of this bin
            # Bin size, distinct particles behind its mock giants, and real giants without a mock in this analogue.
            row = dict(suite=suite, name=name, population=j, n=int(b.sum()), n_part=len(np.unique(c['particle'])), lost=lost)
            near = np.abs(c['R'] - kinfit.R_SIG) < TRUTH_DR   # particles within TRUTH_DR of R = 4 kpc (kinfit.R_SIG)
            # Truth [km/s]: mean v_phi and the dispersions of v_R, v_phi, v_z of those particles.
            row |= dict(t_vphi=c['vphi'][near].mean(), t_sigma_R=c['vR'][near].std(), t_sigma_phi=c['vphi'][near].std(),
                        t_sigma_z=c['vz'][near].std(), t_n=int(near.sum()))
            # Fit only bins of at least 50 mock giants (others keep NaN fit columns), a random subset of at most NMAX.
            if b.sum() >= 50:
                sub = rng.permutation(int(b.sum()))[:NMAX]
                d = kinfit.data(*(c[k][sub] for k in COLS))
                # Record RuntimeWarnings of the fit, to count the non-converged maximisations.
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always', RuntimeWarning)
                    p = ml(d, 'ellipsoid')   # stage-1 maximum likelihood, as for the data
                row['fit_warnings'] = sum('did not converge' in str(w.message) for w in caught)
                # Rotation speed at R = 4 kpc [km/s] and the negative log likelihood of the best fit.
                row |= dict(vrot4=float(kinfit.rotation(p, np.array([kinfit.R_SIG]))[0]), nll=float(kinfit.neg_loglike(p, d, True)))
                # Ellipsoid dispersions, k, outlier fraction (logit), inc and theta [deg], then the whole vector as p_<name>.
                row |= {n: p[P(n)] for n in kinfit.ELLIPSOID + ('k', 'logit_fout', 'inc', 'theta')} | {f'p_{n}': v for n, v in zip(kinfit.PARAMS, p)}
            rows.append(row)
        # Write the analogue's file whole (temporary file, then rename), columns in the fixed order FIELDS.
        tmp = part(suite, name).with_suffix('.tmp')
        pd.DataFrame(rows).reindex(columns=FIELDS).to_csv(tmp, index=False, float_format='%.4f')
        os.replace(tmp, part(suite, name))
        # Progress line: giants per bin, lost giants, fitted and true rotation and sigma_R per bin, time taken.
        print(f'{suite} {name}: {[r["n"] for r in rows]} giants, lost {lost}, '
              f'vrot4 {[round(r.get("vrot4", np.nan), 1) for r in rows]} truth {[round(r["t_vphi"], 1) for r in rows]}, '
              f'sigma_R {[round(r.get("sigma_R", np.nan), 1) for r in rows]} truth {[round(r["t_sigma_R"], 1) for r in rows]} '
              f'({time.time() - t0:.0f} s)', flush=True)
