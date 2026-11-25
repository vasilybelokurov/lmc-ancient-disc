"""Why does one circular speed plus asymmetric drift fit the giant bins worse than free rotation curves (scripts/kin_vc_drift.py)?
Diagnostics and extensions, in this order:

1. Where the deficit comes from: per-star ln L of the 'drift' fit minus the 'free' fit (both joint ML, same stars), summed in bins
   of in-plane radius R and azimuth phi (from the line of nodes, along e_phi) per giant bin, and per survey (APOGEE, BOSS).
2. Matched tracers: the BOSS kinematic giants that are also in the BOSS mass sample (lmcold.mass_samples.samples: the same
   stars, velocity window 150-420 km/s), with h measured from those very stars (selection weights), 'free' and 'drift' refitted;
   and, on all giants, 'drift' with h at the 16th and 84th percentiles of its bootstrap (kin_vc_drift_h.csv) for every bin.
3. Population-dependent streaming: m2c_g = m2c + a_c q_g and m2s_g = m2s + a_s q_g, q_g = Q (a fixed ordering from metal-poor to
   metal-rich, centred), two shared parameters, in 'free + stream' and 'drift + stream'.
4. Thickness (diagnostic): each bin feels an effective circular speed v_c,g^2 = v_c^2 (1 - beta (sigma_z,g / SIGMA_REF)^2), with
   sigma_z,g its own fitted vertical dispersion and beta one shared parameter ('drift + beta', 'drift + stream + beta'): a thick
   population feels a weaker vertically averaged radial force. A phenomenological test, not a vertical-force integration.

All fits: scripts/kin_vc_drift.joint_fit (geometry, systemic motion and streaming shared and free; CURVE shared for the drift models;
OWN per bin; h fixed). Writes results/kin_vc_drift_ext.csv (variant, population, n, nll, dlnL against 'free' of the same stars, vrot4,
vc4 (the bin's effective v_c at 4 kpc), sigma_R, sigma_phi, sigma_z, dlnsig_dR, k, h, extra parameters, success, grad),
results/kin_vc_drift_deficit.csv (population, coordinate, bin, n, dlnL) (figures: scripts/fig_kin_vc_drift.py); fitted vectors in data/kin_vc_drift_ext_vectors.npz.

The numbers of the KB page 'Rotation and the circular speed' (section 'Where one circular speed is not enough') are written from
these tables by scripts/paper_text_vc_drift.py. Diagnostic (not in the Letter).

Inputs: the release giants (lmcold.samples.giants(), Zenodo bundle 1); results/kin_vc_drift_h.csv (scripts/kin_vc_drift.py); the
joint ellipsoid fit results/kin_mcmc_data_ellipsoid.csv (through kin_vc_drift.joint_params); the BOSS mass sample
(lmcold.mass_samples.samples) with data/giant_foreground.parquet.

Run: .venv/bin/python scripts/kin_vc_drift_extensions.py   (after scripts/kin_vc_drift.py; about 5 h in the run of 2026-11-09)
"""
# %% [markdown]
# ### Settings
# The ordering of the bins used by the population-dependent streaming, the reference dispersion and bounds of the thickness term,
# and the bins of radius and azimuth of the deficit map.

# %%
import healpy as hp
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from fit_kinematics_mcmc import COLS, ring
from kin_vc_drift import CURVE, P, SHARED_FREE, joint_fit, joint_params, scale_length, radius, vrot4, with_h
from lmcold import geometry as geo
from lmcold import kinfit, samples
from lmcold.mass_samples import samples as mass_samples
from lmcold.sources import DATA, ROOT

Q = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # streaming ordering of the giant bins, metal-poor to metal-rich
SIGMA_REF = 30.0  # km/s, reference vertical dispersion of the thickness term
# Extra shared parameters of each extension: (names, starting values, bounds). a_c, a_s: slopes of m2c, m2s against Q; beta: the
# thickness coefficient (dimensionless).
STREAM_ADD = (('a_c', 'a_s'), (0.0, 0.0), ((-0.5, 0.5), (-0.5, 0.5)))
BETA_ADD = (('beta',), (0.0,), ((-0.5, 0.75),))  # beta < 0.75 keeps 1 - beta (sigma_z / 30)^2 > 0 for sigma_z < 34 km/s
R_BINS, PHI_BINS = np.arange(1., 11., 1.), np.linspace(0, 360, 9)  # kpc (1-kpc bins), deg (45-deg bins)


# %% [markdown]
# ### Helpers
# The two extensions as functions that modify the population vectors inside a joint fit, the 'drift' fit started from the free
# solution, the output rows, and the in-plane azimuth of each star.

# %%


def adjust_stream(ps, v, offset=0):
    """Per-bin m = 2 coefficients: shared value plus slope times Q.

    ps : list of five kinfit.PARAMS vectors (jax arrays); v : the extra parameters of the fit, of which v[offset], v[offset + 1] are
    a_c, a_s. Returns the modified vectors.
    """
    # Bin i: m2c + a_c Q[i] and m2s + a_s Q[i] (jax functional updates).
    return [p.at[P('m2c')].add(v[offset] * Q[i]).at[P('m2s')].add(v[offset + 1] * Q[i]) for i, p in enumerate(ps)]


def adjust_beta(ps, v, offset=0):
    """Effective circular speed per bin: v0 scaled by sqrt(1 - beta (sigma_z / SIGMA_REF)^2), the square of the scale clipped at
    1e-3 from below.

    ps : list of five kinfit.PARAMS vectors (jax arrays); v : the extra parameters of the fit, v[offset] = beta. Returns the
    modified vectors.
    """
    # Each bin's v0 times sqrt(max(1 - beta (sigma_z / SIGMA_REF)^2, 1e-3)), sigma_z being that bin's own fitted value.
    return [p.at[P('v0')].multiply(kinfit.jnp.sqrt(kinfit.jnp.clip(1 - v[offset] * (p[P('sigma_z')] / SIGMA_REF) ** 2, 1e-3)))
            for p in ps]


# Extension name: the `add` argument of kin_vc_drift.joint_fit (names, starts, bounds, function applying them).
ADD = {'stream': (STREAM_ADD[0], STREAM_ADD[1], STREAM_ADD[2], adjust_stream),
       'beta': (BETA_ADD[0], BETA_ADD[1], BETA_ADD[2], adjust_beta),
       'stream + beta': (STREAM_ADD[0] + BETA_ADD[0], STREAM_ADD[1] + BETA_ADD[1], STREAM_ADD[2] + BETA_ADD[2],
                         lambda ps, v: adjust_beta(adjust_stream(ps, v, 0), v, 2))}


def fit_drift(ds, pfree, hs, add=None):
    """'drift' joint ML from the free solution pfree with scale lengths hs: the shared curve alone first (from there the first
    L-BFGS-B step of the full problem can reach the corners of BOUNDS), then everything."""
    # Start: the free vectors with h set and the shared curve (v0 [km/s], R0 [kpc], vslope [1/kpc]) = (90, 2.5, 0) in every bin.
    p0 = with_h(pfree, hs)
    for p in p0:
        p[[P(n) for n in CURVE]] = [90., 2.5, 0.]
    # Fit the curve alone, then everything free in 'drift' (SHARED_FREE + CURVE), with the extension `add` if given.
    p0, _, _, _ = joint_fit(ds, p0, CURVE, own=())
    return joint_fit(ds, p0, SHARED_FREE + CURVE, add=add)


def rows(variant, ps, nll, info, names, ds, ref_nll, add_names=()):
    """Output rows of one fit (one per bin), with dlnL = ref_nll - nll against the 'free' fit of the same stars; prints a summary.

    variant : name of the fit; ps : fitted vectors; nll : -lnL; info : the info dict of joint_fit; names, ds : bin labels and data;
    add_names : names of the extra parameters (info['extra']). vrot4 and vc4 in km/s, at R = 4 kpc.
    """
    # One row per bin: counts, likelihoods, rotation at 4 kpc, dispersions, scale length, streaming, inclination, extra parameters.
    out = []
    for name, p, d in zip(names, ps, ds):
        # vc4: vrot4 with h (index 32 of kinfit.PARAMS) set to 0, i.e. the circular speed without drift (kin_vc_drift.vrot4).
        r = dict(variant=variant, population=name, n=len(np.asarray(d['pmra'])), nll=nll, dlnL=ref_nll - nll, vrot4=vrot4(p),
                 vc4=vrot4(np.r_[p[:32], 0.0]), success=info['success'], grad=info['grad'])
        r |= {n: p[P(n)] for n in ('sigma_R', 'sigma_phi', 'sigma_z', 'dlnsig_dR', 'k', 'h', 'm2c', 'm2s', 'inc')}
        r |= dict(zip(add_names, info['extra']))
        out.append(r)
    print(variant, f'nll {nll:.1f}, dlnL vs free {ref_nll - nll:+.1f}', 'vrot4', [round(x['vrot4'], 1) for x in out],
          'vc4', [round(x['vc4'], 1) for x in out], 'extra', np.round(info['extra'], 3), 'ok' if info['success'] else 'NOT converged',
          f"grad {info['grad']:.2g}", flush=True)
    return out


def azimuth(p, d):
    """In-plane radius [kpc] and azimuth phi [deg] (from the line of nodes, increasing along e_phi) of the stars of d."""
    # R [kpc] and in-plane unit vector e_R (N, 3) of each star from the model; line of nodes and normal of the fitted orientation.
    _, _, _, _, R, eR, _, n = (np.asarray(v) for v in kinfit._model(p, d['rhat'], d['ea'], d['ed']))
    _, _, _, line, nn = (np.asarray(v) for v in geo.symmetry_axis(*(float(v) for v in kinfit.centre(p[0], p[1])),
                                                                     np.deg2rad(p[P('inc')]), np.deg2rad(p[P('theta')])))
    # phi = atan2(e_R . (n x line), e_R . line), in [0, 360) deg.
    return R, np.rad2deg(np.arctan2(eR @ np.cross(nn, line), eR @ line)) % 360


if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data
    # The disc frame of the joint ellipsoid fit, the giants of the five [Fe/H] bins at 2 < $\rho$ < 9 deg and the scale lengths of
    # `results/kin_vc_drift_h.csv`. The BOSS mass sample and the foreground table are read in the matched-tracer step.

    # %%
    frames = joint_params()  # population -> kinfit.PARAMS vector (results/kin_mcmc_data_ellipsoid.csv, stage 2)
    giants = [(f'giants {lab}', ring(cat | dict(dmu=np.full(len(cat['ra']), np.nan)))) for lab, cat in samples.giants()]
    # Bin labels and the kinfit data of each bin (columns COLS of scripts/fit_kinematics_mcmc.py; distances NaN, not used).
    names = [n for n, _ in giants]
    ds = [kinfit.data(*(c[k] for k in COLS)) for _, c in giants]
    H = pd.read_csv(ROOT / 'results/kin_vc_drift_h.csv').set_index('population')  # h, h_p16, h_p84 [kpc] per population
    out, vectors = [], {}  # output rows; fitted vectors per variant (saved to data/kin_vc_drift_ext_vectors.npz)

    # %% [markdown]
    # ### Reference fits and the scale-length bracket
    # 'free' and 'drift' on all giants, and 'drift' with every $h$ at the 16th and at the 84th percentile of its bootstrap.

    # %%
    # 'free': own rotation curve per bin, no drift (h = 0); its -lnL nllf is the reference of every dlnL on all giants.
    pf, _, nllf, inf_f = joint_fit(ds, with_h([frames[n] for n in names], [0.] * 5), SHARED_FREE)
    # Rows of the 'free' fit (dlnL = 0 by definition); fitted vectors kept for the npz file.
    out += rows('free', pf, nllf, inf_f, names, ds, nllf)
    # 'drift' with the measured h, then with every h at its 16th and at its 84th bootstrap percentile.
    for variant, col in (('drift', 'h'), ('drift, h p16', 'h_p16'), ('drift, h p84', 'h_p84')):
        pj, _, nllj, inf_j = fit_drift(ds, pf, H.loc[names, col].to_numpy())
        out += rows(variant, pj, nllj, inf_j, names, ds, nllf)
        vectors[variant] = np.array(pj)
    vectors['free'] = np.array(pf)

    # %% [markdown]
    # ### Where the deficit comes from
    # The per-star log-likelihood of 'drift' minus that of 'free', summed in bins of $R$ and of azimuth $\phi$ and per survey, for
    # each giant bin. Writes `results/kin_vc_drift_deficit.csv`.

    # %%
    deficit = []
    for name, (_, c), d, a, j in zip(names, giants, ds, pf, vectors['drift']):
        # Per-star ln L of 'drift' (j) minus 'free' (a), ellipsoid likelihood; negative where 'drift' fits worse.
        dl = np.asarray(kinfit.loglike_stars(j, d, True)) - np.asarray(kinfit.loglike_stars(a, d, True))
        R, phi = azimuth(j, d)
        # Sum of dl in 1-kpc bins of R and 45-deg bins of phi (stars outside the edges fall in no bin).
        for coord, x, edges in (('R', R, R_BINS), ('phi', phi, PHI_BINS)):
            k = np.digitize(x, edges) - 1
            for i in range(len(edges) - 1):
                deficit.append(dict(population=name, coordinate=coord, lo=edges[i], hi=edges[i + 1], n=int((k == i).sum()),
                                    dlnL=float(dl[k == i].sum())))
        # Sum of dl per survey.
        for sv in ('apogee', 'boss'):
            m = np.asarray(c['survey']) == sv
            deficit.append(dict(population=name, coordinate='survey', lo=np.nan, hi=np.nan, n=int(m.sum()), dlnL=float(dl[m].sum()),
                                survey=sv))
    # Write the deficit table and print the per-survey sums.
    D = pd.DataFrame(deficit)
    D.to_csv(ROOT / 'results/kin_vc_drift_deficit.csv', index=False, float_format='%.3f')
    print(D[D.coordinate == 'survey'][['population', 'survey', 'n', 'dlnL']].round(1).to_string(index=False), flush=True)

    # %% [markdown]
    # ### Matched tracers
    # The BOSS kinematic giants that are also in the BOSS mass sample (matched within 1 arcsec), with $h$ measured from those same
    # stars, refitted with 'free' and 'drift'.

    # %%
    # The BOSS mass sample (ra, dec [deg], selection weight w), joined to the foreground table (coverage fill, Milky Way fraction
    # f_mw).
    b, _, _ = mass_samples()
    fg = pd.read_parquet(DATA / 'giant_foreground.parquet')
    b = b.merge(fg[['source_id', 'fill', 'f_mw']], on='source_id', validate='one_to_one')
    b['W'] = b.w * b['fill'] * (1 - b.f_mw)  # weight: selection x coverage fill x (1 - Milky Way fraction), as kin_vc_drift.py
    unit = lambda ra, dec: np.c_[np.cos(np.deg2rad(dec)) * np.cos(np.deg2rad(ra)), np.cos(np.deg2rad(dec)) * np.sin(np.deg2rad(ra)),  # noqa: E731
                                 np.sin(np.deg2rad(dec))]
    # Unit vectors on the sky; a chord of 5e-6 is ~1 arcsec (4.85e-6 rad).
    tree = cKDTree(unit(b.ra.to_numpy(), b.dec.to_numpy()))
    mds, mhs, mframes = [], [], []
    for name, c in giants:
        # Nearest mass-sample star of each kinematic giant; BOSS giants within 1 arcsec count as matched.
        dist, idx = tree.query(unit(c['ra'], c['dec']))
        m = (np.asarray(c['survey']) == 'boss') & (dist < 5e-6)  # within 1 arcsec
        # Matched kinematic giants (cm) and their mass-sample rows (s); h from those stars' weighted surface density, bootstrapped
        # over
        # HEALPix pixels (nside 64) as in kin_vc_drift.scale_length.
        cm = {k: np.asarray(v)[m] for k, v in c.items()}
        s = b.iloc[idx[m]]
        Rm = radius(frames[name], s.ra.to_numpy(), s.dec.to_numpy())
        mhs.append(scale_length(Rm, s.W.to_numpy(), hp.ang2pix(64, s.ra.to_numpy(), s.dec.to_numpy(), lonlat=True))[0])
        mds.append(kinfit.data(*(cm[k] for k in COLS)))
        print(f'{name}: matched BOSS {m.sum()} of {len(m)}, h {mhs[-1]:.2f} kpc', flush=True)
    # 'free' and 'drift' on the matched stars; dlnL against the matched 'free' fit.
    pmf, _, nllmf, inf_mf = joint_fit(mds, with_h([frames[n] for n in names], [0.] * 5), SHARED_FREE)
    out += rows('matched: free', pmf, nllmf, inf_mf, names, mds, nllmf)
    # 'drift' on the matched stars with their own h; both fits' vectors kept.
    pmj, _, nllmj, inf_mj = fit_drift(mds, pmf, mhs)
    out += rows('matched: drift', pmj, nllmj, inf_mj, names, mds, nllmf)
    vectors['matched: free'], vectors['matched: drift'] = np.array(pmf), np.array(pmj)

    # %% [markdown]
    # ### Population-dependent streaming and thickness
    # On all giants with the measured $h$: 'free + stream', and 'drift' with population-dependent streaming, with the thickness
    # term $\beta$, and with both. Writes `results/kin_vc_drift_ext.csv` and the fitted vectors.

    # %%
    hs = H.loc[names, 'h'].to_numpy()
    # 'free + stream' from the 'free' solution; then 'drift' with each extension (stream, beta, both), all against 'free'.
    pfs, _, nllfs, inf_fs = joint_fit(ds, pf, SHARED_FREE, add=ADD['stream'])
    out += rows('free + stream', pfs, nllfs, inf_fs, names, ds, nllf, ADD['stream'][0])
    vectors['free + stream'] = np.array(pfs)
    for key in ('stream', 'beta', 'stream + beta'):
        pjx, _, nllx, infx = fit_drift(ds, pf, hs, add=ADD[key])
        out += rows(f'drift + {key}', pjx, nllx, infx, names, ds, nllf, ADD[key][0])
        vectors[f'drift + {key}'] = np.array(pjx)
    T = pd.DataFrame(out)
    T.to_csv(ROOT / 'results/kin_vc_drift_ext.csv', index=False, float_format='%.4f')
    # Fitted vectors per variant, keys made valid names (spaces to _, ':' and ',' dropped, '+' to p).
    np.savez(DATA / 'kin_vc_drift_ext_vectors.npz', names=np.array(names), **{k.replace(' ', '_').replace(':', '').replace('+', 'p').replace(',', ''): v
                                                                         for k, v in vectors.items()})
