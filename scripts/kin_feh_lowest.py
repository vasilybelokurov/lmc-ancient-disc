"""Diagnostic: do the LMC's most metal-poor giants ([Fe/H] < -2) rotate, given that BOSS [Fe/H] errors mix in stars from
-2.0..-1.7 and that Milky Way halo stars and any hot LMC component enter the sample?

The paper's v_rot(4 kpc) / sigma_z of the measured [Fe/H] < -1.5 bin (1.76; scripts/paper_kin_support_ellipsoid.py) comes from
the ellipsoid model of lmcold.kinfit fitted to that bin alone. Here the same likelihood is fitted to narrower bins of measured
[Fe/H] inside it. With a few hundred stars the geometry cannot be fitted per bin, so everything except each bin's own rotation
amplitude v0, dispersions sigma_R, sigma_phi, sigma_z, Gaia error scale k and outlier fraction (OWN) is held at the stage-1
posterior of the [Fe/H] < -1.5 bin (data/kin_mcmc_data_ellipsoid_samples.npz): geometry, centre, bulk motion, streaming, and
R0, vslope, dlnsig_dR. Plan and reviews: two independent reviews rejected unfolding the bins with a mixing matrix (the bins are
nearly collinear, the overlap with APOGEE has 23 stars below -2, and a mixture fit does not return mixture moments); instead:

1. NUTS fits of OWN for each measured bin (BINS) and the subsets of SUBSETS; the bin [Fe/H] < -1.5 itself checks that holding
   the rest reproduces the paper's value.
2. Held parameters: OWN refitted (maximum likelihood) at N_GEO draws of the stage-1 posterior (spread added to the errors), and
   once with the joint (stage-2) geometry of the paper.
3. Null test, end to end: mock velocities for the stars of a bin at their own positions and errors, drawn from the ellipsoid
   model (mean and disc-frame covariance of lmcold.kinfit, Gaia and line-of-sight errors, k = 1). A fraction `purity` of the stars
   are true metal-poor stars with v_rot(4 kpc) / sigma_z = `ratio` (dispersions as the interlopers'); the rest are interlopers with
   the kinematics fitted to measured -1.75..-1.5; a fraction `hot` are replaced by a non-rotating component with dispersion
   HOT_SIGMA about the LMC bulk motion (a stand-in for Milky Way halo stars and an LMC spheroid). Each mock is fitted as the
   data (maximum likelihood); N_MOCK realisations per grid point. The observed ratio is compared with the mock distributions.

Reads the release giants (lmcold.samples.giants()) and the posterior samples above. Writes results/kin_feh_lowest.csv (sample,
n, median [Fe/H], v_rot(4 kpc), sigma_z, ratio median and 16th/84th percentiles, spread over the held parameters, ratio with the
joint geometry, convergence) and results/kin_feh_lowest_null.csv (bin, purity, ratio, hot, realisation, fitted ratio); the figure
is scripts/fig_kin_feh_lowest.py. The Letter's numbers for the [Fe/H] < -2 giants come from these tables (paper_text_letter.py).

Run: .venv/bin/python scripts/manifest.py run kin_feh_lowest.py   (about 30 min on 14 cores: 9 NUTS runs, 256 mock fits)
"""
# %% [markdown]
# ### Settings
# Bins of measured [Fe/H] inside the paper's lowest bin, the subsets, the parameters fitted per bin, and the null-test grid.

# %%
import numpyro

numpyro.set_host_device_count(4)  # four chains in parallel; before JAX starts

import numpy as np
import pandas as pd

from fit_kinematics_mcmc import COLS, ring
from lmcold import kinfit, samples
from lmcold.constants import KAPPA
from lmcold.sources import DATA, ROOT

# Bins of measured [Fe/H] (APOGEE scale; -9 = no lower edge): four narrow bins inside the paper's lowest bin, all stars below -2,
# and the paper's whole lowest bin (the check that holding the rest reproduces its value).
BINS = [(-9, -2.2), (-2.2, -2.0), (-2.0, -1.75), (-1.75, -1.5), (-9, -2.0), (-9, -1.5)]
SNR_BEST = 40       # BOSS S/N of the best-measured subset
SUBSETS = {'[Fe/H] < -2.0, BOSS S/N > 40 or APOGEE': lambda c: (c['feh'] < -2.0) & ((c['snr'] > SNR_BEST) | (c['survey'] == 'apogee')),
           '[Fe/H] < -2.0, Teff < 4500 K': lambda c: (c['feh'] < -2.0) & (c['teff'] < 4500),
           '[Fe/H] < -2.0, Teff >= 4500 K': lambda c: (c['feh'] < -2.0) & (c['teff'] >= 4500)}
OWN = ('v0', 'sigma_R', 'sigma_phi', 'sigma_z', 'k', 'logit_fout')   # fitted per bin: rotation amplitude, dispersions [km/s], error scale, outliers
N_GEO = 30          # posterior draws of the held parameters
PURITY = (0.2, 0.4, 0.6, 1.0)     # null test: fraction of a bin's stars that are truly metal-poor (overlap estimate 9/23 = 0.39)
RATIO = (0.0, 0.5, 1.0, 1.76)     # their true v_rot(4 kpc) / sigma_z: none, weak, moderate, the paper's value for [Fe/H] < -1.5
HOT = (0.0, 0.15)   # 0.15: mean Milky Way probability f_mw of the measured [Fe/H] < -2 giants (data/giant_foreground.parquet)
HOT_SIGMA = 100.0   # km/s, isotropic
N_MOCK = 4          # realisations per grid point
NULL_BINS = [(-9, -2.2), (-9, -2.0)]   # bins tested with mocks
INTERLOPER_BIN = (-1.75, -1.5)         # bin whose fitted kinematics the interlopers get (they come from just above -2)


# %% [markdown]
# ### Held parameters and the mock generator
# `held` turns one draw (or the median) of the stage-1 posterior of the [Fe/H] < -1.5 bin into a full parameter vector.
# `mock` draws velocities from the ellipsoid model at the stars of a catalogue.

# %%
def held(z, i=None):
    """Full kinfit parameter vector (33,) from the stage-1 posterior of population 0 in npz z: draw i (flattened over chains), or
    the median if i is None; parameters the run held (the 0 defaults of start(), e.g. m1c) keep kinfit.start() values except
    the ellipsoid ones, which are all sampled."""
    p = kinfit.start()
    # Keys '1|0|<site>': stage 1, population 0; per-population sites end in '_0'. x: all draws of that site, chains flattened.
    for key in z.files:
        if key.startswith('1|0|'):
            x = np.asarray(z[key]).ravel()
            name = key[4:].removesuffix('_0')
            p[kinfit.PARAMS.index(name)] = np.median(x) if i is None else x[i]
    # Terms the paper's ellipsoid model holds at zero: lopsided part, tilted ring, line-of-sight scale, turning plane.
    for n in ('m1c', 'm1s') + kinfit.TILT + ('qlos',) + kinfit.TURN:
        p[kinfit.PARAMS.index(n)] = 0.0
    return p


def mock(cat, q, rng):
    """Mock proper motions and line-of-sight velocities for catalogue dict cat from parameter vector q (ellipsoid model, k = 1,
    no outliers): returns a copy of cat with pmra, pmdec [mas/yr] and vlos [km/s] replaced (vlos stays NaN where it was)."""
    d = kinfit.data(*(cat[k] for k in COLS[:-1]))   # sky bases and Gaia covariances of the stars
    # Model mean at each star: proper motions [mas/yr], line-of-sight velocity [km/s], distance to the disc plane D [kpc].
    mra, mde, mlos, D = (np.asarray(v) for v in kinfit.model(q, d['rhat'], d['ea'], d['ed']))
    C = np.asarray(kinfit.intrinsic_cov(q, d))                      # (N, 3, 3) km^2/s^2, (east, north, line of sight)
    v = np.einsum('nij,nj->ni', np.linalg.cholesky(C), rng.normal(size=(len(D), 3)))   # intrinsic velocity draws [km/s], (N, 3)
    # Gaia proper-motion errors [mas/yr], (N, 2), with their correlation; line-of-sight errors evlos [km/s] below.
    noise = np.einsum('nij,nj->ni', np.linalg.cholesky(np.asarray(d['cov'])), rng.normal(size=(len(D), 2)))
    kD = KAPPA * D                                                  # km/s per mas/yr at the star
    vlos = mlos + v[:, 2] + rng.normal(size=len(D)) * np.asarray(cat['evlos'])
    return dict(cat, pmra=mra + v[:, 0] / kD + noise[:, 0], pmdec=mde + v[:, 1] / kD + noise[:, 1],
                vlos=np.where(np.isfinite(cat['vlos']), vlos, np.nan))


def with_ratio(q, ratio):
    """Copy of q with v0 set so that v_rot(4 kpc) / sigma_z = ratio (R0 and sigma_z of q)."""
    q = q.copy()
    # v_rot(4 kpc) = v0 tanh(R_SIG / R0) (the vslope term vanishes at R_SIG), so v0 = ratio sigma_z / tanh(R_SIG / R0).
    q[kinfit.PARAMS.index('v0')] = ratio * q[kinfit.PARAMS.index('sigma_z')] / np.tanh(kinfit.R_SIG / q[kinfit.PARAMS.index('R0')])
    return q


def ratio_of(p):
    """v_rot(4 kpc) / sigma_z of vector p."""
    g = lambda n: p[..., kinfit.PARAMS.index(n)]  # noqa: E731  (works for one vector (33,) or a stack of draws (M, 33))
    return g('v0') * np.tanh(kinfit.R_SIG / g('R0')) / g('sigma_z')


def ml(cat, p):
    """Maximum likelihood of OWN for catalogue dict cat, the rest held at p."""
    return kinfit.fit(kinfit.data(*(cat[k] for k in COLS[:-1])), free=OWN, p0=p, ellipsoid=True)[0]


def select(cat, keep):
    """Copy of catalogue dict cat (arrays of shape (N,)) keeping the stars where the boolean array keep is True."""
    return {k: np.asarray(v)[keep] for k, v in cat.items()}


# %% [markdown]
# ### Load the data
# The giants of the paper's lowest bin at 2-9 deg and the stage-1 posterior of that bin.

# %%
if __name__ == '__main__':
    # The giants of the paper's lowest bin (measured [Fe/H] < -1.5): positions [deg], proper motions [mas/yr], line-of-sight
    # velocities [km/s], [Fe/H], Teff, S/N, survey; kept at 2 < rho < 9 deg as in the paper's fit (giants have no distance moduli).
    lab, cat = samples.giants()[0]
    cat = ring(cat | dict(dmu=np.full(len(cat['ra']), np.nan)))
    # Posterior samples of the paper's ellipsoid run; population 0 is the lowest bin (checked), whose stage-1 fit sets the held
    # parameters: p_ref, its posterior medians, a full kinfit vector (33,).
    z = np.load(DATA / 'kin_mcmc_data_ellipsoid_samples.npz')
    assert str(z['names'][0]) == f'giants {lab}' == 'giants [Fe/H] < -1.5'
    p_ref = held(z)
    n_draws = np.asarray(z['1|0|inc']).size   # number of posterior draws (chains x draws)
    # The variant with the joint geometry: p_ref with the shared parameters (centre, bulk motion, inclination, line of nodes,
    # streaming) replaced by the stage-2 medians of the joint fit of all populations.
    joint = pd.read_csv(ROOT / 'results/kin_mcmc_data_ellipsoid.csv').query("stage == 2 and population == 'shared'")
    p_joint = p_ref.copy()
    for n, v in zip(joint.param, joint['median']):
        p_joint[kinfit.PARAMS.index(n)] = v

    # %% [markdown]
    # ### Fits of the bins
    # NUTS for OWN in each bin and subset, then the spread over the held parameters and the joint-geometry value.

    # %%
    rng = np.random.default_rng(1)
    geo = rng.choice(n_draws, N_GEO, replace=False)   # posterior draws at which the held parameters are tested
    # The samples to fit: name -> boolean mask over the stars of the lowest bin.
    samples_ = {f'{lo} < [Fe/H] < {hi}'.replace('-9 < ', ''): (cat['feh'] >= lo) & (cat['feh'] < hi) for lo, hi in BINS}
    samples_ |= {name: f(cat) for name, f in SUBSETS.items()}
    rows, fits = [], {}
    fixed = tuple(n for n in kinfit.PARAMS if n not in OWN)   # everything else held at p_ref
    for name, keep in samples_.items():
        c = select(cat, keep)                         # the sample's stars
        d = kinfit.data(*(c[k] for k in COLS[:-1]))   # the arrays the likelihood needs (sky bases, covariances)
        p0 = ml(c, p_ref)                                                    # maximum likelihood, the start of NUTS
        m = kinfit.sample([d], [p0], fixed, 500, 1000, 4, ellipsoid=True)    # 500 warm-up, 1000 kept draws, 4 chains
        s = m.get_samples()
        # Full parameter vectors, one per posterior draw: the held values of p0 with the sampled OWN inserted.
        draws = np.tile(p0, (len(s['v0_0']), 1))
        for n in OWN:
            draws[:, kinfit.PARAMS.index(n)] = np.asarray(s[f'{n}_0'])
        r = ratio_of(draws)                                                  # v_rot(4 kpc) / sigma_z per draw
        vrot = r * draws[:, kinfit.PARAMS.index('sigma_z')]                  # v_rot(4 kpc) [km/s]
        spread = np.std([ratio_of(ml(c, held(z, i))) for i in geo])          # ratio's scatter over the held parameters
        # Summary row; the draws' columns 29, 9 and 10 are sigma_z, k and logit_fout (order of kinfit.PARAMS).
        # Convergence: largest split Gelman-Rubin r_hat over the sampled parameters, and the number of divergent transitions.
        rhat = max(float(np.max(numpyro.diagnostics.split_gelman_rubin(np.asarray(v)))) for v in m.get_samples(group_by_chain=True).values())
        div = int(np.sum(np.asarray(m.get_extra_fields()['diverging'])))
        rows.append(dict(sample=name, n=int(keep.sum()), feh_med=np.median(c['feh']),
                         vrot4=np.median(vrot), vrot4_p16=np.percentile(vrot, 16), vrot4_p84=np.percentile(vrot, 84),
                         sigma_z=np.median(draws[:, 29]), ratio_z=np.median(r), ratio_z_p16=np.percentile(r, 16),
                         ratio_z_p84=np.percentile(r, 84), held_spread=spread, ratio_z_joint=ratio_of(ml(c, p_joint)),
                         k=np.median(draws[:, 9]), fout=np.median(1 / (1 + np.exp(-draws[:, 10]))), r_hat=rhat, divergences=div))
        fits[name] = p0
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)   # one row per sample (columns in the module docstring)
    t.to_csv(ROOT / 'results/kin_feh_lowest.csv', index=False, float_format='%.4f')

    # %% [markdown]
    # ### Null test
    # Mocks of the lowest bins with a purity of true metal-poor stars of given rotation, the rest interlopers, and a hot fraction.

    # %%
    # The interlopers: the fitted kinematics of INTERLOPER_BIN, with true errors (k = 1) and no outliers. The hot component: the
    # same frame and bulk motion, no rotation, isotropic dispersion HOT_SIGMA, constant with radius.
    q_int = fits[f'{INTERLOPER_BIN[0]} < [Fe/H] < {INTERLOPER_BIN[1]}'].copy()
    q_int[kinfit.PARAMS.index('k')], q_int[kinfit.PARAMS.index('logit_fout')] = 1.0, -10.0
    q_hot = q_int.copy()
    for n, v in dict(v0=0.0, sigma_R=HOT_SIGMA, sigma_phi=HOT_SIGMA, sigma_z=HOT_SIGMA, dlnsig_dR=0.0).items():
        q_hot[kinfit.PARAMS.index(n)] = v
    # The grid: for each tested bin, every purity, true ratio and hot fraction, N_MOCK realisations; each row of `null` is one
    # fitted mock.
    null = []
    for lo, hi in NULL_BINS:
        c = select(cat, (cat['feh'] >= lo) & (cat['feh'] < hi))   # the real stars of the bin: positions and errors are kept
        name = f'{lo} < [Fe/H] < {hi}'.replace('-9 < ', '')
        for purity in PURITY:
            for ratio in RATIO:
                for hot in HOT:
                    for j in range(N_MOCK):
                        # One seed per grid point and realisation, so every mock can be reproduced on its own.
                        g = np.random.default_rng([j, int(100 * purity), int(100 * ratio), int(100 * hot), int(-10 * lo)])
                        u = g.random(len(c['ra']))
                        # Velocities of every star under each of the three components: truly metal-poor (rotation `ratio`),
                        # interloper, hot.
                        parts = [mock(c, with_ratio(q_int, ratio), g), mock(c, q_int, g), mock(c, q_hot, g)]
                        # Each star's component: hot with probability `hot`; otherwise metal-poor with probability `purity`.
                        who = np.where(u < hot, 2, np.where(g.random(len(u)) < purity, 0, 1))
                        mc = dict(c, **{k: np.choose(who, [p_[k] for p_ in parts]) for k in ('pmra', 'pmdec', 'vlos')})
                        # Fitted exactly as the data (maximum likelihood of OWN, started from the bin's own fit).
                        null.append(dict(bin=name, purity=purity, ratio=ratio, hot=hot, realisation=j, fit_ratio=ratio_of(ml(mc, fits[name]))))
            print(name, purity, 'done', flush=True)
    nt = pd.DataFrame(null)   # bin, purity, ratio, hot, realisation, fitted ratio
    nt.to_csv(ROOT / 'results/kin_feh_lowest_null.csv', index=False, float_format='%.4f')
