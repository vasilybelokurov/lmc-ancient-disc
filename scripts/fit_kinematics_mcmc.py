"""The kinematic fit: posterior of the LMC centre, bulk motion, disc orientation, rotation curve, streaming and velocity dispersions
of each old population, from stellar proper motions and line-of-sight velocities (lmcold.kinfit.sample, NUTS). Its result tables
give the rotation, dispersions and geometry quoted in the paper (kinematics section and its appendix; KB chapters 6 and 7), and the
centre and disc plane used by Letter Fig. 1 (scripts/paper_letter_signal.py) and by the maps of scripts/paper_kin_residuals.py.

The model in plain words (lmcold.kinfit; equations in KB chapter 6, 'Disc model and fits'). Each population is a thin disc seen in
perspective, its centre at the fixed distance D0 (lmcold.constants.LMC) and offset (dx, dy) deg from the adopted (vdMK14) centre,
its plane set by the inclination inc and the line of nodes theta. Each star is placed where its line of sight meets that plane.
The mean velocity there is the bulk motion of the LMC (systemic proper motion pmra0, pmdec0 and line-of-sight velocity vlos0 of the
centre) plus rotation in the plane, v_rot(R) = v0 tanh(R / R0) [1 + vslope (R - kinfit.R_SIG)], and, where the model frees them,
the m = 2 streaming terms (m2c, m2s, vr0, r2c, r2s: fractions of v_rot) or a lopsided (m = 1) part (m1c, m1s). This one velocity
vector is projected on the star's own east, north and line-of-sight directions, so the perspective effects of the bulk motion are
part of the model. The stars scatter about the mean with a Gaussian distribution: in the 'ellipsoid' model a velocity ellipsoid
aligned with the disc (sigma_R, sigma_phi, sigma_z [km/s] at R_SIG, all three scaled by exp(dlnsig_dR (R - R_SIG))); in the other
models one dispersion sigma in both proper-motion directions and sigma_los along the line of sight (the 'sky-aligned' model).
The likelihood of a star is a mixture of two Gaussians in (proper motion, line-of-sight velocity): with weight 1 - f, covariance =
intrinsic scatter + the Gaia proper-motion covariance scaled by k^2 + the squared line-of-sight error; with weight f, the same
plus an extra width kinfit.S_OUT (mas/yr) in every component, for Milky Way stars and gross errors (f = 1 / (1 + exp(-logit_fout))).
A star without a line-of-sight velocity (all RR Lyrae) enters with its proper-motion part only. The fit is unweighted (the
selection depends on position, not velocity). Priors: uniform within kinfit.BOUNDS. No gravitational potential is used.

Populations: the giants in five [Fe/H] bins (proper motions and line-of-sight velocities) and the RR Lyrae members (proper motions
only) of lmcold.samples, stars at RING[0] < rho < RING[1] deg from the adopted centre. Models (second argument): 'centre' (centre
free, not lopsided), 'lopsided' (adopted centre, lopsided part m1c, m1s free), 'both', or 'stream' (centre and in-plane streaming
kinfit.STREAM free, not lopsided: the sky-aligned model of the KB checks), or 'ellipsoid' (as 'stream', with the disc-frame
velocity ellipsoid sigma_R, sigma_phi, sigma_z and its radial gradient dlnsig_dR in place of sigma and sigma_los: kinfit's ellipsoid
likelihood; and the outer slope vslope of the rotation curve free: in the mocks of scripts/paper_kin_ellipsoid_mock.py, rotation
curves lagging a tanh by the asymmetric drift, the dispersions with vslope free and with vslope = 0 agree to 0.5 km/s). 'ellipsoid'
is the paper's model; 'centre' is Level 1 of the KB. Stage 1: each population on its own.
Stage 2: all populations together, sharing the centre, bulk motion, line-of-sight systemic velocity, inclination, line of nodes,
lopsided part and streaming, each with its own rotation curve, dispersions, error scale and outlier fraction. Both stages start from the
maximum-likelihood fits (lmcold.kinfit.fit), the shared part from the first population. D0 is fixed.
First argument 'mock': the same positions and errors, velocities drawn from the model with one shared truth (centre offset
MOCK_OFFSET, not lopsided, the other shared parameters, streaming included, from the fit to the first population) and each population's own fitted
rotation and dispersions, k = 1, no outliers. 'data': the observed velocities.
Optional third argument 'dist': the RR Lyrae also bring their relative distance moduli (OGLE Wesenheit; lmcold.samples.rrab), which
tie the disc plane to the plane the RR Lyrae lie in (kinfit distance term, with their own zero point dmu0 and scatter sigma_mu);
the run name gets the suffix _dist. For a mock the moduli are drawn from the model with the population's fitted dmu0 and sigma_mu.
Optional argument 'outer': stars at OUTER_INNER < rho < RING[1] only, run name suffix _outer. The Gaia proper-motion error scale k is
larger in the crowded inner disc (1.39 at 2-4 deg for the RR Lyrae, 1.12-1.14 beyond); with one k per population the inner stars
set it, which overstates the errors of the outer stars and, for the RR Lyrae (errors ~64 km/s per component), pushes their
proper-motion dispersion to its lower bound. The proper-motion dispersions of the paper come from this run
(scripts/paper_kin_dispersion_cut.py checks the cut).

Writes results/kin_mcmc_<kind>_<model>.csv (median, 16th and 84th percentiles, r_hat, n_eff per parameter, number of divergent
transitions of the run; truth for the mock), the posterior samples to data/kin_mcmc_<kind>_<model>_samples.npz (keys
'stage|population index or all|site', each of shape (chains, draws); population names in 'names'; for
scripts/paper_kin_corner.py, scripts/kin_model_comparison.py and the KB) and figures/fig_kin_mcmc_<kind>.png (centre and lopsided
part, each model run so far).

Inputs: lmcold.samples (release_giants_apogee and release_giants_boss via samples.giants(), and data/rrab.parquet, see their sources.yaml entries). Model and likelihood:
lmcold.kinfit. The samples (data/*.npz) are large and stay out of git. The posterior samples that the released scripts read (runs
'data stream' and 'data ellipsoid', each at 2-9 deg and with 'outer'; data/sources.yaml entries release_kin_mcmc_*) are in Zenodo
bundle 1, and the result tables (results/kin_mcmc_*.csv) are in the repository, so nothing downstream needs this fit to be rerun.
Figure: one row per model; left, the centre offset (deg east, deg north; east to the left as on the sky) of each stage-1 fit
(coloured) and of the joint fit (black star) with 16th-84th percentile bars; right, the lopsided part (m1c, m1s), or for 'stream'
and 'ellipsoid' the azimuthal m = 2 part (m2c, m2s). For a mock the truth (shared values of the csv) is a black cross. Diagnostic
figure (not in the paper).
Run time: about 30 min per sky-aligned model on 14 cores with nothing else running (four chains in parallel); 'data ellipsoid' about
4 h (four chains, one thread each) and 'data ellipsoid outer' about 2.5 h on one machine. Run one model at a time.

Run: .venv/bin/python scripts/fit_kinematics_mcmc.py data both [dist] [outer]    (kind: mock | data; model: centre | lopsided | both | stream | ellipsoid)
     The paper's runs: data ellipsoid; data ellipsoid outer; data stream; data stream outer; data centre.
"""
# %% [markdown]
# ### Settings
# The sampler set-up (four chains run in parallel), the radial ranges of the stars, and the parameters each model holds fixed.

# %%
import sys

import numpyro

numpyro.set_host_device_count(4)  # four CPU 'devices' so the four chains run in parallel; must precede the JAX import

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpyro.diagnostics import summary

from lmcold import kinfit, samples
from lmcold.constants import KAPPA, LMC
from lmcold.footprint import angular_distance
from lmcold.layout import panel_grid
from lmcold.sources import DATA, ROOT

RING = (2., 9.)  # deg from the adopted centre: inner edge avoids the bar region, outer edge the sparse outskirts
OUTER_INNER = 4.  # deg, inner edge of the 'outer' runs (one Gaia error scale k fits the stars beyond)
MOCK_OFFSET = (0.5, 0.0)  # deg east, north: true centre offset of the 'mock' runs
CHAINS, WARMUP, DRAWS = 4, 500, 1000  # NUTS chains, warm-up steps and kept draws per chain
# Warm-up of the main (2-9 deg) ellipsoid run: with 500 steps R0 and v0 of the -1.5 < [Fe/H] < -1.1 giants mix poorly (stage 1
# r_hat 1.08, 34 effective draws; 2026-11-05), with 1500 r_hat <= 1.008. The outer run keeps 500 (r_hat 1.016; 1.029 with 1500).
WARMUP_ELLIPSOID = 1500
# Catalogue columns passed to kinfit.data, in its argument order.
COLS = ('ra', 'dec', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error', 'pmra_pmdec_corr', 'vlos', 'evlos', 'dmu')
# Parameters held at their starting values in each model (the starting values are 0: adopted centre, symmetric disc).
FIXED = {m: f + kinfit.TILT + ('qlos',) + kinfit.TURN + (() if m == 'ellipsoid' else ('vslope',)) for m, f in dict(centre=('m1c', 'm1s') + kinfit.STREAM, lopsided=('dx', 'dy') + kinfit.STREAM,
                                             both=kinfit.STREAM, stream=('m1c', 'm1s'), ellipsoid=('m1c', 'm1s')).items()}  # one plane, no LOS scale, in every model
MODELS = tuple(FIXED)  # model names in the order of the diagnostic figure's rows

# %% [markdown]
# ### Star selection and starting point
# `ring` keeps the stars in the radial range of the fit. `ml` finds the maximum-likelihood parameters of one population in two
# steps (geometry first with the adopted centre, then everything the model frees); NUTS starts from there.

# %%
def ring(cat, inner=RING[0]):
    """Stars of catalogue dict cat (arrays) at inner < rho < RING[1] deg from the adopted LMC centre."""
    rho = angular_distance(cat['ra'], cat['dec'], LMC['ra0'], LMC['dec0'])  # [deg] from the adopted centre
    keep = (rho > inner) & (rho < RING[1])
    # The same cut applied to every column of the catalogue dict.
    return {k: np.asarray(v)[keep] for k, v in cat.items()}


def ml(d, model):
    """Maximum likelihood: centre and lopsided part fixed first, then the model's own free from there.

    Parameters
    ----------
    d : dict
        kinfit.data() arrays of one population.
    model : str
        'centre', 'lopsided', 'both', 'stream' or 'ellipsoid' (FIXED lists what each holds). 'ellipsoid' starts from the 'stream'
        fit with sigma_R = sigma_phi = sigma and sigma_z = sigma_los (sigma for proper motions only), constant with radius.

    Returns
    -------
    ndarray, shape (33,)
        Maximum-likelihood vector (kinfit.PARAMS order), used to start NUTS.
    """
    # Line-of-sight parameters are free only if the population has line-of-sight velocities (the giants).
    los = ('vlos0', 'sigma_los') if np.any(np.asarray(d['los'])) else ()
    los += ('dmu0', 'sigma_mu') if np.any(np.asarray(d['dist'])) else ()  # distance moduli: their zero point and scatter
    # Step 1: bulk motion, orientation, rotation, dispersion, error scale, outliers (kinfit.BASE without dx, dy), centre adopted.
    pf, _, _ = kinfit.fit(d, free=kinfit.BASE[2:] + los)
    # Step 2: from there, every candidate parameter except those the model holds (FIXED[model]).
    free = kinfit.BASE + los + ('m1c', 'm1s') + kinfit.STREAM
    p = kinfit.fit(d, free=tuple(n for n in free if n not in FIXED[model]), p0=pf)[0]
    if model != 'ellipsoid':
        return p
    # Ellipsoid: (sigma_R, sigma_phi, sigma_z, dlnsig_dR) start at (sigma, sigma, sigma_los or sigma, 0), then step 3 refits with
    # the ellipsoid likelihood, sigma and sigma_los replaced by the ellipsoid and vslope freed.
    g = lambda n: p[kinfit.PARAMS.index(n)]  # noqa: E731
    p[[kinfit.PARAMS.index(n) for n in kinfit.ELLIPSOID]] = [g('sigma'), g('sigma'), g('sigma_los' if los else 'sigma'), 0.0]
    free = tuple(n for n in free if n not in FIXED[model] + ('sigma', 'sigma_los')) + kinfit.ELLIPSOID + ('vslope',)
    return kinfit.fit(d, free=free, p0=p, ellipsoid=True)[0]


# %% [markdown]
# ### Posterior summary table
# `rows` turns a finished NUTS run into one row per population and parameter: median, 16th and 84th percentiles, and the
# convergence diagnostics (Gelman-Rubin r_hat, effective number of draws n_eff, divergent transitions).

# %%
def rows(mcmc, stage, names, truth):
    """One row per (population, parameter); names maps the sample-site suffix to the population label.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        Finished run of kinfit.sample.
    stage : int
        1 (one population) or 2 (joint).
    names : list of str
        Population labels, indexed by the site suffix.
    truth : dict
        (population, parameter) -> true value, for mocks; empty for the data.

    Returns
    -------
    list of dict
        stage, population ('shared' for shared parameters in stage 2), param, median, p16, p84, r_hat (split Gelman-Rubin) and
        n_eff (effective sample size) from numpyro.diagnostics.summary, divergences (number of divergent NUTS transitions of the
        run, all chains), truth (NaN when not a mock).
    """
    # Convergence diagnostics per sample site (numpyro.diagnostics.summary needs the samples grouped by chain).
    s = summary(mcmc.get_samples(group_by_chain=True))
    # Divergent transitions summed over all chains and draws.
    div = int(np.sum(mcmc.get_extra_fields()['diverging']))
    out = []
    for site, x in mcmc.get_samples().items():  # x: draws of all chains pooled
        # Shared sites have no suffix; per-population sites end with '_<population index>'.
        if site in kinfit.SHARED:
            base, pop = site, 'shared' if stage == 2 else names[0]
        else:
            base, idx = site.rsplit('_', 1)
            pop = names[int(idx)]
        x = np.asarray(x)
        # Median and 16th/84th percentiles, diagnostics and the mock truth (NaN for data).
        out.append(dict(stage=stage, population=pop, param=base, median=np.median(x), p16=np.percentile(x, 16),
                        p84=np.percentile(x, 84), r_hat=float(s[site]['r_hat']), n_eff=float(s[site]['n_eff']), divergences=div,
                        truth=truth.get((pop, base), np.nan)))
    return out


# %% [markdown]
# ### The fit: load the stars, then stage 1 and stage 2
# `main` reads the stars (the giant catalogues of Zenodo bundle 1 through `lmcold.samples.giants()`, the RR Lyrae members through
# `lmcold.samples.rrab()`), cuts them to the radial range, finds each population's maximum-likelihood start, and samples the
# posterior of each population on its own (stage 1) and of all populations together with a shared geometry (stage 2). For a
# 'mock' run it first replaces the velocities by draws from the model with a known truth. It writes the summary table
# `results/kin_mcmc_<run>.csv` and the posterior samples `data/kin_mcmc_<run>_samples.npz`.

# %%
def main(kind, model, dist=False, outer=False):
    """Stage-1 and stage-2 posteriors for 'mock' or 'data' and one model, with the RR Lyrae distance moduli if dist, stars beyond
    OUTER_INNER only if outer; writes the csv and npz of the module docstring."""
    run = f'{kind}_{model}' + ('_dist' if dist else '') + ('_outer' if outer else '')   # name of the output files
    inner = OUTER_INNER if outer else RING[0]  # inner edge of the radial range [deg]
    # Warm-up steps: WARMUP_ELLIPSOID for the main ellipsoid run, WARMUP otherwise.
    warmup = WARMUP_ELLIPSOID if model == 'ellipsoid' and not outer else WARMUP
    # Giants have no distance moduli; NaN columns make kinfit skip the terms a population lacks.
    # Load the data: (name, catalogue dict) per population, giants first (five [Fe/H] bins), RR Lyrae last.
    pops = [(f'giants {lab}', ring(cat | dict(dmu=np.full(len(cat['ra']), np.nan)), inner)) for lab, cat in samples.giants()]
    rr = samples.rrab()
    nan = np.full(len(rr['ra']), np.nan)
    # RR Lyrae have no line-of-sight velocities; their moduli are used only with dist.
    pops.append(('RR Lyrae', ring(rr | dict(vlos=nan, evlos=nan, dmu=rr['dmu'] if dist else nan), inner)))
    names = [n for n, _ in pops]
    ds = [kinfit.data(*(c[k] for k in COLS)) for _, c in pops]   # arrays kinfit needs (unit vectors, covariances) per population
    p_ml = [ml(d, model) for d in ds]   # maximum-likelihood starting vectors, shape (33,) each
    truth = {}
    if kind == 'mock':
        # One shared truth: geometry and bulk motion of the first population's fit, centre moved by MOCK_OFFSET, symmetric disc.
        shared = {n: p_ml[0][kinfit.PARAMS.index(n)] for n in kinfit.SHARED} | dict(dx=MOCK_OFFSET[0], dy=MOCK_OFFSET[1], m1c=0., m1s=0.)
        mock_ds = []
        for i, ((name, c), d, p) in enumerate(zip(pops, ds, p_ml)):
            # Truth vector q: the population's own fit with the shared values, k = 1 and no outliers (logit_fout = -8).
            q = p.copy()
            for n, v in (shared | dict(k=1., logit_fout=-8.)).items():
                q[kinfit.PARAMS.index(n)] = v
            # Model mean pmra, pmdec [mas/yr], vlos [km/s] and disc distance D [kpc] of each star under the truth.
            mra, mde, mlos, D = (np.asarray(v) for v in kinfit.model(q, d['rhat'], d['ea'], d['ed']))
            rng = np.random.default_rng(i)  # one fixed seed per population
            # Draw proper motions from the Gaia covariance plus the intrinsic dispersion sigma (q[8], km/s -> mas/yr at D), and
            # line-of-sight velocities with e_vlos and sigma_los (q[12]); only for stars that have a measured velocity.
            cov = np.asarray(d['cov']) + ((q[8] / (KAPPA * D)) ** 2)[:, None, None] * np.eye(2)
            noise = np.einsum('nij,nj->ni', np.linalg.cholesky(cov), rng.normal(size=(len(D), 2)))
            los = mlos + rng.normal(size=len(D)) * np.hypot(c['evlos'], q[12])
            # Distance moduli about dmu0 + 5 log10(D / D0) with scatter sigma_mu (q[15], q[16]), where the star has one.
            mu = q[15] + 5 * np.log10(D / LMC['D0']) + rng.normal(size=len(D)) * q[16]
            # Mock catalogue: observed positions and errors, simulated velocities and moduli, in kinfit.data layout.
            mock_ds.append(kinfit.data(*(dict(c, pmra=mra + noise[:, 0], pmdec=mde + noise[:, 1], vlos=np.where(np.isfinite(c['vlos']), los, np.nan),
                                              dmu=np.where(np.isfinite(c['dmu']), mu, np.nan))[k] for k in COLS)))
            # Truth per (population, parameter), and for the shared parameters under 'shared'.
            truth.update({(name, n): v for n, v in zip(kinfit.PARAMS, q)})
            truth.update({('shared', n): v for n, v in zip(kinfit.PARAMS, q) if n in kinfit.SHARED})
        ds = mock_ds
        # Restart from the maximum likelihood of the mock itself (as for the data), not from the truth.
        p_ml = [ml(d, model) for d in ds]
    # Stage 1: each population on its own, every parameter of the model its own.
    out, draws = [], {}
    for i, (name, d, p) in enumerate(zip(names, ds, p_ml)):
        # NUTS for one population, started at its maximum-likelihood vector; FIXED[model] parameters held.
        m = kinfit.sample([d], [p], FIXED[model], warmup, DRAWS, CHAINS, ellipsoid=model == 'ellipsoid')
        out += rows(m, 1, [name], truth)
        # Keep the samples, shape (CHAINS, DRAWS) per site, under keys '1|<population index>|<site>'.
        draws.update({f'1|{i}|{k}': np.asarray(v) for k, v in m.get_samples(group_by_chain=True).items()})
        print(f'stage 1 {name} done', flush=True)
    # Stage 2: all populations together, shared parameters started from the first population's fit.
    m = kinfit.sample(ds, p_ml, FIXED[model], warmup, DRAWS, CHAINS, ellipsoid=model == 'ellipsoid')
    out += rows(m, 2, names, truth)
    # Stage-2 samples under keys '2|all|<site>'; then save the samples and the summary table.
    draws.update({f'2|all|{k}': np.asarray(v) for k, v in m.get_samples(group_by_chain=True).items()})
    np.savez(DATA / f'kin_mcmc_{run}_samples.npz', names=names, **draws)
    t = pd.DataFrame(out)
    t.to_csv(ROOT / f'results/kin_mcmc_{run}.csv', index=False, float_format='%.5f')
    # Print the geometry and lopsided parameters, and the worst convergence diagnostics of the run.
    print(t[t.param.isin(['dx', 'dy', 'm1c', 'm1s', 'pmra0', 'pmdec0', 'inc', 'theta', 'dmu0', 'sigma_mu'])].to_string(index=False, float_format='%.3f'))
    print('max r_hat', t.r_hat.max(), 'min n_eff', t.n_eff.min(), 'divergences', t.divergences.max())


# %% [markdown]
# ### Diagnostic figure
# The centre offset and the lopsided (or $m=2$) part of every stage-1 fit and of the joint fit, one row per model run so far.

# %%
def figure(kind):
    """Rows: models run so far; left the centre offset, right the lopsided (or, for 'stream', m = 2) part; stage 1 by population,
    stage 2 black star."""
    plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
    # The models of this kind whose result table exists, in MODELS order: one row of panels each.
    runs = [m for m in MODELS if (ROOT / f'results/kin_mcmc_{kind}_{m}.csv').exists()]
    fig, axes = panel_grid(len(runs), 2, width=6.5, left=.6, right=2.2, hgap=.65, vgap=.6, bottom=.5, top=.3, aspect=1.)
    # One colour per stage-1 population: five giant bins (metal-poor first), then RR Lyrae.
    colours = ['#d7191c', '#fdae61', '#a6d96a', '#1a9641', '#2878b5', '#7b3294']
    for row, model in zip(axes, runs):
        t = pd.read_csv(ROOT / f'results/kin_mcmc_{kind}_{model}.csv')
        # Right panel: the m = 2 part for the models that free streaming, the lopsided part otherwise.
        m = ('m2c', 'm2s') if model in ('stream', 'ellipsoid') else ('m1c', 'm1s')
        for ax, (a, b) in zip(row, (('dx', 'dy'), m)):
            for stage in (1, 2):
                s = t[t.stage == stage]
                # Populations that have parameter a in this stage ('shared' alone in stage 2 for shared parameters).
                pops = [p for p in s.population.unique() if ((s.population == p) & (s.param == a)).any()]
                for j, pop in enumerate(pops):
                    # Median with 16th-84th percentile error bars in both coordinates.
                    x, y = (s[(s.population == pop) & (s.param == c)].iloc[0] for c in (a, b))
                    ax.errorbar(x['median'], y['median'], [[y['median'] - y.p16], [y.p84 - y['median']]],
                                [[x['median'] - x.p16], [x.p84 - x['median']]], fmt='*' if stage == 2 else 'o',
                                ms=10 if stage == 2 else 4, lw=.8, color='k' if stage == 2 else colours[j],
                                label='joint (stage 2)' if stage == 2 else pop)
            # Mock truth (shared values of stage 2) as a black cross; grey lines at zero.
            if kind == 'mock':
                tr = t[(t.stage == 2) & (t.population == 'shared')].set_index('param').truth
                ax.plot(tr[a], tr[b], 'kx', ms=9, mew=1.5, label='truth')
            ax.axhline(0, color='.7', lw=.5)
            ax.axvline(0, color='.7', lw=.5)
        # East to the left on the centre panel (reversed x axis), equal aspect.
        row[0].set(xlim=(1.4, -.6), ylim=(-1., 1.), aspect='equal', xlabel='centre offset east [deg]', ylabel='north [deg]',
                   title=f'{kind}, model: {model}')
        row[1].set(xlim=(-.25, .25), ylim=(-.25, .25), aspect='equal', xlabel=rf'${m[0][0]}_{{{m[0][1:]}}}$', ylabel=rf'${m[1][0]}_{{{m[1][1:]}}}$',
                   title='lopsided part' if m[0] == 'm1c' else 'm = 2 part')
    # One legend for all panels (labels collected, duplicates removed), to the right of the first row.
    handles = {}
    for ax in axes.flat:
        handles.update(dict(zip(*ax.get_legend_handles_labels()[::-1])))
    axes[0, 1].legend(handles.values(), handles.keys(), loc='upper left', bbox_to_anchor=(1.04, 1), fontsize=6.5, frameon=False)
    fig.savefig(ROOT / f'figures/fig_kin_mcmc_{kind}.png', dpi=200)


# %% [markdown]
# ### Run
# Arguments: kind ('data' or 'mock'), model, and optionally 'dist' and 'outer' (see the header).

# %%
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], 'dist' in sys.argv[3:], 'outer' in sys.argv[3:])
    figure(sys.argv[1])
