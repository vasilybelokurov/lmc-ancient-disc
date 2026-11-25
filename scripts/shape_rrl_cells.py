"""Flattening of the old, metal-poor population from its density (RR Lyrae sky counts and mean distances, without the noise floor)
against its inclination and rotation-implied flattening from its motions (paper Section 5; KB chapter 'The shape of the old
population', section 'The depth of the RR Lyrae').

The RR Lyrae (data/rrab.parquet members with an OGLE Wesenheit modulus, in the 2-7 deg footprint of lmcold.shape) are summarised in
sky cells (lmcold.shape.cell_stats: counts and median moduli; the scatter of the moduli, and so the noise floor, does not enter) and
fitted with the inclined oblate K0 spheroid (lmcold.shape.fit_cells). The cells measure two numbers (lmcold.shape.projected): the
outline axis ratio on the sky, q_sky (counts), and the near-to-far depth gradient, g (medians). Each is a curve in (inclination, q);
they cross at the free fit. The ridge (best q at each inclination of INC_GRID) gives q at the inclination of the same population
from its motions: the metal-poor giant bins OLD ([Fe/H] < -1.1; the RR Lyrae median is -1.38), stage-1 ellipsoid fits at 2-9 deg
(results/kin_mcmc_data_ellipsoid.csv), and, less precise, the RR Lyrae proper motions at 4-9 deg (..._outer.csv, as Section 4).
Compared with the rotation-implied q of the OLD bins (results/shape_virial.csv, three potentials).

q at the giants' inclination, its bootstrap scatter (NBOOT resamples of the stars) and its sensitivity to a coherent modulus
gradient across their line of nodes (DIPOLE mag at the footprint edge, both signs), the inclination held in each refit.

Reads: data/rrab.parquet (Zenodo bundle 1); results/kin_mcmc_data_ellipsoid.csv and kin_mcmc_data_ellipsoid_outer.csv (posterior
summaries of the kinematic fits, scripts/fit_kinematics_mcmc.py); results/shape_virial.csv (scripts/shape_virial.py, printed for
comparison only).
Writes results/shape_rrl_cells.csv (the free fit and the ridge: inclination, theta, q, h, delta, lnL, dlnL) and results/
shape_rrl_depth.csv (case: data, dipole +-DIPOLE, bootstrap; q, theta at the giants' inclination); the figure is made by
scripts/paper_shape_rrl.py. The tables are read by scripts/paper_shape_virial.py, paper_shape_rrl.py, paper_shape_rrl_residuals.py
and shape_rrl_triaxial.py; the last three also import OLD, kin_inc, motion_incs or virial_q from this file.

Run: .venv/bin/python scripts/shape_rrl_cells.py
"""
# %% [markdown]
# ### Settings
# The grid of inclinations of the ridge, the number of bootstrap resamples, the size of the coherent distance error, and the two
# metal-poor giant bins whose motions give the old population's tilt.

# %%
import numpy as np
import pandas as pd

from lmcold import shape
from lmcold.sources import ROOT

INC_GRID = np.arange(15.0, 60.1, 2.5)  # deg: inclinations 15-60 deg in steps of 2.5 deg at which the ridge fits hold inc fixed
NBOOT = 50  # bootstrap resamples of the stars for q at the kinematic inclination
DIPOLE = 0.02  # mag: coherent modulus offset at rho = 7 deg across the kinematic line of nodes (+ and -)
OLD = ('giants [Fe/H] < -1.5', 'giants -1.5 < [Fe/H] < -1.1')  # the metal-poor giant bins, the RR Lyrae's metallicities

# %% [markdown]
# ### Helpers
# `kin_inc` reads the inclination (or another parameter) of given populations from a table of kinematic posterior summaries;
# `motion_incs` gives the old population's tilt from the motions of the metal-poor giants and of the RR Lyrae; `virial_q` reads the
# axis ratio the motions of the metal-poor giants require.


# %%
def kin_inc(name, populations=('RR Lyrae',), param='inc'):
    """Stage-1 inclination (or another param) [deg] of the given populations in results/{name}.csv: mean of their medians, lowest p16,
    highest p84.

    Parameters
    ----------
    name : str
        Stem of the posterior-summary table in results/ (columns stage, population, param, median, p16, p84).
    populations : tuple of str
        Population labels of that table to combine.
    param : str
        Parameter name, e.g. 'inc' (inclination) or 'theta' (line of nodes, east of north) [deg].

    Returns
    -------
    (float, float, float)
        Mean of the populations' posterior medians, the lowest 16th and the highest 84th percentile.
    """
    # Stage-1 (own-fit) rows of the requested populations and parameter.
    t = pd.read_csv(ROOT / f'results/{name}.csv')
    t = t[(t.stage == 1) & t.population.isin(populations) & (t.param == param)]
    return t['median'].mean(), t.p16.min(), t.p84.max()  # (median, p16, p84) [deg]


def motion_incs():
    """The old population's inclination from its motions: {label: (median, p16, p84)} [deg], the metal-poor giants first (the two OLD
    bins of the 2-9 deg ellipsoid fit), then the RR Lyrae proper motions (the 4-9 deg fit, results/kin_mcmc_data_ellipsoid_outer.csv)."""
    # Giants: the two OLD bins of the 2-9 deg ellipsoid fit combined; RR Lyrae: their own 4-9 deg fit.
    return {'metal-poor giants': kin_inc('kin_mcmc_data_ellipsoid', OLD),
            'RR Lyrae proper motions': kin_inc('kin_mcmc_data_ellipsoid_outer')}


def virial_q():
    """Rotation-implied q of the OLD bins (results/shape_virial.csv, no central floor), all potentials: a DataFrame with one row per
    bin and potential (columns population, potential, q, q_p16, q_p84, ...)."""
    v = pd.read_csv(ROOT / 'results/shape_virial.csv')
    # OLD bins only, the variant without a central floor (r_floor = 0).
    return v[v.population.isin(OLD) & (v.r_floor == 0)]


if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data
    # The RR Lyrae members with a distance modulus from the OGLE Wesenheit magnitude (`dmu_src == 'I'`) inside the footprint of
    # `lmcold.shape` (2-7 deg from the LMC centre, within the modulus window), counted in the sky cells: number of stars, median modulus
    # and its error per cell. The kinematic tables are read later through `motion_incs()` and `kin_inc()`.

    # %%
    t = pd.read_parquet(ROOT / 'data/rrab.parquet')  # RR Lyrae: ra, dec [deg], membership, relative distance modulus dmu [mag], its source dmu_src
    t = t[t.member & (t.dmu_src == 'I')]  # members with an OGLE I-band Wesenheit modulus
    # Footprint of lmcold.shape: 2-7 deg from the centre, modulus within its window.
    t = t[shape.in_footprint(t.ra.to_numpy(), t.dec.to_numpy(), t.dmu.to_numpy())]
    stats = shape.cell_stats(t.ra.to_numpy(), t.dec.to_numpy(), t.dmu.to_numpy())  # per cell: counts, median dmu, its error

    # %% [markdown]
    # ### The free fit and the ridge
    # The oblate model with all five parameters free (inclination, line of nodes, axis ratio $q$, scale length $h$, modulus zero point $\delta$), then
    # refitted with the inclination held at each value of `INC_GRID`: the best $q$ at each inclination. `dlnL` is the log likelihood
    # relative to the best of all these fits.

    # %%
    rows = []  # one dict per fit: fit type, lnL and the CELL_PARAMS (inc [deg], theta [deg], q, h [kpc], delta [mag])
    # Free fit: all five parameters; best is the parameter vector, llb its log likelihood.
    best, llb = shape.fit_cells(stats)
    rows.append(dict(fit='free', lnL=llb, **dict(zip(shape.CELL_PARAMS, best))))
    # Ridge: refit at each inclination of INC_GRID, each started from the free fit.
    start = best
    for inc in INC_GRID:
        x, ll = shape.fit_cells(stats, fixed={'inc': inc}, start=start)
        rows.append(dict(fit='ridge', lnL=ll, **dict(zip(shape.CELL_PARAMS, x))))
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows)
    R['dlnL'] = R.lnL - R.lnL.max()  # log likelihood relative to the best fit
    R.to_csv(ROOT / 'results/shape_rrl_cells.csv', index=False, float_format='%.4f')

    # %% [markdown]
    # ### q at the tilt from the motions
    # The ridge interpolated at the inclination of the metal-poor giants and at that of the RR Lyrae proper motions (median and
    # 16th-84th percentiles), printed.

    # %%
    r = R[R.fit == 'ridge'].sort_values('inc')  # the ridge, increasing inclination (needed by np.interp)
    ki = motion_incs()  # {label: (median, p16, p84)} inclination [deg] from the motions
    for k, (m, lo, hi) in ki.items():
        # Ridge q at the median, 16th and 84th percentile inclinations (linear interpolation between grid points).
        q = np.interp([m, lo, hi], r.inc, r.q)
        print(f'inc from motions ({k}) {m:.1f} ({lo:.1f}-{hi:.1f}) -> q {q[0]:.3f} ({min(q[1:]):.3f}-{max(q[1:]):.3f})')

    # %% [markdown]
    # ### Robustness of q at the giants' tilt
    # With the inclination and line of nodes of the metal-poor giants: the fit to the data; two refits with a coherent modulus error
    # that grows linearly with the sky offset across the line of nodes to $\pm$`DIPOLE` mag at the outer edge of the cells; and
    # `NBOOT` bootstrap resamples of the stars. Written to `results/shape_rrl_depth.csv`.

    # %%
    # q at the kinematic inclination of the metal-poor giants: bootstrap of the stars, and a coherent modulus gradient across their
    # line of nodes (DIPOLE at the footprint edge), each refitted with the inclination held.
    inc, theta = ki['metal-poor giants'][0], kin_inc('kin_mcmc_data_ellipsoid', OLD, 'theta')[0]  # deg
    # Fit to the data with the inclination held at the giants' value; xs[2] is q, xs[1] theta [deg] (shape.CELL_PARAMS order).
    xs = shape.fit_cells(stats, fixed={'inc': inc}, start=best)[0]
    rows = [dict(case='data', q=xs[2], theta=xs[1])]
    from lmcold import geometry as geo
    from lmcold.constants import LMC
    r = shape.rhat(t.ra.to_numpy(), t.dec.to_numpy())  # unit vectors (N, 3) of the stars
    r0, eE, eN = (np.asarray(v) for v in geo.sky_basis(np.deg2rad(LMC['ra0']), np.deg2rad(LMC['dec0'])))  # centre, east, north
    across = shape.rho(r) * np.sin(np.arctan2(r @ eE, r @ eN) - np.deg2rad(theta))  # deg, sky offset across the line of nodes
    for sign in (1, -1):
        # moduli shifted by sign * DIPOLE * (offset across the nodes / outer cell radius CELL_RHO[-1] = 7 deg)
        st = shape.cell_stats(t.ra.to_numpy(), t.dec.to_numpy(), t.dmu.to_numpy() + sign * DIPOLE * across / shape.CELL_RHO[-1])
        x = shape.fit_cells(st, fixed={'inc': inc}, start=xs)[0]
        rows.append(dict(case=f'dipole {sign * DIPOLE:+.2f}', q=x[2], theta=x[1]))
    rng = np.random.default_rng(3)  # fixed seed of the bootstrap
    for b in range(NBOOT):
        k = rng.integers(0, len(t), len(t))  # indices of one bootstrap resample of the stars
        x = shape.fit_cells(shape.cell_stats(t.ra.to_numpy()[k], t.dec.to_numpy()[k], t.dmu.to_numpy()[k]), fixed={'inc': inc}, start=xs)[0]
        rows.append(dict(case='bootstrap', q=x[2], theta=x[1]))
    # Write the cases and print q with its bootstrap spread and the two dipole refits.
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / 'results/shape_rrl_depth.csv', index=False, float_format='%.4f')
    bq = D[D.case == 'bootstrap'].q
    print(f'q at i = {inc:.1f}: {xs[2]:.3f}; bootstrap sd {bq.std():.3f} (16-84: {bq.quantile(0.16):.3f}-{bq.quantile(0.84):.3f}); '
          + ', '.join(f'{c} -> {q:.3f}' for c, q in D[D.case.str.startswith('dipole')][['case', 'q']].to_numpy()))

    # %% [markdown]
    # ### Comparison with the motions
    # The axis ratio the motions of the metal-poor giants require (three potentials) and the free fit, printed side by side.

    # %%
    v = virial_q()  # rotation-implied q of the OLD bins, per potential
    print('virial q:', v[['population', 'potential', 'q', 'q_p16', 'q_p84']].round(3).to_string(index=False))
    print('free fit:', dict(zip(shape.CELL_PARAMS, np.round(best, 3))))
