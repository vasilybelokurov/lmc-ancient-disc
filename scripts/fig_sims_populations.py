"""Diagnostic (Section 6 review): does the analogues' slow rotation of the most metal-poor population depend on how that population or
its sample is defined? Particle values only, no fits. Its table gives the numbers of the paragraph 'The gap does not depend on how
the old population is defined' (scripts/paper_text_sims.py); the figure is a diagnostic, not in the paper.

For each analogue (lmcold.sims, placed and weighted as in scripts/sims_observe.particles: E = expected BOSS giants), the particles at
2 < rho < 9 deg, inside the v_los window and within TRUTH_DR of R = 4 kpc give v_phi / sigma_z (E-weighted mean v_phi over standard
deviation of v_z) for three definitions of the old population, each holding the same fraction F0 of the formed mass within RHO_MASS
as the LMC's [Fe/H] < -1.5 bin (results/giant_mass.csv, 2.2 per cent):
- 'rank': LMC-scale [Fe/H] < -1.5 (the definition of the paper, no [Fe/H] errors);
- 'native': the simulation's own [Fe/H] < -1.5 (fraction of mass then differs; given in the csv);
- 'oldest': the oldest particles holding F0 of the formed mass.
'observed' is the same quantity for the mock giants of the bin (results/sims_observed.csv t_vphi / t_sigma_z: after the cell
resampling and the [Fe/H] errors), so 'observed' against 'rank' isolates the effect of the observing step.

Reads the star particles of every analogue (as scripts/sims_observe.py), the inputs of sims_observe.setup() and
results/sims_observed.csv.
Writes results/sims_populations.csv (suite, name, ratio_<definition>, mass fraction of 'native', formed-mass fraction of the 'oldest'
particles that are also in 'rank', median age [Gyr] of 'rank') and figures/fig_sims_populations.png: (a) 'observed' and (b) 'native'
and 'oldest' against 'rank', with the LMC's fitted v_rot / sigma_z (1.76) marked.

Run: .venv/bin/python scripts/fig_sims_populations.py   (reads the particles of all analogues)
"""
# %% [markdown]
# ### Settings
# The cuts are those of `scripts/sims_observe.py`, imported from it; the LMC's value is marked on the figure.

# %%
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_sims_kinematics import SUITE_COLOURS
from sims_observe import RHO_MASS, TRUTH_DR, VLOS_WINDOW, particles, setup
from lmcold import kinfit, sims
from lmcold.layout import panel_grid
from lmcold.sources import ROOT

LMC_RATIO = 1.76  # v_rot / sigma_z, results/kin_support_ellipsoid.csv, giants [Fe/H] < -1.5


# %% [markdown]
# ### Helper
# `ratio` is the rotational support of a set of particles, $v_\phi/\sigma_z$, each particle weighted by its expected number of giants.

# %%
def ratio(sel, X, V, E):
    """E-weighted mean v_phi / standard deviation of v_z of the selected particles.

    sel: (N,) bool; X [kpc], V [km/s]: (N, 3) disc-frame positions and velocities; E: (N,) expected giants per particle."""
    # Cylindrical radius [kpc] and v_phi = (x v_y - y v_x) / R [km/s] of the selected particles; weights w = E.
    R = np.hypot(X[sel, 0], X[sel, 1])
    vphi = (X[sel, 0] * V[sel, 1] - X[sel, 1] * V[sel, 0]) / R
    w = E[sel]
    vz = V[sel, 2]
    return np.average(vphi, weights=w) / np.sqrt(np.average((vz - np.average(vz, weights=w)) ** 2, weights=w))


if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data
    # The shared inputs of `scripts/sims_observe.py` (frame, metallicity knots, giants per formed mass, surviving-mass grid) and the
    # mock giants' particle values of the most metal-poor bin (`results/sims_observed.csv`, population 0). `F0` is the LMC's formed-mass
    # fraction below [Fe/H] = −1.5.

    # %%
    frame, _, _, knots, nbox, survival = setup()
    # F0: the LMC's cumulative formed-mass fraction at [Fe/H] = -1.5 (interpolated in the knots of sims_observe.lmc_knots).
    F0 = float(np.interp(-1.5, *knots))
    obs = pd.read_csv(ROOT / 'results/sims_observed.csv', dtype={'name': str})
    # Population 0: the most metal-poor bin, [Fe/H] < -1.5; one row per analogue.
    obs = obs[obs.population == 0].set_index(['suite', 'name'])

    # %% [markdown]
    # ### Three definitions of the old population, per analogue
    # For each analogue: the 'oldest' particles holding `F0` of the formed mass inside the 12-deg cone, the 'rank' and 'native'
    # selections, and $v_\phi/\sigma_z$ of each among the particles that could be sample giants near $R$ = 4 kpc.

    # %%
    rows = []
    for suite, name in sims.names():
        # X, V: disc-frame positions [kpc] and velocities [km/s], (N, 3); s: sky observables; rho [deg]; feh: LMC-scale [Fe/H]
        # [dex];
        # E: expected BOSS giants per particle; st: the loaded particles (native feh, age [Gyr], m_formed [Msun]).
        X, V, s, rho, feh, E, st = particles(suite, name, frame, knots, nbox, survival)
        cone = rho < RHO_MASS
        mf = np.where(cone, st['m_formed'], 0.0)   # formed mass [Msun] inside the cone, 0 outside
        o = np.argsort(-st['age'])   # oldest first
        # 'oldest': the oldest particles whose cumulative formed mass stays within F0 of the cone's total.
        oldest = np.zeros(len(o), bool)
        oldest[o[np.cumsum(mf[o]) <= F0 * mf.sum()]] = True
        oldest &= cone
        rank, native = feh < -1.5, cone & (st['feh'] < -1.5)   # LMC-scale [Fe/H] (NaN outside the cone) and the simulation's own
        # Particles that could be sample giants: 2 < rho < 9 deg, v_los in VLOS_WINDOW [km/s], and |R - R_SIG| < TRUTH_DR (R_SIG = 4
        # kpc).
        near = ((rho > 2) & (rho < 9) & (s['vlos'] > VLOS_WINDOW[0]) & (s['vlos'] < VLOS_WINDOW[1])
                & (np.abs(np.hypot(X[:, 0], X[:, 1]) - kinfit.R_SIG) < TRUTH_DR))
        # 'observed' from the mock giants' true v_phi / sigma_z; the three particle definitions from ratio(); mass fractions and
        # median age.
        t = obs.loc[(suite, name)]
        rows.append(dict(suite=suite, name=name, ratio_observed=t.t_vphi / t.t_sigma_z,
                         **{f'ratio_{k}': ratio(near & sel, X, V, E) for k, sel in (('rank', rank), ('native', native), ('oldest', oldest))},
                         native_fraction=mf[native].sum() / mf.sum(), oldest_in_rank=mf[oldest & rank].sum() / mf[oldest].sum(),
                         rank_age=float(np.median(st['age'][rank]))))
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / 'results/sims_populations.csv', index=False, float_format='%.4f')

    # %% [markdown]
    # ### The diagnostic figure
    # (a) the mock giants' particles against the 'rank' particles; (b) the 'native' and 'oldest' definitions against 'rank'; grey line:
    # equality; dashed: the LMC.

    # %%
    plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
    # Two equal panels; points coloured by suite (SUITE_COLOURS of scripts/paper_sims_kinematics.py), x the 'rank' ratio in both.
    fig, ax = panel_grid(1, 2, width=6.0, left=0.55, right=0.1, hgap=0.5, top=0.4)
    # Panel (a): 'observed'; panel (b): 'native' (circles) and 'oldest' (squares); one colour per suite.
    for a, cols, letter in ((ax[0, 0], [('observed', 'o')], 'a'), (ax[0, 1], [('native', 'o'), ('oldest', 's')], 'b')):
        for suite, c in SUITE_COLOURS.items():
            g = t[t.suite == suite]
            for k, m in cols:
                a.scatter(g.ratio_rank, g[f'ratio_{k}'], s=8, marker=m, color=c, lw=0, alpha=0.8,
                          label=f'{suite}, {k}' if len(cols) > 1 or k == 'observed' else None)
        # Axis range of v_phi/sigma_z in both panels; grey line y = x, dashed line the LMC's value; legend above the panel.
        lim = [-0.3, 2.0]
        a.plot(lim, lim, color='0.5', lw=0.6)
        a.axhline(LMC_RATIO, color='k', lw=1, ls='--')
        a.set(xlim=lim, ylim=lim, xlabel=r"particles, LMC-scale [Fe/H] $< -1.5$: $v_\phi/\sigma_z$",
              ylabel=r'$v_\phi/\sigma_z$ (' + ('mock giants' if letter == 'a' else 'native [Fe/H] / oldest') + ')')
        # Panel letter at top left; the table's summary statistics printed after the figure is saved.
        a.text(0.04, 0.92, f'({letter})', transform=a.transAxes)
        a.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=3, fontsize=5, frameon=False)
    fig.savefig(ROOT / 'figures/fig_sims_populations.png')
    print(t.describe().round(2).to_string())
