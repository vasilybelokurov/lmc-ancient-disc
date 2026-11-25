"""Rotational support of the LMC giants and RR Lyrae against metallicity in the disc-frame velocity-ellipsoid model (paper,
kinematics section; the ellipsoid counterpart of scripts/paper_kin_support.py).

Values: stage-1 posteriors of the ellipsoid model (lmcold.kinfit, ellipsoid=True: thin disc with free centre, bulk motion,
inclination, line of nodes, rotation curve v0 tanh(R / R0) [1 + vslope (R - R_SIG)], m = 2 streaming, dispersions sigma_R,
sigma_phi, sigma_z scaled by exp(dlnsig_dR (R - R_SIG)), Gaia error scale k, outliers) from scripts/fit_kinematics_mcmc.py, each
population fitted on its own. All quantities at R = R_SIG = 4 kpc, where the dispersions are the fitted sigma_R, sigma_phi,
sigma_z and the rotation speed is v_rot(4 kpc) = v0 tanh(4 / R0) (better constrained than v0, which trades with R0 and vslope).
Medians and 16th-84th percentiles of the paired posterior draws (data/kin_mcmc_data_<run>_samples.npz).
- Giants (proper motions and line-of-sight velocities): run 'ellipsoid', 2 < rho < 9 deg; all release giants in their measured
  [Fe/H] bin (lmcold.samples.giants()).
- RR Lyrae (proper motions only): run 'ellipsoid_outer', 4 < rho < 9 deg, because one Gaia error scale k cannot describe the
  crowded inner disc (over 2-9 deg the fit drives dlnsig_dR to -0.56 per kpc; scripts/paper_kin_dispersion_cut.py). Their sigma_z
  rests on the proper motions alone and is not measurable (scripts/paper_kin_ellipsoid_mock.py), so it is not plotted.
- APOGEE reference (bins from [Fe/H] = APOGEE_FROM up, where BOSS [Fe/H] errors blur the narrow bins;
  scripts/paper_kin_feh_mixing.py): maximum likelihood (lmcold.kinfit.fit, ellipsoid=True) of OWN on the APOGEE giants alone, the
  rest held at the bin's posterior medians; errors from the inverse Hessian, propagated to v_rot(4 kpc) and the ratios with R0 held.
x: median [Fe/H] of each sample's stars; horizontal bars: 16th-84th percentiles of their [Fe/H].

Writes figures/paper_kin_support_ellipsoid.png and results/kin_support_ellipsoid.csv (population, n, feh_med, feh_p16, feh_p84;
vrot4, sigma_R, sigma_phi, sigma_z, vrot4/sigma_R (ratio_R), vrot4/sigma_z (ratio_z), each median, _p16, _p84; APOGEE reference
values and errors where fitted). Panels (1 x 3, equal): (a) v_rot(4 kpc) [km/s], black, with the [Fe/H] spread (16th-84th
percentiles) as horizontal bars, sample key above; (b) sigma_R, sigma_phi, sigma_z at 4 kpc [km/s];
(c) v_rot / sigma_R and v_rot / sigma_z, from 0. In (b) and (c) colour is the quantity (COMP_COLOURS, key above each panel) and
the marker the sample: giants filled circles joined by thin lines, APOGEE alone open squares joined by dashed lines that start at the
last giant bin without an APOGEE reference, RR Lyrae stars (no sigma_z or
v_rot / sigma_z: proper motions only); quantities offset by DX in [Fe/H] so that the points do not overlap.

Run: .venv/bin/python scripts/paper_kin_support_ellipsoid.py   (after fit_kinematics_mcmc.py data ellipsoid, and data ellipsoid outer;
     13 s in the run of 2026-11-22, docs/run_times.txt)
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fit_kinematics_mcmc import OUTER_INNER, ring
from lmcold import kinfit, samples
from lmcold.layout import panel_grid
from lmcold.sources import DATA, ROOT

# Columns passed to kinfit.data: position [deg], proper motion and errors [mas/yr], correlation, velocity and error [km/s]
COLS = ('ra', 'dec', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error', 'pmra_pmdec_corr', 'vlos', 'evlos')
# Parameters free in the APOGEE-alone fit (module docstring): v0 [km/s], the three dispersions [km/s], k, outlier fraction
OWN = ('v0', 'sigma_R', 'sigma_phi', 'sigma_z', 'k', 'logit_fout')
APOGEE_FROM = -1.1  # [Fe/H]: bins with this lower edge or above get the APOGEE reference
# Quantities summarised per population: v_rot(4 kpc), the dispersions [km/s] and the two ratios
QUANT = ('vrot4', 'sigma_R', 'sigma_phi', 'sigma_z', 'ratio_R', 'ratio_z')
# The three dispersions (keys used to name the APOGEE columns) and their labels
SIG_COLOURS = dict(sigma_R='C0', sigma_phi='C1', sigma_z='C2')
SIG_LABELS = dict(sigma_R=r'$\sigma_R$', sigma_phi=r'$\sigma_\phi$', sigma_z=r'$\sigma_z$')


def draws(run, pop):
    """Paired stage-1 draws of population pop in run: dict of QUANT arrays, and the posterior-median parameter vector."""
    # Draw arrays keyed 'stage|population index|parameter'; i: index of pop in the run's population list
    z = np.load(DATA / f'kin_mcmc_data_{run}_samples.npz')
    i = list(z['names']).index(pop)
    g = lambda n: z[f'1|{i}|{n}_0'].reshape(-1)  # noqa: E731  population's own parameter
    # Per draw: v_rot(4 kpc) = v0 tanh(R_SIG / R0) [km/s], the dispersions [km/s] and the ratios
    d = dict(vrot4=g('v0') * np.tanh(kinfit.R_SIG / g('R0')), sigma_R=g('sigma_R'), sigma_phi=g('sigma_phi'), sigma_z=g('sigma_z'))
    d |= dict(ratio_R=d['vrot4'] / d['sigma_R'], ratio_z=d['vrot4'] / d['sigma_z'])
    # Parameter vector: start values, overwritten by the median of every stage-1 parameter of pop (suffix '_0' dropped)
    p = kinfit.start()
    for key in z.files:
        if key.startswith(f'1|{i}|'):
            n = key.split('|')[2]
            p[kinfit.PARAMS.index(n[:-2] if n.endswith('_0') else n)] = np.median(z[key])
    return d, p


def summary(d):
    """Median and 16th, 84th percentiles of each draw array, as flat columns."""
    out = {}
    for n, v in d.items():
        out |= dict(zip((n, f'{n}_p16', f'{n}_p84'), np.percentile(v, [50, 16, 84])))
    return out


# %% [markdown]
# ### Load the data
# The giant catalogues of Zenodo bundle 1 through `lmcold.samples.giants()`, merged and split into the five [Fe/H] bins; the RR Lyrae
# members through `lmcold.samples.rrab()` and, for their photometric [Fe/H], `data/rrab.parquet` directly. The posterior draws are read
# where they are used, by `draws()`: `data/kin_mcmc_data_ellipsoid_samples.npz` (giants) and
# `data/kin_mcmc_data_ellipsoid_outer_samples.npz` (RR Lyrae).

# %%
giants = samples.giants()  # (label, catalogue dict) per [Fe/H] bin, from data/giants_apogee.parquet and giants_boss.parquet
rrab = samples.rrab()  # RR Lyrae members with Gaia astrometry (data/rrab.parquet)
rr = pd.read_parquet(DATA / 'rrab.parquet')  # the full RR Lyrae table, with the photometric [Fe/H] column feh

# %% [markdown]
# ### The giant bins
# For each [Fe/H] bin: the stars at $2<\rho<9^\circ$, the medians and 16th-84th percentiles of the stage-1 posterior draws of the
# velocity-ellipsoid model, and the [Fe/H] spread of the stars. From [Fe/H] = `APOGEE_FROM` up, the APOGEE giants of the bin are also fitted
# alone by maximum likelihood ($v_0$, the three dispersions, $k$ and the outlier fraction free; the rest held at the bin's posterior
# medians), with errors from the inverse Hessian.

# %%
rows = []
for (lo, _), (label, c) in zip(samples.FEH_BINS, giants):
    # Stars at 2 < rho < 9 deg; draws and the median parameter vector of the bin's stage-1 fit
    c = ring(c)
    pop = f'giants {label}'
    d, p = draws('ellipsoid', pop)
    # Row: number of stars, median and 16th-84th percentiles of [Fe/H] [dex], posterior summaries
    r = dict(population=pop, n=len(c['ra']), feh_med=np.median(c['feh']), feh_p16=np.percentile(c['feh'], 16),
             feh_p84=np.percentile(c['feh'], 84)) | summary(d)
    if lo >= APOGEE_FROM:
        # APOGEE giants of the bin; maximum likelihood of OWN, the rest held at p; v, e: values and errors
        a = c['survey'] == 'apogee'
        pa, ea, _ = kinfit.fit(kinfit.data(*(c[k][a] for k in COLS)), free=OWN, p0=p, ellipsoid=True)
        v = {n: pa[kinfit.PARAMS.index(n)] for n in OWN}
        e = {n: ea[kinfit.PARAMS.index(n)] for n in OWN}
        # tanh(R_SIG / R0) with R0 held, to turn v0 into v_rot(4 kpc) [km/s]
        t4 = np.tanh(kinfit.R_SIG / p[kinfit.PARAMS.index('R0')])
        r |= dict(n_apogee=int(a.sum()), feh_med_apogee=np.median(c['feh'][a]), vrot4_apogee=v['v0'] * t4, vrot4_apogee_err=e['v0'] * t4)
        r |= {f'{n}_apogee': v[n] for n in SIG_COLOURS} | {f'{n}_apogee_err': e[n] for n in SIG_COLOURS}
        # Ratios v_rot/sigma with relative errors added in quadrature
        for q, s in (('ratio_R', 'sigma_R'), ('ratio_z', 'sigma_z')):
            r[f'{q}_apogee'] = v['v0'] * t4 / v[s]
            r[f'{q}_apogee_err'] = r[f'{q}_apogee'] * np.hypot(e['v0'] / v['v0'], e[s] / v[s])
    rows.append(r)

# %% [markdown]
# ### The RR Lyrae and the table
# The RR Lyrae members: the posterior of their fit at $4<\rho<9^\circ$ (run 'ellipsoid_outer') and the spread of their photometric
# [Fe/H]. All rows are written to `results/kin_support_ellipsoid.csv`.

# %%
# RR Lyrae row: stars at 4 < rho < 9 deg (OUTER_INNER), photometric [Fe/H] spread, posterior of run 'ellipsoid_outer'
rr = rr[rr.member]
rows.append(dict(population='RR Lyrae', n=len(ring(rrab, OUTER_INNER)['ra']), feh_med=float(rr.feh.median()),
                 feh_p16=float(rr.feh.quantile(0.16)), feh_p84=float(rr.feh.quantile(0.84))) | summary(draws('ellipsoid_outer', 'RR Lyrae')[0]))
# Write the table and print the main columns and the APOGEE-alone values
t = pd.DataFrame(rows)
t.to_csv(ROOT / 'results/kin_support_ellipsoid.csv', index=False, float_format='%.4f')
print(t[['population', 'n', 'feh_med'] + list(QUANT)].round(2).to_string(index=False))
print(t.filter(regex='population|_apogee$').round(2).to_string(index=False))

# %% [markdown]
# ### The figure
# Three panels against [Fe/H]: rotation speed at $R_\sigma$, the three dispersions, and the two support ratios. Colour marks the
# quantity, the marker the sample; the quantities of one panel are offset slightly in [Fe/H].

# %%
# Three equal panels; g: giant rows, rl: RR Lyrae row; err, fx: asymmetric y and x error bars from the percentiles
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
fig, ax = panel_grid(1, 3, width=7.0, left=0.5, right=0.12, hgap=0.55, top=0.4, bottom=0.42)
g, rl = t[t.population != 'RR Lyrae'], t[t.population == 'RR Lyrae']
err = lambda s, n: [s[n] - s[f'{n}_p16'], s[f'{n}_p84'] - s[n]]  # noqa: E731
fx = lambda s: [s.feh_med - s.feh_p16, s.feh_p84 - s.feh_med]  # noqa: E731  [Fe/H] spread of the sample
# Colour = quantity; the ratio over sigma_X takes the colour of sigma_X.
COMP_COLOURS = dict(sigma_R='#1f5fa8', sigma_phi='#e08a1e', sigma_z='#2e9e57', ratio_R='#1f5fa8', ratio_z='#2e9e57')
COMP_LABELS = dict(sigma_R=r'$\sigma_R$', sigma_phi=r'$\sigma_\phi$', sigma_z=r'$\sigma_z$', ratio_R=r'$v_{\rm rot}/\sigma_R$',
                   ratio_z=r'$v_{\rm rot}/\sigma_z$')
DX = 0.035  # [Fe/H] offset between the quantities of one panel


def draw(a, n, colour, dx=0.0, xbars=False, with_rrl=True, label=None):
    """Plot quantity n of table t on axes a in colour, offset by dx in [Fe/H]; xbars: [Fe/H] spread as horizontal bars;
    with_rrl: also plot the RR Lyrae; label: legend label of the giant points."""
    # Giants (filled circles on a thin line), APOGEE alone (open squares) and RR Lyrae (star) for quantity n.
    a.plot(g.feh_med + dx, g[n], '-', color=colour, lw=0.6, alpha=0.6)
    # APOGEE branch (dashed): from the last bin without an APOGEE reference, where the bins are wide and mixing is small, through
    # the APOGEE points
    # Bins with an APOGEE reference; first: the last bin without one
    ap = g[g.feh_med_apogee.notna()]
    first = g[g.feh_med_apogee.isna()].iloc[-1]
    a.plot(np.r_[first.feh_med, ap.feh_med_apogee] + dx, np.r_[first[n], ap[f'{n}_apogee']], '--', color=colour, lw=0.6, alpha=0.6)
    # Giants (filled circles), APOGEE alone (open squares), RR Lyrae (star) with their error bars
    a.errorbar(g.feh_med + dx, g[n], xerr=fx(g) if xbars else None, yerr=err(g, n), fmt='o', color=colour, ms=4, lw=0.8, label=label)
    a.errorbar(g.feh_med_apogee + dx, g[f'{n}_apogee'], yerr=g[f'{n}_apogee_err'], fmt='s', color=colour, mfc='white', ms=4, lw=0.8)
    if with_rrl:
        a.errorbar(rl.feh_med + dx, rl[n], xerr=fx(rl) if xbars else None, yerr=err(rl, n), fmt='*', color=colour, ms=8, lw=0.8)


# (a) v_rot(4 kpc); (b) the dispersions (no RR Lyrae sigma_z); (c) the ratios (no RR Lyrae v_rot/sigma_z)
draw(ax[0, 0], 'vrot4', 'k', xbars=True)
for n, dx in (('sigma_R', -DX), ('sigma_phi', 0.0), ('sigma_z', DX)):
    draw(ax[0, 1], n, COMP_COLOURS[n], dx, with_rrl=n != 'sigma_z', label=COMP_LABELS[n])
for n, dx in (('ratio_R', -DX / 2), ('ratio_z', DX / 2)):
    draw(ax[0, 2], n, COMP_COLOURS[n], dx, with_rrl=n != 'ratio_z', label=COMP_LABELS[n])
# Sample key (marker shapes, black) above (a); quantity keys (colours) above (b) and (c).
for fmt, mfc, ms, lab in (('o', 'k', 4, 'giants'), ('s', 'white', 4, 'APOGEE alone'), ('*', 'k', 8, 'RR Lyrae')):
    ax[0, 0].errorbar([], [], fmt=fmt, color='k', mfc=mfc, ms=ms, lw=0.8, label=lab)
ax[0, 0].legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=6, handletextpad=0.2, columnspacing=0.8)
for a in ax[0, 1:]:
    a.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=7, handletextpad=0.2, columnspacing=1.2)
# Axis labels, [Fe/H] range and y limits of each panel
for a, lab in zip(ax[0], (r'$v_{\rm rot}(4\,{\rm kpc})$ [km s$^{-1}$]', r'dispersion at 4 kpc [km s$^{-1}$]',
                          r'rotational support at 4 kpc')):
    a.set(xlabel='[Fe/H]', ylabel=lab, xlim=(-2.2, -0.2))
ax[0, 0].set_ylim(45, 85)
ax[0, 1].set_ylim(8, 44)
ax[0, 2].set_ylim(0, 8.5)  # rotational support from zero
for a, letter, (lx, ly) in zip(ax[0], 'abc', ((0.04, 0.92), (0.9, 0.92), (0.04, 0.92))):  # panel letters in empty corners
    a.text(lx, ly, f'({letter})', transform=a.transAxes, fontsize=7)
fig.savefig(ROOT / 'figures/paper_kin_support_ellipsoid.png', dpi=300)
