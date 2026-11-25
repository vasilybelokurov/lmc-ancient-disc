"""Paper figure (Section 6): rotation, dispersion and rotational support of the LMC giants against the simulated analogues observed
and fitted like them, against the cumulative fraction of formed mass by metallicity rank (the quantity the simulations' [Fe/H] is
matched in).

x: the LMC's cumulative formed-mass fraction below each giant bin's median [Fe/H] (sims_observe.lmc_knots; log axis); thin vertical
lines and the top axis mark the bin edges in LMC [Fe/H]. LMC (results/kin_support_ellipsoid.csv, the stage-1 ellipsoid fits of
Fig. 11): one black band per quantity spanning the 16th-84th percentiles of APOGEE + BOSS and, where fitted ([Fe/H] > -1.1), APOGEE
alone +- 1 sigma. RR Lyrae (same table, fitted at 4 < rho < 9 deg): median and 16th-84th percentiles, at the x of their median
[Fe/H] with a horizontal bar over their 16th-84th [Fe/H] percentiles (photometric [Fe/H], placed on the giants' formed-mass scale);
proper motions alone do not measure sigma_z, so in (b) and (c) they show sigma_R and v_rot / sigma_R instead (RRL_QUANT; PI,
2026-11-16: the best available). Simulations (results/sims_observed.csv, scripts/sims_observe.py): each analogue observed like the data (positions,
errors and number of the real giants, metallicity matched in cumulative formed mass) and fitted with the same model, bin by bin;
TNG50 (681): median and a probability band whose opacity is proportional to the probability density of the analogues (density_band);
Auriga (5) and FIRE-2 (3): one line per analogue. All are the fitted values of the five bins, joined by a monotone cubic in log x
(smooth()). Bins with fewer than NMIN_SIM mock giants are not shown. All at R = 4 kpc.

Two versions, panels 1 x 3, equal: (a) v_rot(4 kpc), (b) sigma_z(4 kpc) [km/s], (c) v_rot / sigma_z (linear); legend in the lower left of (b).
figures/paper_sims_kinematics.png (main text): x = the stellar mass formed before each population, f_j x M_formed (f_j the
LMC formed-mass fraction below bin j's median [Fe/H]; M_formed the total formed mass: the LMC's within RHO_MASS, results/
giant_mass.csv; each analogue its own, results/sims_formed_mass.csv from scripts/build_sims_formed_mass.py), so the LMC and every analogue have their own x; the TNG50 median and band
are drawn at the median TNG50 mass of each bin; the vertical lines and top axis are the LMC bin edges; numbers in (c): per cent of
the analogues whose bins span the LMC's own mass that are at or above the lower edge of the LMC band there (counts_at_mass();
results/sims_mass_counts.csv; none span the most metal-poor bin's, which is left blank).
figures/paper_sims_kinematics_fraction.png (appendix; letter Fig. 2): x = f_j, the quantity the simulations are matched in, the same for all
galaxies; numbers in (c): per cent of the TNG50 analogues (681) at or above the lower edge of the LMC band in each bin (no Auriga or
FIRE-2 analogue reaches it).
Definitions table (paper appendix): for each bin, the number of analogues at or above the LMC with rotational support defined as
v_rot / sigma_z, v_rot / sigma_R or v_rot / sigma_RMS (sigma_RMS^2 = (sigma_R^2 + sigma_phi^2 + sigma_z^2) / 3), fitted or from the
mock giants' particles within 0.5 kpc of R = 4 kpc; the LMC value is the lower of APOGEE + BOSS (16th percentile; median for
sigma_RMS) and APOGEE alone (-1 sigma; central for sigma_RMS). 'any': analogues reaching the lowest LMC value under any of the six.
Recovery (appendix): fitted against particle truth for every analogue and bin shown (v_rot, sigma_z, v_rot / sigma_z), colour =
suite.
Writes figures/paper_sims_kinematics.png, figures/paper_sims_kinematics_fraction.png, figures/paper_sims_recovery.png (1 x 3),
results/sims_mass_counts.csv, results/sims_kinematics_summary.csv (suite,
population, quantity, median, p16, p84, max, n_analogues, n_above_lmc: analogues above the lower edge of the LMC band) and
results/sims_definitions.csv.

Run: .venv/bin/python scripts/paper_sims_kinematics.py   (after scripts/sims_observe.py and scripts/build_sims_formed_mass.py; 3 s in
the recorded run of docs/run_times.txt)
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sims_observe import lmc_knots
from lmcold.layout import panel_grid
from lmcold.sources import ROOT

NMIN_SIM = 300  # mock giants: smaller bins give unstable fits
# (column, axis label) of the three panels: v_rot(4 kpc) [km/s], sigma_z(4 kpc) [km/s], v_rot / sigma_z.
QUANT = [('vrot4', r'$v_{\rm rot}(4\,{\rm kpc})$ [km s$^{-1}$]'), ('sigma_z', r'$\sigma_z(4\,{\rm kpc})$ [km s$^{-1}$]'),
         ('ratio_z', r'$v_{\rm rot}/\sigma_z$')]
LMC_COLOUR = 'k'   # LMC measurement bands black (PI, letter Fig. 2)
RRL_LABEL = r'LMC RR Lyrae ($\sigma_R$ in b, c)'
RRL_QUANT = {'vrot4': 'vrot4', 'sigma_z': 'sigma_R', 'ratio_z': 'ratio_R'}  # the RR Lyrae column shown in each panel
PGRID = np.linspace(0.002, 0.998, 400)  # probability levels of the band
BAND_ALPHA = 0.7  # opacity at the peak density
CONTOURS = {(0.16, 0.84): '-', (0.025, 0.975): '--'}  # quantile pairs of the band drawn as lines (68 and 95 per cent)
YMAX_RATIO = 6.0   # upper limit of the v_rot/sigma_z axis (panel c): room for the top of the LMC band (APOGEE alone, 7.7, is cut)
SUITE_COLOURS = {'TNG50': '#4c72b0', 'Auriga': '#dd8452', 'FIRE-2': '#55a868'}  # one colour per simulation suite


def lmc_band(t, q):
    """Lower and upper edge of the LMC band of quantity q per bin: the giants' 16th-84th percentiles and APOGEE +- 1 sigma."""
    lo, hi = t[f'{q}_p16'].to_numpy(), t[f'{q}_p84'].to_numpy()  # APOGEE + BOSS 16th-84th percentiles
    # APOGEE alone (value and error; columns present only for the bins where it was fitted) widens the band.
    a, e = t.get(f'{q}_apogee'), t.get(f'{q}_apogee_err')
    if a is not None:
        lo, hi = np.fmin(lo, (a - e).to_numpy()), np.fmax(hi, (a + e).to_numpy())
    return lo, hi


def rms(r, prefix='', suffix=''):
    """sigma_RMS of a table with columns <prefix>sigma_R<suffix>, ..._phi, ..._z."""
    return np.sqrt(sum(r[f'{prefix}sigma_{c}{suffix}'] ** 2 for c in ('R', 'phi', 'z')) / 3)


def smooth(x, y, xs):
    """y (per bin at x; along the first axis) at xs, joined by a monotone cubic (PCHIP, no overshoot) in log x."""
    from scipy.interpolate import PchipInterpolator
    return PchipInterpolator(np.log10(x), y, axis=0)(np.log10(xs))


def density_band(a, x, v, colour, label):
    """Draw the distribution of the analogues' values v (DataFrame: analogue x bin) as a band whose opacity is proportional to the
    probability density: per bin a Gaussian KDE gives the quantile function on PGRID; between bins each quantile is joined by
    smooth() (so the distribution shifts rather than splitting into two; quantiles kept non-decreasing in p); the density dp/dy on a fine y grid sets
    the opacity, scaled to its maximum in each column (BAND_ALPHA at the peak); the quantiles of CONTOURS are drawn as thin lines."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    from scipy.stats import gaussian_kde
    Q = []  # quantile function of each bin on PGRID
    for j in range(len(x)):
        u = v[j].dropna().to_numpy()  # values of the analogues in bin j
        # KDE evaluated on a y grid reaching 3 sd beyond the extreme values; its normalised cumulative sum is the CDF.
        ygrid = np.linspace(u.min() - 3 * u.std(), u.max() + 3 * u.std(), 2000)
        cdf = np.cumsum(gaussian_kde(u)(ygrid))
        Q.append(np.interp(PGRID, cdf / cdf[-1], ygrid))
    Q = np.array(Q)  # shape (bins, len(PGRID))
    lx = np.log10(x)
    xs = np.linspace(lx[0], lx[-1], 400)  # fine grid in log x between the first and last bin
    Qx = np.maximum.accumulate(smooth(x, Q, 10 ** xs).T, axis=0)  # (p, x)
    lo, hi = a.get_ylim()
    ys = np.linspace(lo, hi, 600)  # y grid over the current axis range
    # Density dp/dy per x column: derivative of the inverted quantile function; normalised to 1 at each column's peak.
    dens = np.array([np.gradient(np.interp(ys, Qx[:, i], PGRID), ys) for i in range(len(xs))]).T
    dens /= dens.max(axis=0, keepdims=True)
    # Colour map from transparent to the suite colour at opacity BAND_ALPHA.
    cmap = LinearSegmentedColormap.from_list('band', [(*to_rgb(colour), 0.0), (*to_rgb(colour), BAND_ALPHA)])
    a.pcolormesh(10 ** xs, ys, np.where(dens < 0.02, 0, dens), cmap=cmap, shading='nearest', lw=0,
                  edgecolors='none', antialiased=False, zorder=1, rasterized=True)  # tails below 2% of the peak blank
    # The quantiles of CONTOURS as thin lines (solid 68, dashed 95 per cent).
    for levels, ls in CONTOURS.items():
        for pk in levels:
            y = np.array([np.interp(pk, PGRID, Qx[:, i]) for i in range(len(xs))])
            a.plot(10 ** xs, y, ls, color=colour, lw=0.5, zorder=1.2)
    a.fill_between([], [], color=colour, alpha=BAND_ALPHA / 2, lw=0, label=label)  # legend entry


def formed_masses():
    """Total formed stellar mass [Msun] of every analogue (results/sims_formed_mass.csv, scripts/build_sims_formed_mass.py), as a
    Series indexed by (suite, name)."""
    return pd.read_csv(ROOT / 'results/sims_formed_mass.csv', dtype={'name': str}).set_index(['suite', 'name']).m_formed


def kinematics(ax, t, s, x_lmc, X, edges, xlabel, letters, counts, rrl, x_rrl):
    """One row of the figure (module docstring) on the three axes ax: the LMC bins at x_lmc, the RR Lyrae (row rrl of the kinematics
    table) at x_rrl (x of their 16th, 50th, 84th [Fe/H] percentiles), each analogue's bins at X (DataFrame
    indexed by (suite, name), columns = bins), bin edges = (x positions, LMC [Fe/H] labels). The TNG50 median and band are drawn at
    the median TNG50 x of each bin; counts: the numbers written above the LMC band in the v_rot / sigma_z panel. Returns the summary
    rows."""
    rows = []
    for a, (q, lab), letter in zip(ax, QUANT, letters):
        lo, hi = lmc_band(t, q)  # LMC band edges per bin
        for suite, c in SUITE_COLOURS.items():
            # v: values of quantity q, analogue x bin; xv: their x positions in the same layout.
            v = s[s.suite == suite].pivot(index='name', columns='population', values=q).reindex(columns=range(len(x_lmc)))
            xv = X.loc[suite].reindex(v.index)
            # TNG50: median over the analogues at the median x of each bin; Auriga, FIRE-2: one line per analogue.
            if suite == 'TNG50':
                x_tng = xv.median().to_numpy()
                xs = np.geomspace(x_tng[0], x_tng[-1], 200)
                a.plot(xs, smooth(x_tng, v.median().to_numpy(), xs), '-', color=c, lw=1.2, zorder=2)
            else:
                for i, (name, r) in enumerate(v.iterrows()):
                    xa = xv.loc[name].to_numpy()
                    xs = np.geomspace(xa[0], xa[-1], 200)
                    a.plot(xs, smooth(xa, r.to_numpy(), xs), '-', color=c, lw=0.8, label=suite if i == 0 else None)
            # Summary per bin over the analogues of this suite (results/sims_kinematics_summary.csv).
            for k in range(len(x_lmc)):
                col = v[k].dropna()
                rows.append(dict(suite=suite, population=k, quantity=q, median=col.median(), p16=col.quantile(.16),
                                 p84=col.quantile(.84), max=col.max(), n_analogues=len(col), n_above_lmc=int((col >= lo[k]).sum())))
        # Thin vertical lines at the LMC bin edges; log x axis; y from 0.
        for e in edges[0]:
            a.axvline(e, color='0.85', lw=0.5, zorder=0)
        a.set(xlabel=xlabel, ylabel=lab, xscale='log')
        a.set_ylim(0, YMAX_RATIO if q == 'ratio_z' else 1.08 * max(a.get_ylim()[1], np.nanmax(hi)))
        if q == 'ratio_z':
            for k in range(len(x_lmc)):  # above-left of the LMC band at each bin, further left where the band then rises steeply
                above = hi[k] + 0.5 < YMAX_RATIO   # (> 1 to the next bin); where the band leaves the axis, left of it under the top edge
                steep = k + 1 < len(x_lmc) and hi[k + 1] - hi[k] > 1
                a.text(x_lmc[k] * (0.8 if steep else 1.04) if above else x_lmc[k] / 1.6, hi[k] + 0.2 if above else YMAX_RATIO - 0.25,
                       counts[k], ha='right', va='bottom' if above else 'center', fontsize=6)
        # TNG50 probability band (drawn after the y range is set: it fills the axis range).
        density_band(a, x_tng, s[s.suite == 'TNG50'].pivot(index='name', columns='population', values=q),
                     SUITE_COLOURS['TNG50'], 'TNG50 (median, 68, 95 per cent)')
        # one slightly transparent colour, no edge; at least 2 per cent of the (linear) axis so the narrow metal-poor part shows
        pad = np.clip(0.01 * a.get_ylim()[1] - (hi - lo) / 2, 0, None)
        a.fill_between(x_lmc, lo - pad, hi + pad, color=LMC_COLOUR, lw=0, alpha=0.75, label='LMC giants (APOGEE + BOSS)', zorder=1.5)
        # RR Lyrae: open diamond at their median [Fe/H] x, bars over 16th-84th percentiles in x and y.
        r = RRL_QUANT[q]
        a.errorbar(x_rrl[1], rrl[r], xerr=[[x_rrl[1] - x_rrl[0]], [x_rrl[2] - x_rrl[1]]],
                   yerr=[[rrl[r] - rrl[f'{r}_p16']], [rrl[f'{r}_p84'] - rrl[r]]], fmt='D', ms=3.5, mfc='w', mec=LMC_COLOUR,
                   ecolor=LMC_COLOUR, elinewidth=0.7, capsize=0, zorder=3, label=RRL_LABEL)
        a.tick_params(axis='x', which='both', top=False)   # the top edge carries only the [Fe/H] bin edges
        # Top axis: the LMC [Fe/H] of the bin edges.
        top = a.secondary_xaxis('top')
        top.set_xticks(edges[0], [f'{f:g}' for f in edges[1]], fontsize=7)
        top.tick_params(direction='in')
        top.minorticks_off()
        top.set_xlabel('[Fe/H] (LMC)', fontsize=7)
        # Panel letter; x range padded by factors 1.6 and 1.25 beyond the outermost points.
        a.text(0.04, 0.92, f'({letter})', transform=a.transAxes)
        a.set_xlim(min(x_lmc[0], np.nanmin(X.to_numpy())) / 1.6, 1.25 * max(x_lmc[-1], np.nanmax(X.to_numpy())))
    return rows


def percent(p):
    """Fraction p as the text written in panel (c): '0%', one decimal below 1 per cent, else whole per cent."""
    return '0%' if p == 0 else (f'{100 * p:.1f}%' if p < 0.01 else f'{100 * p:.0f}%')


def counts_at_mass(t, s, x_lmc, X, q='ratio_z'):
    """Analogues at or above the lower edge of the LMC band at the LMC's own formed mass below each bin: each analogue's values,
    joined between its bins linearly in log x, read at x_lmc[k]; analogues whose bins do not span x_lmc[k] are not counted.
    Returns a DataFrame (population, n_covering, n_above_lmc)."""
    lo, _ = lmc_band(t, q)
    # Values (analogue x bin) and the x of each analogue's bins in the same order.
    v = s.pivot_table(index=['suite', 'name'], columns='population', values=q).reindex(columns=range(len(x_lmc)))
    xv = X.reindex(v.index).to_numpy()
    rows = []
    for k, xk in enumerate(x_lmc):
        n = above = 0
        for xa, ya in zip(xv, v.to_numpy()):
            ok = np.isfinite(ya) & np.isfinite(xa)  # bins shown (>= NMIN_SIM mock giants)
            # Count the analogue only if its shown bins span xk; interpolate linearly in log x.
            if ok.sum() >= 2 and xa[ok].min() <= xk <= xa[ok].max():
                n += 1
                above += np.interp(np.log10(xk), np.log10(xa[ok]), ya[ok]) >= lo[k]
        rows.append(dict(population=t.population[k], n_covering=n, n_above_lmc=int(above)))
    return pd.DataFrame(rows)


def definitions(t, s):
    """Number of analogues at or above the LMC per bin and definition (module docstring)."""
    # LMC value per definition and bin: the lower of APOGEE + BOSS and APOGEE alone.
    lmc = {'z': np.fmin(t.ratio_z_p16, t.ratio_z_apogee - t.ratio_z_apogee_err),
           'R': np.fmin(t.ratio_R_p16, t.ratio_R_apogee - t.ratio_R_apogee_err),
           'RMS': np.fmin(t.vrot4 / rms(t), t.vrot4_apogee / rms(t, suffix='_apogee'))}
    # Analogue values: fitted (first three) and from the mock giants' particles (t_* columns).
    sim = {'z': s.vrot4 / s.sigma_z, 'R': s.vrot4 / s.sigma_R, 'RMS': s.vrot4 / rms(s),
           'z particles': s.t_vphi / s.t_sigma_z, 'R particles': s.t_vphi / s.t_sigma_R, 'RMS particles': s.t_vphi / rms(s, 't_')}
    floor = pd.concat(lmc, axis=1).min(axis=1)  # lowest LMC value of the three definitions, per bin ('any')
    rows = []
    for j in range(len(t)):
        b = (s.population == j) & (s.n >= NMIN_SIM)  # analogues shown in bin j
        r = dict(population=t.population[j], n_analogues=int(b.sum()))
        for k, v in sim.items():
            r[k] = int((v[b] >= lmc[k.split()[0]][j]).sum())
        r['any'] = int((pd.concat(sim, axis=1)[b].max(axis=1) >= floor[j]).sum())
        r |= {f'lmc_{k}': float(v[j]) for k, v in lmc.items()}
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data
    # The figure reads four result tables and no catalogue or simulation: the simulations are needed only to remake
    # `results/sims_observed.csv` (by `scripts/sims_observe.py`, which observes and fits the TNG50, Auriga and FIRE-2 analogues like the
    # LMC giants) and `results/sims_formed_mass.csv` (by `scripts/build_sims_formed_mass.py`). The LMC side is
    # `results/kin_support_ellipsoid.csv` (stage-1 ellipsoid fits of the giants and RR Lyrae) and `results/giant_mass.csv` (formed mass
    # per [Fe/H] bin), read here directly and through `lmc_knots` (imported from `scripts/sims_observe.py`).

    # %%
    t = pd.read_csv(ROOT / 'results/kin_support_ellipsoid.csv')   # LMC: v_rot(4 kpc), sigma_R, sigma_phi, sigma_z per giant bin and RR Lyrae
    s = pd.read_csv(ROOT / 'results/sims_observed.csv', dtype={'name': str})   # one row per analogue and bin: fitted and particle values
    M_formed = formed_masses()   # results/sims_formed_mass.csv: total formed stellar mass of each analogue
    m_lmc = pd.read_csv(ROOT / 'results/giant_mass.csv').cumulative.iloc[-1]   # LMC formed mass within RHO_MASS
    knots = lmc_knots()   # results/giant_mass.csv: the LMC's cumulative formed-mass fraction against [Fe/H]

    # %% [markdown]
    # ### The LMC and the observed analogues
    # `t` holds the LMC's stage-1 ellipsoid fits at $R=4$ kpc (five giant [Fe/H] bins and the RR Lyrae, split here); `s` one row per
    # analogue and bin: the fitted $v_{\rm rot}(4\,{\rm kpc})$, $\sigma_R$, $\sigma_\phi$, $\sigma_z$ and the particle values `t_*` of the
    # same mock giants. Bins of fewer than `NMIN_SIM` mock giants are not used.

    # %%
    rrl = t[t.population == 'RR Lyrae'].iloc[0]  # the RR Lyrae row
    t = t[t.population.str.startswith('giants')].reset_index(drop=True)  # the five giant bins, metal-poor first
    x = t.feh_med.to_numpy()  # median [Fe/H] [dex] of each giant bin
    s['ratio_z'] = s.vrot4 / s.sigma_z  # fitted rotational support of each analogue and bin
    # Bins with fewer than NMIN_SIM mock giants are blanked.
    s.loc[s.n < NMIN_SIM, ['vrot4', 'sigma_z', 'ratio_z']] = np.nan

    # %% [markdown]
    # ### Where each population lies: formed-mass fraction and formed mass
    # The LMC's cumulative formed-mass fraction against [Fe/H] (`knots`, from `lmc_knots`) places each bin at the
    # fraction $f_j$ of the formed mass below its median [Fe/H]. In the formed-mass version each galaxy's $x$ is $f_j$ times its own total
    # formed mass (`M_formed`). `counts_at_mass` counts, at the LMC's own formed mass below each bin, the analogues
    # whose bins span that mass and that lie at or above the LMC band.

    # %%
    plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
    feh_edges = np.array([-1.5, -1.1, -0.8, -0.5])   # dex, inner edges of the five giant bins (lmcold.samples.FEH_BINS)
    # LMC cumulative formed-mass fraction below each bin's median [Fe/H], below each bin edge, and at the RR Lyrae percentiles.
    frac, frac_edges = np.interp(x, *knots), np.interp(feh_edges, *knots)
    frac_rrl = np.interp(rrl[['feh_p16', 'feh_med', 'feh_p84']].to_numpy(float), *knots)
    # x of every analogue's bins: f_j (fraction version, the same for all) or f_j x its own formed mass [Msun].
    analogues = pd.MultiIndex.from_frame(s[['suite', 'name']].drop_duplicates())
    X_frac = pd.DataFrame(np.tile(frac, (len(analogues), 1)), index=analogues)
    X_mass = X_frac.mul(M_formed.reindex(analogues), axis=0)
    lo, _ = lmc_band(t, 'ratio_z')  # lower edge of the LMC v_rot / sigma_z band per bin
    n = s.pivot_table(index=['suite', 'name'], columns='population', values='ratio_z').reindex(columns=range(len(x)))
    C = counts_at_mass(t, s, frac * m_lmc, X_mass)  # counts at the LMC's own formed mass below each bin
    tng = n.loc['TNG50']   # only TNG50 analogues reach the LMC band (results/sims_kinematics_summary.csv)
    C.to_csv(ROOT / 'results/sims_mass_counts.csv', index=False)
    print(C.to_string(index=False))

    # %% [markdown]
    # ### The two figures of rotation, dispersion and rotational support
    # One loop draws both versions: against the formed mass (`figures/paper_sims_kinematics.png`) and against the formed-mass fraction
    # (`figures/paper_sims_kinematics_fraction.png`). The per-bin summary over the analogues of the fraction version is kept for
    # `results/sims_kinematics_summary.csv`.

    # %%
    # Per version: file name, analogue x, LMC x, bin-edge x, x label, numbers for panel (c), RR Lyrae x.
    for name, X_, x_, edges_, xlabel, counts, xr in (
            ('paper_sims_kinematics', X_mass, frac * m_lmc, frac_edges * m_lmc, r'formed stellar mass [M$_\odot$]',
             [percent(c / nc) if nc else '' for c, nc in zip(C.n_above_lmc, C.n_covering)], frac_rrl * m_lmc),
            ('paper_sims_kinematics_fraction', X_frac, frac, frac_edges, 'cumulative fraction of formed mass',
             [percent((tng[k] >= lo[k]).sum() / tng[k].notna().sum()) for k in range(len(x))], frac_rrl)):
        fig, ax = panel_grid(1, 3, width=7.0, left=0.5, right=0.1, hgap=0.55, top=0.45)
        r = kinematics(ax[0], t, s, x_, X_, (edges_, feh_edges), xlabel, 'abc', counts, rrl, xr)
        if name.endswith('fraction'):
            rows = r  # summary per bin, bin-matched
        # Legend in (b): entries from (c) plus the RR Lyrae entry from (a), in a fixed order.
        h, l = ax[0, 2].get_legend_handles_labels()
        hr, lr = ax[0, 0].get_legend_handles_labels()
        h, l = h + [hr[lr.index(RRL_LABEL)]], l + [RRL_LABEL]
        order = [l.index(k) for k in ('LMC giants (APOGEE + BOSS)', RRL_LABEL, 'TNG50 (median, 68, 95 per cent)', 'Auriga', 'FIRE-2')]
        ax[0, 1].legend([h[i] for i in order], [l[i] for i in order], loc='lower left', fontsize=5, frameon=False)   # empty: sigma_z > 30 at the metal-poor end
        fig.savefig(ROOT / f'figures/{name}.png')

    # %% [markdown]
    # ### Other definitions of rotational support
    # For each bin, the number of analogues at or above the LMC with $v_{\rm rot}$ divided by $\sigma_z$, $\sigma_R$ or $\sigma_{\rm RMS}$,
    # fitted or from the particles (`definitions`; `results/sims_definitions.csv`).

    # %%
    D = definitions(t, s)  # one row per giant bin: analogue counts per definition and the LMC values
    D.to_csv(ROOT / 'results/sims_definitions.csv', index=False, float_format='%.3f')
    print(D.round(2).to_string(index=False))

    # %% [markdown]
    # ### Recovery of the analogues' own kinematics
    # The fitted values of every analogue and bin against the particle values of the same mock giants; then the summary table.

    # %%
    fig, ax = panel_grid(1, 3, width=7.0, left=0.5, right=0.1, hgap=0.55, top=0.4)
    s['t_ratio_z'] = s.t_vphi / s.t_sigma_z  # particle v_phi / sigma_z of the mock giants near R = 4 kpc
    for a, (q, lab), letter in zip(ax[0], QUANT, 'abc'):
        # Fitted (y) against particle (x) value per analogue and bin; the particle column of v_rot is t_vphi.
        for suite, c in SUITE_COLOURS.items():
            g = s[s.suite == suite]
            a.scatter(g[f't_{"vphi" if q == "vrot4" else q}'], g[q], s=4, color=c, lw=0, label=suite)
        # Common range for both axes from 0 (or the smallest value) to 1.05 x the largest; grey line y = x.
        t = s[[f't_{"vphi" if q == "vrot4" else q}', q]]
        lim = [min(0, np.nanmin(t.to_numpy()) * 1.05), np.nanmax(t.to_numpy()) * 1.05]
        a.plot(lim, lim, color='0.5', lw=0.6)
        a.set(xlim=lim, ylim=lim, xlabel='particles: ' + lab, ylabel='fitted: ' + lab)
        a.text(0.94, 0.06, f'({letter})', transform=a.transAxes, ha='right')
    ax[0, 0].legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=3, fontsize=6, frameon=False)
    fig.savefig(ROOT / 'figures/paper_sims_recovery.png')
    # Recovery numbers: median and 16-84 range of fitted - particle v_rot [km/s], median fitted / particle sigma_z.
    d = s.vrot4 - s.t_vphi
    print(f'recovery: v_rot - truth {np.nanmedian(d):+.1f} (16-84: {np.nanpercentile(d, 16):+.1f}, {np.nanpercentile(d, 84):+.1f}) km/s; '
          f'sigma_z / truth {np.nanmedian(s.sigma_z / s.t_sigma_z):.3f}')
    print('ratio fitted - particles per bin (p16, p50, p84):')
    for j in range(len(x)):
        print(j, np.nanpercentile((s.ratio_z - s.t_ratio_z)[s.population == j], [16, 50, 84]).round(2))
    # Summary of the fraction version, one row per suite x bin x quantity.
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / 'results/sims_kinematics_summary.csv', index=False, float_format='%.3f')
    print(R[R.quantity == 'ratio_z'].round(2).to_string(index=False))
