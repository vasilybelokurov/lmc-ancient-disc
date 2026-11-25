"""Our [Fe/H] profiles against published LMC metallicity gradients (lmcold.constants.LMC_FEH_GRADIENT_LITERATURE, _LEVEL_).

Run scripts/paper_feh_profile.py first. For each published gradient we fit a straight line to our median [Fe/H] in the 1-kpc
in-plane bins whose centres lie inside the published radial range, weighted by the bootstrap errors of the medians (floor 0.005
dex); the error is the formal one of that fit. Giants: the BOSS giants with G < 16.5 ('BOSS G<16.5' of paper_feh_profile.py), the
reliable BOSS profile (fainter, hot, low-S/N giants have overestimated [Fe/H]; Section 3), and APOGEE; our own slopes are also given
for all BOSS giants (G < 17.5) as a check. Ranges with fewer than three bins (Skowron+16 beyond 8 kpc) get no fit.
*: published range given in degrees, converted at 0.873 kpc/deg. (a) giants: BOSS and APOGEE; (b) RRab; (c) median against projected distance
with the mean levels quoted in degrees by Carrera et al. (2008) and Massana et al. (2024; Gaia XP [M/H]) and the
Muñoz et al. (2023) APOGEE slope per degree drawn through our BOSS profile at the middle of its range.
Writes figures/paper_feh_literature.png and results/feh_gradients.csv.

Inputs: results/feh_profile.csv (median [Fe/H] per 1-kpc bin of in-plane radius R, with bootstrap median_lo/median_hi) and
results/feh_profile_rho.csv (the same per 1-deg bin of projected distance rho), both from scripts/paper_feh_profile.py; the
published values in lmcold.constants (LMC_FEH_GRADIENT_LITERATURE [dex/kpc], LMC_FEH_GRADIENT_DEG_LITERATURE [dex/deg],
LMC_FEH_LEVEL_LITERATURE).
Output CSV results/feh_gradients.csv, one row per published gradient and sample, plus our own slopes over 0-6, 6-10 and 0-10 kpc
('this work'): label, sample, r_lo, r_hi (range, kpc or, for the [dex/deg] rows, deg), slope_lit, err_lit (published), slope,
err (ours), and slope_unweighted for the [dex/deg] rows.
Panels (1 x 3; legends above the panels):
(a) x: published giant gradients (labels: study and radial range); y: d[Fe/H]/dR [dex/kpc]; black: published value (square with
    error; x where no error is quoted; a quoted interval of slopes is shown by its midpoint and half-range plus the error);
    coloured circles: our BOSS and APOGEE slopes over the same range.
(b) the same for the RRab (Skowron+16 inner, middle, outer).
(c) x: rho [deg], 0-12; y: median [Fe/H] [dex]; BOSS and APOGEE profiles with bootstrap errors; horizontal segments: Carrera+08
    (mean) and Massana+24 (XP [M/H]) levels over their quoted ranges; purple dashed: the Munoz+23 slope through our BOSS profile.
Placement: appendix E (docs/figures.md). Run: .venv/bin/python scripts/paper_feh_literature.py (after paper_feh_profile.py)
Run time: 1 s on a MacBook (M-series), docs/run_times.txt.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmcold.constants import LMC_FEH_GRADIENT_DEG_LITERATURE, LMC_FEH_GRADIENT_LITERATURE, LMC_FEH_LEVEL_LITERATURE
from lmcold.layout import panel_grid
from lmcold.sources import ROOT

# %% [markdown]
# ### Load the data
# Our [Fe/H] profiles written by `scripts/paper_feh_profile.py`: `results/feh_profile.csv` (against in-plane radius) and
# `results/feh_profile_rho.csv` (against projected distance). The published gradients and levels are not read from a file: they
# are listed in `lmcold.constants`.

# %%
# results/feh_profile.csv: median [Fe/H] and its bootstrap range per sample and bin of in-plane radius R [kpc]
t = pd.read_csv(ROOT / 'results/feh_profile.csv')
# results/feh_profile_rho.csv: the same per bin of projected distance rho [deg]
p = pd.read_csv(ROOT / 'results/feh_profile_rho.csv')

# %% [markdown]
# ### Our profiles
# Median [Fe/H] per bin of in-plane radius and of projected distance, with the error of each median.

# %%
BRIGHT = 'BOSS G<16.5'  # sample name in results/feh_profile*.csv (paper_feh_profile.G_BRIGHT)
GIANTS = [BRIGHT, 'APOGEE']  # giant samples compared with the published giant gradients
t['R'] = .5 * (t.R_lo + t.R_hi)  # bin centre [kpc]
t['e'] = (.5 * (t.median_hi - t.median_lo)).clip(.005)  # error of the median: half the 16-84 bootstrap range, floor 0.005 dex
p['x'], p['e'] = .5 * (p.rho_lo + p.rho_hi), (.5 * (p.median_hi - p.median_lo)).clip(.005)  # bin centre [deg], error [dex]


def slope(sample, lo, hi):
    """Weighted straight-line fit of our median [Fe/H] against R over the bins with centres in (lo, hi).

    Parameters
    ----------
    sample : str
        'BOSS', 'APOGEE' or 'RRab' (rows of results/feh_profile.csv).
    lo, hi : float
        Radial range [kpc]; bin centres strictly inside are used.

    Returns
    -------
    slope, err : float
        Gradient [dex/kpc] and its formal error (polyfit weights 1 / e, unscaled covariance); NaN for fewer than three bins.
    """
    q = t[(t['sample'] == sample) & (t.R > lo) & (t.R < hi)]
    if len(q) < 3:  # a line through two bins has no meaningful error
        return np.nan, np.nan
    # Line [Fe/H] = p[0] R + p[1]; p[0] is the slope [dex/kpc].
    p, cov = np.polyfit(q.R, q['median'], 1, w=1 / q.e, cov='unscaled')
    return p[0], np.sqrt(cov[0, 0])


# %% [markdown]
# ### Our slopes over each published range
# Written to `results/feh_gradients.csv`, with our own inner and outer slopes.

# %%
lit = pd.DataFrame(LMC_FEH_GRADIENT_LITERATURE)  # one row per published gradient [dex/kpc]: label, tracer, r_lo, r_hi, slope, err
rows = []  # rows of results/feh_gradients.csv
# Our slope over each published range, for the giant samples (giant gradients) or the RRab (RR Lyrae gradients).
for _, L in lit.iterrows():
    for sample in (GIANTS if L.tracer == 'giants' else ['RRab']):
        rows.append(dict(label=L.label, sample=sample, r_lo=L.r_lo, r_hi=L.r_hi, slope_lit=L.slope, err_lit=L.err,
                         **dict(zip(['slope', 'err'], slope(sample, L.r_lo, L.r_hi)))))
for sample in GIANTS + ['BOSS', 'RRab']:  # our own inner and outer slopes, either side of the break near 6 kpc (all BOSS as a check)
    for lo, hi in [(0., 6.), (6., 10.), (0., 10.)]:
        rows.append(dict(label='this work', sample=sample, r_lo=lo, r_hi=hi, **dict(zip(['slope', 'err'], slope(sample, lo, hi)))))
for L in LMC_FEH_GRADIENT_DEG_LITERATURE:  # [dex/deg], against projected distance (no minimum number of bins)
    for sample in GIANTS:
        q = p[(p['sample'] == sample) & (p.x > L['rho_lo']) & (p.x < L['rho_hi'])]
        # Weighted slope [dex/deg] with formal error; the unweighted slope is stored too.
        c, cov = np.polyfit(q.x, q['median'], 1, w=1 / q.e, cov='unscaled')
        rows.append(dict(label=L['label'] + ' [dex/deg]', sample=sample, r_lo=L['rho_lo'], r_hi=L['rho_hi'], slope_lit=L['slope'],
                         slope=c[0], err=np.sqrt(cov[0, 0]), slope_unweighted=np.polyfit(q.x, q['median'], 1)[0]))
g = pd.DataFrame(rows)
g.to_csv(ROOT / 'results/feh_gradients.csv', index=False)
print(g.round(4).to_string(index=False))

# %% [markdown]
# ### The figure

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
colours = {BRIGHT: '#2878b5', 'APOGEE': '#d75a20', 'RRab': '#3a923a'}  # one colour per sample
# Three equal panels (sizes in inches); the large bottom margin holds the rotated tick labels of (a) and (b).
fig, axes = panel_grid(1, 3, width=7, left=.55, right=.1, hgap=.55, bottom=1.05, top=.6)
for ax, tracer, samples in [(axes[0, 0], 'giants', GIANTS), (axes[0, 1], 'RRab', ['RRab'])]:
    L = lit[lit.tracer == tracer].reset_index(drop=True)  # the published gradients of this tracer; x position = row index
    for i, r in L.iterrows():
        # A quoted interval of slopes (slope to slope_hi) is drawn at its midpoint with half its range added to the error.
        lo, hi = sorted([r.slope, r.slope if pd.isna(r.get('slope_hi')) else r.slope_hi])
        e = 0. if r.err is None or pd.isna(r.err) else r.err
        ax.errorbar(i - .15, .5 * (lo + hi), [[.5 * (hi - lo) + e]] * 2, fmt='s' if e else 'x', ms=4, lw=1, color='k',
                    label='published' if i == 0 else None)
        # Our slopes over the same range, offset to the right of the published value, one per sample.
        for j, s in enumerate(samples):
            q = g[(g.label == r.label) & (g['sample'] == s)].iloc[0]
            ax.errorbar(i + .1 + .15 * j, q.slope, q.err, fmt='o', ms=4, lw=1, color=colours[s],
                        label=f"this work, {s.replace('<', ' < ')}" if i == 0 else None)
    ax.axhline(0, color='0.7', lw=.6, zorder=0)
    # Tick labels: study (without the Skowron+16 range names) and radial range; * marks ranges converted from degrees.
    ax.set_xticks(range(len(L)))
    ax.set_xticklabels([f"{r.label.replace(' inner', '').replace(' middle', '').replace(' outer', '')}\n"
                        f"{r.r_lo:g}–{r.r_hi:g} kpc{'*' if r.range_from_deg else ''}" for _, r in L.iterrows()],
                       rotation=60, ha='right', fontsize=6.5)
    ax.set(xlim=(-.6, len(L) - .4), ylabel=r'd[Fe/H]/dR [dex kpc$^{-1}$]')
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.08), ncol=1, frameon=False, fontsize=6.5)
axes[0, 0].set_title('(a) Giants', pad=46)
axes[0, 1].set_title('(b) RR Lyrae', pad=46)
# (c) Our median [Fe/H] against projected distance, with bootstrap errors.
ax = axes[0, 2]
for s in GIANTS:
    q = p[p['sample'] == s]
    ax.errorbar(.5 * (q.rho_lo + q.rho_hi), q['median'], [q['median'] - q.median_lo, q.median_hi - q['median']], fmt='-o', ms=3,
                lw=1, color=colours[s], label=f"this work, {s.replace('<', ' < ')} median")
# Published mean levels as horizontal segments over their quoted rho ranges; one legend entry per study.
for name, style in [('Carrera+08', dict(color='k', lw=2.2)), ('Massana+24', dict(color='0.55', lw=2.2, ls=(0, (3, 1))))]:
    for k, L in enumerate(x for x in LMC_FEH_LEVEL_LITERATURE if x['label'] == name):
        ax.plot([L['rho_lo'], L['rho_hi']], [L['feh']] * 2, **style, label=name + (' (XP [M/H])' if 'Massana' in name else ' (mean)')
                if k == 0 else None)
bq = p[p['sample'] == BRIGHT]  # our bright-BOSS profile against rho
for L in LMC_FEH_GRADIENT_DEG_LITERATURE:  # through our BOSS profile at the middle of the range: slope only
    rr = np.array([L['rho_lo'], L['rho_hi']])  # the published range [deg]
    # Our BOSS median at the middle of the range (linear interpolation between bin centres) anchors the published slope.
    y0 = np.interp(rr.mean(), .5 * (bq.rho_lo + bq.rho_hi), bq['median'])
    ax.plot(rr, y0 + L['slope'] * (rr - rr.mean()), color='#7b4fa0', lw=1.1, ls=(0, (4, 1.5)),
            label=f"{L['label']} ({L['slope']:g} dex/deg)")
# Axes, title and the legend above the panel.
ax.set(xlabel=r'$\rho$ [deg]', ylabel='[Fe/H]', xlim=(0, 12))
ax.set_title('(c) Against projected distance', pad=46)
ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.08), ncol=2, frameon=False, fontsize=6.5, columnspacing=.8, handlelength=1.6)
fig.savefig(ROOT / 'figures/paper_feh_literature.png', dpi=200)
