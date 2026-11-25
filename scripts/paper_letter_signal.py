"""Letter Fig. 1 (prototype): the rotation of the LMC's populations seen in the data, with only the bulk motion removed.

Each star's internal velocity is its proper motion and line-of-sight velocity minus those of the LMC's frame with no internal
motion: the centre, systemic proper motion and velocity, inclination and line of nodes of the joint Level-2 ellipsoid fit
(results/kin_mcmc_data_ellipsoid.csv, stage 2, shared parameters, written by scripts/fit_kinematics_mcmc.py data ellipsoid, KB
chapter 6; v0 = 0, which also removes the streaming, proportional to v0).
This removes the bulk motion and its perspective, nothing else (the 'data' rows of scripts/paper_kin_residuals.py). The proper-
motion part is projected on the sky-polar basis about the fitted centre (lmcold.kinfit.residual_polar, each star at its disc
distance): v_t, 90 deg east of the direction away from the centre (negative: clockwise on the sky).
Stars: the RR Lyrae members (lmcold.samples.rrab, proper motions only) and the giants in the five [Fe/H] bins of
lmcold.samples.giants, at 2 < rho < 9 deg from the vdMK14 centre (fit_kinematics_mcmc.ring); stars more than CLIP km/s (plus three
times their own proper-motion error) from their population's Level-2 model are dropped, as in paper_kin_residuals.py.
Reference disc model: the metal-poor giants' ([Fe/H] < -1.5) Level-2 disc (the shared frame and their rotation curve),
noiseless, evaluated on a grid (every 1 deg in position angle, radii 3.25-6.75 deg every 0.5 deg about the fitted centre),
put through the same steps and averaged over the radii at each position angle: the shape that the disc geometry alone gives.

Panels 1 x 3, titles above the panels as Fig. sims_sidm, legends inside the panels (the giant bins' colours are keyed by the shading of (b)):
(a) the stars on the sky, tangent-plane coordinates about the fitted centre (east left): the giants of the five bins by survey
    (merged as lmcold.samples.merge_giants: a star in both keeps its APOGEE row), 0.25-deg pixels holding at least one
    giant of BOSS or APOGEE in the colours SURVEY at 70 per cent opacity (APOGEE drawn on top); [Fe/H] < -1.5 giants (both surveys) as dots, RR Lyrae density contours (0.5-deg pixels smoothed by one pixel,
    10 and 40 per cent of the peak; none within RING[0] plus one smoothing length of the vdMK14 centre, where the selection leaves
    no stars); dashed circles PROFILE_RING, dotted the line of nodes;
(b) formed stellar mass against [Fe/H], corrected for selection, coverage, giants per formed mass (PARSEC isochrones, Kroupa
    IMF), BOSS [Fe/H] response and foreground (results/giant_present_mass.csv, scripts/build_giant_mass.py; from the BOSS counts,
    split above -1 by the APOGEE [Fe/H]; bars and band 16th-84th percentiles of the bootstrap). Left axis: cumulative mass below
    each bin's upper edge (thick line with band), with the total at [Fe/H] = 0, and the published totals
    (lmcold.constants.LMC_MASS_LITERATURE: open triangles formed, squares present mass, in order of year, as
    scripts/paper_mass_summary.py) to the right. Right axis (unlabelled; the legend names it): fraction of the total per dex in each 0.25-dex bin (the first bin,
    [Fe/H] < -2, closed at -2.5), corrected (markers with bars) and of the observed giants of panel (a), uncorrected (dotted);
    the five kinematic bins shaded in their colours;
(c) against position angle (north through east) at PROFILE_RING, median per SECTOR-deg bin with 1.2533 sd / sqrt(n) errors (bins
    of >= NMIN stars), one velocity axis: v_t and v_r of every population as a band (median +- error; RR Lyrae and
    the five giant bins), v_los of the five giant bins likewise, and the reference disc model (black: v_t solid, v_r dotted,
    v_los dashed). v_r, the sky-plane component away from the centre, is out of phase with v_los, so the two do not overlap.
Writes figures/paper_letter_signal.png and results/letter_signal_counts.csv (the sample sizes of (a): BOSS and APOGEE giants, [Fe/H] < -1.5 giants, RR Lyrae; for scripts/paper_text_letter.py).
Run: .venv/bin/python scripts/manifest.py run paper_letter_signal.py
"""
# %% [markdown]
# ### Settings
# The cuts and plotting choices.

# %%
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter

from fit_kinematics_mcmc import RING, ring
from lmcold import geometry, kinfit, samples
from lmcold.constants import KAPPA, LMC, LMC_MASS_LITERATURE
from lmcold.layout import panel_grid
from lmcold.sources import ROOT

# Columns passed to lmcold.kinfit.data: position [deg], proper motion and errors [mas/yr], correlation, v_los and error [km/s].
COLS = ('ra', 'dec', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error', 'pmra_pmdec_corr', 'vlos', 'evlos')
CLIP = 150.0                         # km/s, as paper_kin_residuals.py
CLIP_ERR = 3                         # proper-motion components: CLIP plus CLIP_ERR times the star's own error
MEDIAN_SE = 1.2533                   # error of a median: MEDIAN_SE sd / sqrt(n), sqrt(pi / 2) for a Gaussian
PIX, PIX_ALPHA = 0.25, 0.7           # deg, opacity: survey pixels of (a)
RR_PIX, RR_SMOOTH, RR_LEVELS = 0.5, 1.0, (0.1, 0.4)   # deg, pixels, fractions of the peak: RR Lyrae contours of (a)
REF_DPA, REF_RHO = 1.0, (3.25, 6.75, 0.5)   # deg: reference disc grid in position angle, and radii (first, last, step)
SECTOR = 30.0                        # deg, position-angle bins
NMIN = 8                             # minimum stars in a position-angle bin of (c)
PROFILE_RING = (3.0, 7.0)            # deg, inner and outer radius of the ring of (c) about the fitted centre
COLOURS = ['#e66101', '#1b1b6b', '#3a5fcd', '#1a9e9e', '#7f7f7f', '#b0b0b0']   # RR Lyrae orange; giants indigo, royal blue, teal (metal-poor), greys (metal-rich)
SURVEY = dict(BOSS='#e3e3e3', APOGEE='#a9b9c9')   # panel (a): pixels with giants of each survey (pale grey, pale slate)

# %% [markdown]
# ### Load the data
# Everything the figure uses is read here; the rest of the script only computes and draws. The stars are the RR Lyrae and giant
# catalogues of Zenodo bundle 1, read through `lmcold.samples.rrab()` (RRab members with a Gaia proper motion) and
# `lmcold.samples.giants()` (APOGEE and BOSS merged, a star in both keeping its APOGEE row, split into the five [Fe/H] bins); both are
# cut to 2-9 deg from the LMC centre in the next cells. The fitted disc frame of each population is the joint ellipsoid fit
# `results/kin_mcmc_data_ellipsoid.csv`, and the stellar mass formed per [Fe/H] bin is `results/giant_present_mass.csv`; the
# published LMC mass totals of panel (b) are constants of `lmcold.constants`, not a file.

# %%
FIT = pd.read_csv(ROOT / 'results/kin_mcmc_data_ellipsoid.csv')   # joint ellipsoid fit: posterior medians per stage, population, parameter
rr = samples.rrab()                  # data/rrab.parquet: RRab members (positions, proper motions and their errors)
giants = samples.giants()            # data/giants_apogee.parquet + data/giants_boss.parquet: (label, catalogue) per [Fe/H] bin
gm = pd.read_csv(ROOT / 'results/giant_present_mass.csv').query('formed_mass > 0')   # formed mass per [Fe/H] bin, with bootstrap percentiles

# %% [markdown]
# ### Helpers
# Four helpers: `params` takes the fitted disc frame of a population from the fit table, `internal` removes the LMC's bulk motion from
# each star and splits its velocity into $v_t$, $v_r$ and $v_{\rm los}$, `mock` does the same for the noiseless disc model, and
# `profile` takes the median of a velocity component in bins of position angle.

# %%
def params(pop):
    """Stage-2 posterior medians of the joint ellipsoid fit (FIT): shared parameters plus population pop's own, as a kinfit.PARAMS vector."""
    p = kinfit.start()   # default vector; every fitted entry is overwritten below
    for _, r in FIT[(FIT.stage == 2) & FIT.population.isin(['shared', pop])].iterrows():
        p[kinfit.PARAMS.index(r.param)] = r['median']
    return p


def internal(name, c):
    """Position angle [deg] and distance rho [deg] about the fitted centre, v_t, v_r and v_los [km/s] of catalogue c with the frame
    removed; clipped stars dropped."""
    l2 = params(name)   # the population's full Level-2 model
    # The frame alone: the same vector with no rotation (v0 = 0).
    frame = l2.copy()
    frame[kinfit.PARAMS.index('v0')] = 0.0
    d = kinfit.data(*(c[k] for k in COLS))
    out = []
    # Residuals [km/s] from the frame (plotted) and from the full model (for the clipping): v_t, v_r on the sky, v_los.
    for p in (frame, l2):
        pa, vt, vr = kinfit.residual_polar(p, c)
        vl = np.asarray(c['vlos']) - np.asarray(kinfit.model(p, d['rhat'], d['ea'], d['ed'])[2])
        out.append((np.asarray(pa), np.asarray(vt), np.asarray(vr), vl))
    (pa, vt, vr, vl), (_, rt, rr, rl) = out
    D = np.asarray(kinfit.model(l2, d['rhat'], d['ea'], d['ed'])[3])   # kpc, disc distance of each star
    # Proper-motion error as a velocity [km/s]: KAPPA [km/s per (mas/yr kpc)] x D x quadrature sum of the two errors.
    epm = KAPPA * D * np.hypot(c['pmra_error'], c['pmdec_error'])
    # Keep stars within CLIP (+ CLIP_ERR x error) of the full model in v_t and v_r, and within CLIP in v_los (NaN v_los passes).
    ok = (np.abs(rt) < CLIP + CLIP_ERR * epm) & (np.abs(rr) < CLIP + CLIP_ERR * epm) & ~(np.abs(np.nan_to_num(rl)) > CLIP)
    # Fitted centre [deg] from the offsets dx, dy, and tangent-plane x, y [deg] of each star about it (rho = hypot(x, y)).
    cra, cdec = (float(v) for v in kinfit.centre(*l2[:2]))
    x, y = (np.rad2deg(np.asarray(v)) for v in geometry.tangent_plane_coords(np.deg2rad(c['ra']), np.deg2rad(c['dec']), cra, cdec))
    print(f'{name}: {ok.sum()} stars ({(~ok).sum()} clipped)', flush=True)
    return dict(pa=pa[ok], rho=np.hypot(x, y)[ok], v_t=vt[ok], v_r=vr[ok], v_los=vl[ok])


def mock(name, c):
    """internal() of the noiseless Level-2 disc of population name at the stars of c (line of sight only where c has one)."""
    d = kinfit.data(*(c[k] for k in COLS))
    # Model proper motions [mas/yr] and v_los [km/s] at the stars, used in place of the measured ones.
    mra, mde, mvl, _ = (np.asarray(v) for v in kinfit.model(params(name), d['rhat'], d['ea'], d['ed']))
    return internal(name, c | dict(pmra=mra, pmdec=mde, vlos=np.where(np.isfinite(c['vlos']), mvl, np.nan)))


def profile(s, comp):
    """Median and error of comp per SECTOR-deg position-angle bin at PROFILE_RING."""
    pa0 = np.arange(0, 360, SECTOR)   # deg, lower edges of the position-angle bins
    m, e = np.full(len(pa0), np.nan), np.full(len(pa0), np.nan)
    for j, a in enumerate(pa0):
        # Stars in the ring and in this bin with a finite value.
        k = (s['rho'] > PROFILE_RING[0]) & (s['rho'] < PROFILE_RING[1]) & (s['pa'] >= a) & (s['pa'] < a + SECTOR) & np.isfinite(s[comp])
        if k.sum() >= NMIN:
            m[j], e[j] = np.median(s[comp][k]), MEDIAN_SE * np.std(s[comp][k]) / np.sqrt(k.sum())
    return pa0 + SECTOR / 2, m, e


# %% [markdown]
# ### The stars and the reference disc
# The RR Lyrae (proper motions only) and the giants in five [Fe/H] bins, all at 2 < $\rho$ < 9 deg from the LMC centre. The reference
# disc is the fitted model of the [Fe/H] < -1.5 giants, evaluated without noise on a grid of sky positions.

# %%
# (name, catalogue) of the six populations; the RR Lyrae get NaN v_los and evlos (proper motions only).
POPS = [('RR Lyrae', ring(rr | dict(vlos=np.full(len(rr['ra']), np.nan), evlos=np.full(len(rr['ra']), np.nan))))]
POPS += [(f'giants {k}', ring(c)) for k, c in giants]
theta = params('RR Lyrae')[kinfit.PARAMS.index('theta')]   # shared by all populations
# Reference disc on a fine grid: points every 1 deg in position angle at radii 3.25-6.75 deg about the fitted centre.
P0 = params('giants [Fe/H] < -1.5')
# Unit vector to the fitted centre and the east and north unit vectors there.
r0, eE0, eN0 = (np.asarray(v) for v in geometry.sky_basis(*kinfit.centre(*P0[:2])))
gpa, grho = np.meshgrid(np.deg2rad(np.arange(REF_DPA / 2, 360, REF_DPA)), np.deg2rad(np.arange(REF_RHO[0], REF_RHO[1] + REF_RHO[2] / 2, REF_RHO[2])))
# (N, 3) unit vectors of the grid points: rho from the centre along position angle pa (north through east).
u = np.cos(grho.ravel())[:, None] * r0 + np.sin(grho.ravel())[:, None] * (np.sin(gpa.ravel())[:, None] * eE0 + np.cos(gpa.ravel())[:, None] * eN0)
zero = np.zeros(len(u))
# Grid catalogue: positions from u; motions and errors are dummies (replaced by the model in mock).
grid = dict(ra=np.rad2deg(np.arctan2(u[:, 1], u[:, 0])) % 360, dec=np.rad2deg(np.arcsin(u[:, 2])), pmra=zero, pmdec=zero,
            pmra_error=zero, pmdec_error=zero, pmra_pmdec_corr=zero, vlos=zero, evlos=zero)
ref = mock('giants [Fe/H] < -1.5', grid)   # the reference disc's v_t, v_r, v_los on the grid
ref_pa = np.arange(REF_DPA / 2, 360, REF_DPA)   # deg, centres of the 1-deg position-angle bins
# 1-deg bin of each grid point's position angle, for the average over radii.
ref_bin = np.clip(np.floor(ref['pa']).astype(int), 0, 359)


# %% [markdown]
# ### (a) The tracers on the sky
# Sky pixels holding BOSS or APOGEE giants, the [Fe/H] < -1.5 giants as dots, and density contours of the RR Lyrae, about the fitted
# centre.

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
fig, ax = panel_grid(1, 3, width=7.0, left=0.5, right=0.1, hgap=0.75, bottom=0.42, top=0.3, aspect=1.0)

# (a) Where the stars are: tangent-plane coordinates [deg] about the fitted centre (east to the left).
cra, cdec = (float(v) for v in kinfit.centre(*P0[:2]))


def sky(c):
    """Tangent-plane coordinates x (east), y (north) [deg] of catalogue c about the fitted centre (cra, cdec)."""
    x, y = (np.rad2deg(np.asarray(v)) for v in geometry.tangent_plane_coords(np.deg2rad(c['ra']), np.deg2rad(c['dec']), cra, cdec))
    return x, y


a = ax[0, 0]
# All giants of the five bins in one catalogue (positions and survey).
G = {k: np.concatenate([c[k] for name, c in POPS[1:]]) for k in ('ra', 'dec', 'survey')}
gx, gy = sky(G)
edges = np.arange(-9.5, 9.51, PIX)   # deg, pixel edges in x and y
boss = G['survey'] == 'boss'
for sel, col in ((boss, SURVEY['BOSS']), (~boss, SURVEY['APOGEE'])):   # pixels with at least one giant; APOGEE on top
    h = np.histogram2d(gx[sel], gy[sel], [edges, edges])[0].T
    a.pcolormesh(edges, edges, np.where(h > 0, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap([col]), alpha=PIX_ALPHA, lw=0,
                 rasterized=True)
# [Fe/H] < -1.5 giants (POPS[1]) as dots.
mx, my = sky(POPS[1][1])
a.plot(mx, my, '.', color=COLOURS[1], ms=2.0, mew=0)
# RR Lyrae (POPS[0]): counts in RR_PIX-deg pixels, smoothed, contoured at RR_LEVELS of the peak.
rx, ry = sky(POPS[0][1])
redges = np.arange(-9.5, 9.51, RR_PIX)
hr = gaussian_filter(np.histogram2d(rx, ry, [redges, redges])[0].T, RR_SMOOTH)   # smoothed by RR_SMOOTH pixels
rc = 0.5 * (redges[1:] + redges[:-1])   # deg, pixel centres
vx, vy = sky(dict(ra=np.array([LMC['ra0']]), dec=np.array([LMC['dec0']])))   # the vdMK14 centre, origin of the RING cut
# No contours inside the inner edge of the selection (plus one smoothing length): the empty disc there is the cut, not the LMC.
hr = np.ma.masked_where(np.hypot(rc[None, :] - vx[0], rc[:, None] - vy[0]) < RING[0] + RR_SMOOTH * RR_PIX, hr)
a.contour(rc, rc, hr, levels=hr.max() * np.array(RR_LEVELS),
          colors=COLOURS[0], linewidths=0.8)
# Dashed circles at the radii of PROFILE_RING, and the line of nodes at position angle theta (dotted).
t = np.linspace(0, 2 * np.pi, 200)
for r in PROFILE_RING:
    a.plot(r * np.sin(t), r * np.cos(t), 'k--', lw=0.5)
L = np.array([-9.5, 9.5])
a.plot(L * np.sin(np.deg2rad(theta)), L * np.cos(np.deg2rad(theta)), color='0.3', lw=0.6, ls=':')
# Legend inside the panel, lower left, with the number of giants per survey.
a.legend([Patch(color=SURVEY['BOSS']), Patch(color=SURVEY['APOGEE']), Line2D([], [], ls='', marker='.', color=COLOURS[1], ms=4),
          Line2D([], [], color=COLOURS[0], lw=0.8)],
         [f'BOSS giants ({boss.sum()})', f'APOGEE giants ({(~boss).sum()})', r'[Fe/H] < -1.5 giants', 'RR Lyrae'],
         fontsize=4.5, frameon=False, loc='lower left', ncol=2, borderaxespad=0.3, handlelength=1.2, columnspacing=0.8)
a.set(xlim=(10.75, -10.75), ylim=(-11.5, 10), xticks=np.arange(-10, 11, 5), yticks=np.arange(-10, 11, 5),   # room for the legend below the stars
       xlabel=r'$x$ (east) [deg]', ylabel=r'$y$ (north) [deg]')
a.set_title('(a) tracers on the sky', loc='left', fontsize=8)   # as Fig. sims_sidm

# %% [markdown]
# ### (b) The stellar mass formed against [Fe/H]
# From `results/giant_present_mass.csv` (read above): the giants counted per [Fe/H] bin and corrected for selection, coverage, foreground and the
# number of giants per unit formed mass; cumulative (left axis) and per dex (right axis).

# %%
# (b) Formed stellar mass against [Fe/H] (results/giant_present_mass.csv; bars and band 16th-84th bootstrap percentiles). Left axis:
# cumulative mass below each bin's upper edge (thick, band), ending at the total at [Fe/H] = 0, and the published totals
# (lmcold.constants.LMC_MASS_LITERATURE, as paper_mass_summary.py; open triangles formed, squares present mass) to the right in order
# of year. Right axis (unlabelled; the legend names it): fraction of the total per dex in each 0.25-dex bin (the open first bin closed at -2.5, so 0.5 dex wide, as
# paper_mass.py), corrected (markers, bars) and of the observed giants of panel (a) (APOGEE + BOSS, uncorrected; dotted).
# Kinematic bins shaded.
a = ax[0, 1]
# Cumulative formed mass [Msun] at each bin's upper edge, its band, and the total marked at [Fe/H] = 0.
a.fill_between(gm.feh_hi, gm.cumulative_p16, gm.cumulative_p84, color='k', alpha=0.15, lw=0)
a.plot(gm.feh_hi, gm.cumulative, 'k-', lw=1.4, label='cumulative mass (left)')
a.plot(0.0, gm.cumulative.iloc[-1], 'ko', ms=3.5)
LIT = sorted(LMC_MASS_LITERATURE, key=lambda r: r['label'][-5:-1])   # by the year in the label
# Published totals at x = 0.1, 0.16, ...: triangles formed, squares present mass.
for k, r in enumerate(LIT):
    a.plot(0.1 + 0.06 * k, r['mass'], '^' if r['kind'] == 'formed' else 's', color='k', mfc='white', ms=3.5, mew=0.7)
a.text(0.27, 3.6e9, 'published\ntotals', ha='center', va='bottom', fontsize=4.5)
# The five kinematic bins shaded in their colours, clipped to the axis range.
for (lo, hi), col in zip(samples.FEH_BINS, COLOURS[1:]):
    a.axvspan(max(lo, -2.5), min(hi, 0.0), color=col, alpha=0.2, lw=0)
a.set(yscale='log', xlim=(-2.5, 0.55), ylim=(1.5e6, 1.2e10), xlabel='[Fe/H]', ylabel=r'cumulative formed mass [M$_\odot$]')
b = a.twinx()   # right axis: fraction per dex
width = gm.feh_hi.clip(upper=0.0) - gm.feh_lo.clip(lower=-2.5)
tot = gm.formed_mass.sum()   # Msun, total formed mass
# Corrected fraction per dex, with the 16th-84th percentile bars.
b.errorbar(gm.feh, gm.formed_mass / tot / width, yerr=[(gm.formed_mass - gm.mass_p16) / tot / width, (gm.mass_p84 - gm.formed_mass) / tot / width],
           fmt='o-', color='0.25', ms=2.5, lw=0.6, elinewidth=0.6, label='per dex, corrected (right)')
# Observed giants of (a): counts per mass bin, as a fraction per dex.
feh = np.concatenate([c['feh'] for name, c in POPS[1:]])
n = np.array([((feh >= lo) & (feh < hi)).sum() for lo, hi in zip(gm.feh_lo, gm.feh_hi)])
b.plot(gm.feh, n / n.sum() / width, 's:', color='0.5', ms=2.5, lw=0.8, mfc='none', label='per dex, observed (right)')
b.set(yscale='log', ylim=(2e-3, 2e2))
# One legend for both axes (right-axis entries reversed).
h, l = a.get_legend_handles_labels()
h2, l2 = b.get_legend_handles_labels()
a.legend(h + h2[::-1], l + l2[::-1], fontsize=4.5, frameon=False, loc='upper left', bbox_to_anchor=(0, 0.9), handlelength=1.8)   # empty there
a.set_title('(b) stellar mass formed against [Fe/H]', loc='left', fontsize=8)

# %% [markdown]
# ### (c) The velocities against position angle
# For every population, the medians of $v_t$, $v_r$ and $v_{\rm los}$ in 30-deg sectors at 3-7 deg from the fitted centre, with the
# reference disc in black. The figure is saved here, with the sample sizes of (a) for the Letter's text.

# %%
# (c) Velocities against position angle.
a = ax[0, 2]
a.set_title('(c) rotation signatures', loc='left', fontsize=8)
# Reference disc: mean over radii of each component in each 1-deg position-angle bin (black lines).
for comp, ls, lab in (('v_t', '-', r'disc model, $v_t$'), ('v_r', ':', r'disc model, $v_r$'), ('v_los', '--', r'disc model, $v_{\rm los}$')):
    a.plot(ref_pa, [ref[comp][ref_bin == i].mean() for i in range(360)], 'k' + ls, lw=1.0, label=lab, zorder=3)
# Each population: band of median +- error and the median line per sector, for every component it has.
for (name, c), col in zip(POPS, COLOURS):
    s = internal(name, c)
    for comp in ('v_t', 'v_r', 'v_los'):
        pa, m, e = profile(s, comp)
        if np.isfinite(m).any():   # skips v_los of the RR Lyrae
            rrl = name == 'RR Lyrae'   # on top of the giants; the giants paler, so the disc model stays visible
            a.fill_between(pa, m - e, m + e, color=col, alpha=0.4 if rrl else 0.15, lw=0, zorder=2.5 if rrl else 1)
            a.plot(pa, m, '-', color=col, lw=1.1 if rrl else 0.8, alpha=1 if rrl else 0.8, zorder=2.6 if rrl else 1.1)
            # colours keyed in (a) (RR Lyrae) and by the shading of (b) (giant bins)
a.axhline(0, color='0.5', lw=0.5, ls=':')
# Component labels, placed by hand in empty regions.
a.text(265, 38, r'$v_{\rm los}$', ha='center', va='center', fontsize=10)   # above the v_los maximum (< 33)
a.text(150, 27, r'$v_r$', ha='center', va='center', fontsize=10)   # empty: v_r < 22 and v_los < -10 there
a.text(255, -36, r'$v_t$', ha='center', va='center', fontsize=10)   # empty: between the two groups
a.set(xlim=(0, 360), xticks=np.arange(0, 361, 90), ylim=(-97, 45), xlabel='position angle [deg]', ylabel=r'velocity [km s$^{-1}$]')
a.legend(*(v[:3] for v in a.get_legend_handles_labels()), fontsize=4.5, frameon=False, loc='lower center', ncol=3,
         handlelength=1.3, columnspacing=0.6, handletextpad=0.4)   # disc model only; empty there: every v_t > -86
fig.savefig(ROOT / 'figures/paper_letter_signal.png', dpi=250)
# Sample sizes: BOSS and APOGEE giants, [Fe/H] < -1.5 giants, RR Lyrae (after the ring cut).
pd.DataFrame([dict(boss=int(boss.sum()), apogee=int((~boss).sum()), mp_giants=len(POPS[1][1]['ra']), rrl=len(POPS[0][1]['ra']))]).to_csv(
    ROOT / 'results/letter_signal_counts.csv', index=False)   # the sample sizes of (a), for the text (scripts/paper_text_letter.py)
print('wrote figures/paper_letter_signal.png')
