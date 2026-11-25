"""Paper figure (appendix, Sect. 2.2): BOSS coverage gaps and non-members in the parent.

Uncovered parent stars are those in selection groups with fewer than selection.NMIN_GOOD good spectra (data/boss_weights.parquet).
(a) Fraction of the parent with a good spectrum per HEALPix nside=64 pixel; red outlines: pixels with most of their parent in
uncovered groups. (b) Fraction of XP [M/H] below a threshold, uncovered
parent over covered parent reweighted to the same ring (0-3-6-12 deg) x G (edges 12, 15.5, 16.25, 16.75, 17.1, 17.5) mix, i.e. the
assumption of the coverage factor; band: 16th-84th percentiles of a bootstrap over uncovered groups. (c) Weighted fraction of good
spectra that are not LMC giants (Teff >= 5500 K or log g >= 3.5, or v_los outside 150-420 km/s), BOSS per ring x G and APOGEE per
field type. Also (json only): the fraction of APOGEE velocity members (good-spectrum giants, 150 < v < 420 km/s; APOGEE targeted
without astrometry) that fail the carton's astrometric cuts, by field type; and the sensitivity of the weighted BOSS member counts
(final weights x ring x G coverage factor) below [Fe/H] = -1.5, -2 and in total to the coverage threshold NMIN_GOOD = 2, 3, 5.
Writes figures/paper_selection_gaps.png and results/selection_gaps.json.

Inputs: data/boss_weights.parquet (release_boss_weights), data/apogee_weights.parquet (release_apogee_weights),
data/apogee_parent.parquet (release_apogee_parent; APOGEE candidates with nbr = bright-neighbour flag, astro_ok = passes the
carton's astrometric cuts, spectroscopic columns), data/spec_giants.parquet (scripts/build_spec_giants.py; BOSS and APOGEE
v [km/s], teff [K], logg), lmcold.mass_samples.samples() (BOSS members, for the coverage-threshold test).
Note: (c) uses the raw BOSS velocity v of data/spec_giants.parquet, not the calibrated v_los of the mass samples, and keeps the
first row per Gaia source_id.
Panels (1 x 3, square):
(a) x (east, increasing to the left), y (north) [deg], +-12.5, orthographic about the LMC centre; nside-64 pixels coloured by
    good spectra / parent (log, 1e-3 to 1); red outlines: pixels where more than half of the parent is in uncovered groups.
(b) x: XP [M/H] threshold, -2 to -0.5 in 0.125 steps; y: fraction below the threshold (log); orange: uncovered parent with
    16th-84th percentile band from 300 bootstraps over uncovered groups; blue: covered parent reweighted to the ring x G mix of
    the uncovered stars.
(c) x: G bin (labels <15.5, 15.5-16.25, 16.25-16.75, 16.75-17.1, 17.1-17.5); y: weighted per cent of good spectra that are not
    LMC giants, 0-7; BOSS per ring (0-3, 3-6, 6-12 deg); horizontal lines: APOGEE LMC1-17 (dashed) and LMC18-35 (dotted).
JSON keys: parent, uncovered, uncovered_share, uncovered_by_ring, xp_available, xp_gaps, boss_nonmember, apogee_nonmember,
sparse_groups, coverage_factor (parent / covered per ring x G bin and per ring), parent_xp_fraction_covered,
astrometric_loss_of_members, coverage_threshold (weighted member counts relative to NMIN_GOOD = 3).
Placement: appendix (Sect. 2.2; docs/figures.md). Run: .venv/bin/python scripts/paper_selection_gaps.py   (about 4 s on a laptop, docs/run_times.txt)
"""
import json

import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from lmcold import giants, selection
from lmcold.groups import bin_index
from lmcold.mass_samples import samples
from lmcold.constants import LMC
from lmcold.footprint import angular_distance, lmc_xy
from lmcold.layout import panel_grid
from lmcold.sources import DATA, ROOT

plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
BLUE, ORANGE = '#2878b5', '#d75a20'  # line colours: covered / BOSS, uncovered / outer ring
G_EDGES = [15.5, 16.25, 16.75, 17.1]  # G bin edges [mag] (inner edges of 12-15.5-16.25-16.75-17.1-17.5)
THRESHOLDS = np.arange(-2.0, -0.49, .125)  # XP [M/H] thresholds [dex] of panel (b): -2 to -0.5 in 0.125 steps
rng = np.random.default_rng(5)  # fixed seed of the bootstrap over uncovered groups

# %% [markdown]
# ### Load the data
# The selection weights of the two samples (`data/boss_weights.parquet`, `data/apogee_weights.parquet`), the APOGEE targeting
# candidates (`data/apogee_parent.parquet`) and the spectroscopic parameters (`data/spec_giants.parquet`), all from Zenodo
# bundle 1; and the BOSS members of the mass samples through `lmcold.mass_samples.samples()`.

# %%
# data/boss_weights.parquet: the BOSS parent with selection group, good/observed flags, weight w, G and XP [M/H]
b = pd.read_parquet(DATA / 'boss_weights.parquet')
# data/spec_giants.parquet: BOSS and APOGEE spectra with raw velocity v [km/s], teff [K], logg
v = pd.read_parquet(DATA / 'spec_giants.parquet')
# data/apogee_weights.parquet: the APOGEE candidates with field, quality flag and weight w
a = pd.read_parquet(DATA / 'apogee_weights.parquet')
# data/apogee_parent.parquet: APOGEE candidates with bright-neighbour flag nbr, astrometric-cut flag astro_ok, spectra; one row per
# star (observed first)
ap = pd.read_parquet(DATA / 'apogee_parent.parquet').sort_values(['apogee_id', 'observed'], ascending=[True, False]).drop_duplicates('apogee_id')
# lmcold.mass_samples.samples(): the BOSS members (calibrated [Fe/H]) for the coverage-threshold test
members, _, _ = samples()

# %% [markdown]
# ### The BOSS parent
# Covered and uncovered selection groups, rings and $G$ bins of the parent stars.

# %%
# Covered: the star's selection group holds at least NMIN_GOOD good spectra.
b['covered'] = b.group.map(b.groupby('group').good.sum()).to_numpy() >= selection.NMIN_GOOD
# Projected distance [deg] from the adopted LMC centre.
b['rho'] = angular_distance(b.ra.to_numpy(), b.dec.to_numpy(), LMC['ra0'], LMC['dec0'])
b['ring'] = np.digitize(b.rho, [3, 6])  # 0: < 3 deg, 1: 3-6 deg, 2: >= 6 deg
b['gb'] = np.digitize(b.g, G_EDGES)  # G bin 0-4
# Counts of the parent and of the uncovered stars, overall and per ring; share with an XP [M/H] among covered and uncovered.
out = dict(parent=len(b), uncovered=int((~b.covered).sum()), uncovered_share=float((~b.covered).mean()),
           uncovered_by_ring={int(k): int(v) for k, v in (~b.covered).groupby(b.ring).sum().items()},
           xp_available=dict(covered=float(np.isfinite(b.mh_xp[b.covered]).mean()), uncovered=float(np.isfinite(b.mh_xp[~b.covered]).mean())))

# %% [markdown]
# ### XP metallicities of the gaps
# Uncovered parent stars against covered ones reweighted to the same ring and $G$ mix, with a bootstrap over uncovered groups.

# %%
# (b) XP metallicities of the gaps against the covered stars of the same ring x G mix.
x = b[np.isfinite(b.mh_xp)]  # parent stars with an XP [M/H]
u, c = x[~x.covered], x[x.covered]  # uncovered and covered
mix = u.groupby(['ring', 'gb']).size() / len(u)
# Weight of each covered star: share of its ring x G cell among the uncovered stars over its share among the covered stars.
wc = (mix / (c.groupby(['ring', 'gb']).size() / len(c))).reindex(pd.MultiIndex.from_frame(c[['ring', 'gb']])).fillna(0).to_numpy()
frac = lambda mh, w: np.array([np.average(mh < t, weights=w) for t in THRESHOLDS])  # noqa: E731  fraction below each threshold
# Fractions below each threshold: covered (reweighted) and uncovered (unweighted).
fc, fu = frac(c.mh_xp.to_numpy(), wc), frac(u.mh_xp.to_numpy(), None)
groups = u.group.unique()
boot = []
# 300 bootstrap draws of the uncovered groups (with replacement); a group drawn n times enters with weight n.
for _ in range(300):
    n = pd.Series(rng.choice(groups, len(groups))).value_counts()  # times each uncovered group is drawn
    s = u[u.group.isin(n.index)]
    boot.append(frac(s.mh_xp.to_numpy(), s.group.map(n).to_numpy()))
lo, hi = np.percentile(np.array(boot), [16, 84], axis=0)  # band of panel (b)
out['xp_gaps'] = [dict(below=float(t), uncovered=float(a), uncovered_p16=float(l), uncovered_p84=float(h), covered_same_mix=float(z))
                  for t, a, l, h, z in zip(THRESHOLDS, fu, lo, hi, fc)]

# %% [markdown]
# ### Non-members, sparse groups, coverage factors and astrometric losses
# The numbers written to `results/selection_gaps.json`.

# %%
# (c) Non-members among the weighted good spectra.
# First BOSS row per star (raw velocity, Teff, log g), joined to the good spectra with a weight.
vb = v[v.survey == 'boss'][['source_id', 'v', 'teff', 'logg']].drop_duplicates('source_id')
g = b[b.good.astype(bool) & (b.w > 0)].merge(vb, on='source_id', how='left')
# LMC giant: the giant cuts of lmcold.giants.is_giant and the 150-420 km/s window of lmcold.mass_samples; NaN fails.
g['member'] = (g.teff < 5500) & (g.logg < 3.5) & (g.v > 150) & (g.v < 420)
# Weighted non-member fraction per ring x G bin (panel c), overall and per ring.
boss_nm = g.groupby(['ring', 'gb']).apply(lambda s: 1 - np.average(s.member, weights=s.w), include_groups=False)
out['boss_nonmember'] = dict(all=float(1 - np.average(g.member, weights=g.w)),
                             by_ring={int(k): float(1 - np.average(s.member, weights=s.w)) for k, s in g.groupby('ring')})
# APOGEE: the same per field type, for weighted good-quality spectra (velocity v from the weights table).
va = v[v.survey == 'apogee'][['source_id', 'teff', 'logg']].drop_duplicates('source_id')
a = a[a.quality.astype(bool) & (a.w > 0)].merge(va, on='source_id', how='left')
a['member'] = (a.teff < 5500) & (a.logg < 3.5) & (a.v > 150) & (a.v < 420)
a['type'] = np.where(a.field.str[3:].astype(int) <= 17, 'LMC1-17', 'LMC18-35')  # field number from 'LMCn'
out['apogee_nonmember'] = {k: dict(n=len(s), unweighted=float(1 - s.member.mean()), weighted=float(1 - np.average(s.member, weights=s.w)))
                           for k, s in a.groupby('type')}
# Sparsely sampled covered groups (the holes of the sky maps): good / parent < 1 per cent.
gs = b.groupby('group').agg(parent=('good', 'size'), good=('good', 'sum'))
sparse = gs[(gs.good >= selection.NMIN_GOOD) & (gs.good / gs.parent < 0.01)]
inside = b.group.isin(sparse.index)
out['sparse_groups'] = dict(groups=len(sparse), parent_share=float(sparse.parent.sum() / len(b)), good=int(sparse.good.sum()),
                            median_rho=float(b.rho[inside].median()), max_w=float(b.w.max()))
# Coverage factor: parent over covered parent per ring x G bin and per ring.
out['coverage_factor'] = {f'ring{r}': np.round(b[b.ring == r].groupby('gb').size() / b[(b.ring == r) & b.covered].groupby('gb').size(), 3).tolist()
                          for r in range(3)} | {'per_ring': np.round(b.groupby('ring').size() / b[b.covered].groupby('ring').size(), 3).tolist()}
# Fraction of the covered parent with XP [M/H] below -1, -1.5 and -2.
out['parent_xp_fraction_covered'] = {f'<{x}': float(np.mean(b.mh_xp[b.covered & np.isfinite(b.mh_xp)] < x)) for x in (-1.0, -1.5, -2.0)}

# Members lost to the astrometric cuts, from the fields APOGEE targeted without astrometry.
ap = ap[~ap.nbr & ap.observed & ap.snr.notna()]  # one row per star (observed first), no bright neighbour, with a spectrum
# Good spectrum at S/N >= 30, giant cuts, 150 < v < 420 km/s.
ap = ap[giants.apogee_spectrum_ok(ap, 30) & giants.is_giant(ap) & (ap.v > 150) & (ap.v < 420)]
ap['type'] = np.where(ap.field.str[3:].astype(int) <= 17, 'LMC1-17', 'LMC18-35')
out['astrometric_loss_of_members'] = {k: dict(n=len(t), fail=float(1 - t.astro_ok.mean())) for k, t in ap.groupby('type')}

# %% [markdown]
# ### Coverage threshold
# Weighted member counts with two, three or five good spectra required per covered group.

# %%
# Coverage threshold: members (lmcold.mass_samples) weighted with NMIN_GOOD = 2, 3, 5.
pos = pd.Index(b.source_id).get_indexer(members.source_id)  # row of each member in the parent table
# Selection-group index, good and observed flags, and pass groups, as in scripts/boss_selection.py.
s_, good, obs = pd.factorize(b.group)[0], b.good.to_numpy(), b.observed.to_numpy()
pg = selection.boss_pass_groups(b.g, b.mh_xp, b.ra, b.dec)
# Coverage-factor cell: ring x G bin (12-15.5-16.25-16.75-17.1-17.5).
key = b.ring.to_numpy() * 10 + bin_index(b.g, [12.0, 15.5, 16.25, 16.75, 17.1, 17.5])
ones, sens = np.ones(len(b)), {}
for nmin in (2, 3, 5):
    selection.NMIN_GOOD = nmin  # module constant read at call time
    # Final weight (group weight x pass factor), times the ring x G coverage factor, at the members.
    w = selection.group_weights(s_, good, ones) * selection.pass_factor(s_, obs, good, pg, ones)
    w = (w * selection.coverage_factor([key], np.bincount(s_, good)[s_] >= nmin, 1, ones))[pos]
    # Weighted member counts below [Fe/H] = -1.5, below -2, and in total.
    sens[nmin] = [float(w[members.feh.to_numpy() < x].sum()) for x in (-1.5, -2.0)] + [float(w.sum())]
selection.NMIN_GOOD = 3  # restore the adopted value
# Counts relative to those with NMIN_GOOD = 3; then write and print the JSON.
out['coverage_threshold'] = {f'nmin={k}': dict(zip(['<-1.5', '<-2', 'all'], np.round(np.array(v) / np.array(sens[3]), 4).tolist())) for k, v in sens.items()}
json.dump(out, open(ROOT / 'results/selection_gaps.json', 'w'), indent=1)
print(json.dumps(out, indent=1))

# %% [markdown]
# ### The figure
# (a) good spectra over parent on the sky; (b) XP metallicities of the gaps; (c) non-members.

# %%
# Three equal panels in one row (sizes in inches).
fig, axes = panel_grid(1, 3, width=7, left=.5, right=.1, hgap=.62, bottom=.42, top=.62)
# (a) Per nside-64 pixel (ring order): fraction of the parent with a good spectrum and fraction in uncovered groups.
pix = hp.ang2pix(64, b.ra.to_numpy(), b.dec.to_numpy(), lonlat=True)
per = pd.DataFrame(dict(pix=pix, good=b.good.to_numpy(), unc=~b.covered.to_numpy())).groupby('pix').agg(good=('good', 'mean'), unc=('unc', 'mean'))
# Pixel centres [deg], projected about the LMC centre (orthographic x east, y north [deg]).
ra, dec = hp.pix2ang(64, per.index.to_numpy(), lonlat=True)
px, py = lmc_xy(ra, dec)
sc = axes[0, 0].scatter(px, py, c=np.clip(per.good, 1e-3, 1), s=7.5, marker='s', cmap='viridis', norm=LogNorm(1e-3, 1), lw=0, rasterized=True)
u = per.unc.to_numpy() > 0.5  # pixels with more than half of their parent in uncovered groups: red outlines
axes[0, 0].scatter(px[u], py[u], s=7.5, marker='s', facecolor='none', edgecolor='r', lw=.5)
# East to the left; colour bar above the panel.
axes[0, 0].set(xlim=(12.5, -12.5), ylim=(-12.5, 12.5), xlabel='$x$ (east) [deg]', ylabel='$y$ (north) [deg]')
axes[0, 0].set_title('(a) good spectra / parent (red: uncovered)', pad=22)
fig.colorbar(sc, cax=axes[0, 0].inset_axes([0, 1.03, 1, .04]), orientation='horizontal').ax.xaxis.set_ticks_position('top')
# (b) Fraction below each XP [M/H] threshold: uncovered with its bootstrap band, covered reweighted to the same mix.
ax = axes[0, 1]
ax.fill_between(THRESHOLDS, lo, hi, color=ORANGE, alpha=.25, lw=0)
ax.plot(THRESHOLDS, fu, '-o', ms=2.5, color=ORANGE, label='uncovered groups')
ax.plot(THRESHOLDS, fc, '-', color=BLUE, label='covered, same ring and $G$ mix')
ax.set(yscale='log', xlabel=r'[M/H]$_{\rm XP}$ threshold', ylabel=r'fraction below threshold', xlim=(-2.05, -.45))
ax.set_title('(b) metallicity of the gaps', pad=22)
ax.legend(loc='lower right')
# (c) Weighted per cent of non-members per G bin, one line per ring (offset slightly in x); APOGEE as horizontal lines.
ax = axes[0, 2]
labels = ['$<$15.5', '15.5–16.25', '16.25–16.75', '16.75–17.1', '17.1–17.5']
for r, (name, colour) in enumerate(zip(['0–3°', '3–6°', '6–12°'], ['0.1', BLUE, ORANGE])):
    y = [boss_nm.get((r, k), np.nan) for k in range(5)]
    ax.plot(np.arange(5) + (r - 1) * .08, 100 * np.array(y), '-o', ms=3, color=colour, label=f'BOSS {name}')
for k, (t, s) in enumerate(out['apogee_nonmember'].items()):
    ax.axhline(100 * s['weighted'], color='0.5', ls=['--', ':'][k], lw=.8, label=f'APOGEE {t}')
ax.set(xticks=range(5), ylabel='non-members among good spectra (%)', xlabel='$G$', ylim=(0, 7))
ax.set_xticklabels(labels, rotation=30, fontsize=6)
ax.set_title('(c) non-members', pad=22)
ax.legend(loc='upper left', fontsize=6)
fig.savefig(ROOT / 'figures/paper_selection_gaps.png')
print('Saved paper_selection_gaps.png')
