"""Paper figure (appendix): BOSSNet over-estimates [Fe/H] of hot, low-S/N giants, which the calibration sample barely contains.

(a) Stars with both spectra (lmcold.mass_samples.overlap), APOGEE [Fe/H] -1 to -0.5: calibrated BOSS minus APOGEE [Fe/H] against
    BOSS Teff, for S/N < 25 and >= 25; medians in Teff bins with >= 8 stars.
(b) Weighted fraction of giants with [Fe/H] > -0.5 against dereddened BP-RP (lmcold.mass_samples.with_colour) at 16.25 <= G <= 17.5
    and 6 <= rho <= 12 deg (where APOGEE has faint giants; the metal-rich fraction falls with radius): BOSS (weights w x coverage x
    (1 - f_MW)) at S/N < 25 and >= 25, APOGEE (weights w).
(c) BOSS Teff distribution of the BOSS mass sample (weighted) and of the stars with both spectra, for S/N < 25 and >= 25.
Writes results/boss_hot_bias.json and figures/paper_boss_hot_bias.png.

Why: this bias is the reason the BOSS mass above [Fe/H] = -1 is shared among the reporting bins as the APOGEE giants of the same
ring x G x colour cell (lmcold.mass_samples.metal_rich_split, used by scripts/build_giant_mass.py).

Inputs (all derived; not in data/sources.yaml as files of their own):
- lmcold.mass_samples.samples(): BOSS and APOGEE mass samples (velocity-selected members with selection weights w) from
  data/boss_weights.parquet, data/apogee_weights.parquet, data/spec_giants.parquet and results/boss_calibration.json; BOSS
  [Fe/H] and v_los are on the APOGEE scale.
- data/giant_foreground.parquet (scripts/build_giant_mass.py): per BOSS giant the coverage fill factor `fill` (parent /
  covered parent, lmcold.selection.coverage_factor) and the Milky Way foreground fraction `f_mw` (lmcold.foreground).
- data/boss_parent.parquet (via with_colour): Gaia G, BP, RP and SFD E(B-V) for the dereddened colour c0 and rho.
Note: with_colour takes photometry from the BOSS parent, so APOGEE giants outside the BOSS parent drop out of (b).

Panels (1 x 3; legends above each panel):
(a) x: BOSS Teff [K], 3700-5300; y: calibrated BOSS - APOGEE [Fe/H] [dex], -0.5 to 0.6; points: overlap stars with APOGEE
    [Fe/H] in [-1, -0.5); lines: median per 200-K bin (>= 8 stars), orange S/N < 25, blue S/N >= 25.
(b) x: dereddened Gaia BP-RP [mag], 0.9-2.2; y: weighted fraction with [Fe/H] > -0.5, 0-0.85; BOSS S/N < 25, BOSS S/N >= 25,
    APOGEE; bins with >= 20 stars.
(c) x: BOSS Teff [K]; y: fraction per 200-K bin. Solid: BOSS mass sample weighted by W, normalised to the total weight of the
    sample; dashed: overlap stars, normalised to the number of overlap stars (so each pair of S/N curves sums to 1).
Output JSON results/boss_hot_bias.json: the bin edges, the curves of (a) and (b) (offset_*, n_overlap_*, frac_*, n_*), the
weight fraction of the BOSS sample that is hot (Teff > 4300 K) and low S/N, the same fraction among the overlap stars, and the
weighted fractions above [Fe/H] = -0.5 for BOSS (hot low-S/N, rest, all) and APOGEE.
Run: .venv/bin/python scripts/paper_boss_hot_bias.py   (about 2 s on a laptop, docs/run_times.txt)
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmcold.layout import panel_grid
from lmcold.mass_samples import samples, overlap, with_colour
from lmcold.sources import DATA, ROOT

SNR_SPLIT = 25.  # BOSS S/N separating the low- and high-S/N subsets (sample S/N >= lmcold.giants.BOSS_SNR_MIN = 16)
T_EDGES = np.arange(3700, 5301, 200)  # K, BOSS Teff bins of (a) and (c)
C_EDGES = np.array([.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.2])  # mag, dereddened BP-RP bins of (b)

# %% [markdown]
# ### Load the data
# The BOSS and APOGEE mass samples through `lmcold.mass_samples.samples()` (built from the catalogues of Zenodo bundle 1), and the
# coverage fill and foreground fraction of each BOSS giant from `data/giant_foreground.parquet` (written by
# `scripts/build_giant_mass.py`).

# %%
# lmcold.mass_samples.samples(): BOSS and APOGEE giants with calibrated [Fe/H], Teff, S/N and selection weight w
b, a, _ = samples()
# data/giant_foreground.parquet: per BOSS giant the coverage fill and the Milky Way foreground fraction f_mw
fg = pd.read_parquet(DATA / 'giant_foreground.parquet')

# %% [markdown]
# ### The mass samples and the stars with both spectra
# BOSS weights include the coverage fill and the foreground fraction; APOGEE uses its selection weights.

# %%
# BOSS weight W = selection weight x coverage fill x LMC-member probability (1 - foreground fraction), as in the BOSS mass.
b = b.merge(fg[['source_id', 'fill', 'f_mw']], on='source_id', validate='one_to_one').assign(W=lambda t: t.w * t['fill'] * (1 - t.f_mw))
a = a.assign(W=a.w)  # APOGEE: selection weight only
# Overlap stars: feh_b (calibrated BOSS), feh_a (APOGEE), BOSS g and snr, plus BOSS Teff; d = BOSS - APOGEE [Fe/H] [dex].
o = overlap(b, a).merge(b[['source_id', 'teff']], on='source_id', validate='one_to_one')
o['d'] = o.feh_b - o.feh_a
b, a = with_colour(b), with_colour(a)  # adds g, c0 (dereddened BP-RP) and rho [deg]


# %% [markdown]
# ### Binned statistics

# %%
def binned(x, y, w, edges, f, nmin):
    """Statistic f(y, w) per bin of x; NaN for bins with fewer than nmin stars.

    Parameters
    ----------
    x : array-like
        Binned quantity (BOSS Teff [K] or dereddened BP-RP [mag]).
    y, w : numpy.ndarray
        Values ([Fe/H] or its BOSS - APOGEE difference [dex]) and weights per star.
    edges : numpy.ndarray
        Bin edges in x; stars outside them are ignored.
    f : callable
        f(y, w) -> float, evaluated on the stars of one bin.
    nmin : int
        Minimum number of stars (unweighted) in a bin.

    Returns
    -------
    numpy.ndarray
        One value per bin, shape (len(edges) - 1,).
    """
    # Used for the median offsets of panel (a) (f = mid) and the metal-rich fractions of panel (b) (f = frac).
    k_ = np.digitize(x, edges) - 1   # bin index of each star (-1 or len(edges) - 1: outside)
    # f on the stars of each bin with at least nmin stars, NaN otherwise.
    return np.array([f(y[k_ == i], w[k_ == i]) if (k_ == i).sum() >= nmin else np.nan for i in range(len(edges) - 1)])


# %% [markdown]
# ### The curves and `results/boss_hot_bias.json`
# Offsets against $T_{\rm eff}$, metal-rich fractions against colour, and the weight of hot, low-S/N giants.

# %%
frac = lambda y, w: float(np.sum(w[y > -.5]) / np.sum(w))  # noqa: E731  weighted fraction with [Fe/H] > -0.5
mid = lambda y, w: float(np.median(y))  # noqa: E731  unweighted median (w ignored)
rich = o[(o.feh_a >= -1) & (o.feh_a < -.5)]  # overlap stars with APOGEE [Fe/H] in [-1, -0.5): panel (a)
# JSON output, started with the settings.
out = dict(snr_split=SNR_SPLIT, t_edges=T_EDGES.tolist(), c_edges=C_EDGES.tolist())
# Panel (b) subsamples: faint giants (16.25 <= G <= 17.5) at rho >= 6 deg (the parent ends at 12 deg).
faint_b, faint_a = (t[t.g.between(16.25, 17.5) & (t.rho >= 6)] for t in (b, a))
curves = {}   # name -> (bins,) array
# Panel (a): median offset (bins with >= 8 stars) and star counts per Teff bin, low and high S/N.
for lab, s in [('low', rich[rich.snr < SNR_SPLIT]), ('high', rich[rich.snr >= SNR_SPLIT])]:
    curves[f'offset_{lab}'] = binned(s.teff, s.d.to_numpy(), np.ones(len(s)), T_EDGES, mid, 8)
    curves[f'n_overlap_{lab}'] = np.histogram(s.teff, T_EDGES)[0]
# Panel (b): weighted fraction with [Fe/H] > -0.5 (bins with >= 20 stars) and star counts per colour bin.
for lab, s in [('boss_low', faint_b[faint_b.snr < SNR_SPLIT]), ('boss_high', faint_b[faint_b.snr >= SNR_SPLIT]), ('apogee', faint_a)]:
    curves[f'frac_{lab}'] = binned(s.c0, s.feh.to_numpy(), s.W.to_numpy(), C_EDGES, frac, 20)
    curves[f'n_{lab}'] = np.histogram(s.c0, C_EDGES)[0]
# "Hot, low-S/N" BOSS giants: Teff > 4300 K (above it the offset of panel a becomes positive) and S/N < 25.
hot = (b.teff > 4300) & (b.snr < SNR_SPLIT)
# Curves, the weight fraction of hot low-S/N BOSS giants, the same fraction among overlap stars (unweighted), and the
# metal-rich fractions of BOSS (hot low-S/N, the rest, all) and APOGEE.
out.update({k_: v.tolist() for k_, v in curves.items()},
           target_weight_hot_lowsnr=float(b.W[hot].sum() / b.W.sum()),
           overlap_fraction_hot_lowsnr=float(((o.teff > 4300) & (o.snr < SNR_SPLIT)).mean()),
           boss_frac_above_m05={'hot_lowsnr': frac(b.feh[hot].to_numpy(), b.W[hot].to_numpy()),
                                'rest': frac(b.feh[~hot].to_numpy(), b.W[~hot].to_numpy()), 'all': frac(b.feh.to_numpy(), b.W.to_numpy())},
           apogee_frac_above_m05=frac(a.feh.to_numpy(), a.W.to_numpy()))
(ROOT / 'results/boss_hot_bias.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

# %% [markdown]
# ### The figure

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
fig, axes = panel_grid(1, 3, width=7, left=.55, right=.1, hgap=.6, bottom=.5, top=.6)
tm, cm = .5 * (T_EDGES[1:] + T_EDGES[:-1]), .5 * (C_EDGES[1:] + C_EDGES[:-1])  # bin centres [K], [mag]
# (a) BOSS - APOGEE [Fe/H] against BOSS Teff for the overlap stars at APOGEE -1..-0.5.
ax = axes[0, 0]
# Points: individual stars; line: median per Teff bin.
for lab, c, mk in [('low', '#d75a20', 'o'), ('high', '#2878b5', 's')]:
    s = rich[(rich.snr < SNR_SPLIT) if lab == 'low' else (rich.snr >= SNR_SPLIT)]
    ax.scatter(s.teff, s.d, s=2, color=c, alpha=.3, lw=0)
    ax.plot(tm, curves[f'offset_{lab}'], '-' + mk, ms=3, color=c,
            label=f"S/N {'<' if lab == 'low' else '≥'} {SNR_SPLIT:g} ({len(s)} stars)")
ax.axhline(0, color='.6', lw=.7)
ax.set(xlabel=r'BOSS $T_{\rm eff}$ [K]', ylabel='BOSS − APOGEE [Fe/H]', ylim=(-.5, .6), xlim=(3700, 5300),
       title='(a) APOGEE [Fe/H] −1 to −0.5')
# (b) Weighted metal-rich fraction against dereddened colour for faint outer giants.
ax = axes[0, 1]
# (curve key, colour, line and marker style, legend text) per sample.
for lab, c, ls, txt in [('boss_low', '#d75a20', '-o', f'BOSS, S/N < {SNR_SPLIT:g}'), ('boss_high', '#2878b5', '-s', f'BOSS, S/N ≥ {SNR_SPLIT:g}'),
                        ('apogee', 'k', '--^', 'APOGEE')]:
    ax.plot(cm, curves[f'frac_{lab}'], ls, ms=3, color=c, label=txt)
ax.set(xlabel=r'$(G_{\rm BP}-G_{\rm RP})_0$', ylabel='weighted fraction with [Fe/H] > −0.5', ylim=(0, .85),
       title=r'(b) $16.25\leq G\leq17.5$, $\rho\geq6^\circ$')
# (c) Teff distributions: BOSS mass sample (weighted) against the overlap stars that set the calibration.
ax = axes[0, 2]
for lab, c in [('low', '#d75a20'), ('high', '#2878b5')]:
    # S/N subset of the BOSS sample (m) and of the overlap stars (mo).
    m = (b.snr < SNR_SPLIT) if lab == 'low' else (b.snr >= SNR_SPLIT)
    mo = (o.snr < SNR_SPLIT) if lab == 'low' else (o.snr >= SNR_SPLIT)
    ax.hist(b.teff[m], T_EDGES, weights=b.W[m] / b.W.sum(), histtype='step', color=c, lw=1.3,
            label=f"BOSS sample, S/N {'<' if lab == 'low' else '≥'} {SNR_SPLIT:g}")
    ax.hist(o.teff[mo], T_EDGES, weights=np.full(mo.sum(), 1 / len(o)), histtype='step', color=c, lw=1, ls='--',
            label=f"both spectra, S/N {'<' if lab == 'low' else '≥'} {SNR_SPLIT:g}")
ax.set(xlabel=r'BOSS $T_{\rm eff}$ [K]', ylabel='fraction', xlim=(3700, 5300), title='(c) Who is calibrated')
# Legends above each panel.
for ax in axes[0]:
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.12), ncol=1, frameon=False, fontsize=6.5)
fig.savefig(ROOT / 'figures/paper_boss_hot_bias.png', dpi=200)
