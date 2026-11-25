"""Do BOSS [Fe/H] errors change the kinematics of the [Fe/H] bins? Forward check of the adopted treatment.

BOSSNet over-estimates the [Fe/H] of hot giants: calibrated BOSS - APOGEE at APOGEE -1..-0.5 is within 0.03 dex below 4300 K,
+0.05 at 4300-4500 K and +0.14..+0.17 at 4500-4700 K (paper App. B, scripts/paper_boss_hot_bias.py). Hot, more metal-poor and
kinematically hotter giants could thus enter the metal-rich bins. APOGEE gives a reference: its [Fe/H] has no such bias.

Every giant in its measured bin (lmcold.samples.giants(), release catalogues, 2 < rho < 9 deg). For each [Fe/H] bin: first the L2 fit of the whole bin (lmcold.kinfit,
symmetric disc then + STREAM, PM + LOS). Then, with the geometry, bulk motion, vlos0, R0 and
streaming held at that fit, v0, sigma, sigma_los, k and the outlier fraction refitted for three subsets: APOGEE, BOSS Teff < 4300 K
('BOSS cool') and BOSS Teff >= 4300 K ('BOSS hot'), each restricted to the shared sky cells: HEALPix nside 32 (~1.8 deg) cells
holding >= MIN_APOGEE APOGEE stars of the bin. The same fits of the two BOSS subsets on the whole ring are also made.
Leakage expected from the [Fe/H] errors: on the stars in both surveys, the fraction of those in each measured (BOSS) bin whose
APOGEE [Fe/H] lies in another bin, for BOSS cool and BOSS hot separately (binomial errors; few hot stars in the overlap).
Forward prediction (why APOGEE alone is the reference from -1.1 up while all giants are the default sample): the BOSS value of a
measured bin j is APOGEE's, mixed over the true bins t with the overlap fractions P(t | j) of the class:
sigma_pred^2 = sum_t P(t | j) sigma_APOGEE,t^2 and v0_pred = sum_t P(t | j) v0_APOGEE,t (mean-velocity differences between the
bins neglected in sigma_pred). Unweighted overlap fractions (the overlap is brighter and has higher S/N than the BOSS sample).

Writes results/kin_feh_contamination.csv and figures/paper_kin_feh_mixing.png: four equal panels against the bin's median
[Fe/H]: (a) sigma_los and (b) v0 [km/s] of APOGEE, BOSS cool and BOSS hot in the shared cells (BOSS on the whole ring as open
symbols) and the forward predictions for BOSS cool and hot (crosses); (c) fraction of hot stars among the BOSS stars of each
bin; (d) leakage fraction from the overlap, BOSS cool and hot.
Paper figure (appendix).

Run: .venv/bin/python scripts/paper_kin_feh_mixing.py   (about 2.5 min: 153 s in the run of 2026-11-22, docs/run_times.txt)
"""
import healpy as hp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fit_kinematics_mcmc import ring
from lmcold import kinfit, samples
from lmcold.layout import panel_grid
from lmcold.sources import ROOT, path

# Columns passed to lmcold.kinfit.data: degrees, mas/yr, km/s.
COLS = ('ra', 'dec', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error', 'pmra_pmdec_corr', 'vlos', 'evlos')
FREE = kinfit.BASE + ('vlos0', 'sigma_los')  # proper-motion model plus systemic velocity and line-of-sight dispersion
OWN = ('v0', 'sigma', 'sigma_los', 'k', 'logit_fout')  # refitted per subset; the rest held at the bin's L2 fit
T_HOT = 4300.  # K: BOSS [Fe/H] unbiased below, biased high above (paper App. B)
NSIDE = 32  # HEALPix cells of ~1.8 deg
MIN_APOGEE = 5  # APOGEE stars of the bin needed for a cell to count as shared
EDGES = [e for e, _ in samples.FEH_BINS] + [samples.FEH_BINS[-1][1]]   # dex, the six edges of the five [Fe/H] bins


def sub(c, keep):
    """Catalogue dict c restricted to the boolean mask keep."""
    return {k: v[keep] for k, v in c.items()}


def l2(c):
    """Maximum-likelihood L2 fit (symmetric disc, then + STREAM) of catalogue dict c."""
    d = kinfit.data(*(c[k] for k in COLS))
    # Symmetric disc with the centre (dx, dy, the first two of FREE) fixed, then free, then + STREAM; returns the parameter vector.
    p, _, _ = kinfit.fit(d, free=FREE[2:])
    p, _, _ = kinfit.fit(d, free=FREE, p0=p)
    return kinfit.fit(d, free=FREE + kinfit.STREAM, p0=p)[0]


def own(c, p):
    """v0, sigma, sigma_los, k, outliers of catalogue dict c refitted with the rest held at p: (p, err)."""
    p, e, _ = kinfit.fit(kinfit.data(*(c[k] for k in COLS)), free=OWN, p0=p)
    return p, e


# %% [markdown]
# ### Load the data
# The giant catalogues of Zenodo bundle 1: through `lmcold.samples.giants()`, merged and split into the five [Fe/H] bins, and, for
# the stars observed by both surveys, the APOGEE and BOSS catalogues (`data/giants_apogee.parquet`, `data/giants_boss.parquet`) joined
# on the Gaia source_id.

# %%
giants = samples.giants()  # (label, catalogue dict) per [Fe/H] bin, from data/giants_apogee.parquet and giants_boss.parquet
# Stars in both surveys: APOGEE columns with suffix _a, BOSS columns (calibrated [Fe/H]) with suffix _b.
m = pd.read_parquet(path('release_giants_apogee')).merge(pd.read_parquet(path('release_giants_boss')), on='source_id', suffixes=('_a', '_b'))

# %% [markdown]
# ### Fits of the survey subsets
# For each [Fe/H] bin: the Level-2 maximum-likelihood fit of the whole bin, then $v_0$, $\sigma$, $\sigma_{\rm los}$, $k$ and the
# outlier fraction refitted for APOGEE, BOSS cool ($T_{\rm eff}$ < `T_HOT`) and BOSS hot giants in the HEALPix cells shared with
# APOGEE, and for the two BOSS subsets on the whole ring.

# %%
rows = []   # one dict per (bin, subset)
for label, c in giants:
    c = ring(c)   # 2 < rho < 9 deg
    p = l2(c)   # L2 fit of the whole bin: the held parameters
    pix = hp.ang2pix(NSIDE, c['ra'], c['dec'], lonlat=True)   # HEALPix cell (ring ordering) of each star
    apo = c['survey'] == 'apogee'
    # Shared cells: those with >= MIN_APOGEE APOGEE stars of the bin.
    cells, counts = np.unique(pix[apo], return_counts=True)
    shared = np.isin(pix, cells[counts >= MIN_APOGEE])
    # Masks of the five subsets: three in the shared cells, two BOSS subsets on the whole ring.
    subsets = {'APOGEE': apo & shared, 'BOSS cool': ~apo & (c['teff'] < T_HOT) & shared, 'BOSS hot': ~apo & (c['teff'] >= T_HOT) & shared,
               'BOSS cool, ring': ~apo & (c['teff'] < T_HOT), 'BOSS hot, ring': ~apo & (c['teff'] >= T_HOT)}
    for name, keep in subsets.items():
        q, e = own(sub(c, keep), p)
        # Subset size, the bin's median [Fe/H] [dex] and hot fraction of its BOSS stars, refitted parameters and errors.
        r = dict(bin=label, subset=name, n=int(keep.sum()), feh_med=np.median(c['feh']), hot_frac=np.mean(c['teff'][~apo] >= T_HOT))
        r |= {n: q[kinfit.PARAMS.index(n)] for n in OWN} | {f'{n}_err': e[kinfit.PARAMS.index(n)] for n in OWN}
        rows.append(r)
        print(f"{label:22s} {name:16s} N {r['n']:6d}  v0 {r['v0']:6.1f} +- {r['v0_err']:4.1f}  sigma_los {r['sigma_los']:5.1f} +- "
              f"{r['sigma_los_err']:4.1f}", flush=True)
t = pd.DataFrame(rows)   # one row per (bin, subset)

# %% [markdown]
# ### Leakage and forward prediction
# On the stars observed by both surveys: the fraction of each measured (BOSS) bin whose APOGEE [Fe/H] lies in another bin, and the
# BOSS dispersion and rotation predicted by mixing the APOGEE values in those proportions,
# $\sigma_{\rm pred}^2=\sum_tP(t\,|\,j)\,\sigma_{{\rm APOGEE},t}^2$.

# %%
# Leakage from the [Fe/H] errors, on the stars in both surveys (BOSS calibrated value vs APOGEE).
# mb: measured (BOSS) bin, tb: 'true' (APOGEE) bin of each star (-1 or 5: outside the bins).
mb, tb = np.digitize(m.feh_b, EDGES) - 1, np.digitize(m.feh_a, EDGES) - 1
leak = []   # one dict per (bin, class)
for j, (label, _) in enumerate(zip(t.bin.unique(), samples.FEH_BINS)):
    # Class by BOSS Teff: cool below T_HOT, hot at or above.
    for cls, k in (('cool', m.teff_b < T_HOT), ('hot', m.teff_b >= T_HOT)):
        n = int((k & (mb == j)).sum())   # overlap stars of the class measured in bin j
        # Fraction whose APOGEE [Fe/H] is in another bin, with its binomial error (max(n, 1) avoids division by zero).
        f = ((tb != j) & k & (mb == j)).sum() / max(n, 1)
        leak.append(dict(bin=label, cls=cls, n=n, frac=f, err=np.sqrt(f * (1 - f) / max(n, 1))))
lk = pd.DataFrame(leak)
# Forward prediction of the BOSS values from APOGEE's (see the module docstring).
apo = t[t.subset == 'APOGEE']   # APOGEE values per bin (shared cells), in bin order
pred = []
for cls, k in (('cool', m.teff_b < T_HOT), ('hot', m.teff_b >= T_HOT)):
    for j, label in enumerate(apo.bin):
        # P(t | j): fractions of the class's overlap stars of measured bin j in each APOGEE bin t; then the mixed sigma_los, v0.
        f = np.bincount(tb[k & (mb == j)], minlength=len(apo)) / max((k & (mb == j)).sum(), 1)
        pred.append(dict(bin=label, cls=cls, sigma_los_pred=np.sqrt(f @ apo.sigma_los.to_numpy() ** 2), v0_pred=f @ apo.v0.to_numpy()))
pr = pd.DataFrame(pred)   # one row per (bin, class): predicted sigma_los, v0 [km/s]
pr.to_csv(ROOT / 'results/kin_feh_forward.csv', index=False, float_format='%.5f')
# Print the predictions next to the measured BOSS cool and hot values ('BOSS cool'[5:] = 'cool').
print(pr.merge(t[t.subset.isin(['BOSS cool', 'BOSS hot'])].assign(cls=lambda x: x.subset.str[5:])[['bin', 'cls', 'sigma_los', 'v0']],
               on=['bin', 'cls']).to_string(index=False, float_format='%.2f'))
t.to_csv(ROOT / 'results/kin_feh_contamination.csv', index=False, float_format='%.5f')
lk.to_csv(ROOT / 'results/kin_feh_leakage.csv', index=False, float_format='%.5f')
print(lk.to_string(index=False, float_format='%.2f'))

# %% [markdown]
# ### The figure
# Four panels against the median [Fe/H] of each bin: (a) $\sigma_{\rm los}$, (b) $v_0$, (c) the hot fraction of the BOSS giants,
# (d) the leakage fraction.

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
fig, ax = panel_grid(1, 4, width=9.0, left=0.5, right=0.15, hgap=0.6, top=0.45, bottom=0.45)
# Subset -> (colour, marker, marker face colour); 'none': open symbols for the whole-ring BOSS fits.
style = {'APOGEE': ('k', 'o', 'k'), 'BOSS cool': ('C0', 's', 'C0'), 'BOSS hot': ('C3', '^', 'C3'),
         'BOSS cool, ring': ('C0', 's', 'none'), 'BOSS hot, ring': ('C3', '^', 'none')}
fe = t.groupby('bin', sort=False).feh_med.first()   # median [Fe/H] of each bin [dex]
# (a), (b): measured values, subsets offset by 0.012 dex in x.
for j, (name, (col, mk, mfc)) in enumerate(style.items()):
    s = t[t.subset == name]
    x = s.feh_med + 0.012 * (j - 2)
    ax[0, 0].errorbar(x, s.sigma_los, yerr=s.sigma_los_err, fmt=mk, color=col, mfc=mfc, ms=4, lw=0.7, label=name)
    ax[0, 1].errorbar(x, s.v0, yerr=s.v0_err, fmt=mk, color=col, mfc=mfc, ms=4, lw=0.7)
# (a), (b): forward predictions as crosses, offset by 0.03 dex.
for cls, col in (('cool', 'C0'), ('hot', 'C3')):
    s = pr[pr.cls == cls]
    ax[0, 0].plot(fe.loc[s.bin] + 0.03, s.sigma_los_pred, 'x', color=col, ms=5, label=f'BOSS {cls}, predicted')
    ax[0, 1].plot(fe.loc[s.bin] + 0.03, s.v0_pred, 'x', color=col, ms=5)
# (c) hot fraction of BOSS stars; (d) leakage with binomial errors.
ax[0, 2].plot(fe, t.groupby('bin', sort=False).hot_frac.first(), 'o', color='k', ms=4)
for cls, col, mk in (('cool', 'C0', 's'), ('hot', 'C3', '^')):
    s = lk[lk.cls == cls]
    ax[0, 3].errorbar(fe.loc[s.bin], s.frac, yerr=s.err, fmt=mk, color=col, ms=4, lw=0.7, label=f'BOSS {cls}')
ax[0, 0].legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=4)  # above the panel
ax[0, 3].legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, fontsize=5)
for a, letter in zip(ax[0], 'abcd'):  # empty corners: lower left in (a) and (c), upper left in (b) and (d)
    a.text(0.04, 0.06 if letter in 'ac' else 0.92, f'({letter})', transform=a.transAxes, fontsize=7)
# Axis labels.
for a, lab in zip(ax[0], (r'$\sigma_{\rm los}$ [km s$^{-1}$]', r'$v_0$ [km s$^{-1}$]', r'BOSS fraction with $T_{\rm eff} \geq 4300$ K',
                          'fraction in another bin (APOGEE)')):
    a.set(xlabel='median [Fe/H] of the bin', ylabel=lab)
ax[0, 3].set_ylim(0, 1)
fig.savefig(ROOT / 'figures/paper_kin_feh_mixing.png', dpi=200)
