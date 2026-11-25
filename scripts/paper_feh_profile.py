"""Metallicity against in-plane radius for the BOSS and APOGEE giants and the RRab.

Radius R [kpc]: distance from the centre in the adopted disc plane (lmcold.constants.LMC: inclination, line of nodes, D0), each
star placed on the plane. The 12-deg BOSS footprint covers all azimuths only to R ~ 10 kpc, so bins stop there.
Giants (lmcold.mass_samples.samples) are counted with their selection weights; BOSS also with the coverage factor and one minus the
Milky Way foreground fraction (data/giant_foreground.parquet, from build_giant_mass.py). APOGEE covers only its 35 fields.
RRab members are unweighted: the Gaia+OGLE union is >= 96 per cent complete at all radii (Sect. 2.2.1).
(a, b) weighted median [Fe/H] of the giants (BOSS also with G < G_BRIGHT) and the RRab, with the published gradients (lmcold.constants.
LMC_FEH_GRADIENT_LITERATURE) drawn over their own radial ranges through our profile at the middle of each range (slopes only);
(c) surface density of the BOSS giants [kpc^-2 on the disc plane] in three ranges of true [Fe/H]: per R bin, the weighted counts
in fine measured bins are fitted through the BOSS response (lmcold.mass_samples.boss_response re-weighted to the bin's G mix,
lmcold.mass_response.recover_mass), so no star is corrected individually; the medians in (a) are of measured [Fe/H]. Dashed:
exponentials fitted to ln(Sigma) over FIT_RANGE; scale lengths in results/feh_profile.json with their bootstrap percentiles and
the foreground set to zero and doubled. Percentiles 16-84 of [Fe/H] are in the csv. Open symbols: RRab members per kpc^2 below and
above [Fe/H] = RR_SPLIT (photometric, Crestani+21 scale; unweighted, Poisson errors), to RR_R_MAX, with exponentials over
RR_FIT_RANGE; their scale lengths are in results/feh_profile.json.
Errors: 16th-84th percentiles of NBOOT draws of nside-64 (0.9-deg) sky pixels with the weights fixed and, for (c), the overlap
stars of the response resampled; nside 16 would leave one or two pixels in the inner bins.
Writes figures/paper_feh_profile.png, results/feh_profile.csv, results/feh_profile.json and, against projected distance rho [deg] for
scripts/paper_feh_literature.py, results/feh_profile_rho.csv.

Inputs:
- lmcold.mass_samples.samples(): BOSS and APOGEE mass samples (velocity-selected giants with selection weights w; BOSS [Fe/H] on
  the APOGEE scale) from data/boss_weights.parquet (sources.yaml release_boss_weights), data/apogee_weights.parquet
  (release_apogee_weights), data/spec_giants.parquet and results/boss_calibration.json.
- data/giant_foreground.parquet (scripts/build_giant_mass.py): per BOSS giant `fill` (coverage factor) and `f_mw`.
- data/rrab.parquet (release_rrab, scripts/build_rrab.py): RRab with `member` flag and adopted photometric `feh`.
Outputs:
- results/feh_profile.csv (bins of R [kpc]) and results/feh_profile_rho.csv (bins of rho [deg]): one row per sample (BOSS,
  APOGEE, RRab, 'BOSS G<16.5') and bin, with columns sample, R_lo/R_hi (or rho_lo/rho_hi), n (stars), median, p16, p84
  (weighted percentiles of [Fe/H] in the bin), median_lo/median_hi (16th/84th percentiles of the median over the bootstrap);
  BOSS in R also sigma_<range>, sigma_<range>_lo, sigma_<range>_hi [giants kpc^-2] for range lt15, 15to10, gt10.
- results/feh_profile.json: per giant range h [kpc], h_p16, h_p84 (bootstrap), h_foreground_x0, h_foreground_x2; fit range, NBOOT,
  number of skipped bootstrap bins; per RRab range (rr_lt15, rr_gt15) h [kpc], fit range and counts per R bin.
Panels (1 x 3; legends above the panels; x: in-plane radius R [kpc], 0-10):
(a) y: weighted median [Fe/H] [dex] of BOSS (solid), BOSS G < 16.5 (dashed, open), APOGEE, error bars the bootstrap 16-84 range
    of the median; published giant gradients as line segments over their radial ranges (slopes only).
(b) the same for the RRab with the Skowron+16 inner, middle and outer segments.
(c) y: number per kpc^2 on the disc plane (log); filled circles: BOSS giants in true [Fe/H] < -1.5, -1.5 to -1, > -1 with
    exponential fits over FIT_RANGE (dashed); open squares: RRab below and above RR_SPLIT with exponential fits over RR_FIT_RANGE
    (dotted). Legend gives the scale lengths h.
Figure: paper Fig. 5 (Sect. 3, main text; docs/figures.md).
Run: .venv/bin/python scripts/paper_feh_profile.py
Run time: 11 s on a MacBook (M-series), docs/run_times.txt.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import healpy as hp
import numpy as np
import pandas as pd
from lmcold.constants import LMC, LMC_FEH_GRADIENT_LITERATURE
from lmcold.footprint import angular_distance
from lmcold.layout import panel_grid
from lmcold.maps import wquantile
from lmcold.mass_response import recover_mass
from lmcold.mass_samples import samples, overlap, boss_response, MEASURED_EDGES, TRUE_EDGES
from lmcold.perspective import disc_radius
from lmcold.sources import DATA, ROOT

# %% [markdown]
# ### Settings and helpers
# Radial bins, the true [Fe/H] ranges and fit ranges, and the helpers for the profiles, the scale lengths and the bootstrap.

# %%
R_EDGES = np.arange(0., 10.5, 1.)  # kpc, 1-kpc in-plane radius bins 0-10 kpc
# BOSS reaches G = 17.5; 93 per cent of the APOGEE weight is at G < 16.5 (its faint groups are uncovered), and the two medians agree
# at G < 16.5 (scripts/fig_feh_profile_glim.py), so the BOSS median is also shown to this limit.
G_BRIGHT = 16.5
RHO_EDGES = np.arange(0., 12.5, 1.)  # projected distance [deg], for literature quoted in degrees
NBOOT = 200  # bootstrap draws of sky pixels
R_MID = .5 * (R_EDGES[:-1] + R_EDGES[1:])
# True [Fe/H] ranges for the surface densities: indices of the TRUE_EDGES bins (-3, -2, -1.5, -1, 0.5) they sum.
RANGES = {'lt15': (0, 1), '15to10': (2,), 'gt10': (3,)}
FIT_RANGE = (2., 10.)  # kpc; inside 2 kpc the bar adds to the metal-rich profile
# RRab members lie within rho < 10 deg, which covers all azimuths of the disc plane to R = 8 kpc (93 per cent of 8-9 kpc, 32 per cent
# of 9-10 kpc; points uniform on the sky), so their profiles stop there.
RR_SPLIT, RR_R_MAX = -1.5, 8.  # RRab [Fe/H] split [dex] and outer radius of the RRab profiles [kpc]
RR_FIT_RANGE = (2., RR_R_MAX)
rng = np.random.default_rng(1)  # fixed seed: the bootstrap is reproducible


def profile(s, o=None, col='R', edges=R_EDGES):
    """Per bin of s[col]: median, p16, p84 of [Fe/H]; with the overlap stars o (BOSS), also the surface density [giants kpc^-2]
    in the RANGES of true [Fe/H], from the counts in fine measured bins fitted through the response (Sect. 2.3).

    Parameters
    ----------
    s : pandas.DataFrame
        Sample with columns feh [dex], W (weight per star), col, and for the surface densities g (Gaia G) and snr (BOSS S/N).
    o : pandas.DataFrame or None, optional
        Overlap stars (lmcold.mass_samples.overlap) defining the BOSS response; None skips the surface densities.
    col : str, optional
        Binned column: 'R' (in-plane radius [kpc]) or 'rho' (projected distance [deg]).
    edges : numpy.ndarray, optional
        Bin edges in `col`.

    Returns
    -------
    numpy.ndarray
        Shape (len(edges) - 1, 3 + len(RANGES)): weighted median, 16th and 84th percentiles of [Fe/H] [dex], then the surface
        density per RANGES entry [weighted giants kpc^-2, annulus area pi (R_hi^2 - R_lo^2) on the disc plane]. NaN for bins with
        total weight <= 0 or fewer than 20 stars, and for densities where the response cannot be built.
    """
    k = np.digitize(s[col], edges) - 1
    out = np.full((len(edges) - 1, 3 + len(RANGES)), np.nan)
    for i in range(len(edges) - 1):
        t = s[k == i]
        if t.W.sum() <= 0 or len(t) < 20:
            continue
        x, w = t.feh.to_numpy(), t.W.to_numpy()
        out[i, :3] = wquantile(x, w, [.5, .16, .84])
        if o is not None and (x < -1.5).any():  # boss_response needs stars on both sides of -1.5
            resp = boss_response(o, x, t.g.to_numpy(), t.snr.to_numpy(), w)
            if not np.isfinite(resp).all():  # a resampled overlap can leave a column empty after G reweighting; draw skipped
                continue
            counts = np.histogram(x, MEASURED_EDGES, weights=w)[0]  # weighted counts in 0.125-dex measured [Fe/H] bins
            true = recover_mass(counts, resp, np.ones(len(TRUE_EDGES) - 1))  # eta = 1: returns counts
            area = np.pi * (edges[i + 1] ** 2 - edges[i] ** 2)
            out[i, 3:] = [true[list(j)].sum() / area for j in RANGES.values()]
    return out


def scale_lengths(v):
    """Exponential scale length [kpc] of each RANGES profile, from a straight line in ln(Sigma) over FIT_RANGE.

    Unweighted least squares on the bin centres R_MID strictly inside FIT_RANGE; Sigma(R) ~ exp(-R / h), so h = -1 / slope.

    Parameters
    ----------
    v : numpy.ndarray
        Output of `profile` in R_EDGES bins, shape (len(R_MID), 3 + len(RANGES)).

    Returns
    -------
    list of float
        Scale length h [kpc] per RANGES entry.
    """
    m = (R_MID > FIT_RANGE[0]) & (R_MID < FIT_RANGE[1])
    return [-1 / np.polyfit(R_MID[m], np.log(v[m, 3 + j]), 1)[0] for j in range(len(RANGES))]


def boot(s, o=None, col='R', edges=R_EDGES):
    """NBOOT draws of nside-64 sky pixels; the overlap stars that define the response are resampled in each draw.

    Each draw picks len(upix) pixels with replacement among the occupied ones and multiplies the weight of every star by the
    number of times its pixel was drawn (the selection weights themselves are not recomputed).

    Parameters
    ----------
    s : pandas.DataFrame
        Sample as for `profile`, plus `pix` (HEALPix nside-64 index, ring ordering).
    o, col, edges
        As for `profile`.

    Returns
    -------
    numpy.ndarray
        Shape (NBOOT, len(edges) - 1, 3 + len(RANGES)): `profile` of each draw.
    """
    pix = s.pix.to_numpy()
    upix, inv = np.unique(pix, return_inverse=True)
    draws = []
    for _ in range(NBOOT):
        m = np.bincount(rng.integers(0, len(upix), len(upix)), minlength=len(upix))[inv]
        t = s.assign(W=s.W * m)
        draws.append(profile(t[t.W > 0], None if o is None else o.sample(len(o), replace=True, random_state=rng), col, edges))
    return np.array(draws)


# %% [markdown]
# ### Load the data
# The BOSS and APOGEE mass samples through `lmcold.mass_samples.samples()` (built from the catalogues of Zenodo bundle 1), the
# coverage fill and foreground fraction of each BOSS giant (`data/giant_foreground.parquet`, written by
# `scripts/build_giant_mass.py`), and the RRab catalogue `data/rrab.parquet` (Zenodo bundle 1).

# %%
# lmcold.mass_samples.samples(): BOSS and APOGEE giants with calibrated [Fe/H], g, snr and selection weight w
b, a, _ = samples()
# data/giant_foreground.parquet: per BOSS giant the coverage fill and the Milky Way foreground fraction f_mw
fg = pd.read_parquet(DATA / 'giant_foreground.parquet')
# data/rrab.parquet: RRab with member flag and photometric [Fe/H]
r = pd.read_parquet(DATA / 'rrab.parquet')

# %% [markdown]
# ### Samples and weights

# %%
# --- Samples and weights ---
o = overlap(b, a)  # stars with both spectra: feh_b (calibrated BOSS), feh_a (APOGEE), BOSS g and snr
b = b.merge(fg[['source_id', 'fill', 'f_mw']], on='source_id', validate='one_to_one')
b['W'] = b.w * b['fill'] * (1 - b.f_mw)  # BOSS: selection weight x coverage fill x (1 - Milky Way foreground fraction)
a['W'] = a.w  # APOGEE: selection weight only (its 35 fields)
r = r[r.member & np.isfinite(r.feh)].assign(W=1.)
for s in (b, a, r):
    # R: in-plane radius [kpc] with each star placed on the adopted disc plane; rho: projected distance [deg] from (ra0, dec0).
    s['R'] = disc_radius(s.ra.to_numpy(), s.dec.to_numpy())
    s['rho'] = angular_distance(s.ra.to_numpy(), s.dec.to_numpy(), LMC['ra0'], LMC['dec0'])
    s['pix'] = hp.ang2pix(64, s.ra.to_numpy(), s.dec.to_numpy(), lonlat=True)

# %% [markdown]
# ### Profiles and bootstrap errors
# Against projected distance and in-plane radius; written to `results/feh_profile_rho.csv` and `results/feh_profile.csv`.

# %%
# --- Profiles and bootstrap errors, against rho (for the appendix) and R (this figure; `t` and `summary` keep the R results) ---
for col, edges, out in [('rho', RHO_EDGES, 'feh_profile_rho.csv'), ('R', R_EDGES, 'feh_profile.csv')]:
    rows = []
    extra = [(f'BOSS G<{G_BRIGHT}', b[b.g < G_BRIGHT])]  # last, so the other bootstraps keep their draws
    for name, s in [('BOSS', b), ('APOGEE', a), ('RRab', r)] + extra:
        oo = o if (name, col) == ('BOSS', 'R') else None  # surface densities only for the full BOSS sample against R
        state = rng.bit_generator.state
        v, d = profile(s, oo, col, edges), boot(s, oo, col, edges)
        if col == 'rho' and name.endswith(str(G_BRIGHT)):  # added later: restore the generator so the R bootstraps keep their draws
            rng.bit_generator.state = state
        e = np.nanpercentile(d, [16, 84], axis=0)
        n = np.bincount(np.digitize(s[col], edges) - 1, minlength=len(edges))[:len(edges) - 1]
        for i in range(len(edges) - 1):
            row = {'sample': name, col + '_lo': edges[i], col + '_hi': edges[i + 1], 'n': n[i], 'median': v[i, 0], 'p16': v[i, 1],
                   'p84': v[i, 2], 'median_lo': e[0, i, 0], 'median_hi': e[1, i, 0]}
            if oo is not None:
                for j, key in enumerate(RANGES):
                    row.update({f'sigma_{key}': v[i, 3 + j], f'sigma_{key}_lo': e[0, i, 3 + j], f'sigma_{key}_hi': e[1, i, 3 + j]})
            rows.append(row)
        if oo is not None:
            h = np.array([scale_lengths(x) for x in d])
            summary = {'fit_range_kpc': FIT_RANGE, 'nboot': NBOOT, 'skipped_bins': int(np.isnan(d[:, :, 3]).sum())}
            for j, key in enumerate(RANGES):
                summary[key] = dict(h=scale_lengths(v)[j], h_p16=np.nanpercentile(h[:, j], 16), h_p84=np.nanpercentile(h[:, j], 84))
            for k_fg in (0., 2.):  # foreground bracket, as for the masses
                vf = profile(s.assign(W=s.w * s['fill'] * np.clip(1 - k_fg * s.f_mw, 0, None)), o, col, edges)
                for j, key in enumerate(RANGES):
                    summary[key][f'h_foreground_x{k_fg:g}'] = scale_lengths(vf)[j]
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / 'results' / out, index=False)
# %% [markdown]
# ### RR Lyrae surface densities
# Counts per ring below and above the [Fe/H] split, with their scale lengths; written to `results/feh_profile.json`.

# %%
# --- RRab surface density below and above RR_SPLIT, 0-8 kpc: members per kpc^2 on the disc plane ---
rr_bins = R_EDGES[R_EDGES <= RR_R_MAX]
rr_mid = .5 * (rr_bins[1:] + rr_bins[:-1])
rr_n = {key: np.histogram(r.R[m], rr_bins)[0] for key, m in [('rr_lt15', r.feh < RR_SPLIT), ('rr_gt15', r.feh >= RR_SPLIT)]}
rr_area = np.pi * np.diff(rr_bins ** 2)
fit = (rr_mid > RR_FIT_RANGE[0]) & (rr_mid < RR_FIT_RANGE[1])
for key, n in rr_n.items():
    # ln(N / area) has error 1 / sqrt(N) for Poisson counts, so polyfit weights (1 / sigma) are sqrt(N).
    cf = np.polyfit(rr_mid[fit], np.log(n[fit] / rr_area[fit]), 1, w=np.sqrt(n[fit]))
    summary[key] = dict(h=-1 / cf[0], fit_range_kpc=RR_FIT_RANGE, n=n.tolist())
(ROOT / 'results/feh_profile.json').write_text(json.dumps(summary, indent=1))
print(t.round(3).to_string(index=False))
print(json.dumps(summary, indent=1))

# %% [markdown]
# ### The figure

# %%
# --- Figure ---
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
colours = dict(BOSS='#2878b5', APOGEE='#d75a20', RRab='#3a923a')
# Horizontal offsets [kpc] of each sample's points, so their error bars do not overlap.
shift = {'BOSS': 0., f'BOSS G<{G_BRIGHT}': .06, 'APOGEE': .12, 'RRab': -.12}
# Published gradients: one colour and dash pattern per study.
LIT_STYLE = {'Povick+25': ('#000000', (0, (6, 2))), 'Grady+21': ('#7b3294', (0, (1, 1.2))), 'Cioni 09': ('#a6611a', (0, (5, 1.5, 1, 1.5))),
             'Choudhury+16': ('#d01c8b', (0, (2.5, 1))), 'Choudhury+21': ('#4d9221', (0, (8, 1.5, 1, 1.5, 1, 1.5))),
             'Skowron+16': ('#000000', (0, (6, 2)))}
fig, axes = panel_grid(1, 3, width=7, left=.55, right=.1, hgap=.62, bottom=.5, top=.95)
for ax, samples_, tracer in [(axes[0, 0], ['BOSS', f'BOSS G<{G_BRIGHT}', 'APOGEE'], 'giants'), (axes[0, 1], ['RRab'], 'RRab')]:
    for name in samples_:
        q = t[t['sample'] == name]
        x = .5 * (q.R_lo + q.R_hi) + shift[name]
        bright = name.endswith(str(G_BRIGHT))
        ax.errorbar(x, q['median'], [q['median'] - q.median_lo, q.median_hi - q['median']], fmt='--o' if bright else '-o', ms=3, lw=1,
                    color=colours[name.split()[0]], mfc='white' if bright else None,
                    label=f'BOSS G < {G_BRIGHT} median' if bright else f'{name} median', zorder=3)
    # Published gradients, each over its own range and through our profile (BOSS for giants) at the middle of that range:
    # only the slopes are compared, so no zero points are needed.
    ref = t[t['sample'] == samples_[0]]
    for L in (x for x in LMC_FEH_GRADIENT_LITERATURE if x['tracer'] == tracer):
        hi = min(L['r_hi'], R_EDGES[-1])
        mid = .5 * (L['r_lo'] + hi)
        slope = L['slope'] if 'slope_hi' not in L else .5 * (L['slope'] + L['slope_hi'])
        y0 = np.interp(mid, .5 * (ref.R_lo + ref.R_hi), ref['median'])
        rr = np.array([L['r_lo'], hi])
        study = L['label'].split(' inner')[0].split(' middle')[0].split(' outer')[0]
        c, ls = LIT_STYLE[study]
        yy = y0 + slope * (rr - mid)
        ax.plot(rr, yy, color=c, lw=1.4, ls=ls, zorder=2, label=None if L['label'].endswith((' middle', ' outer')) else study)
    ax.set(xlabel='R [kpc]', ylabel='median [Fe/H]', xlim=(0, 10))
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=2, frameon=False, fontsize=6.5, handlelength=3.5,
              columnspacing=1.)
axes[0, 0].set_title('(a) Giants', pad=62)
axes[0, 1].set_title('(b) RR Lyrae', pad=62)
# (c) Surface densities: giants by true [Fe/H] (response-corrected), RRab by photometric [Fe/H].
ax = axes[0, 2]
q = t[t['sample'] == 'BOSS']
for key, label, c in [('lt15', '[Fe/H] < −1.5', '#3b2c85'), ('15to10', '−1.5 < [Fe/H] < −1', '#2a9d8f'), ('gt10', '[Fe/H] > −1', '#e09f3e')]:
    y = q[f'sigma_{key}']
    ax.errorbar(R_MID, y, [y - q[f'sigma_{key}_lo'], q[f'sigma_{key}_hi'] - y], fmt='o', ms=3, lw=1, color=c,
                label=f"giants {label}: $h={summary[key]['h']:.2f}$ kpc")
    m = (R_MID > FIT_RANGE[0]) & (R_MID < FIT_RANGE[1])
    cf = np.polyfit(R_MID[m], np.log(y[m]), 1)
    rr = np.array(FIT_RANGE)
    ax.plot(rr, np.exp(np.polyval(cf, rr)), color=c, lw=.9, ls=(0, (4, 1.5)))
for key, label, c in [('rr_lt15', f'RRab [Fe/H] < {RR_SPLIT:g}', '#3b2c85'), ('rr_gt15', f'RRab [Fe/H] > {RR_SPLIT:g}', '#2a9d8f')]:
    n = rr_n[key]
    ax.errorbar(rr_mid, n / rr_area, np.sqrt(n) / rr_area, fmt='s', ms=3.5, mfc='white', lw=1, color=c,
                label=f"{label}: $h={summary[key]['h']:.2f}$ kpc".replace('-', '−'))
    cf = np.polyfit(rr_mid[fit], np.log(n[fit] / rr_area[fit]), 1, w=np.sqrt(n[fit]))
    ax.plot(RR_FIT_RANGE, np.exp(np.polyval(cf, RR_FIT_RANGE)), color=c, lw=.9, ls=(0, (1, 1)))
ax.set(xlabel='R [kpc]', ylabel=r'number per kpc$^2$', xlim=(0, 10), yscale='log')
ax.set_title('(c) Giants and RR Lyrae by metallicity', pad=62)
ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=1, frameon=False, fontsize=6.5)
fig.savefig(ROOT / 'figures/paper_feh_profile.png', dpi=200)
