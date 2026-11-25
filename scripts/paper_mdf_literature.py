"""Paper figure (appendix E): metallicity distributions of LMC giants by number, ours against published ones in the same places.

Ours: BOSS giants (calibrated [Fe/H], weights w x coverage x (1 - f_MW)) and APOGEE giants (weights w) of lmcold.mass_samples, as
measured (no response correction, which acts on binned counts, not on stars). Published (scripts/fetch_mdf_literature.py):
(a) bar: Cole et al. (2005) Ca II triplet [Fe/H]; ours within FIELD_R of the centre of each of their seven fields.
(b) disc fields 5, 6 and 8 deg north of the bar: Carrera et al. (2008) Ca II triplet, [Fe/H] from their equation (1) on the CG97
    scale, -2.95 + 0.38 Sum(Ca) + 0.23 M_I, M_I = I - LMC_DISTANCE_MODULUS - A_I (A_I = 1.505 E(B-V), lmcold.extinction), radial
    velocity members (170-380 km/s, their criterion); ours within FIELD_R of each field. Their 3-deg field holds 4 BOSS giants and is
    left out. In (a) and (b) our stars are reweighted so that each field has its share of the published sample.
(c) whole LMC within 12 deg: APOGEE as targeted (unweighted) and with its selection weights; BOSS weighted.
(d) the same BOSS giants with Gaia XP photometric metallicities: Omkumar et al. (2025) Stroemgren [Fe/H] and Andrae et al. (2023)
    [M/H] (to [Fe/H] with the alpha trend, as in paper_mass.py), BOSS weights; BOSS spectroscopic [Fe/H] of the same stars.
Scale check: stars of Cole+05 and Carrera+08 also in BOSS or APOGEE (within MATCH arcsec), median and robust sd of survey minus
Ca II triplet [Fe/H]. Histograms per dex of [Fe/H], unit area; medians in the legends. Writes results/mdf_literature.json and figures/paper_mdf_literature.png.

Inputs: lmcold.mass_samples.samples() (BOSS and APOGEE mass samples; BOSS also carries its XP [M/H], mh_xp);
data/giant_foreground.parquet (scripts/build_giant_mass.py: coverage fill and foreground fraction f_mw per BOSS giant);
results/mass_alpha_trend.csv (scripts/build_giant_mass.py); sources.yaml mdf_cole05, mdf_carrera08 (columns used: ra, dec,
sum_ca, Imag, rv) and mdf_omkumar25 (source_id, feh). SFD E(B-V) for the Carrera+08 stars comes from the `dustmaps` package
(SFDQuery); its map files are the data/sources.yaml entry sfd_dust_map, downloaded on first use.
Panels (2 x 2; x: [Fe/H] [dex], -2.5 to 0.6; y: fraction per dex, histograms in 0.1-dex bins normalised to unit area over the
plotted range): (a) Cole+05 bar fields with BOSS and APOGEE in the same fields; (b) Carrera+08 fields at 5, 6, 8 deg with
BOSS and APOGEE; (c) APOGEE unweighted and weighted, BOSS weighted, rho < 12 deg; (d) BOSS giants with XP [M/H]: Omkumar+25,
Andrae+23 (converted) and BOSS spectroscopic [Fe/H], all with the BOSS weights. Legends (above each panel) give the number of
stars and the weighted median.
Output JSON: scale_check (n, median and robust sd of survey minus literature [Fe/H] for matched stars) and, per panel and curve,
n, weighted median, and weighted fractions above -0.5 and below -1. Paper appendix E (Sect. E2; docs/figures.md).
Run: .venv/bin/python scripts/paper_mdf_literature.py
Run time: 2 s on a MacBook (M-series), docs/run_times.txt.
"""
import json
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
from dustmaps.std_paths import data_dir

from lmcold.constants import LMC, LMC_DISTANCE_MODULUS
from lmcold.extinction import rc_map, true_ebv
from lmcold.layout import panel_grid
from lmcold.maps import wquantile
from lmcold.mass_samples import samples
from lmcold.sources import DATA, ROOT

# %% [markdown]
# ### Settings

# %%
FIELD_R = .5  # deg, radius about each literature field centre within which our stars are taken
MATCH = 1.5  # arcsec, positional match radius for the scale check
EDGES = np.arange(-2.5, .61, .1)  # [Fe/H] histogram edges, 0.1 dex

# %% [markdown]
# ### Load the data
# The BOSS and APOGEE mass samples through `lmcold.mass_samples.samples()` (built from the catalogues of Zenodo bundle 1), the
# coverage fill and foreground fraction of each BOSS giant (`data/giant_foreground.parquet`, written by
# `scripts/build_giant_mass.py`), the published Ca II triplet samples of Cole et al. (2005) and Carrera et al. (2008)
# (`data/mdf_cole05.csv`, `data/mdf_carrera08.csv`) and the APOGEE [alpha/Fe] trend `results/mass_alpha_trend.csv`. Read where
# they are used: the SFD and Skowron et al. (2021) reddening maps (`dustmaps`, `lmcold.extinction.rc_map()`) for the Carrera
# stars, and the Omkumar et al. (2025) XP [Fe/H] (`data/mdf_omkumar25.parquet`), joined to the BOSS giants on Gaia source_id.

# %%
# lmcold.mass_samples.samples(): BOSS and APOGEE giants with calibrated [Fe/H], XP [M/H] and selection weight w
b, a, _ = samples()
# data/giant_foreground.parquet: per BOSS giant the coverage fill and the Milky Way foreground fraction f_mw
fg = pd.read_parquet(DATA / 'giant_foreground.parquet')
# data/mdf_cole05.csv: Cole et al. (2005) bar giants with [Fe/H]; field = name before '/'
cole = pd.read_csv(DATA / 'mdf_cole05.csv').assign(field=lambda t: t.Field.str.split('/').str[0])  # field name before '/'
# data/mdf_carrera08.csv: Carrera et al. (2008) disc giants (Ca II triplet sum_ca, I magnitude, rv), velocity members kept
car = pd.read_csv(DATA / 'mdf_carrera08.csv').query('170 <= rv <= 380')  # radial-velocity members [km/s], their criterion
# results/mass_alpha_trend.csv: the APOGEE [alpha/Fe] trend against [Fe/H]
trend = pd.read_csv(ROOT / 'results/mass_alpha_trend.csv')
# The SFD map (data/sources.yaml sfd_dust_map, two 64-MB files): downloaded into the dustmaps data folder if it is not there yet,
# from the copy at github.com/kbarbary/sfddata (dustmaps.sfd.fetch() fails: the Harvard Dataverse answers 403); both files have the
# sha256 recorded in sources.yaml (checked 2026-11-24)
for name in ('SFD_dust_4096_ngp.fits', 'SFD_dust_4096_sgp.fits'):
    sfd_file = Path(data_dir()) / 'sfd' / name
    if not sfd_file.exists():
        sfd_file.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(f'https://github.com/kbarbary/sfddata/raw/master/{name}', sfd_file)

# %% [markdown]
# ### Our giants
# BOSS and APOGEE giants with their weights.

# %%
# BOSS weight: selection weight x coverage fill x (1 - Milky Way foreground fraction); APOGEE: selection weight.
b = b.merge(fg[['source_id', 'fill', 'f_mw']], on='source_id', validate='one_to_one').assign(W=lambda t: t.w * t['fill'] * (1 - t.f_mw))
a = a.assign(W=a.w)
centre = SkyCoord(LMC['ra0'], LMC['dec0'], unit='deg')

# %% [markdown]
# ### The published samples
# The bar fields of Cole et al. (2005) and the disc fields of Carrera et al. (2008), with [Fe/H] from their equation (1).

# %%
pos = SkyCoord(car.ra, car.dec, unit='deg')
ebv = true_ebv(car.ra.to_numpy(), car.dec.to_numpy(), SFDQuery()(pos), rc_map())  # Skowron+21 map inside it, SFD outside
# Carrera+08 Eq. (1) with M_I from the photometric LMC distance modulus; field: 1, 2, 3 for projected distance 4-5.5, 5.5-7 and
# > 7 deg (their 5, 6 and 8 deg fields), 0 for the 3-deg field (< 4 deg), which is dropped.
car = car.assign(feh=-2.95 + .38 * car.sum_ca + .23 * (car.Imag - LMC_DISTANCE_MODULUS - 1.505 * ebv),
                 field=np.digitize(pos.separation(centre).deg, [4, 5.5, 7]))
car = car[car.field > 0]


def in_fields(lit, t):
    """Stars of t within FIELD_R of the centre of each field of lit, reweighted to the share of lit in that field.

    Parameters
    ----------
    lit : pandas.DataFrame
        Literature stars with ra, dec [deg] and `field`; a field's centre is the mean ra, dec of its stars.
    t : pandas.DataFrame
        Our sample, with ra, dec [deg] and weight W.

    Returns
    -------
    pandas.DataFrame
        Rows of t in the fields (a star near two field centres appears twice), with W rescaled so that the total weight in each
        field equals that field's fraction of the literature stars (total 1).
    """
    parts = []
    for f, d in lit.groupby('field'):
        # Separation [deg] of each of our stars from the field centre (mean position of the field's literature stars).
        sep = SkyCoord(t.ra, t.dec, unit='deg').separation(SkyCoord(d.ra.mean(), d.dec.mean(), unit='deg')).deg
        s = t[sep < FIELD_R]
        # Our weights in the field normalised to sum to the field's share of the literature sample.
        parts.append(s.assign(W=s.W / s.W.sum() * len(d) / len(lit)))
    return pd.concat(parts)


# %% [markdown]
# ### Gaia XP metallicities of the same BOSS giants

# %%
# Panel (d): BOSS giants with XP [M/H] (Andrae+23) and with Omkumar+25 [Fe/H] (inner join on Gaia source_id).
xp = b.dropna(subset=['mh_xp'])
# [Fe/H] = [M/H] - log10(0.638 x 10^[alpha/Fe] + 0.362) (Salaris1993 in refs.bib), alpha trend interpolated at [M/H].
xp = xp.assign(feh_xp=xp.mh_xp - np.log10(.638 * 10 ** np.interp(xp.mh_xp, trend.feh, trend.alpha) + .362))
xp = xp.merge(pd.read_parquet(DATA / 'mdf_omkumar25.parquet', columns=['source_id', 'feh']).rename(columns={'feh': 'feh_omk'}),
              on='source_id', validate='one_to_one')
# Line colours of the curves; one() gives unit weights to the unweighted samples.
blue, orange, purple, green = '#2878b5', '#d75a20', '#7b3294', '#1b9e77'
one = lambda t: np.ones(len(t))  # noqa: E731
# Per panel: title and curves (values, weights, colour, line style, label).
panels = [
    ('(a) Bar', [(cole.feh, one(cole), 'k', '-', 'Cole+05, Ca II triplet'),
                 (in_fields(cole, b).feh, in_fields(cole, b).W, blue, '-', 'BOSS, as measured'),
                 (in_fields(cole, a).feh, in_fields(cole, a).W, orange, '--', 'APOGEE')]),
    (r'(b) Fields 5, 6, 8$^\circ$ north', [(car.feh, one(car), 'k', '-', 'Carrera+08, Ca II triplet'),
                                       (in_fields(car, b).feh, in_fields(car, b).W, blue, '-', 'BOSS, as measured'),
                                       (in_fields(car, a).feh, in_fields(car, a).W, orange, '--', 'APOGEE')]),
    (r'(c) $\rho<12^\circ$, spectroscopic', [(a.feh, one(a), orange, ':', 'APOGEE, as targeted'),
                                            (a.feh, a.W, orange, '--', 'APOGEE, weighted'), (b.feh, b.W, blue, '-', 'BOSS, weighted, as measured')]),
    ('(d) Same BOSS giants, Gaia XP', [(xp.feh_omk, xp.W, green, '-', 'Omkumar+25, XP Strömgren'),
                                      (xp.feh_xp, xp.W, purple, '--', 'Andrae+23, XP'), (xp.feh, xp.W, blue, '-', 'BOSS, as measured')]),
]
# %% [markdown]
# ### Scale check
# Stars in both a published sample and ours: survey minus Ca II triplet [Fe/H].

# %%
# Scale check: each literature star matched to its nearest survey star; pairs within MATCH arcsec give survey - literature [Fe/H].
out = {'scale_check': {}}
for lit_name, lit in [('Cole+05', cole), ('Carrera+08', car)]:
    for name, t in [('BOSS', b), ('APOGEE', a)]:
        # i: index of the nearest survey star; d: its separation (angle).
        i, d, _ = SkyCoord(lit.ra, lit.dec, unit='deg').match_to_catalog_sky(SkyCoord(t.ra, t.dec, unit='deg'))
        m = d < MATCH * u.arcsec
        diff = t.feh.to_numpy()[i[m]] - lit.feh.to_numpy()[m]
        out['scale_check'][f'{name} - {lit_name}'] = dict(n=int(m.sum()), median=float(np.median(diff)) if m.any() else None,
                                                         robust_sd=float(1.4826 * np.median(np.abs(diff - np.median(diff)))) if m.any() else None)
# %% [markdown]
# ### The figure
# Histograms per panel; medians and fractions written to `results/mdf_literature.json`.

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
# 2 x 2 equal panels (sizes in inches); the large vertical gap holds the legends above the lower panels.
fig, axes = panel_grid(2, 2, width=7, left=.55, right=.1, hgap=.6, vgap=1.25, bottom=.5, top=.9, aspect=.75)
for ax, (title, curves) in zip(axes.flat, panels):
    out[title] = {}  # JSON entries of this panel, one per curve
    for x, w, colour, ls, lab in curves:
        x, w = np.asarray(x), np.asarray(w)  # [Fe/H] [dex] and weights of one curve
        h = np.histogram(x, EDGES, weights=w)[0]  # weighted counts per 0.1-dex bin
        med = wquantile(x, w, .5)  # weighted median [Fe/H]
        # Number of stars, weighted median, weighted fractions with [Fe/H] > -0.5 and < -1 (over all stars, not only EDGES).
        out[title][lab] = dict(n=len(x), median=float(med), frac_above_m05=float(w[x > -.5].sum() / w.sum()),
                               frac_below_m1=float(w[x < -1].sum() / w.sum()))
        # Histogram normalised to unit area over the plotted range: fraction per bin divided by the 0.1-dex bin width.
        ax.stairs(h / h.sum() / .1, EDGES, color=colour, ls=ls, lw=1.3, label=f'{lab} ({len(x):,}; median {med:.2f})')
    # Axes, panel title, and the legend above the panel.
    ax.set(xlim=(-2.5, .6), ylim=(0, None), xlabel='[Fe/H]', ylabel='fraction per dex', title=title)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.12), frameon=False, fontsize=6.5)
# Write the numbers, print them, save the figure.
(ROOT / 'results/mdf_literature.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
fig.savefig(ROOT / 'figures/paper_mdf_literature.png', dpi=200)
