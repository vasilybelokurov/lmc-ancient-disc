"""Paper figure: RRab photometric metallicities. (a) OGLE I-band against Gaia G-band (Muraveva+25) [Fe/H]; (b) their difference
against their mean, with the adopted constant shift and the Deming slope; (c) both against the spectroscopy of Gratton et al.
(2004), on the C21 scale, for the stars with all three. Reads data/rrab.parquet, results/rrab_feh_transform.csv and the
spectroscopic inputs (lmcold.spectra); writes figures/paper_rrab_feh.png.

Inputs: data/rrab.parquet (sources.yaml release_rrab): feh_I (OGLE I band, Crestani et al. 2021 scale) and feh_G_m25 (Gaia G
band, Muraveva et al. 2025 relation) [dex]; results/rrab_feh_transform.csv (scripts/fig_rrab_feh_transform.py): the median I - G
shift (column shift [dex]) and the Deming fit I = a + b G (deming_b, deming_b_err); the Gratton et al. (2004) spectroscopic
[Fe/H] (sources.yaml spec_gratton04 and the Di Fabrizio et al. 2005 positions, read by lmcold.spectra.read_lmc_rrl_spectra),
converted from the Harris (1996) ~ Zinn & West scale to the C21 scale with lmcold.rrlyrae.zw_to_c21 and matched to the RRab
within 2 arcsec.
Panels (1 x 3, square):
(a) x: [Fe/H]_G [dex], y: [Fe/H]_I [dex], both -3.4 to 0.2; 2D histogram (0.05 dex, log colour) of the stars with both; dotted
    one-to-one line; orange: the adopted constant shift.
(b) x: mean of the two [Fe/H] [dex], -3 to 0; y: I - G [dex], -1.5 to 1.9; 2D histogram; black: running median (solid) and
    16th/84th percentiles (dashed) in 0.15-dex bins with >= 30 stars; orange: the shift; the Deming slope is quoted above.
(c) x: spectroscopic [Fe/H] (C21 scale) [dex], -2.4 to -0.4; y: photometric [Fe/H] [dex]; blue: OGLE I, orange: Gaia G + shift;
    legend: median offset (photometric minus spectroscopic) and robust sd (1.4826 x MAD) [dex].
Placement: appendix (Sect. 2.1; docs/figures.md). Run: .venv/bin/python scripts/paper_rrab_feh.py   (about 2 s on a laptop, docs/run_times.txt)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from matplotlib.colors import LogNorm

from lmcold.layout import panel_grid
from lmcold.rrlyrae import zw_to_c21
from lmcold.sources import DATA, ROOT
from lmcold.spectra import read_lmc_rrl_spectra

plt.style.use(ROOT / "src" / "lmcold" / "paper.mplstyle")
# colours of the OGLE I and Gaia G points
BLUE, ORANGE = "#2a78d6", "#eb6834"

# %% [markdown]
# ### Load the data
# The RRab catalogue `data/rrab.parquet` (Zenodo bundle 1), the I - G shift and Deming fit from `results/rrab_feh_transform.csv`
# (written by `scripts/fig_rrab_feh_transform.py`), and the published RR Lyrae spectroscopy through
# `lmcold.spectra.read_lmc_rrl_spectra()` (the VizieR tables under `data/raw/spec/`).

# %%
# data/rrab.parquet: RRab with feh_I (OGLE I band) and feh_G_m25 (Gaia G band) [dex]
t = pd.read_parquet(DATA / "rrab.parquet")
# results/rrab_feh_transform.csv: median I - G shift [dex] and Deming slope deming_b, deming_b_err (one row)
tr = pd.read_csv(ROOT / "results" / "rrab_feh_transform.csv").iloc[0]
# lmcold.spectra.read_lmc_rrl_spectra(): spectroscopic [Fe/H] of LMC RR Lyrae with positions and source study (src)
g = read_lmc_rrl_spectra()

# %% [markdown]
# ### The two photometric scales and the spectroscopy
# RRab with both $I$- and $G$-band [Fe/H]; the Gratton et al. (2004) stars matched within 2 arcsec, on the
# Crestani et al. (2021) scale.

# %%
ov = t[t.feh_I.notna() & t.feh_G_m25.notna()]  # RRab with both photometric estimates
# mean and difference I - G of the two estimates [dex], (N,)
mean, diff = 0.5 * (ov.feh_I + ov.feh_G_m25), ov.feh_I - ov.feh_G_m25

g = g[g.src == "G04"]  # Gratton et al. (2004) only
# positions of the spectroscopic stars [deg]
cg = SkyCoord(g.ra, g.dec, unit="deg")
# Nearest RRab to each spectroscopic star; G is put on the I scale with the constant shift.
# j: index of the nearest RRab; sep: separation (angle)
j, sep, _ = cg.match_to_catalog_sky(SkyCoord(t.ra, t.dec, unit="deg"))
# spec: spectroscopic [Fe/H] converted to the C21 scale; I, G: photometric [Fe/H] of the matched RRab [dex]
s = pd.DataFrame(dict(spec=zw_to_c21(g.feh.to_numpy()), I=t.feh_I.to_numpy()[j], G=t.feh_G_m25.to_numpy()[j] + tr["shift"]))
# matches within 2 arcsec with all three [Fe/H] values
s = s[(sep.arcsec < 2) & s.I.notna() & s.G.notna() & s.spec.notna()]

# %% [markdown]
# ### Panel (a): $I$-band against $G$-band [Fe/H]

# %%
# one row of three square panels
fig, (ax,) = panel_grid(1, 3)
a = ax[0]
# 2D histogram of the stars with both estimates, 0.05-dex bins, log colour scale
h = a.hist2d(ov.feh_G_m25, ov.feh_I, bins=np.arange(-3.4, 0.21, 0.05), norm=LogNorm(), cmap="Blues", rasterized=True)
# dotted one-to-one line and the adopted shift (orange), over the axis range [dex]
x = np.array([-3.4, 0.2])
a.plot(x, x, color="0.4", lw=0.8, ls=":")
a.plot(x, x + tr["shift"], color=ORANGE, lw=1, label=f"$+{tr['shift']:.2f}$ dex")
a.set(xlabel=r"[Fe/H]$_G$ (Muraveva et al. 2025)", ylabel=r"[Fe/H]$_I$ (OGLE)", aspect="equal", xlim=(-3.4, 0.2), ylim=(-3.4, 0.2))
# legend above the panel, right-aligned
a.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.1)
# panel letter and number of stars above the panel
a.text(0, 1.03, f"(a)  N = {len(ov)}", transform=a.transAxes)

# %% [markdown]
# ### Panel (b): their difference against their mean
# Running median and 16th/84th percentiles, the adopted shift and the Deming slope.

# %%
a = ax[1]
# 2D histogram of (mean, difference): 0.05-dex bins in the mean, 0.04-dex bins in the difference
a.hist2d(mean, diff, bins=[np.arange(-3.0, 0.01, 0.05), np.arange(-1.5, 1.91, 0.04)], norm=LogNorm(), cmap="Blues", rasterized=True)
b = np.arange(-2.6, -0.49, 0.15)  # bins in the mean [Fe/H] [dex] for the running percentiles
# bin index of each star
i = np.digitize(mean, b)
# 16th, 50th, 84th percentiles of I - G in bins with at least 30 stars (NaN otherwise)
q = np.array([np.percentile(diff[i == k], [16, 50, 84]) if (i == k).sum() >= 30 else [np.nan] * 3 for k in range(1, len(b))])
# bin centres [dex]
xc = 0.5 * (b[1:] + b[:-1])
# running median (solid) and 16th/84th percentiles (dashed)
a.plot(xc, q[:, 1], color="k", lw=1)
a.plot(xc, q[:, 0], color="k", lw=0.6, ls="--")
a.plot(xc, q[:, 2], color="k", lw=0.6, ls="--")
# the adopted constant shift
a.axhline(tr["shift"], color=ORANGE, lw=1)
a.set(xlabel=r"([Fe/H]$_I$ + [Fe/H]$_G$)/2", ylabel=r"[Fe/H]$_I$ $-$ [Fe/H]$_G$", xlim=(-3.0, 0.0), ylim=(-1.5, 1.9))
# the Deming slope of I against G (from fig_rrab_feh_transform.py), above the panel
a.text(1, 1.03, f"Deming slope ${tr['deming_b']:.3f} \\pm {tr['deming_b_err']:.3f}$", transform=a.transAxes, ha="right")
a.text(0, 1.03, "(b)", transform=a.transAxes)

# %% [markdown]
# ### Panel (c): against the spectroscopy

# %%
a = ax[2]
# OGLE I (blue) and shifted Gaia G (orange) against the spectroscopic [Fe/H]
for k, col, lab in (("I", BLUE, "OGLE $I$"), ("G", ORANGE, f"Gaia $G + {tr['shift']:.2f}$")):
    # photometric minus spectroscopic [dex]
    d = s[k] - s.spec
    # legend label: median offset and robust sd (1.4826 x median absolute deviation) [dex]
    a.scatter(s.spec, s[k], s=7, color=col, edgecolor="white", lw=0.3,
              label=f"{lab}: {d.median():+.2f}, {1.4826 * (d - d.median()).abs().median():.2f}")
# dotted one-to-one line
x = np.array([-2.4, -0.4])
a.plot(x, x, color="0.4", lw=0.8, ls=":")
a.set(xlabel=r"[Fe/H]$_{\rm spec}$ (Gratton et al. 2004, C21 scale)", ylabel=r"[Fe/H]$_{\rm phot}$", xlim=(-2.4, -0.4), ylim=(-2.6, -0.2))
# legend above the panel; its title gives the number of matched stars
a.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), handletextpad=0.1, borderaxespad=0.1,
         title=f"{len(s)} stars: offset, $\\sigma$", title_fontsize=7, alignment="right")
a.text(0, 1.03, "(c)", transform=a.transAxes)

fig.savefig(ROOT / "figures" / "paper_rrab_feh.png")
