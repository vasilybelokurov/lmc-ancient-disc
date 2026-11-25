"""BOSS metallicity response on the APOGEE scale, measured on the stars with both spectra, and its closure test (§2.3).

Overlap: the mass samples of both surveys (lmcold.mass_samples), same Gaia source; APOGEE [Fe/H] is the reference (true) scale.
The calibrated BOSS values are biased by -0.15 to -0.2 dex at -2 < [Fe/H] < -1.5 and unbiased on either side, and BOSSNet errors
shrink towards the bulk of its training set, so the response is measured directly (lmcold.mass_response.empirical_response),
not modelled as independent Gaussian errors (that model predicts a broadened MDF peak, which BOSS does not show).
G: the overlap is brighter than the BOSS sample, so its stars are weighted to the G distribution of the target sample
(lmcold.mass_samples.boss_response).
Closure: common parent (Gaia IDs in covered BOSS and APOGEE parents, as in paper_giant_counts). Predicted BOSS counts
R @ (APOGEE counts in the true bins) against the observed BOSS counts, in the reporting bins and below -2 and -1.5; errors
from 500 joint bootstraps of nside-16 pixels (both surveys, weights fixed) and overlap stars. Also by APOGEE field type
(LMC1-17 used the Washington + DDO51 giant flag, which is not reproduced; apogee_selection T7) and with the response measured
on the overlap stars of one half of the sky (nside-8 pixel parity) and tested on the other. The overlap stars are themselves
part of the common parent, so the full-sky closure is not independent of the response; the split-sky test is.
Writes results/boss_response.json and figures/paper_boss_response.png (appendix: overlap residuals; closure by field type).
The common-parent closure is panel (b) of figures/paper_mass_method.png.

Inputs (through lmcold.mass_samples.samples): data/boss_weights.parquet (release_boss_weights), data/apogee_weights.parquet
(release_apogee_weights), data/spec_giants.parquet and results/boss_calibration.json (scripts/build_giants.py); here also
apogee_weights.parquet for the APOGEE field of each common-parent star. Both samples: good giant spectra with positive weight
and 150 < v_los < 420 km/s (APOGEE also quality and covered).
Response matrix R: column j = distribution of calibrated BOSS [Fe/H] over the 0.125-dex measured bins (MEASURED_EDGES) for
overlap stars with APOGEE [Fe/H] in true bin j (TRUE_EDGES: < -2, -2..-1.5, -1.5..-1, > -1). Predicted BOSS counts
= R @ (weighted APOGEE counts per true bin), summed into the reporting bins (REPORT_EDGES); chi2 uses the bootstrap covariance
of observed minus predicted. results/boss_response.json: overlap size, median BOSS - APOGEE per reporting bin, the bin edges
and, per closure case, chi2, dof, observed / predicted / APOGEE counts per reporting bin, the 16th and 84th percentiles of the
bootstrap ratio, and the totals below -2 and -1.5.

Run: .venv/bin/python scripts/boss_response.py   (after boss_selection.py, apogee_selection.py, build_giants.py)
"""
import json

import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmcold.layout import panel_grid
from lmcold.mass_samples import samples, boss_response, overlap, FEH_CENTRES, FEH_EDGES, TRUE_EDGES, MEASURED_EDGES
from lmcold.sources import DATA, ROOT

# %% [markdown]
# ### Settings and helpers
# Number of joint bootstrap draws, the reporting [Fe/H] bins, and two helpers: the response matrix for a target sample
# (`lmcold.mass_samples.boss_response`) and the sum of the fine measured bins into the reporting bins.

# %%
NBOOT = 500  # joint bootstrap draws
# Reporting bins [dex]: FEH_EDGES with the open ends closed at -3 and +0.5 (centres FEH_CENTRES).
REPORT_EDGES = [-3., *FEH_EDGES[1:-1], .5]
# Reporting bin of each measured bin, by its upper edge; the last reporting bin also takes any measured bin beyond it.
GROUP = np.minimum(np.searchsorted(REPORT_EDGES, MEASURED_EDGES[1:]) - 1, len(FEH_CENTRES) - 1)  # reporting bin per measured bin


def response(o, t, w):
    """BOSS response matrix for target sample t, measured on overlap stars o.

    Parameters
    ----------
    o : pandas.DataFrame
        Overlap stars (lmcold.mass_samples.overlap: feh_b, feh_a, g, snr).
    t : pandas.DataFrame
        Target BOSS stars (feh [dex, calibrated], g [mag], snr).
    w : numpy.ndarray
        Selection weights of t.

    Returns
    -------
    numpy.ndarray
        Shape (len(MEASURED_EDGES) - 1, len(TRUE_EDGES) - 1); each column sums to 1.
    """
    # lmcold.mass_samples.boss_response weights the overlap stars to the G distribution of the target (module docstring).
    # Arguments: overlap stars, target [Fe/H] [dex], G [mag], S/N, and weights.
    return boss_response(o, t.feh, t.g, t.snr, w)


def coarse(x):
    """Measured-bin counts summed into the reporting bins (last axis).

    Parameters
    ----------
    x : numpy.ndarray
        Counts with the measured bins on the last axis.

    Returns
    -------
    numpy.ndarray
        Same leading shape, last axis len(FEH_CENTRES).
    """
    # For each reporting bin i, sum the measured bins whose GROUP is i; stack them on a new last axis.
    return np.stack([x[..., GROUP == i].sum(-1) for i in range(len(FEH_CENTRES))], -1)


# %% [markdown]
# ### Load the data
# The BOSS and APOGEE mass samples and their common parent through `lmcold.mass_samples.samples()` (built from the catalogues of
# Zenodo bundle 1), and the APOGEE field of each covered candidate from `data/apogee_weights.parquet`.

# %%
# lmcold.mass_samples.samples(): BOSS and APOGEE giants (calibrated [Fe/H], g, snr, weight w) and the source_id of the common parent
b, a, common = samples()
# data/apogee_weights.parquet: APOGEE field name ('LMC<n>') of each covered candidate, indexed by source_id
field = pd.read_parquet(DATA / "apogee_weights.parquet").query("covered").set_index("source_id").field

# %% [markdown]
# ### Stars observed by both surveys
# Calibrated BOSS minus APOGEE [Fe/H] of the stars with both spectra, and its median per reporting bin of APOGEE [Fe/H]
# (panel a).

# %%
# Overlap residuals per reporting bin of APOGEE [Fe/H] (panel a, and `bias` in the json).
o = overlap(b, a)   # stars in both samples (same Gaia source): feh_b (BOSS), feh_a (APOGEE) [dex], g, snr, ra, dec
d = (o.feh_b - o.feh_a).to_numpy()   # (N,) BOSS - APOGEE [Fe/H] [dex]
k = np.digitize(o.feh_a, REPORT_EDGES) - 1   # reporting bin of each star's APOGEE [Fe/H]
# Number of overlap stars and median residual per reporting bin (NaN for an empty bin).
bias = pd.DataFrame(dict(feh=FEH_CENTRES, n=[int(np.sum(k == i)) for i in range(len(FEH_CENTRES))],
                         median=[float(np.median(d[k == i])) if np.any(k == i) else np.nan for i in range(len(FEH_CENTRES))]))
print(bias.round(3).to_string(index=False))

# %% [markdown]
# ### The common parent and the closure test
# Both samples restricted to the parent common to the two surveys, with the APOGEE fields LMC1-17 flagged; the bootstrap draws
# of nside-16 sky pixels; and the closure test: observed BOSS counts against the APOGEE counts passed through the response.

# %%
# Restrict both samples to the common parent; wd: APOGEE field LMC1-17 (field names 'LMC<n>'), Washington + DDO51 targeting.
bc, ac = b[b.source_id.isin(common)].copy(), a[a.source_id.isin(common)].copy()
bc["wd"], ac["wd"] = bc.source_id.map(field).str[3:].astype(int) <= 17, ac.field.str[3:].astype(int) <= 17
# nside-16 nested HEALPix pixel of each star (the bootstrap resamples these pixels).
pix_b = hp.ang2pix(16, bc.ra.to_numpy(), bc.dec.to_numpy(), lonlat=True, nest=True)
pix_a = hp.ang2pix(16, ac.ra.to_numpy(), ac.dec.to_numpy(), lonlat=True, nest=True)
# Bootstrap: each row of draws gives how many times each nside-16 pixel (of either sample) is drawn.
upix = np.unique(np.concatenate([pix_b, pix_a]))   # sorted pixels occupied by either sample
rng = np.random.default_rng(20261027)   # fixed seed
# (NBOOT, len(upix)): multinomial draw of len(upix) pixels with equal probability, i.e. resampling with replacement.
draws = rng.multinomial(len(upix), np.ones(len(upix)) / len(upix), size=NBOOT)


def per_pixel(t, pix, edges, sel):
    """Weighted counts per pixel and [Fe/H] bin.

    Parameters
    ----------
    t : pandas.DataFrame
        Sample with columns feh [dex] and w (selection weight).
    pix : numpy.ndarray
        nside-16 nested pixel of each row of t.
    edges : array_like
        [Fe/H] bin edges; values are clipped into [-2.999, 0.499] first, so the end bins collect the tails.
    sel : numpy.ndarray of bool
        Rows to use.

    Returns
    -------
    numpy.ndarray
        Shape (len(upix), len(edges) - 1).
    """
    m = np.zeros((len(upix), len(edges) - 1))
    # [Fe/H] bin of each selected star, then its weight added to its (pixel, bin) cell.
    j = np.digitize(np.clip(t.feh.to_numpy()[sel], -2.999, 0.499), edges) - 1
    np.add.at(m, (np.searchsorted(upix, pix[sel]), j), t.w.to_numpy()[sel])
    return m


def closure(sel_b, sel_a, o_fit):
    """Observed and predicted BOSS counts in the reporting bins, with joint bootstrap draws (pixels and overlap stars).

    Parameters
    ----------
    sel_b, sel_a : numpy.ndarray of bool
        Rows of the common-parent BOSS (bc) and APOGEE (ac) samples to use.
    o_fit : pandas.DataFrame
        Overlap stars the response is measured on.

    Returns
    -------
    dict
        chi2 and dof over bins with >= 1 weighted star in APOGEE or BOSS; per reporting bin the observed, predicted and APOGEE
        counts and the 16th/84th percentiles of observed / predicted over the draws; below_-2.0 and below_-1.5 totals.
    """
    # Per-pixel weighted counts: BOSS in the fine measured bins, APOGEE in the true bins; APOGEE in the reporting bins (all sky).
    mb, ma = per_pixel(bc, pix_b, MEASURED_EDGES, sel_b), per_pixel(ac, pix_a, TRUE_EDGES, sel_a)
    direct = per_pixel(ac, pix_a, REPORT_EDGES, sel_a).sum(0)
    t, w = bc[sel_b], bc.w.to_numpy()[sel_b]   # BOSS target sample and its weights, for the G weighting of the response
    # Observed BOSS counts, and the prediction R @ (APOGEE counts per true bin), both in the reporting bins.
    obs, pred = coarse(mb.sum(0)), coarse(response(o_fit, t, w) @ ma.sum(0))
    dobs = coarse(draws @ mb)  # (NBOOT, reporting bins)
    # Each draw also resamples the overlap stars with replacement before measuring the response.
    dpred = np.array([coarse(response(o_fit.iloc[rng.integers(0, len(o_fit), len(o_fit))], t, w) @ (dr @ ma)) for dr in draws])
    use = (direct >= 1) | (obs >= 1)   # bins entering chi2
    # Bootstrap covariance of observed - predicted; chi2 = r^T C^-1 r over the used bins.
    cov = np.cov((dobs - dpred)[:, use].T)
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = dobs / dpred   # (NBOOT, reporting bins) observed / predicted per draw
    res = dict(chi2=float((obs - pred)[use] @ np.linalg.solve(cov, (obs - pred)[use])), dof=int(use.sum()),
               observed=obs.tolist(), predicted=pred.tolist(), apogee=direct.tolist(),
               ratio_p16=np.nanpercentile(rb, 16, axis=0).tolist(), ratio_p84=np.nanpercentile(rb, 84, axis=0).tolist())
    for x, n in ((-2.0, 1), (-1.5, 3)):  # the first n reporting bins lie below x
        rat = dobs[:, :n].sum(1) / dpred[:, :n].sum(1)   # (NBOOT,) ratio of the totals below x
        res[f"below_{x}"] = dict(observed=float(obs[:n].sum()), predicted=float(pred[:n].sum()), apogee=float(direct[:n].sum()),
                                 ratio=float(obs[:n].sum() / pred[:n].sum()), ratio_p16_p84=np.percentile(rat, [16, 84]).tolist())
    return res


def half(t):
    """Sky half of each row of t: parity (0 or 1) of its nside-8 nested HEALPix pixel."""
    return hp.ang2pix(8, t.ra.to_numpy(), t.dec.to_numpy(), lonlat=True, nest=True) % 2


# %% [markdown]
# ### Closure on all stars, by field type and across the sky
# All stars, LMC1-17 and LMC18-35 separately, and the response measured on one half of the sky and tested on the other;
# written to `results/boss_response.json`.

# %%
every_b, every_a = np.ones(len(bc), bool), np.ones(len(ac), bool)   # all rows of the common-parent samples
# Summary for the json: overlap size, residuals per bin, bin edges, and closure on all stars and per APOGEE field type.
out = dict(overlap=len(o), bias=bias.to_dict("list"), true_edges=TRUE_EDGES, measured_edges=MEASURED_EDGES.tolist(),
           closure={"all": closure(every_b, every_a, o),
                    "LMC1-17 (W+D)": closure(bc.wd.to_numpy(), ac.wd.to_numpy(), o),
                    "LMC18-35": closure(~bc.wd.to_numpy(), ~ac.wd.to_numpy(), o)})
# Split-sky test: response from the overlap stars of half h, closure on the stars of the other half.
for h in (0, 1):
    out["closure"][f"response from sky half {h}, tested on {1 - h}"] = closure(half(bc) == 1 - h, half(ac) == 1 - h, o[half(o) == h])
(ROOT / "results/boss_response.json").write_text(json.dumps(out, indent=1))
# Print chi2 and the totals below [Fe/H] = -2 and -1.5 of each closure case.
for name, c in out["closure"].items():
    print(f"{name}: chi2 {c['chi2']:.1f}/{c['dof']}; " + "; ".join(
        f"{x}: BOSS {c[x]['observed']:.1f} predicted {c[x]['predicted']:.1f} APOGEE {c[x]['apogee']:.1f} ratio {c[x]['ratio']:.2f} "
        f"[{c[x]['ratio_p16_p84'][0]:.2f}, {c[x]['ratio_p16_p84'][1]:.2f}]" for x in ("below_-2.0", "below_-1.5")))

# %% [markdown]
# ### The figure

# %%
# Figure: (a) overlap residuals against APOGEE [Fe/H]; (b) observed / predicted BOSS counts by APOGEE field type, bins with
# >= 3 weighted APOGEE stars.
plt.style.use(ROOT / "src/lmcold/paper.mplstyle")
fig, ax = panel_grid(1, 2, width=5.0, left=0.6, right=0.1, hgap=0.6, bottom=0.45, top=0.6)
blue, orange, green, grey = "#2878b5", "#d75a20", "#33976b", "0.6"
# Panel (a): every overlap star, and the median residual of bins with at least 3 stars.
a0 = ax[0, 0]
a0.plot(o.feh_a, d, ".", ms=1, color=grey, rasterized=True)
ok = bias.n >= 3
a0.plot(bias.feh[ok], bias["median"][ok], "o-", ms=3, color=blue, lw=0.8, label="median per bin")
a0.axhline(0, color="0.3", lw=0.6, ls="--")
a0.set(xlim=(-2.8, 0), ylim=(-0.8, 0.8), xlabel=r"[Fe/H]$_{\rm APOGEE}$", ylabel=r"[Fe/H]$_{\rm BOSS}-$[Fe/H]$_{\rm APOGEE}$",
       title="(a) Stars with both spectra")
a0.legend(loc="lower left", bbox_to_anchor=(0, 1.12), borderaxespad=0)   # legend above the panel
# Panel (b): observed / predicted per reporting bin with its 16-84th percentile range, field types offset by +-0.03 dex.
a1 = ax[0, 1]
for name, lab, col, off in (("LMC1-17 (W+D)", "LMC1–17 (W+D)", orange, -0.03), ("LMC18-35", "LMC18–35", green, 0.03)):
    c = out["closure"][name]
    obs, ref = np.array(c["observed"]), np.array(c["predicted"])
    good = (np.array(c["apogee"]) >= 3) & (ref > 0)   # bins with >= 3 weighted APOGEE stars
    a1.vlines(FEH_CENTRES[good] + off, np.array(c["ratio_p16"])[good], np.array(c["ratio_p84"])[good], color=col, lw=0.8)
    a1.plot(FEH_CENTRES[good] + off, obs[good] / ref[good], "o", ms=3, color=col, label=lab)
a1.axhline(1, color="0.3", lw=0.6, ls="--")
a1.set(xlim=(-2.45, -0.1), ylim=(0, 3.2), xlabel="[Fe/H]", xticks=[-2.25, -1.5, -1, -0.5], xticklabels=["<−2", "−1.5", "−1", "−0.5"],
       ylabel="BOSS observed / expected", title="(b) By APOGEE field type")
a1.legend(loc="lower left", bbox_to_anchor=(0, 1.12), borderaxespad=0, ncol=2)
fig.savefig(ROOT / "figures/paper_boss_response.png")
