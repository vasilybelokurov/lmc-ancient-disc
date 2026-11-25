"""Is the OGLE I minus Gaia G (Muraveva+25) [Fe/H] difference a constant shift, or does it depend on [Fe/H]?

On the RRab with both estimates: the difference against I, G and their mean (binning on one noisy estimate alone produces a slope),
and a Deming fit I = a + b G with equal errors in both (the spectra give similar per-star errors, about 0.2-0.25 dex), bootstrapped.
Writes figures/rrab_feh_transform.png and results/rrab_feh_transform.csv.

Diagnostic (docs/figures.md: "RRab [Fe/H] scales"); not a paper figure. scripts/paper_text_data.py takes the shift and the Deming
slope of the text from results/rrab_feh_transform.csv.

Input: data/rrab.parquet (sources.yaml key release_rrab): feh_I (OGLE I band) and feh_G_m25 (Gaia G band, Muraveva+25) [dex].
Stars whose I - G difference lies more than 4 robust SD (1.4826 x MAD) from its median are dropped first.

Figure (1 x 4 panels, matplotlib constrained layout): (1)-(3) I minus G [dex] against I-band [Fe/H], G-band [Fe/H] and their
mean [dex]: running median (line) and 16th-84th percentiles (band) in 0.15-dex bins with >= 30 stars, dashed line the overall
median shift; (4) normalised histograms of I-band [Fe/H], G plus the median shift, and G transformed by the Deming fit [dex].

results/rrab_feh_transform.csv (one row): N, clipped (stars dropped), shift (median I - G) [dex], deming_a [dex], deming_b and
their bootstrap SDs (500 resamplings of the stars).

Run: .venv/bin/python scripts/fig_rrab_feh_transform.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmcold.sources import DATA, ROOT


# %% [markdown]
# ### Deming regression
# Slope and intercept of a straight line for equal errors in both variables.

# %%
def deming(x, y):
    """Slope and intercept of y = a + b x for equal error variances in x and y.

    Parameters
    ----------
    x, y : numpy.ndarray
        Paired values with equal (unknown) error variances.

    Returns
    -------
    tuple of float
        (a, b): intercept (units of y) and slope. Note the order: intercept first.
    """
    # Population variances and covariance (ddof = 0).
    sxx, syy, sxy = np.var(x), np.var(y), np.cov(x, y, bias=True)[0, 1]
    # Deming slope for an error-variance ratio of 1 (orthogonal regression); the line passes through the means.
    b = (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy**2)) / (2 * sxy)
    return np.mean(y) - b * np.mean(x), b


# %% [markdown]
# ### Load the data
# The RR Lyrae catalogue `data/rrab.parquet` (`scripts/build_rrab.py`).

# %%
t = pd.read_parquet(DATA / "rrab.parquet")   # one row per RR Lyrae; feh_I, feh_G_m25 [dex]

# %% [markdown]
# ### The stars with both estimates
# Stars with $I$-band and $G$-band [Fe/H]; gross outliers of the difference are
# dropped.

# %%
ov = t[t.feh_I.notna() & t.feh_G_m25.notna()]       # stars with both estimates
d = ov.feh_I - ov.feh_G_m25                          # I minus G [dex]
mad = 1.4826 * np.median(np.abs(d - d.median()))     # robust SD of the difference [dex]
ov = ov[np.abs(d - d.median()) < 4 * mad]  # drop gross outliers (wrong period or phi31) before the moment-based fit

# %% [markdown]
# ### Shift and Deming fit
# The median difference and the Deming fit with its bootstrap errors, written to
# `results/rrab_feh_transform.csv`.

# %%
x, y = ov.feh_G_m25.to_numpy(), ov.feh_I.to_numpy()   # G-band and I-band [Fe/H] [dex], shape (N,)
a, b = deming(x, y)                                   # I = a + b G
# 500 bootstrap resamples of the stars (seed 2): shape (500, 2) of (a, b).
rng = np.random.default_rng(2)
boot = np.array([deming(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(500))])
shift = np.median(y - x)                              # median I - G [dex]
# One-row table: sample size, number clipped, shift, Deming a and b with bootstrap SDs.
res = pd.DataFrame([dict(N=len(x), clipped=len(t[t.feh_I.notna() & t.feh_G_m25.notna()]) - len(x), shift=shift,
                         deming_a=a, deming_a_err=boot[:, 0].std(), deming_b=b, deming_b_err=boot[:, 1].std())])
res.to_csv(ROOT / "results" / "rrab_feh_transform.csv", index=False, float_format="%.4f")
print(res.to_string(index=False, float_format="%.4f"))

# %% [markdown]
# ### The figure
# The difference against each estimate and their mean, and the distributions after each transform.

# %%
fig, ax = plt.subplots(1, 4, figsize=(17, 4.4), constrained_layout=True)
bins = np.arange(-2.6, -0.39, 0.15)   # 0.15-dex bins of [Fe/H], -2.6 to -0.5
for a_, (xv, xl) in zip(ax[:3], ((y, "[Fe/H] OGLE I"), (x, "[Fe/H] G, Muraveva+25"), ((x + y) / 2, "mean of the two"))):
    # Bin index of each star and bin centres; 16th, 50th, 84th percentiles of I - G in bins with >= 30 stars.
    i = np.digitize(xv, bins)
    xc = 0.5 * (bins[1:] + bins[:-1])
    q = np.array([np.percentile((y - x)[i == k], [16, 50, 84]) if (i == k).sum() >= 30 else [np.nan] * 3
                  for k in range(1, len(bins))])
    a_.fill_between(xc, q[:, 0], q[:, 2], color="#2a78d6", alpha=0.2, lw=0)
    a_.plot(xc, q[:, 1], color="#2a78d6", lw=2)
    a_.axhline(shift, color="0.3", lw=1, ls="--", label=f"median {shift:+.3f}")
    a_.set(xlabel=xl, ylabel="I minus G [dex]", ylim=(-0.8, 1.2), title=f"difference against {xl}")
ax[0].legend(frameon=False)
# Panel 4: normalised histograms (0.1-dex bins) of I, G + shift and G through the Deming line.
a_ = ax[3]
h = np.arange(-3.2, 0.21, 0.1)
a_.hist(y, h, histtype="step", lw=2, color="#2a78d6", density=True, label="OGLE I")
a_.hist(x + shift, h, histtype="step", lw=2, color="#eb6834", density=True, label=f"G {shift:+.2f}")
a_.hist(a + b * x, h, histtype="step", lw=2, color="#1baf7a", density=True, label=f"G Deming: {a:+.2f} + {b:.2f} G")
a_.set(xlabel="[Fe/H]", ylabel="density", title="I against transformed G (same stars)")
a_.legend(frameon=False, fontsize=8)
fig.savefig(ROOT / "figures" / "rrab_feh_transform.png", dpi=150)
