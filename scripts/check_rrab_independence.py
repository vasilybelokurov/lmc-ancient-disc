"""Check of the capture-recapture assumption: do OGLE (own search) and Gaia miss RRab independently?

Within each radial ring inside the OGLE-IV footprint, the Lincoln-Petersen total N = n_O n_G / n_both is compared with the sum
of the estimates made separately in quartiles of a stellar property. If detection probabilities vary with that property in both
surveys (heterogeneity), the unstratified estimate is biased low and the stratified sum is larger. Properties: period (measured by
both surveys), mean magnitude (Gaia G where Gaia has the star, else OGLE I + 0.6), and amplitude (OGLE I-band amplitude where
available, else Gaia peak-to-peak G scaled to I by the median ratio on the stars with both). Writes results/rrab_independence.csv.

Input: data/rrab.parquet (release_rrab); OGLE-IV footprint from ogle4_fields_lmc / ogle4_fields_mbr (lmcold.ogle.in_footprint).
OGLE detections are ogle_own (OGLE's own discoveries, independent of Gaia), Gaia detections in_gaia.
Output columns per ring (rho_lo, rho_hi [deg]): n_union (stars found by either), N_lp (unstratified Lincoln-Petersen total),
N_lp_by_<P|mag|amp> (sum over quartiles), C_union and C_union_by_<prop> = n_union / estimate (completeness of the union).
Its C_union_by_mag column is the union completeness (green squares) of figures/paper_selection.png (a); the other columns give
the largest change from splitting by period or amplitude, quoted in the selection-function text (scripts/paper_text_data.py).

Run: .venv/bin/python scripts/check_rrab_independence.py   (after build_rrab.py)
"""
import numpy as np
import pandas as pd

from lmcold import ogle
from lmcold.sources import DATA, ROOT

# %% [markdown]
# ### Load the data
# The RR Lyrae catalogue `data/rrab.parquet` (`scripts/build_rrab.py`).

# %%
# One row per RRab: ra, dec [deg], rho [deg from the LMC centre], P [d], I, phot_g_mean_mag [mag], amp_I, peak_to_peak_g [mag],
# ogle_own, in_gaia (detection flags).
t = pd.read_parquet(DATA / "rrab.parquet")

# %% [markdown]
# ### RRab inside the OGLE-IV footprint
# Stars within 12 deg and inside the OGLE-IV fields, with a mean magnitude on the
# $G$ scale and an amplitude on the $I$ scale for every star.

# %%
t = t[t.rho < 12].copy()  # deg; the outer ring edge below
t = t[ogle.in_footprint(t.ra.to_numpy(), t.dec.to_numpy())]   # inside the OGLE-IV LMC and bridge fields
both = t.amp_I.notna() & t.peak_to_peak_g.notna()   # stars with both amplitudes, for the G-to-I amplitude ratio
# Mean magnitude on the G scale: Gaia G, else OGLE I + 0.6 mag (fixed offset adopted for RRab here).
t["mag"] = t.phot_g_mean_mag.astype(float).fillna(t.I + 0.6)
# Amplitude on the I scale [mag]: OGLE amp_I, else Gaia peak-to-peak G times the median amp_I / peak-to-peak G ratio.
t["amp"] = t.amp_I.fillna(t.peak_to_peak_g * np.median(t.amp_I[both] / t.peak_to_peak_g[both]))


# %% [markdown]
# ### The Lincoln-Petersen estimate
# $N=n_{\rm O}n_{\rm G}/n_{\rm OG}$ from OGLE's own detections and the Gaia detections.

# %%
def lp(s):
    """Lincoln-Petersen estimate n_O n_G / n_both of the total number of RRab in subsample s (NaN if n_both = 0).

    Parameters
    ----------
    s : pandas.DataFrame
        Rows with boolean columns ogle_own and in_gaia.

    Returns
    -------
    float
    """
    # Stars found by OGLE's own search, by Gaia, and by both.
    nO, nG, nB = s.ogle_own.sum(), s.in_gaia.sum(), (s.ogle_own & s.in_gaia).sum()
    # N = n_O n_G / n_both; NaN when no star is found by both (estimate undefined).
    return nO * nG / nB if nB else np.nan


# %% [markdown]
# ### Estimates per ring
# In each ring, the unstratified estimate and the sums over quartiles of period, magnitude and amplitude,
# and the completeness of the union; written to `results/rrab_independence.csv`.

# %%
rows = []   # one dict per ring
# Rings in rho [deg].
for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 6), (6, 12)):
    s = t[(t.rho >= lo) & (t.rho < hi)]
    # Stars found by either survey, and the unstratified estimate.
    row = dict(rho_lo=lo, rho_hi=hi, n_union=int((s.ogle_own | s.in_gaia).sum()), N_lp=lp(s))
    for prop in ("P", "mag", "amp"):
        q = pd.qcut(s[prop], 4, labels=False, duplicates="drop")  # quartile of the property within the ring
        # Sum of the estimates made separately in each quartile.
        row[f"N_lp_by_{prop}"] = sum(lp(s[q == k]) for k in range(int(q.max()) + 1))
    # Completeness of the union: n_union over each estimate.
    row.update({f"C_union_by_{p}": row["n_union"] / row[f"N_lp_by_{p}"] for p in ("P", "mag", "amp")}, C_union=row["n_union"] / row["N_lp"])
    rows.append(row)
r = pd.DataFrame(rows)   # one row per ring
r.to_csv(ROOT / "results" / "rrab_independence.csv", index=False, float_format="%.4f")
print(r[["rho_lo", "rho_hi", "n_union", "N_lp", "N_lp_by_P", "N_lp_by_mag", "N_lp_by_amp", "C_union", "C_union_by_P",
         "C_union_by_mag", "C_union_by_amp"]].to_string(index=False, float_format="%.3f"))
