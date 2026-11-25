"""Paper table (Sect. 2.3): formed and present mass per [Fe/H] bin and below each upper edge, with bootstrap percentiles and model
ranges. Reads results/giant_present_mass.csv (build_giant_mass.py, build_present_mass.py); writes results/table_mass.tex.

One LaTeX row per reporting [Fe/H] bin (lmcold.mass_samples.FEH_EDGES, open ends shown as < -2 and >= 0), with 10 columns:
bin; formed mass and its model range; present mass (living stars and remnants) and its model range; upper edge of the bin;
cumulative formed mass below that edge and its range; cumulative present mass and its range. Masses in 10^6 Msun, as
value^{+(p84 - value)}_{-(value - p16)} (16th-84th percentiles of the pixel and response bootstrap) and [model min, max].
Run: .venv/bin/python scripts/paper_table_mass.py
"""
import numpy as np
import pandas as pd

from lmcold.sources import ROOT

# %% [markdown]
# ### Load the data
# Formed and present mass per reporting [Fe/H] bin, with bootstrap percentiles and model ranges.

# %%
t = pd.read_csv(ROOT / "results/giant_present_mass.csv")   # one row per bin; masses in Msun


# %% [markdown]
# ### Cell format

# %%
def cell(v, lo, hi, mlo, mhi):
    """value_{-lo}^{+hi} [model range], in 1e6 Msun.

    Parameters
    ----------
    v, lo, hi : float
        Value and its 16th and 84th bootstrap percentiles [Msun].
    mlo, mhi : float
        Minimum and maximum over the model alternatives [Msun].

    Returns
    -------
    str
        Two LaTeX cells ('value with errors & [min, max]') in 10^6 Msun, with 2, 1 or 0 decimals for values below 10, below 100
        and above.
    """
    # all inputs from Msun to 10^6 Msun
    v, lo, hi, mlo, mhi = (x / 1e6 for x in (v, lo, hi, mlo, mhi))
    # decimals: 2 below 10, 1 below 100, 0 above
    d = 2 if v < 10 else (1 if v < 100 else 0)
    # format string with d decimals, e.g. '{:.2f}'
    f = f"{{:.{d}f}}"
    # value with the upper (p84 - v) and lower (v - p16) errors, then the model range in brackets
    return f"${f.format(v)}^{{+{f.format(hi - v)}}}_{{-{f.format(v - lo)}}}$ & [{f.format(mlo)}, {f.format(mhi)}]"


# %% [markdown]
# ### Write the rows
# One row per bin: the bin, its formed and present mass, its upper edge, and the cumulative masses below that edge.

# %%
# one LaTeX row (string) per reporting [Fe/H] bin
rows = []
for r in t.itertuples():
    # bin label: open lower edge (feh_lo = -inf) -> '< -2', open upper edge (feh_hi = +inf) -> '>= 0', else 'lo to hi' [dex]
    lo = "$<-2$" if not np.isfinite(r.feh_lo) else (r"$\geq 0$" if not np.isfinite(r.feh_hi) else f"${r.feh_lo:.2f}$ to ${r.feh_hi:.2f}$")
    # upper edge of the bin [dex], or 'all' for the last (open) bin
    up = "all" if not np.isfinite(r.feh_hi) else f"${r.feh_hi:.2f}$"
    # columns: bin, formed mass, present mass, upper edge, cumulative formed mass, cumulative present mass (with ranges)
    rows.append(" & ".join([lo, cell(r.formed_mass, r.mass_p16, r.mass_p84, r.mass_model_min, r.mass_model_max),
                            cell(r.present_mass, r.present_mass_p16, r.present_mass_p84, r.present_mass_model_min, r.present_mass_model_max),
                            up, cell(r.cumulative, r.cumulative_p16, r.cumulative_p84, r.cumulative_model_min, r.cumulative_model_max),
                            cell(r.present_cumulative, r.present_cumulative_p16, r.present_cumulative_p84, r.present_cumulative_model_min,
                                 r.present_cumulative_model_max)]) + r" \\")
# the table body only; the table environment and header are in the paper source
(ROOT / "results/table_mass.tex").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
