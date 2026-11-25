"""Paper table (Sect. 2.3): summary of the LMC stellar mass, this work and published estimates.

This work: formed, present and living mass within 12 deg, and formed and present mass below [Fe/H] = -2, -1.5 and -1, each with the
16th-84th bootstrap percentiles and the range of the model alternatives (results/giant_present_mass.csv from build_giant_mass.py
and build_present_mass.py), and the fraction of formed mass above -0.5 (bootstrap: results/giant_mass_bootstrap.csv; models:
results/giant_mass_variants.csv). Published: lmcold.constants.LMC_MASS_LITERATURE, each as published (formed or present), with its
aperture, method and initial mass function; their source files are checked against data/sources.yaml.
Writes results/table_mass_summary.tex (table rows).

Inputs: results/giant_present_mass.csv (one row per reporting [Fe/H] bin; formed, present and living masses [Msun], their
cumulative sums, bootstrap _p16/_p84 and model _model_min/_model_max), results/giant_mass_bootstrap.csv (formed mass per bin
[Msun] for each bootstrap draw, columns = bin centres), results/giant_mass_variants.csv (formed mass per bin for each model
alternative: conversion, prior, alpha_model, foreground, filled). Published values: lmcold.constants.LMC_MASS_LITERATURE; each
source file is checked through lmcold.sources.path('lmc_mass_reference_<key>'), which raises if its sha256 differs.
Output rows (five LaTeX columns: label & formed & present & aperture/method & IMF), mass cells as
$(v^{+}_{-}) x 10^n$ [model min--max] in Msun; the metal-rich fraction in per cent. Paper Table 3 (Sect. 2.3).
Run: .venv/bin/python scripts/paper_table_mass_summary.py
"""
import numpy as np
import pandas as pd
from lmcold.constants import LMC_MASS_LITERATURE
from lmcold.sources import ROOT, path

# %% [markdown]
# ### Load the data
# Each published source is checked against data/sources.yaml; the masses per [Fe/H] bin come from
# results/giant_present_mass.csv.

# %%
for r in LMC_MASS_LITERATURE:  # checksum check of each published source (the return value is not needed)
    path('lmc_mass_reference_' + r.get('source_key', r['key']))
# One row per reporting [Fe/H] bin (feh: centre, feh_hi: upper edge [dex]); masses [Msun].
t = pd.read_csv(ROOT / 'results/giant_present_mass.csv')


# %% [markdown]
# ### Cell formats
# A mass with its bootstrap errors and model range in powers of ten, and the cumulative mass below a bin edge.

# %%
def sci(v, lo, hi, mlo, mhi):
    """$(v^{+}_{-})\\times10^n$ [model range] with two significant decimals.

    Parameters
    ----------
    v, lo, hi : float
        Value and its 16th and 84th bootstrap percentiles [Msun].
    mlo, mhi : float
        Minimum and maximum over the model alternatives [Msun].

    Returns
    -------
    str
        LaTeX cell; all numbers scaled by 10^n with n = floor(log10 v) and given to two decimals.
    """
    # Common power of ten n; the value, its errors (p84 - v, v - p16) and the model range in units of 10^n.
    n = int(np.floor(np.log10(v)))
    v, lo, hi, mlo, mhi = (x / 10 ** n for x in (v, lo, hi, mlo, mhi))
    # LaTeX: value with upper and lower errors, power of ten, then the model range in brackets.
    return (rf"$({v:.2f}^{{+{hi - v:.2f}}}_{{-{v - lo:.2f}}})\times10^{{{n}}}$ [{mlo:.2f}--{mhi:.2f}]")


def row(i, pre):
    """Cumulative mass below the upper edge of bin i, for prefix '' (formed), 'present_' or 'living_'.

    Parameters
    ----------
    i : int
        Row of results/giant_present_mass.csv (reporting bin); the last row gives the total.
    pre : str
        Column prefix: '' (formed), 'present_' (living stars and remnants) or 'living_'.

    Returns
    -------
    str
        LaTeX cell from `sci`.
    """
    # The column stem: 'cumulative' for formed mass, '<pre>cumulative' otherwise; suffixes _p16, _p84, _model_min, _model_max.
    r = t.iloc[i]
    # Cumulative mass below the bin's upper edge, with its bootstrap percentiles and model range.
    key = 'cumulative' if pre == '' else pre + 'cumulative'
    return sci(r[key], r[key + '_p16'], r[key + '_p84'], r[key + '_model_min'], r[key + '_model_max'])


# %% [markdown]
# ### This work
# Total formed, present and living mass, the mass below [Fe/H] = -1, -1.5 and -2, and the metal-rich fraction.

# %%
# Last row: the cumulative mass of all bins, i.e. the total within 12 deg.
last = len(t) - 1
below = {f'{t.feh_hi[i]:g}': i for i in range(last)}  # upper bin edge as text ('-2', '-1.5', ...) -> row index
# Section header, the totals (formed, present; living stars only), then the cumulative masses below -1, -1.5 and -2.
rows = [r"\multicolumn{5}{l}{\textit{This work: BOSS giants within $12^\circ$, PARSEC conversion}}\\",
        rf"All & {row(last, '')} & {row(last, 'present_')} & $\rho<12^\circ$ & Kroupa\\",
        rf"Living stars & -- & {row(last, 'living_')} & & \\"]
for edge in ('-1', '-1.5', '-2'):
    rows.append(rf"$\feh<{edge}$ & {row(below[edge], '')} & {row(below[edge], 'present_')} & & \\")
# Fraction of formed mass in the bins above [Fe/H] = -0.5: reference, bootstrap 16th-84th percentiles, and the range over the
# model alternatives (filled, i.e. with the full-coverage extrapolation).
rich = t.feh.to_numpy() > -.5  # bin centres above -0.5
# Bootstrap draws (n_boot, n_bins) [Msun]; fb: the metal-rich fraction in each draw.
boot = pd.read_csv(ROOT / 'results/giant_mass_bootstrap.csv').to_numpy()
fb = boot[:, rich].sum(1) / boot.sum(1)
# Metal-rich fraction for each model alternative (conversion, prior, alpha_model, foreground); f0: the reference value.
v = pd.read_csv(ROOT / 'results/giant_mass_variants.csv').query('filled')
fm = v.groupby(['conversion', 'prior', 'alpha_model', 'foreground']).apply(
    lambda d: d.mass[d.feh > -.5].sum() / d.mass.sum(), include_groups=False)
f0 = t.formed_mass[rich].sum() / t.formed_mass.sum()
p16, p84 = np.percentile(fb, [16, 84])
# Fraction in per cent, integer: reference, bootstrap errors, [model min--max].
rows.append(rf"$\feh>-0.5$, fraction & ${100 * f0:.0f}^{{+{100 * (p84 - f0):.0f}}}_{{-{100 * (f0 - p16):.0f}}}$ per cent "
            rf"[{100 * fm.min():.0f}--{100 * fm.max():.0f}] & & & \\")
# %% [markdown]
# ### Published estimates
# One row per entry of lmcold.constants.LMC_MASS_LITERATURE, as published (formed or present mass).

# %%
rows.append(r"\hline")
rows.append(r"\multicolumn{5}{l}{\textit{Published estimates, as published}}\\")
for r in LMC_MASS_LITERATURE:
    # Published mass [Msun] in units of 1e9, put in the formed or the present column according to its kind.
    m = rf"${r['mass'] / 1e9:.2f}\times10^{{9}}$"
    formed, present = (m, '--') if r['kind'] == 'formed' else ('--', m)
    rows.append(rf"\citet{{{r['bibkey']}}} & {formed} & {present} & {r['summary']} & {r['imf_short']}\\")
# Write the rows (the paper's table file holds the rest) and print them.
(ROOT / 'results/table_mass_summary.tex').write_text('\n'.join(rows) + '\n')
print('\n'.join(rows))
