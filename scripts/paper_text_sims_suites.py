"""Paper text: the numbers of the Section 6.4 paragraph on Auriga and FIRE-2 (no figure), as LaTeX macros.

Inputs: results/sims_birth.csv (scripts/build_sims_birth.py: birth cohorts of COHORT Gyr, TNG50 and Auriga, from BirthPos/BirthVel)
and the research-repository tables in results/research/ (data/sources.yaml: research_suites_early): fire_early_birth.csv (FIRE-2
birth vectors, cohorts of 0.1 Gyr), auriga_early_hist.csv, fire_early_hist.csv (v_phi/sigma_z of 100-Myr cohorts after birth;
first point = the output after birth), auriga_early_coh.csv, fire_early_coh.csv (coherence C of 100-Myr cohorts),
auriga_stageD_rules.csv (loss of C at 0.2-0.35 Gyr per Auriga cohort and variant of the gravity, as Fig. sims_scramble b).
Definitions (as Fig. sims_heating; per galaxy the median in a bin, then the median over the galaxies):
- at birth: v_phi/sigma_z of the cohorts in 0.5-Gyr bins of lookback time at birth, 8-12 Gyr ago; the range (lowest, highest)
  of the bin medians, per suite;
- the oldest stars: cohorts born > 10 Gyr ago, v_phi/sigma_z at the first output and after 0.25 Gyr (0.25-Gyr bins of time since
  birth); coherence of the cohorts born at z > 1.5 (> 9.4 Gyr ago) at birth and after 0.5 Gyr (0.1-Gyr bins); FIRE-2 cohorts born
  < 9 Gyr ago, C at birth;
- Auriga halo tests: summed loss of a variant over the cohorts / summed loss with the full frozen gravity (E_split): dark matter made
  round (G_bar), axisymmetric about the stars' spin (D0_fr), the halo axisymmetric about its minor axis (H1) for cohorts born less and
  more than 45 deg from alignment.
Writes results/macros_sims_suites.tex (copied to the paper's tables/; \\input at the top of Section 6).
Run: .venv/bin/python scripts/manifest.py run paper_text_sims_suites.py"""

# %% [markdown]
# ### Settings and helpers
# The time windows of the definitions, the macro store M, and the median over galaxies of the per-galaxy medians.

# %%
import numpy as np
import pandas as pd

from lmcold.sources import ROOT

RES = ROOT / 'results/research'   # research-repository tables of data/sources.yaml entry research_suites_early
BIRTH_LB = (8, 12)   # Gyr: lookback times at birth of the 'at birth' ranges
OLD_LB = 10          # Gyr: 'the oldest stars' are born more than OLD_LB Gyr ago
T_AFTER = 0.25       # Gyr after birth of the 'after' value (and the width of the time bins)
M = {}   # macro store: LaTeX macro name -> formatted value (string)


def put(name, value, d=0):
    """Store value in M under macro name, formatted with d decimals."""
    M[name] = f'{value:.{d}f}'


def gal_median(d, by, q):
    """Median of column q per (bin by, galaxy halo), then the median over galaxies per bin; a Series indexed by the bin."""
    return d.groupby([by, 'halo'])[q].median().groupby(level=0).median()


# %% [markdown]
# ### At birth
# v_phi/sigma_z at birth, 8-12 Gyr ago, for TNG50, Auriga and FIRE-2.

# %%
# TNG50 and Auriga birth cohorts: vs = v_phi/sigma_z at birth, lb = lookback time at birth (column age) [Gyr]
b = pd.read_csv(ROOT / 'results/sims_birth.csv', dtype={'halo': str}).dropna(subset=['vphi_birth'])
b = b.assign(vs=b.vphi_birth / b.sz_birth, lb=b.age)
# FIRE-2 birth vectors, the 0.1-Gyr cohorts only, with the same ratio vs
f = pd.read_csv(RES / 'fire_early_birth.csv').query('width == 0.1')
f = f.assign(vs=f.vphi / f.sz)
for name, d in (('Tng', b[b.suite == 'TNG50']), ('Au', b[b.suite == 'Auriga']), ('Fire', f)):
    # Cohorts born 8-12 Gyr ago, binned in 0.5 Gyr of lookback (lbb = lower bin edge [Gyr])
    d = d[(d.lb >= BIRTH_LB[0]) & (d.lb < BIRTH_LB[1])].assign(lbb=lambda t: np.floor(t.lb * 2) / 2)
    # Median over galaxies per bin; the macros are the lowest and highest bin medians, 1 decimal
    m = gal_median(d, 'lbb', 'vs')
    put(f'ssBirth{name}Lo', m.min(), 1); put(f'ssBirth{name}Hi', m.max(), 1)

# %% [markdown]
# ### The oldest stars
# v_phi/sigma_z and coherence of the oldest cohorts at birth and soon after, Auriga and FIRE-2.

# %%
for name in ('Auriga', 'Fire'):
    # Macro tag; h: v_phi/sigma_z of 100-Myr cohorts versus time since birth dt [Gyr]
    tag = 'Au' if name == 'Auriga' else 'Fire'
    h = pd.read_csv(RES / f'{name.lower()}_early_hist.csv', dtype={'halo': str})
    # Cohorts born > OLD_LB Gyr ago; dt (negative values set to 0) rounded to 0.25-Gyr bins (dtb [Gyr])
    h = h[h.lb_birth > OLD_LB].assign(vs=lambda t: t.vphi / t.sz, dtb=lambda t: (t.dt.clip(lower=0) / T_AFTER).round() * T_AFTER)
    # Galaxy medians at the first output (dtb = 0) and T_AFTER after birth
    m = gal_median(h, 'dtb', 'vs')
    put(f'ssOld{tag}Birth', m[0.0], 2); put(f'ssOld{tag}After', m[T_AFTER], 2)
    # Coherence C of 100-Myr cohorts versus time since birth, in 0.1-Gyr bins
    c = pd.read_csv(RES / f'{name.lower()}_early_coh.csv', dtype={'halo': str})
    c = c.assign(dtb=(c.dt / 0.1).round() * 0.1)
    # Cohorts born at z > 1.5 (> 9.4 Gyr ago): C in the first bin and in the last bin up to 0.5 Gyr after birth
    m = gal_median(c[c.lb_birth > 9.4], 'dtb', 'C')
    put(f'ssCoh{tag}Birth', m.iloc[0], 2); put(f'ssCoh{tag}End', m[m.index <= 0.5 + 1e-9].iloc[-1], 2)
    # FIRE-2 only: C at birth of the cohorts born < 9 Gyr ago
    if tag == 'Fire':
        put('ssCohFireLate', gal_median(c[c.lb_birth < 9], 'dtb', 'C').iloc[0], 2)

# %% [markdown]
# ### Auriga halo tests
# Loss of coherence with modified gravity, relative to the full frozen gravity.

# %%
# One row per Auriga cohort: loss of C under each gravity variant (columns E_split, G_bar, D0_fr, H1) and
# the misalignment [deg]; lo, hi: cohorts born less and more than 45 deg from alignment
s = pd.read_csv(RES / 'auriga_stageD_rules.csv')
lo, hi = s[s.misalign < 45], s[s.misalign >= 45]
# Number of cohorts and the summed loss of each variant over the summed loss with the full frozen gravity
put('ssAuN', len(s)); put('ssAuRound', s.G_bar.sum() / s.E_split.sum(), 2); put('ssAuAxi', s.D0_fr.sum() / s.E_split.sum(), 2)
put('ssAuHOneAligned', lo.H1.sum() / lo.E_split.sum(), 2); put('ssAuHOneTilted', hi.H1.sum() / hi.E_split.sum(), 2)

# %% [markdown]
# ### Write the macros
# One \newcommand per entry of M into results/macros_sims_suites.tex.

# %%
# One \newcommand line per macro; written to results/macros_sims_suites.tex and printed
text = ''.join(f'\\newcommand{{\\{n}}}{{{v}}}\n' for n, v in M.items())
(ROOT / 'results/macros_sims_suites.tex').write_text(text)
print(text)
