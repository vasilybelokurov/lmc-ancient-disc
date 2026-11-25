"""Paper text: the sentences on the NUTS sampling of the kinematic fits (Section 4.2 'Posteriors' and Appendix app:kin_sky), from
the run settings (scripts/fit_kinematics_mcmc.py: CHAINS, DRAWS, WARMUP, WARMUP_ELLIPSOID) and the diagnostics in the result tables
(r_hat, n_eff, divergences per stage and population).

- results/text_mcmc_ellipsoid.tex: velocity-ellipsoid fits (results/kin_mcmc_data_ellipsoid.csv, giants and RR Lyrae at 2-9 deg;
  kin_mcmc_data_ellipsoid_outer.csv, 4-9 deg, RR Lyrae dispersions).
- results/text_mcmc_stream.tex: sky-aligned fits (kin_mcmc_data_stream.csv, kin_mcmc_data_stream_outer.csv).
Bounds: max r_hat rounded up to 3 decimals, min n_eff rounded down. Each sentence names the fits with divergent transitions; the
script stops if their number differs from what the sentence's wording covers (one fit), so the text cannot silently go wrong.
Copied to the paper's tables/ and \\input in the text.
Run: .venv/bin/python scripts/manifest.py run paper_text_mcmc.py
"""

# %% [markdown]
# ### Settings and helpers
# The run settings of the fits and two helpers: the diagnostics of a set of runs, and a check that exactly one fit has divergent
# transitions.

# %%
import math

import pandas as pd

from fit_kinematics_mcmc import CHAINS, DRAWS, WARMUP, WARMUP_ELLIPSOID
from lmcold.sources import ROOT
from lmcold.tex import GROUP_LABEL, count

WORD = {4: 'four', 8: 'eight'}  # number of chains as written in the text


def diagnostics(*runs):
    """(max r_hat rounded up to 3 decimals, min n_eff rounded down, [(stage, population, divergences) with divergences > 0])."""
    # All rows of the given runs (one per stage x population x parameter), tagged with the run name.
    t = pd.concat([pd.read_csv(ROOT / f'results/kin_mcmc_data_{r}.csv').assign(run=r) for r in runs])
    # Divergences of each fit (the same value on every row of a fit), and the fits that have any.
    d = t.groupby(['run', 'stage', 'population']).divergences.max()
    div = [(s, p, int(n)) for (_, s, p), n in d[d > 0].items()]
    # Bounds safe for the text: largest r_hat rounded up, smallest n_eff rounded down.
    return math.ceil(t.r_hat.max() * 1000) / 1000, math.floor(t.n_eff.min()), div


def one_fit(div, what):
    """The single fit with divergences; stop if there is not exactly one (the wording assumes it)."""
    # Stop rather than write a wrong sentence.
    if len(div) != 1:
        raise SystemExit(f'{what}: {len(div)} fits with divergent transitions, the sentence is written for one: {div}')
    return div[0]


# %% [markdown]
# ### Velocity-ellipsoid fits
# The sentence on the sampling of the velocity-ellipsoid fits.

# %%
# r: max r_hat, n: min n_eff over the two ellipsoid runs; the one fit with divergences (stage, population, number).
r, n, div = diagnostics('ellipsoid', 'ellipsoid_outer')
stage, pop, nd = one_fit(div, 'ellipsoid')
# Name of that fit in the text (giant groups by their paper label, lmcold.tex.GROUP_LABEL).
who = 'the RR Lyrae' if pop == 'RR Lyrae' else 'giants, ' + GROUP_LABEL[pop]
# The sentence: sampler, chains, draws and warm-up from fit_kinematics_mcmc.py, then the diagnostics.
ellipsoid = (f'We sample the posteriors with NUTS \\citep{{Hoffman2014}} in {{\\sc numpyro}} \\citep{{Phan2019}}, {WORD[CHAINS]} chains '
             f'of {count(DRAWS)} draws after {count(WARMUP_ELLIPSOID)} warm-up steps ({count(WARMUP)} for the RR Lyrae fit at '
             f'$4<\\rho<9^\\circ$). Every run has $\\hat R\\leq{r:.3f}$ and at least {count(n)} effective draws per parameter; one '
             f'stage-{stage} fit ({who}) has {nd} divergent transitions among its {count(CHAINS * DRAWS)} draws, the others none.')

# %% [markdown]
# ### Sky-aligned fits
# The sentence on the sky-aligned fits; the script stops if the divergences are not in the RR Lyrae fit the sentence names.

# %%
# The same diagnostics for the two sky-aligned runs; exactly one fit with divergences, and it must be the RR Lyrae fit.
r, n, div = diagnostics('stream', 'stream_outer')
stage, pop, nd = one_fit(div, 'stream')
if pop != 'RR Lyrae':
    raise SystemExit(f'stream: the sentence names the RR Lyrae fit at 4-9 deg, but the divergences are in {pop}')
# The sentence of the sky-aligned fits.
stream = (f'The sky-aligned fits sampled the posteriors with $\\hat R\\leq{r:.3f}$ and at least {count(n)} effective draws per '
          f'parameter; the RR Lyrae fit at $4<\\rho<9^\\circ$ had {nd} divergent transitions.')

# %% [markdown]
# ### Write the sentences
# Each sentence into results/text_mcmc_<name>.tex.

# %%
# One .tex file per sentence (copied to the paper's tables/), and print it.
for name, text in (('ellipsoid', ellipsoid), ('stream', stream)):
    (ROOT / f'results/text_mcmc_{name}.tex').write_text(text + '\n')
    print(text)
