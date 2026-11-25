"""Why the early discs of the simulated analogues are measured at 0.5 < R < 3 kpc and not at the LMC's 4 kpc (Letter, Section 3):
results/sims_birth_radii.csv, read by scripts/paper_text_letter.py for the Letter's numbers on the radii (the Fig. 3 page of the
knowledge base quotes them).

Stars and frames exactly as scripts/build_sims_birth.py (the table of Fig. sims_birth): the star particles within R_TODAY of the
centre today (Auriga: in situ only; TNG50: all, no assembly catalogue), born OLD_AGES Gyr ago (the window of the Letter's birth
numbers, scripts/paper_text_letter.py), in cohorts of COHORT Gyr; birth positions relative to the progenitor track in the frame of
each cohort's spin within R_BIRTH (lmcold.sims.progenitor_track, birth_phase_space); today's positions in the z = 0 disc frame
(lmcold.sims.disc_frame). Two annuli in cylindrical radius R, each with |z| below its outer edge: INNER = 0.5-3 kpc (as Fig.
sims_birth and Fig. letter_scramble) and OUTER = 3-5 kpc (around the LMC's R = 4 kpc, lmcold.kinfit.R_SIG).

Per analogue:
- n_inner, n_outer: star particles of the old cohorts at birth in each annulus;
- coh, coh_inner, coh_outer: old cohorts (at least 20 stars within R_TODAY), and those with at least MIN_IN particles at birth in
  each annulus (MIN_IN: the minimum for a birth value in build_sims_birth.py);
- n_today: old star particles at 3 < R < 5 kpc today (OUTER annulus in the z = 0 frame), and the mass fractions of them born in
  INNER (f_born_inner) and in OUTER (f_born_outer); NaN where n_today < 20.
Reads the same particles as scripts/build_sims_birth.py (TNG50 cutouts and birth files, Auriga level-3 snapshots).
Writes results/sims_birth_radii.csv (suite, halo, n_inner, n_outer, coh, coh_inner, coh_outer, n_today, f_born_inner,
f_born_outer).
Run: .venv/bin/python scripts/manifest.py run sims_birth_radii.py   (about 10 min on WORKERS cores; 49 s in the recorded run of
docs/run_times.txt)
"""
# %% [markdown]
# ### Settings
# The cohort width, birth sphere and minimum numbers are imported from `scripts/build_sims_birth.py`, so the stars are the same.

# %%
from multiprocessing import Pool

import numpy as np
import pandas as pd

from build_sims_birth import COHORT, MIN_IN, R_BIRTH, R_TODAY
from lmcold import sims
from lmcold.sources import ROOT

OLD_AGES = (10.0, 12.0)   # Gyr: the cohorts of the Letter's birth numbers (paper_text_letter.OLD_AGES)
INNER, OUTER = (0.5, 3.0), (3.0, 5.0)   # kpc: the annulus of Figs sims_birth and letter_scramble, and one around R = 4 kpc
MIN_TODAY = 20   # old stars at 3-5 kpc today for the born-where fractions
WORKERS = 6   # parallel processes, one analogue each


# %% [markdown]
# ### Helpers
# `in_annulus` selects the points in one of the two annuli; `radii` counts, for one analogue, the old stars in each annulus at birth
# and where the old stars now at 3–5 kpc were born.

# %%
def in_annulus(P, ann):
    """Mask of the points P (N, 3) [kpc] with ann[0] < R < ann[1] (cylindrical) and |z| < ann[1]."""
    R = np.hypot(P[:, 0], P[:, 1])
    return (R > ann[0]) & (R < ann[1]) & (np.abs(P[:, 2]) < ann[1])


def radii(job):
    """One row of results/sims_birth_radii.csv for analogue job = (suite, name)."""
    suite, name = job
    d = sims.birth(suite, name)
    X, _, _ = sims.disc_frame(d['x'], d['v'], d['m'])   # positions today [kpc], z = 0 disc frame
    # Stars within R_TODAY [kpc] today and not ex situ; the progenitor's track (centre, velocity against time) from their birth
    # data.
    use = (np.linalg.norm(d['x'], axis=1) < R_TODAY) & ~d['exsitu']
    track = sims.progenitor_track(d['birth_x'][use], d['a_form'][use], d['birth_v'][use])
    # Counters of the output row; per cohort, the annulus masks at birth, the OUTER mask today and the masses [Msun] are collected.
    row = dict(suite=suite, halo=name, n_inner=0, n_outer=0, coh=0, coh_inner=0, coh_outer=0)
    born_inner, born_outer, today, m = [], [], [], []
    for lo in np.arange(OLD_AGES[0], OLD_AGES[1], COHORT):
        # Indices of the stars of this cohort (age lo to lo + COHORT [Gyr]); cohorts with fewer than 20 stars are skipped.
        i = np.flatnonzero(use & (d['age'] >= lo) & (d['age'] < lo + COHORT))
        if len(i) < 20:
            continue
        P, _, _ = sims.birth_phase_space(d['birth_x'][i], d['a_form'][i], d['birth_v'][i], d['m'][i], track, R_BIRTH)   # birth positions [kpc]
        # Masks (n,) of the cohort's stars born in INNER (a) and OUTER (b); counts and cohorts with at least MIN_IN stars in each.
        a, b = in_annulus(P, INNER), in_annulus(P, OUTER)
        row['coh'] += 1
        row['n_inner'] += int(a.sum()); row['n_outer'] += int(b.sum())
        row['coh_inner'] += int(a.sum() >= MIN_IN); row['coh_outer'] += int(b.sum() >= MIN_IN)
        born_inner.append(a); born_outer.append(b); today.append(in_annulus(X[i], OUTER)); m.append(d['m'][i])
    row['n_today'], row['f_born_inner'], row['f_born_outer'] = 0, np.nan, np.nan
    if today:
        # All old cohorts together: t marks the stars in OUTER today; fractions only with at least MIN_TODAY of them.
        a, b, t, m = map(np.concatenate, (born_inner, born_outer, today, m))
        row['n_today'] = int(t.sum())
        if t.sum() >= MIN_TODAY:
            row['f_born_inner'] = m[t & a].sum() / m[t].sum()   # mass fraction of today's 3-5 kpc old stars born at 0.5-3 kpc
            row['f_born_outer'] = m[t & b].sum() / m[t].sum()
    return row


if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data and count every analogue
    # The TNG50 and Auriga analogues, one per process (each reads its particles through `lmcold.sims.birth`); then a summary per suite.

    # %%
    # (suite, name) of every TNG50 and Auriga analogue; one row each, written to the csv.
    jobs = [s for s in sims.names() if s[0] in ('TNG50', 'Auriga')]
    with Pool(WORKERS) as pool:
        t = pd.DataFrame(pool.map(radii, jobs))
    t.to_csv(ROOT / 'results/sims_birth_radii.csv', index=False, float_format='%.4f')
    # Summary per suite: particles and cohorts in INNER -> OUTER, and where today's 3-5 kpc old stars were born.
    for s, g in t.groupby('suite'):
        print(f"{s}: {len(g)} analogues; particles {g.n_inner.sum()} -> {g.n_outer.sum()} ({g.n_outer.sum() / g.n_inner.sum():.2f}); "
              f"cohorts with >= {MIN_IN}: {g.coh_inner.sum()} -> {g.coh_outer.sum()}; analogues with any: "
              f"{(g.coh_inner > 0).sum()} -> {(g.coh_outer > 0).sum()}; of the old stars at 3-5 kpc today born at 0.5-3 kpc "
              f"{g.f_born_inner.median():.2f}, at 3-5 kpc {g.f_born_outer.median():.2f} (medians over analogues)")
