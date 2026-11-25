---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Fig. 3: the simulated stars at birth and today

Every number in the text and caption is written by `scripts/paper_text_letter.py` from the result tables
(`results/macros_letter.tex`); the code cells are the figure script itself (`scripts/paper_sims_birth.py`), step by step.

**What it shows.** {fig}`paper_sims_birth` asks whether the stars of simulated analogues of the Large Magellanic Cloud (LMC) are born
without rotation or lose it later. The TNG50 ([Pillepich et al. 2019](https://doi.org/10.1093/mnras/stz2338)) and Auriga
([Grand et al. 2024](https://doi.org/10.1093/mnras/stae1598)) simulations record the position and velocity of every star particle
at birth, so the same stars can be measured when they formed and today. The stars of each analogue are grouped in cohorts, stars
born within the same {glue:text}`ltCohort` Gyr. For each cohort the figure shows the mass-weighted mean azimuthal velocity $v_\phi$
(its rotation), the vertical velocity dispersion $\sigma_z$ (the spread of the velocities perpendicular to the disc plane) and their
ratio $v_\phi/\sigma_z$, the rotational support. At birth these are measured relative to the progenitor galaxy, about the cohort's
own spin, at cylindrical radii {glue:text}`ltRMinBirth` < $R$ < {glue:text}`ltRBirth` kpc; today they are measured in the frame of
the galaxy's disc, within {glue:text}`ltRToday` kpc. The LMC's five giant [Fe/H] bins ([Fe/H]: the logarithm of the iron-to-hydrogen
ratio relative to the Sun's) are placed at the ages that the age–metallicity relation of [Povick et al.
(2024)](https://doi.org/10.1093/mnras/stae1949) gives them, with their $v_{\rm rot}$, $\sigma_z$ and $v_{\rm rot}/\sigma_z$ at
$R$ = {glue:text}`ltRSig` kpc from the fits of the Letter's Fig. 2. The simulated stars are born as rotation-supported as the
LMC's old stars are today, but the oldest of them have since lost most of their rotation: what fails in the simulations is not how
the stars are born but what happens to them afterwards.

**Starts from:** the quick route.

## Making the figure, step by step

<!-- script: paper_sims_birth.py -->

**Figure (paper_sims_birth).** The rotational support of the stars of simulated LMC analogues at birth (blue, dashed) and today
(red, solid), against age, in cohorts of {glue:text}`ltCohort` Gyr: (a) $v_\phi$, (b) $\sigma_z$, (c) $v_\phi/\sigma_z$. TNG50:
median and 16th–84th percentiles over {glue:text}`ltNTngBirth` analogues; Auriga: one thin line per galaxy. Squares: the LMC bins at
their mean ages (bars: 16th–84th percentiles). The simulated values are intrinsic, at {glue:text}`ltRMinBirth` < $R$ <
{glue:text}`ltRBirth` kpc at birth and within {glue:text}`ltRToday` kpc today; the LMC's are measured at $R$ = {glue:text}`ltRSig`
kpc. The simulated stars are born as rotation-supported as the LMC's old stars are today, but the oldest of them have since lost
most of their rotation.

## How the inputs are made

The figure script reads the birth table `results/sims_birth.csv` and the LMC fits `results/kin_support_ellipsoid.csv`. The logged
steps upstream of the figure, nearest first:

```{code-cell} python
:tags: [remove-input]
import kbtools as kb
kb.upstream('paper_sims_birth.png')
```

The birth table comes from `python scripts/build_sims_birth.py` (about a minute on 6 cores; needs the TNG50 birth files and the
Auriga snapshots); its code, and that of `scripts/sims_birth_radii.py` and `scripts/fire2_birth.py`, is shown step by step, not run,
in the chapter 'Simulated analogues observed like the LMC'. Why the simulated stars are measured at {glue:text}`ltRMinBirth`–{glue:text}`ltRBirth` kpc at birth, not at
the LMC's {glue:text}`ltRSig` kpc: `scripts/sims_birth_radii.py`. In the young progenitors TNG50 has only
{glue:text}`ltRadPartRatio` times as many star particles at 3–5 kpc as at {glue:text}`ltRMinBirth`–{glue:text}`ltRBirth` kpc, and
only {glue:text}`ltRadBornOuter` per cent of the old stars seen today at the LMC's radius were born at 3–5 kpc.
