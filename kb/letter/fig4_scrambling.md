---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Fig. 4: how the early discs lose their rotation, and why

Every number in the text and caption is written by `scripts/paper_text_letter.py` from the result tables
(`results/macros_letter.tex`); the code cells are the figure script itself (`scripts/paper_letter_scramble.py`), step by step.

**What it shows.** {fig}`paper_letter_scramble` follows the earliest stars of the progenitors of simulated analogues of the Large
Magellanic Cloud (LMC) in TNG50 ([Pillepich et al. 2019](https://doi.org/10.1093/mnras/stz2338)) and asks how fast they lose their
rotation, what removes it, and why only early. It uses {glue:text}`ltNCohorts` cohorts born at redshifts $z$ =
{glue:text}`ltZSnapA`–{glue:text}`ltZSnapB` in {glue:text}`ltNCohortGal` of the {glue:text}`ltNFull` progenitors with outputs at
every snapshot, and the dark matter of a larger sample of {glue:text}`ltNSample` progenitors that includes them. A cohort is the
stars formed in the {glue:text}`ltWindow` Myr before a simulation output (its birth snapshot), followed for {glue:text}`ltDtMax`
Gyr. Its rotational support is $v_\phi/\sigma_z$, the mean azimuthal velocity over the vertical velocity dispersion, measured about
the cohort's spin at birth. The same stars are also moved as test particles through their progenitor's gravity, computed with
AGAMA ([Vasiliev 2019](https://doi.org/10.1093/mnras/sty2672)): frozen at birth, with only the dark matter aspherical, and made
axisymmetric about the cohort's spin. Time is also given in orbital periods $T_{\rm orb}$ at {glue:text}`ltROrb` kpc
($T_{\rm orb}$ = {glue:text}`ltTOrb` Gyr).

The shape of the dark-matter halo is measured by $\epsilon_2$: the amplitude of the quadrupole of the dark-matter potential at
$r$ = {glue:text}`ltSetRZero` kpc that is not axisymmetric about the disc's spin, relative to the monopole. It grows with a tilt of
the halo relative to the disc and with the halo's elongation in the disc plane, but not with a flattening along the spin. The
misalignment $\psi$ is the angle between the minor axis of the dark matter within {glue:text}`ltSetRIn` kpc and the spin of the
stellar disc.

(a) The rotation of the early cohorts is halved within one orbital period. The test particles keep their rotation only when the
gravity is made axisymmetric about their spin, so the loss is caused by the non-axisymmetric gravity of the young halo. (b) The
rotation that survives falls with $\epsilon_2$ at birth. (c) Followed in lookback time (the time before the present), the halo
settles, lining up with the disc and becoming rounder, only after the LMC's oldest stars formed.

**Starts from:** the quick route.

## Making the figure, step by step

<!-- script: paper_letter_scramble.py -->

**Figure (paper_letter_scramble).** How the early TNG50 cohorts lose their rotation, and why. (a) $v_\phi/\sigma_z$ about each
cohort's spin at birth against time since birth (inner labels: orbital periods at {glue:text}`ltROrb` kpc) for the real stars
(black) and for test particles in the progenitor's gravity frozen at birth (orange), with only the dark matter aspherical (red) and
made axisymmetric about the cohort's spin (purple); blue band: the star-forming gas; dotted: the LMC's [Fe/H] <
{glue:text}`ltFehA` giants (at $R$ = {glue:text}`ltRSig` kpc, for reference). (b) $v_\phi/\sigma_z$ after
{glue:text}`ltTAfterA`–{glue:text}`ltTAfterB` Gyr against the dark-matter quadrupole $\epsilon_2$ at birth; dashed: at birth. (c)
Misalignment $\psi$ of the halo's minor axis with the disc spin (red, left), and $\epsilon_2$ (black) with its in-plane part (blue
dashed; right), against lookback time. The rotation is halved within one orbital period, and mostly lost within two, because the
halo is aspherical and misaligned with the disc; it settles only after the LMC's oldest stars formed.

## How the inputs are made

The figure script reads the cohorts and their test-particle runs (`results/sims_scramble.csv`), the star-forming gas
(`results/sims_heating_gas.csv`), the halo shapes (`results/sims_halo_settle.csv`), the orbital periods
(`results/sims_vcirc.csv`) and the LMC fits (`results/kin_support_ellipsoid.csv`). The logged steps upstream of the figure,
nearest first:

```{code-cell} python
:tags: [remove-input]
import kbtools as kb
kb.upstream('paper_letter_scramble.png')
```

The test-particle experiment (`scripts/sims_scramble.py`, a few minutes) and the halo shapes (`scripts/sims_halo_settle.py`) use
AGAMA multipole potentials built from the TNG50 progenitors (`scripts/fetch_tng50_progenitors.py`). For an introduction to
potentials and orbit integration with AGAMA, see the
[Galactic Dynamics with AGAMA](https://vasilybelokurov.github.io/galactic-dynamics-tutorial/) tutorial. The same experiment on the
Auriga analogues: `scripts/auriga_scramble.py`.

## Try it: the experiment for one cohort

The test-particle experiment needs the TNG50 particles, which are too large for this site. One cohort can be repeated here from a
few small files: the stars formed in the {glue:text}`ltWindow` Myr before snapshot 33 in progenitor 670104 (the $z$ = 0 subhalo id in
TNG50-1). Its real stars lose their rotation within {glue:text}`ltDtMax` Gyr, like most early cohorts in panel (a) of
{fig}`paper_letter_scramble`.

The gravity of the progenitor is taken at the cohort's birth snapshot and held fixed. It is used in two forms: complete (all the dark
matter, gas and stars), and with every part that is not axisymmetric about the cohort's spin removed, separately for the dark matter
and for the baryons. The stars are then moved as test particles through each form for {glue:text}`ltDtMax` Gyr, and their
$v_\phi/\sigma_z$ about the cohort's spin at birth is measured. If the non-axisymmetric gravity removes the rotation, the test
particles lose it in the complete gravity and keep it in the axisymmetric one.

### The inputs for one cohort

The step below, run on the full simulation, picks the cohort and its stars as `scripts/sims_scramble.py` does and writes what the two
runs need: the AGAMA multipole coefficients of the two forms of the gravity, the stars' positions, velocities and masses at birth in
the frame aligned with the cohort's spin, and the times of the later snapshots.

<!-- script: build_scramble_demo.py show: the TNG50 particle cutouts (scripts/fetch_tng50_progenitors.py); a few seconds -->

### The two runs

The orbit integration needs AGAMA. On this site it is installed; in Google Colab the next cell installs a ready-made copy (the
wheel of the [Galactic Dynamics with AGAMA](https://vasilybelokurov.github.io/galactic-dynamics-tutorial/) tutorial).

```{code-cell} python
# Install AGAMA only if it is missing (e.g. in Colab): a ready-made Linux wheel for this Python version.
import importlib.util, subprocess, sys
if importlib.util.find_spec('agama') is None:
    v = f'cp{sys.version_info.major}{sys.version_info.minor}'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'https://github.com/vasilybelokurov/galactic-dynamics-tutorial/'
                    f'releases/download/agama-wheels/agama-1.0-{v}-{v}-linux_x86_64.whl'], check=True)
```

The script below runs the two forms of the gravity, compares the result with the full experiment (`results/sims_scramble.csv`) and
stops if they disagree; {fig}`scramble_demo` shows the result.

<!-- script: scramble_demo.py -->

**Figure (scramble_demo).** $v_\phi/\sigma_z$ about the cohort's spin at birth against time since birth, for the cohort born at
snapshot 33 in TNG50 progenitor 670104: test particles in the progenitor's gravity frozen at birth, complete (orange) and made
axisymmetric about the cohort's spin (purple), computed here (lines) and in the full experiment (open circles), and the real stars
(black squares). The test particles lose their rotation in the complete gravity, as the real stars do, and keep it in the
axisymmetric gravity.
