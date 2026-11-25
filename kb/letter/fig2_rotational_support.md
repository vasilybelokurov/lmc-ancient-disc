---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Fig. 2: rotational support against metallicity, the LMC and its simulated analogues

Every number in the text and captions is written by `scripts/paper_text_letter.py` from the result tables
(`results/macros_letter.tex`); the code cells are the figure script itself (`scripts/paper_sims_kinematics.py`), step by step.

**What it shows.** {fig}`paper_sims_kinematics_fraction` compares how strongly the stars of the Large Magellanic Cloud (LMC) and of
simulated LMC analogues are supported by rotation, as a function of their metallicity. Each population of stars is fitted with a
thin, inclined disc model. In cylindrical coordinates $(R,\phi,z)$ in the disc frame, with $z$ along the disc's spin, its mean
velocity is a rotation curve $v_{\rm rot}(R)$, and about this mean the velocities scatter with dispersions $\sigma_R$ (radial),
$\sigma_\phi$ (azimuthal) and $\sigma_z$ (vertical). The figure shows $v_{\rm rot}$, $\sigma_z$ and the rotational support
$v_{\rm rot}/\sigma_z$, all at $R$ = {glue:text}`ltRSig` kpc. For the LMC the populations are its red giants (spectra from
[APOGEE DR17](https://doi.org/10.3847/1538-4365/ac4414) and SDSS-V BOSS, [Kollmeier et al.
2026](https://doi.org/10.3847/1538-3881/ae0576); proper motions from [*Gaia* DR3](https://doi.org/10.1051/0004-6361/202243940)) in
five bins of iron abundance [Fe/H] (the logarithm of the iron-to-hydrogen ratio relative to the Sun's), and its RR Lyrae, old
pulsating stars that have proper motions only and so give the radial dispersion $\sigma_R$ but not $\sigma_z$.

The analogues are {glue:text}`ltNTng` central galaxies of the TNG50 simulation ([Pillepich et al.
2019](https://doi.org/10.1093/mnras/stz2338)) with stellar masses of {glue:text}`ltMstarLo`–{glue:text}`ltMstarHi` × 10⁹ M☉,
{glue:text}`ltNAu` dwarf galaxies of the Auriga zoom-in simulations ([Grand et al. 2024](https://doi.org/10.1093/mnras/stae1598))
and {glue:text}`ltNFire` of FIRE-2 ([Wetzel et al. 2023](https://doi.org/10.3847/1538-4365/acb99a)). Each is observed with the
LMC's samples and errors and fitted with the same model. Their metallicities are matched to the LMC's by rank in formed mass, so the
horizontal axis is the cumulative fraction of formed mass: for each [Fe/H] bin, the fraction of the galaxy's formed stellar mass in
stars more metal-poor than the bin's median. Almost no analogue is as rotation-supported as the LMC's metal-poor stars.

The script also saves the same figure against formed stellar mass instead of the fraction (`figures/paper_sims_kinematics.png`,
used in the extended paper) and {fig}`paper_sims_recovery`, which tests whether the fits recover the analogues' true kinematics.

**Starts from:** the quick route.

## Making the figure, step by step

<!-- script: paper_sims_kinematics.py -->

**Figure (paper_sims_kinematics_fraction).** The rotational support of the LMC's stars against that of simulated analogues observed
and fitted in the same way, at $R$ = {glue:text}`ltRSig` kpc: (a) $v_{\rm rot}$, (b) $\sigma_z$, (c) $v_{\rm rot}/\sigma_z$, against
the cumulative fraction of the formed mass below each metallicity bin; top axis: the bin edges in LMC [Fe/H]. Black bands: the LMC
giants (16th–84th percentiles; above [Fe/H] = {glue:text}`ltApoFeh` extended to APOGEE alone, hence the widening). Blue:
{glue:text}`ltNTng` TNG50 analogues (median, thick; 68 and 95 per cent ranges, thin and dashed); orange: {glue:text}`ltNAu` Auriga,
green: {glue:text}`ltNFire` FIRE-2 analogues. Numbers in (c): per cent of TNG50 analogues reaching the LMC band. Diamonds: the RR
Lyrae, with $\sigma_R$ in (b) and (c). Almost no analogue is as rotation-supported as the LMC at [Fe/H] < {glue:text}`ltFehB`; only
TNG50 matches its metal-rich stars.

## How the inputs are made

The figure script reads the LMC fits (`results/kin_support_ellipsoid.csv`), the fits of the analogues
(`results/sims_observed.csv`), and the formed masses of the LMC and of the analogues (`results/giant_mass.csv`,
`results/sims_formed_mass.csv`). The logged steps upstream of the figure, nearest first:

```{code-cell} python
:tags: [remove-input]
import kbtools as kb
kb.upstream('paper_sims_kinematics_fraction.png')
```

Long steps, not run here:
- the LMC fits of each metallicity bin: `python scripts/fit_kinematics_mcmc.py data ellipsoid` and
  `... data ellipsoid outer` (RR Lyrae), about 4 h each;
- the mock observation and fit of every analogue: `python scripts/sims_observe.py` (about 13 h on 5 cores; needs the simulation
  particles, `scripts/fetch_tng50.py`, `fetch_auriga.py`, `fetch_fire2.py`); it writes `results/sims_observed.csv`;
- the total formed mass of every analogue: `python scripts/build_sims_formed_mass.py` (seconds; needs the simulation particles).

The code of the last two is shown step by step, not run, in the chapter 'Simulated analogues observed like the LMC'.
