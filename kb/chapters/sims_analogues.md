---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Simulated analogues observed like the LMC

Every number on this page is written by a script: `scripts/paper_text_sims.py` (`results/macros_sims.tex`) and
`scripts/paper_text_letter.py` (`results/macros_letter.tex`), from the result tables; the code cells are the figure script
`scripts/paper_sims_kinematics.py`, step by step.

The kinematics chapter found that the oldest, most metal-poor red giants of the LMC rotate at
$v_{\rm rot}(4\,{\rm kpc})\approx$ {glue:text}`ltKinVrotMP` km/s with a vertical velocity dispersion $\sigma_z\approx$ {glue:text}`ltKinSigMP` km/s:
rotation dominates even in the most metal-poor population. Here $v_{\rm rot}(4\,{\rm kpc})$ is the mean azimuthal velocity of a
population of stars, from its fitted rotation curve, at cylindrical radius $R$ = {glue:text}`ltRSig` kpc in the plane of the disc; $\sigma_z$ is the
dispersion of the velocities perpendicular to that plane, at the same radius; and [Fe/H], the iron abundance, is the logarithm of a
star's iron-to-hydrogen ratio relative to the Sun's. The ratio $v_{\rm rot}/\sigma_z$, the rotational support, measures how far a
population is supported against gravity by rotation rather than by random motions: a cold disc has $v_{\rm rot}/\sigma_z$ of several,
a hot spheroid well below 1. The giants are divided, as in the Letter, into five bins of [Fe/H] with edges at {glue:text}`ltFehA`, {glue:text}`ltFehB`, {glue:text}`ltFehC`
and {glue:text}`ltFehD`.

Here we ask whether galaxies of the LMC's mass formed in cosmological simulations have the same property. A like-for-like answer
needs the simulations measured as the LMC was: the same tracer, the same sky positions and errors, the same metallicity bins and the
same kinematic model. We therefore 'observe' every simulated galaxy and compare rotation, dispersion and rotational support against
metallicity.

## The simulations

We use three public suites of hydrodynamical simulations, which follow dark matter, gas and stars together, with LMC-mass galaxies
and star particles (each a population of stars of one age and metallicity) at redshift $z=0$.

- [TNG50-1](https://www.tng-project.org/) ([Nelson et al. 2019a](https://doi.org/10.1093/mnras/stz2306);
  [Pillepich et al. 2019](https://doi.org/10.1093/mnras/stz2338); [Nelson et al. 2019b](https://doi.org/10.1186/s40668-019-0028-x)),
  the highest-resolution run of the IllustrisTNG cosmological volumes: all {glue:text}`ltNTng` central galaxies (the main galaxy of
  their group, not a satellite) with bound stellar mass {glue:text}`ltMstarLo`–{glue:text}`ltMstarHi` × 10⁹ M☉ and subhalo mass (all mass bound to the galaxy)
  ${glue:text}`ltMtotLo`$–${glue:text}`ltMtotHi`$ M☉; baryonic mass resolution 8.5 × 10⁴ M☉ (Pillepich et al. 2019).
- Auriga, level 3 ([Grand et al. 2024](https://doi.org/10.1093/mnras/stae1598)):
  zoom-in simulations, which resimulate one halo of a cosmological volume at high resolution; we use the five dwarf-galaxy zooms
  Au{glue:text}`ltAuNames` (haloes of ≈10¹¹ M☉).
- FIRE-2 ([Wetzel et al. 2023](https://doi.org/10.3847/1538-4365/acb99a)): the public
  zooms {glue:text}`ltFireNames` at 7100 M☉ resolution.

The TNG50 analogues are selected by mass alone, as $z=0$ centrals; none is required to have an SMC-like companion or an LMC-like
orbit. Their dark-matter particles have 4.5 × 10⁵ M☉ (Pillepich et al. 2019); those of the Auriga and FIRE-2 zooms,
{glue:text}`smDmAuLo`–{glue:text}`smDmAuHi` × 10⁴ and ${glue:text}`smDmFire`$ M☉ (read from the snapshots), {glue:text}`smDmFactor` times less. Stellar metallicities are
$[{\rm Fe/H}]=\log_{10}[(X_{\rm Fe}/X_{\rm H})/(X_{\rm Fe}/X_{\rm H})_\odot]$, with $X$ the mass fractions of iron and hydrogen of the
particle and the solar abundance of [Asplund et al. (2021)](https://doi.org/10.1051/0004-6361/202140445). The formed (birth) mass of
a star particle is the mass of stars it formed, before stellar evolution returned part of it to the gas. It is taken from the
snapshot where available (Auriga) and otherwise from the present mass divided by the present-to-formed mass fraction of a stellar
population with the [Kroupa (2001)](https://doi.org/10.1046/j.1365-8711.2001.04022.x) initial mass function, of the particle's age and
metallicity (the conversion of the stellar-mass chapter); for Auriga this estimate is {glue:text}`smAuFormedLo`–{glue:text}`smAuFormedHi` per cent below the snapshot's birth
masses.

## Observing the simulations

For every simulated galaxy we build a mock version of the sample of red giants and fit it exactly as the data. The data are the
giants with SDSS-V BOSS or APOGEE spectra (two spectroscopic surveys; BOSS is the optical, APOGEE the near-infrared one) and Gaia
proper motions, described in the data chapters.

1. *Frame and placement.* Star particles within {glue:text}`ltRMax` kpc of the galaxy centre are rotated so that $z$ lies along the angular
   momentum of the stars within {glue:text}`ltAlignR` kpc, with velocities relative to their mean. The galaxy is then placed at the LMC: the centre,
   inclination (the tilt of the disc plane to the sky), line of nodes (where the disc plane crosses the sky), systemic proper motion
   and line-of-sight velocity (the motion of the LMC's centre) of the joint kinematic fit (kinematics chapter),
   distance {glue:text}`ltDist` kpc, with the simulation's sense of rotation mapped onto the LMC's. Each particle keeps its own 3D position, so
   the thickness of the simulated galaxy enters the projection as it does for the LMC.
2. *Tracer.* Each particle contributes $E=m_{\rm formed}\,\eta_{\rm BOSS}(\tau,[{\rm Fe/H}])$ red giants in the BOSS selection box,
   where $m_{\rm formed}$ is its formed mass, $\tau$ its age and $\eta_{\rm BOSS}$ the number of giants in the box per unit formed mass
   of the stellar-mass chapter.
3. *Metallicity.* The simulated metallicity distributions differ from the LMC's in zero point and shape, and the difference is partly
   a matter of nucleosynthetic yields. We therefore compare populations of the same *rank*: each particle's [Fe/H] is mapped onto the
   LMC scale so that the cumulative formed mass below it, as a fraction of the galaxy's formed mass within {glue:text}`smRhoMass`° of its centre
   on the sky, equals the LMC's (stellar-mass chapter). A simulated bin then holds the same fraction of its galaxy's stellar mass as
   the LMC bin of the same [Fe/H]: the most metal-poor bin, [Fe/H] < {glue:text}`ltFehA`, holds the {glue:text}`smFirstFrac` per cent of the formed mass lowest in
   metallicity. These are old stars (median age {glue:text}`smRankAge` Gyr), but not simply the oldest: only {glue:text}`smOldestInRank` per cent of the oldest
   stars holding the same fraction of the mass by age fall in this bin. The mapping is applied to the LMC's measured distribution,
   and BOSS errors are then added (step 4), so the simulated bins are blurred somewhat more than the LMC's; this mixes metal-richer,
   more rotating stars into the metal-poor bin.
4. *Where and how well.* The real giants fix where we look. Each real giant at {glue:text}`ltRingIn`° < $\rho$ < {glue:text}`ltRingOut`°, where $\rho$ is the
   angular distance from the LMC centre, lies in a cell of the fit frame ({glue:text}`ltCellR` kpc in in-plane radius × {glue:text}`ltCellPhi`° in azimuth); for
   each, one mock giant is drawn from the particles of the same cell with probability $\propto E$, inside the same velocity window
   ({glue:text}`ltVlosLo` < $v_{\rm los}$ < {glue:text}`ltVlosHi` km/s, with $v_{\rm los}$ the line-of-sight velocity). It inherits the real giant's proper-motion
   covariance, scaled by the Gaia error factor $k$ fitted for the real bin (the factor by which the kinematic model inflates the
   catalogued proper-motion errors), and its line-of-sight error, and its measured [Fe/H] is the mapped value plus an error: none for
   an APOGEE giant, a random BOSS − APOGEE difference of the stars observed by both surveys for a BOSS giant. The mock sample
   therefore has the size, sky distribution, error budget and metallicity blurring of the real one. Its [Fe/H] bins differ in size
   from the real ones (for [Fe/H] < {glue:text}`ltFehA`, a median {glue:text}`smNMockMP` mock giants against {glue:text}`smNRealMP`), because the analogues' stars of a
   given [Fe/H] rank differ in age and so in giants per unit mass.
5. *Fit.* Each mock [Fe/H] bin is fitted on its own with the ellipsoid model of the kinematics chapter (a thin rotating disc whose
   velocity ellipsoid has dispersions $\sigma_R$, $\sigma_\phi$, $\sigma_z$ along the radial, azimuthal and vertical directions),
   by maximum likelihood with all parameters free (stage 1 of the fit); bins with more than {glue:text}`ltNMaxFit` mock giants are fitted on a
   random {glue:text}`ltNMaxFit`.

The procedure recovers the particles' own kinematics ({fig}`paper_sims_recovery`), measured as the mean $v_\phi$ and $\sigma_z$ of the
particles that became the mock giants, within {glue:text}`smTruthDR` kpc of $R$ = {glue:text}`ltRSig` kpc. Over all {glue:text}`smNBins` analogue bins the fitted
$v_{\rm rot}(4\,{\rm kpc})$ exceeds the particles' by {glue:text}`smRecDv` km/s on the median (16th–84th percentiles {glue:text}`smRecDvLo` to {glue:text}`smRecDvHi` km/s) and
$\sigma_z$ by {glue:text}`smRecSig` per cent. The metal-poor bin is the noisiest ({glue:text}`smRecMPLo` to {glue:text}`smRecMPHi` km/s in $v_{\rm rot}$, {glue:text}`smRecSzMPLo` to {glue:text}`smRecSzMPHi` per
cent in $\sigma_z$), but its $v_{\rm rot}/\sigma_z$ is recovered to {glue:text}`smRecRatioMP` ({glue:text}`smRecRatioMPLo` to {glue:text}`smRecRatioMPHi`). The model does not allow
counter-rotation, so a bin whose particles counter-rotate is fitted with $v_{\rm rot}=0$.

### The mock observation, step by step

`scripts/sims_observe.py` carries out steps 1–5 for every analogue and records the particle values against which the fits are
checked. Its function `setup` reads what all analogues share: the frame of the joint kinematic fit, the real giants with their errors,
cells and error factors $k$, the differences between the BOSS and APOGEE [Fe/H] of the stars observed by both surveys, the LMC's
cumulative formed mass against [Fe/H], and the number of BOSS giants per unit formed mass. `particles` does steps 1–3 for one
analogue: it turns the galaxy into the frame of its spin, places it at the LMC, and gives every particle its LMC-scale [Fe/H] and its
expected number of giants $E$. `observe` does step 4: for each real giant it draws one particle of the same cell with probability
proportional to $E$ and gives it that giant's errors. The main loop splits the mock giants into the five [Fe/H] bins, takes the mean
$v_\phi$ and the dispersions of their particles near $R$ = {glue:text}`ltRSig` kpc, and fits each bin (step 5).

Each analogue is written to a file of its own, so that the work can be shared among parallel processes and restarted;
`python scripts/sims_observe.py merge` then joins the files into `results/sims_observed.csv`, the table from which all the figures of
this chapter are drawn.

<!-- script: sims_observe.py show: the star particles of all simulated analogues (scripts/fetch_tng50.py, fetch_auriga.py, fetch_fire2.py); about 13 h on 5 parallel processes -->

### The formed mass of each analogue

{fig}`paper_sims_kinematics` places each population of an analogue at the stellar mass formed before it: its cumulative fraction of
the formed mass (step 3) times the galaxy's total formed mass. `scripts/build_sims_formed_mass.py` gives that total, the sum of the
formed masses of the star particles within {glue:text}`ltRMax` kpc of the centre (`results/sims_formed_mass.csv`).

<!-- script: build_sims_formed_mass.py show: the star particles of all simulated analogues (scripts/fetch_tng50.py, fetch_auriga.py, fetch_fire2.py); seconds -->

## Rotation and dispersion against metallicity

{fig}`paper_sims_kinematics` compares the rotation, dispersion and rotational support of the observed analogues with those of the LMC
giants. Three choices make the comparison local and like-for-like. The kinematics are read at one radius, $R$ = {glue:text}`ltRSig` kpc, inside the
footprint of the real giants, rather than averaged over each galaxy's own density profile: the analogues' metal-rich stars are more
centrally concentrated than the LMC's, and a galaxy-wide average would mix their kinematics with their concentration. The
simulations are matched to the LMC in the cumulative fraction of the formed mass below each population in metallicity (step 3), so the
most metal-poor bin is the first {glue:text}`smFirstFrac` per cent of the mass. {fig}`paper_sims_kinematics` shows the populations against the stellar
mass formed before them, that fraction times the galaxy's total formed mass, the quantity in which disc settling is usually discussed
([Kassin et al. 2012](https://doi.org/10.1088/0004-637X/758/2/106); Pillepich et al. 2019); the analogues formed {glue:text}`smMassRatio` times more stars
than the LMC (median ${glue:text}`smMSim`$ against ${glue:text}`smMLmc`$ M☉), so each of their bins lies at {glue:text}`smMassRatio` times the LMC's mass. The same comparison
against the mass fraction, where every bin of every galaxy is at the same position, is {fig}`paper_sims_kinematics_fraction`. And
rotational support is $v_{\rm rot}/\sigma_z$: at the LMC's inclination $\sigma_z$ rests mostly on the line-of-sight velocities, so it
is the dispersion least sensitive to the proper-motion error scale and to in-plane streaming, and, because the analogues are more
anisotropic than the LMC ($\sigma_R/\sigma_z$ = {glue:text}`smAnisoSim` against {glue:text}`smAnisoLmc` for [Fe/H] < {glue:text}`ltFehA`), it is the definition most favourable to
them. Other definitions are compared in {tab}`sims_definitions`. The mock giants are drawn at the real giants' positions with all bins
together, so within a bin their spatial distribution is the analogue's own.

At the metal-rich end, the TNG50 analogues bracket the LMC ($v_{\rm rot}$ = {glue:text}`smVrotMRLo`–{glue:text}`smVrotMRHi` km/s for the 16th–84th percentiles in the
bin [Fe/H] > {glue:text}`ltFehD`, against {glue:text}`ltKinVrotMR` km/s), and {glue:text}`smNAuFast` of the five Auriga analogues rotate as fast ({glue:text}`smAuFast` km/s); all Auriga and
FIRE-2 analogues are hotter than the LMC at every metallicity.

At the metal-poor end no analogue resembles the LMC. In the bin [Fe/H] < {glue:text}`ltFehA` the analogues rotate at $v_{\rm rot}$ = {glue:text}`smTngVrotMP` km/s
(TNG50 median; 16th–84th percentiles {glue:text}`smTngVrotMPLo`–{glue:text}`smTngVrotMPHi` km/s; Auriga {glue:text}`smAuVrotMPLo`–{glue:text}`smAuVrotMPHi`, FIRE-2 {glue:text}`smFireVrotMPLo`–{glue:text}`smFireVrotMPHi` km/s, lowest to highest) against
the LMC's {glue:text}`ltKinVrotMP` km/s, and are hotter, $\sigma_z$ = {glue:text}`smTngSzMP` km/s (TNG50; {glue:text}`smTngSzMPLo`–{glue:text}`smTngSzMPHi`; Auriga {glue:text}`smAuSzMP`, FIRE-2 {glue:text}`smFireSzMP`) against {glue:text}`ltKinSigMP` km/s; the
coldest has {glue:text}`smColdSz` km/s. Their rotational support is $v_{\rm rot}/\sigma_z$ = {glue:text}`smTngRatioMP` (TNG50; {glue:text}`smTngRatioMPLo`–{glue:text}`smTngRatioMPHi`), {glue:text}`smAuRatioMP` (Auriga) and
{glue:text}`smFireRatioMP` (FIRE-2); the highest of all {glue:text}`smNAll` analogues is {glue:text}`smRatioMaxMP`, against the LMC's {glue:text}`smLmcRatioMP` ± {glue:text}`smLmcRatioMPErr`, and the highest of their
particle values is {glue:text}`smPartMaxMP`. In the next bin, {glue:text}`ltFehA` < [Fe/H] < {glue:text}`ltFehB` ({glue:text}`smFirstFrac`–{glue:text}`smFracB` per cent of the mass), {glue:text}`smNAboveOne` analogues
have $v_{\rm rot}/\sigma_z$ > 1 and {glue:text}`ltTngNAboveB` reaches the LMC's {glue:text}`smLmcRatioB` ({glue:text}`smRatioMaxB`). At the LMC's own formed mass below that bin,
${glue:text}`smMassB`$ M☉, none does: TNG50 populations with the same mass formed before them have $v_{\rm rot}$ = {glue:text}`smTngVrotAtB` km/s,
$\sigma_z$ = {glue:text}`smTngSzAtB` km/s and $v_{\rm rot}/\sigma_z$ = {glue:text}`smTngRatioAtB` (median), against the LMC's {glue:text}`smLmcVrotB` km/s, {glue:text}`smLmcSzB` km/s and {glue:text}`smLmcRatioB`. The LMC's most
metal-poor giants (${glue:text}`smMassA`$ M☉ formed before them) lie below the mass of the first bin of all but {glue:text}`smNCoverA` analogues, and
{glue:text}`smNAboveA` of those {glue:text}`smNCoverA` reach the LMC there. From the third bin on, which begins at {glue:text}`smFracB` per cent of the mass and ends at {glue:text}`smFracC` per
cent, the analogues reach the LMC: {glue:text}`smNMassC`, {glue:text}`smNMassD` and {glue:text}`smNMassE` of the {glue:text}`smNCovC`, {glue:text}`smNCovD` and {glue:text}`smNCovE` analogues whose bins span the LMC's mass in the
three metal-richer bins lie at or above its band there ({glue:text}`smNBinC`, {glue:text}`smNBinD` and {glue:text}`smNBinE` bin by bin). In the simulations rotational
support rises steeply with metallicity, from a hot, slowly rotating metal-poor component to a disc; in the LMC it changes by a factor
of {glue:text}`smFactorLo`–{glue:text}`smFactorHi` over the same range (to {glue:text}`ltKinRatioMR` for APOGEE+BOSS and {glue:text}`ltKinRatioMRApo` for APOGEE alone at [Fe/H] > {glue:text}`ltFehD`), and the
metal-poor giants are already rotation-dominated. The shape of their velocity ellipsoid, on the other hand, is ordinary: the LMC's
$\sigma_z/\sigma_R$ = {glue:text}`smShapeLmc` for [Fe/H] < {glue:text}`ltFehA` lies inside the analogues' range ({glue:text}`smShapeLo`–{glue:text}`smShapeHi`, 16th–84th percentiles).

The gap does not depend on how the old population is defined. Particle values at $R$ = {glue:text}`ltRSig` kpc, without fits or the resampling of
step 4, give a median $v_\phi/\sigma_z$ of {glue:text}`smPartRank` for the mapped bin, {glue:text}`smPartNative` for the simulations' own [Fe/H] < {glue:text}`ltFehA`, and {glue:text}`smPartOldest`
for the oldest stars holding the same fraction of the formed mass, with maxima of {glue:text}`smPartRankMax`, {glue:text}`smPartNativeMax` and {glue:text}`smPartOldestMax` among the
{glue:text}`smNAll` analogues, all below the LMC's {glue:text}`smLmcRatioMP` (`results/sims_populations.csv`, from `scripts/fig_sims_populations.py`). Nor does
it depend on the dispersion used ({tab}`sims_definitions`): with $\sigma_R$ or the root-mean-square of the three components, fitted or
from the particles, no analogue reaches the LMC in the most metal-poor bin, and in the next bin {glue:text}`smDefAny` reach its lowest value under
at least one of the six definitions, {glue:text}`smDefLo` to {glue:text}`smDefHi` under any single one. The choice matters only where the simulated discs begin: in
the third bin {glue:text}`smDefCLo` to {glue:text}`smDefCHi` analogues reach the LMC, depending on the definition.

<!-- script: paper_sims_kinematics.py -->

**Figure (paper_sims_recovery).** Simulated analogues observed and fitted like the giants: fitted (a) $v_{\rm rot}(4\,{\rm kpc})$
(km/s), (b) $\sigma_z(4\,{\rm kpc})$ (km/s) and (c) $v_{\rm rot}/\sigma_z$ against the mean $v_\phi$, the $\sigma_z$ and their ratio
for the mock giants' particles within {glue:text}`smTruthDR` kpc of $R$ = {glue:text}`ltRSig` kpc, for every analogue and [Fe/H] bin with at least {glue:text}`ltNMinSim` mock
giants; blue: TNG50, orange: Auriga, green: FIRE-2; grey line: equality. Points at fitted $v_{\rm rot}=0$ are bins whose particles
counter-rotate. `scripts/paper_sims_kinematics.py`

**Figure (paper_sims_kinematics).** The LMC giants against simulated LMC analogues observed and fitted like them, at the radius
$R$ = {glue:text}`ltRSig` kpc: (a) mean rotation $v_{\rm rot}(4\,{\rm kpc})$ (km/s), (b) vertical velocity dispersion $\sigma_z(4\,{\rm kpc})$
(km/s), (c) rotational support $v_{\rm rot}/\sigma_z$, against the stellar mass formed before each population (M☉, logarithmic
axis): its cumulative fraction of the formed mass in metallicity, the quantity the simulations are matched in, times the galaxy's
total formed mass (${glue:text}`smMLmc`$ M☉ for the LMC within {glue:text}`smRhoMass`°, each analogue its own within {glue:text}`ltRMax` kpc), so every galaxy has its
own $x$; the TNG50 band is drawn at the median TNG50 mass of each bin. The top axis and the thin vertical grey lines mark the LMC's
[Fe/H] bin edges. Numbers in (c): per cent of the analogues whose bins span the LMC's mass that lie at or above the LMC band there.
Black band: the LMC giants (the band spans the 16th–84th percentiles of APOGEE+BOSS and, where fitted, [Fe/H] > {glue:text}`ltApoFeh`, APOGEE alone
±1σ, which is colder). White diamonds: the RR Lyrae, placed at their median [Fe/H] with a horizontal bar over its 16th–84th
percentiles; their proper motions alone do not measure $\sigma_z$, so (b) and (c) show their $\sigma_R$ and $v_{\rm rot}/\sigma_R$
instead. Blue: the {glue:text}`ltNTng` TNG50 analogues, median (thick line), 16th–84th and 2.5th–97.5th percentiles (thin solid and dashed
lines) and probability density (shading, opacity proportional to the density in each column; kernel density estimate per bin);
orange: the {glue:text}`ltNAu` Auriga analogues; green: the {glue:text}`ltNFire` FIRE-2 analogues, one line each. All simulated values are fitted like the
LMC's, in its five [Fe/H] bins, and joined between bins by a monotone cubic in the logarithm of the mass.
`scripts/paper_sims_kinematics.py`

```{code-cell} python
:tags: [remove-input]
import pandas as pd
# Per giant bin: number of analogues at or above the LMC with v_rot over sigma_z, sigma_R or sigma_RMS, fitted or from the
# particles; 'any': analogues reaching the lowest LMC value under any of the six (scripts/paper_sims_kinematics.py)
d = pd.read_csv('results/sims_definitions.csv')
t = d.set_index(d.population.str.replace('giants ', ''))[['z', 'z particles', 'R', 'R particles', 'RMS', 'RMS particles', 'any']]
t.columns = ['σz fit', 'σz part.', 'σR fit', 'σR part.', 'σRMS fit', 'σRMS part.', 'any']
t.index.name = None
t
```

**Table (sims_definitions).** Simulated analogues, of {glue:text}`smNAll`, whose rotational support at $R$ = {glue:text}`ltRSig` kpc reaches the LMC's, per giant
[Fe/H] bin, for six definitions: $v_{\rm rot}$ over $\sigma_z$, $\sigma_R$ or
$\sigma_{\rm RMS}=[(\sigma_R^2+\sigma_\phi^2+\sigma_z^2)/3]^{1/2}$, fitted (fit) or from the mock giants' particles within {glue:text}`smTruthDR` kpc of
{glue:text}`ltRSig` kpc (part.). The LMC value is the lower of APOGEE+BOSS (16th percentile) and APOGEE alone (−1σ); for $\sigma_{\rm RMS}$,
the lower of their central values. Any: analogues reaching the lowest LMC value under any definition (`results/sims_definitions.csv`,
written by `scripts/paper_sims_kinematics.py`).

### Other definitions of the old population

`scripts/fig_sims_populations.py` gives the particle values quoted above for the three definitions of the old population. For each
analogue it takes the particles that could be sample giants near $R$ = {glue:text}`ltRSig` kpc and measures their $v_\phi/\sigma_z$,
each particle weighted by its expected number of giants, for the old population defined by the LMC-scale [Fe/H] (the definition of
this chapter), by the simulation's own [Fe/H], and as the oldest stars holding the same fraction of the formed mass; it also takes
the same quantity for the particles of the mock giants. The text's numbers come from its table, `results/sims_populations.csv`; its
figure, each definition against the first, is a diagnostic and not part of the paper.

<!-- script: fig_sims_populations.py show: the star particles of all simulated analogues and results/sims_observed.csv (scripts/sims_observe.py) -->

## Details of the mock observation

- *Selection* (`scripts/fetch_tng50.py`, `lmcold.sims`): TNG50-1 centrals (`primary_flag = 1`) in the mass ranges above; the Auriga
  level-3 dwarfs and the FIRE-2 runs named above.
- *Placement* (`scripts/sims_observe.py`, `lmcold.sims.place`): star particles within `sims.R_MAX` = {glue:text}`ltRMax` kpc, rotated to the angular
  momentum of the stars within `sims.ALIGN_R` = {glue:text}`ltAlignR` kpc, put at the LMC's centre, distance, orientation and systemic motion of
  the joint fit, each particle at its own 3D position.
- *Tracer weight*: BOSS-selected red giants per particle from its formed mass, age and metallicity (`results/giant_conversion.csv`),
  from the same PARSEC isochrones and Kroupa initial mass function as the LMC's stellar mass.
- *Metallicity*: the LMC [Fe/H] of the same rank in cumulative formed mass (`lmcold.sims.lmc_scale`).
- *Sampling*: one mock giant per real giant, from the particles in the same disc cell (`CELL_R` × `CELL_PHI`) and line-of-sight
  velocity window (`VLOS_WINDOW`); errors as in step 4.
- *Fit and display*: each bin on its own, maximum likelihood, at most `NMAX` stars; bins of fewer than `NMIN_SIM` mock giants are not
  shown. The alternative definitions of rotational support are in `results/sims_definitions.csv`.

## The stars at birth

The next chapter asks whether the simulated old stars were born without rotation or lost it later. Its tables of the stars at
birth are made from the same analogues by the three scripts below.

### Birth cohorts in TNG50 and Auriga

TNG50 and Auriga record the position and velocity of every star particle at its birth. `scripts/build_sims_birth.py` groups the
stars within {glue:text}`ltRToday` kpc of the centre today (in Auriga, only those formed in the galaxy itself) into cohorts of
{glue:text}`ltCohort` Gyr of age. The galaxy moves through the simulation volume while it forms stars, so the birth positions and
velocities are taken relative to a track of the galaxy's centre and velocity against formation time, built from the stars
themselves. Each cohort is rotated to its own spin, and its mass-weighted $v_\phi$ and $\sigma_z$ are measured at
{glue:text}`ltRMinBirth` < $R$ < {glue:text}`ltRBirth` kpc; the same stars are measured today in the frame of the galaxy's present disc.
The table, `results/sims_birth.csv`, is the input of the Letter's Fig. 3 and of the birth points of the next chapter's figure of the
progenitors.

<!-- script: build_sims_birth.py show: the TNG50 birth files and the Auriga snapshots (scripts/fetch_tng50.py, fetch_auriga.py); about a minute on 6 cores -->

### Why the birth values are measured inside the LMC's radius

The birth values are measured at {glue:text}`ltRMinBirth`–{glue:text}`ltRBirth` kpc, inside the radius $R$ = {glue:text}`ltRSig` kpc at
which the LMC is measured. `scripts/sims_birth_radii.py` uses the same stars and frames to count the old star particles at birth in
that annulus and in an annulus around the LMC's radius, and to find where the old stars now near the LMC's radius were born
(`results/sims_birth_radii.csv`; its numbers are quoted on the page of the Letter's Fig. 3).

<!-- script: sims_birth_radii.py show: the TNG50 birth files and the Auriga snapshots (scripts/fetch_tng50.py, fetch_auriga.py); under a minute on 6 cores -->

### FIRE-2 at birth

The FIRE-2 release gives, for every star particle, its position and velocity at formation relative to its host galaxy.
`scripts/fire2_birth.py` groups the stars of the FIRE-2 analogues by the look-back time of their formation and measures the rotation
and vertical dispersion of each cohort at birth about the spin of the cohort's own stars, as for TNG50 and Auriga
(`results/research/fire_early_birth.csv`, the FIRE-2 birth values of the Letter and of the next chapter).

<!-- script: fire2_birth.py show: the FIRE-2 snapshots and host-coordinate files (scripts/fetch_fire2.py); seconds -->
