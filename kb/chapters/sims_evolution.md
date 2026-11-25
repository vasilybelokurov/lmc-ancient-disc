---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# How the simulated old stars lose their rotation

Every number on this page is written by a script: `scripts/paper_text_sims_birth.py` (`results/macros_sims_birth.tex`),
`scripts/paper_text_sims_suites.py` (`results/macros_sims_suites.tex`), `scripts/paper_text_letter.py` (`results/macros_letter.tex`)
and `scripts/paper_text_sims.py` (`results/macros_sims.tex`), from the result tables; the code cells are the figure scripts
`scripts/paper_sims_*.py`, step by step.

The previous chapter showed that no simulated LMC analogue, observed and fitted like the LMC, has a rotating metal-poor population:
their most metal-poor stars have a rotational support $v_{\rm rot}/\sigma_z\approx$ {glue:text}`ltTngRatioMP` against the LMC's {glue:text}`ltKinRatioMP`. Here $v_\phi$ (or
$v_{\rm rot}$ for a fitted rotation curve) is the mean azimuthal velocity of a group of stars about the spin axis of the galaxy, $\sigma_z$
the dispersion of their velocities along that axis, and $R$ the cylindrical radius from it. Times are given as redshift $z$ ($z=0$
today, larger in the past) or in Gyr. This chapter asks whether the simulated
old stars were born hot or became hot, and what heated them. A cohort is a group of star particles born within a short interval of
time; the look-back time is the time before the present at which an event happened, and the age of a star today equals the
look-back time of its birth.

## Born rotating

Both TNG50 ([Pillepich et al. 2019](https://doi.org/10.1093/mnras/stz2338)) and Auriga
([Grand et al. 2024](https://doi.org/10.1093/mnras/stae1598)) record each star particle's birth position and velocity, so we can ask
whether the old stars formed hot or were heated later ({fig}`paper_sims_birth`). They formed rotating. Measured relative to the galaxy
they formed in, the stars now {glue:text}`smOldAgeLo`–{glue:text}`smOldAgeHi` Gyr old had $v_\phi$ = {glue:text}`sbVphiBirth` km/s, $\sigma_z$ = {glue:text}`sbSzBirth` km/s and $v_\phi/\sigma_z$ = {glue:text}`sbVsBirth` in the median
TNG50 analogue (16th–84th percentiles over the analogues {glue:text}`sbVsBirthLo`–{glue:text}`sbVsBirthHi`; Auriga {glue:text}`sbAuVsBirth`), and {glue:text}`sbNReach` of the {glue:text}`sbNGal` TNG50 analogues
with such stars reached the LMC's {glue:text}`sbLmcVs`. Today the same stars have $v_\phi$ = {glue:text}`sbVphiToday` km/s and $\sigma_z$ = {glue:text}`sbSzToday` km/s ($v_\phi/\sigma_z$ = {glue:text}`sbVsToday`,
at most {glue:text}`sbVsTodayHi` at the 84th percentile; Auriga {glue:text}`sbAuVsToday`), and none reaches it. The LMC's three metal-poor bins lie below the TNG50
median at birth, but within its 16th–84th percentiles, at the ages that their metallicity implies ($v_{\rm rot}/\sigma_z$ = {glue:text}`sbLmcVsOne`, {glue:text}`sbLmcVsTwo`
and {glue:text}`sbLmcVsThree` against {glue:text}`sbTngVsOne`, {glue:text}`sbTngVsTwo` and {glue:text}`sbTngVsThree` at {glue:text}`sbLmcAgeOne`, {glue:text}`sbLmcAgeTwo` and {glue:text}`sbLmcAgeThree` Gyr; {fig}`paper_sims_birth`c): the LMC's old population still has most of
the rotational support with which the simulated old stars were born. Their vertical dispersion at birth, {glue:text}`sbSzBirth` km/s, is lower than
the LMC's old giants have today ({glue:text}`sbLmcSzLo`–{glue:text}`sbLmcSzHi` km/s), so if the LMC's old stars were born as the simulated ones were, they too were
heated, but less: $\sigma_z^2$ grew by about {glue:text}`sbHeatLmc` km² s⁻², against {glue:text}`sbHeatSims` in the simulations. TNG50 forms its young stars as cold as the
LMC's metal-rich giants ($\sigma_z\approx$ {glue:text}`sbYoungSz` km/s at birth); Auriga forms stars of every age with $\sigma_z\approx$ {glue:text}`sbAuSzLo`–{glue:text}`sbAuSzHi` km/s. Rotation at
birth is measured less well than dispersion: for stars younger than {glue:text}`smYoungAge` Gyr, which have had little time to change, TNG50 gives
$v_\phi$ = {glue:text}`sbYoungVphiBirth` km/s at birth against {glue:text}`sbYoungVphiToday` km/s today, so the birth rotation may be low by up to that difference.

<!-- script: paper_sims_birth.py -->

{fig}`paper_sims_heating` follows the main progenitors of {glue:text}`ltNSample` of the TNG50 analogues through time (the main progenitor is
the galaxy's most massive ancestor at each earlier time, from the merger tree), to see in what conditions the old stars were born and
when and how they lost their rotation. At birth (panel a), rotation is set by the stellar mass the galaxy had at the time: stars born
in progenitors of $10^{{glue:text}`smDexA`}$–$10^{{glue:text}`smDexB`}$ M☉ rotate at $v_\phi$ = {glue:text}`shVphiLow` km/s, those born at $10^{{glue:text}`smDexC`}$–$10^{{glue:text}`smDexD`}$ M☉ at {glue:text}`shVphiHigh` km/s, while their
vertical dispersion stays at {glue:text}`shSzLo`–{glue:text}`shSzHi` km/s. The star-forming gas rotates faster than the stars born from it at every mass ({glue:text}`shGasVphiLow`
and {glue:text}`shGasVphiHigh` km/s in the same two ranges) and is as cold, $\sigma_z$ = {glue:text}`shGasSzLo`–{glue:text}`shGasSzHi` km/s, close to its effective sound speed ({glue:text}`shGasCs` km/s; in
TNG50 the pressure of the star-forming gas is set by an effective equation of state); its $v_\phi/\sigma_z$ rises from {glue:text}`shGasVsFirst` at
$z$ = {glue:text}`smGasZFirst` to {glue:text}`shGasVsLast` at $z$ = {glue:text}`shGasZLast`. The LMC's metal-poor giants rotate at {glue:text}`shLmcVrotLo`–{glue:text}`shLmcVrotHi` km/s although the LMC had formed only
$10^{{glue:text}`shLmcLogMLo`}$–$10^{{glue:text}`shLmcLogMHi`}$ M☉ in more metal-poor stars, a mass at which TNG50's stars are born rotating at {glue:text}`shTngVphiLo`–{glue:text}`shTngVphiHi` km/s. Followed
after birth (panel b), the stars born more than {glue:text}`smAgeOld` Gyr ago lose most of their rotation before the first tracked snapshot, a
median {glue:text}`shFirstStep` Gyr later ($v_\phi/\sigma_z$ from {glue:text}`shOldBirth` to {glue:text}`shOldFirst`, and {glue:text}`shOldTwo` after {glue:text}`smLaterOld` Gyr); those born {glue:text}`smAgeMid`–{glue:text}`smAgeOld` Gyr ago fall from
{glue:text}`shMidBirth` to {glue:text}`shMidFirst`; those born in the last {glue:text}`smAgeYoung` Gyr keep it ({glue:text}`shYoungBirth` to {glue:text}`shYoungFirst`) and lose it only slowly ({glue:text}`shYoungThree` after
{glue:text}`smLaterYoung` Gyr). The {glue:text}`smWindowHeat`-Myr cohorts born at $z$ = {glue:text}`ltZSnapB`–{glue:text}`ltZSnapA` in {glue:text}`ltNFull` progenitors followed at every snapshot (panel c)
lose most of the alignment of their orbital planes within {glue:text}`smCohEnd` Gyr: the coherence

$$
C=\frac{\left|\sum m\,\mathbf{j}\right|}{\sum m\,|\mathbf{j}|}
$$

falls from {glue:text}`shCohBirth` to {glue:text}`shCohEnd`, while the late control cohorts keep {glue:text}`shCohLate`. Here the sums run over the stars of a cohort,
$m$ is a star particle's mass and $\mathbf{j}=\mathbf{x}\times\mathbf{v}$ its angular momentum per unit mass, with positions $\mathbf{x}$ from
the progenitor's centre and velocities $\mathbf{v}$ relative to the mean velocity of its stars within {glue:text}`ltAlignR` kpc; a cohort is the
stars formed in the {glue:text}`smWindowHeat` Myr before a snapshot within {glue:text}`ltRCohort` kpc of the centre, identified by their particle numbers and
measured at each later snapshot within {glue:text}`ltAlignR` kpc. $C$ compares only the directions of the orbits: it is 1 when all lie in one
plane and turn the same way, does not change if the stars speed up or slow down or the frame is rotated, and is $\approx N^{-1/2}$ for
$N$ randomly oriented orbits ({glue:text}`shCohRandom` for the median cohort of {glue:text}`shCohN` stars). Panel (c) shows, in {glue:text}`smCohBin`-Gyr bins, the median over the
galaxies of each galaxy's median. The old stars are not slowed down; their orbits are turned in different directions, and the
cohort's rotation cancels. In the simulations this happens to the stars born more than {glue:text}`smAgeOld` Gyr ago in galaxies of every mass they
then had; the LMC's old stars kept their rotation, as do the late control cohorts, born at $z\leq$ {glue:text}`smLateZHi`.

<!-- script: paper_sims_heating.py -->

**Figure (paper_sims_heating).** Birth and early evolution of the stars of TNG50 LMC analogues, from the main progenitors of
{glue:text}`ltNSample` of them. (a) Conditions at birth against the logarithm of the stellar mass of the main progenitor at the time (M☉, from the
merger tree): mean rotation $v_\phi$ (solid) and vertical dispersion $\sigma_z$ (dashed), in km/s, of stars at birth (blue: medians over
the galaxies of the {glue:text}`sbCohort`-Gyr cohorts of {fig}`paper_sims_birth`, 16th–84th percentiles shaded) and of the star-forming gas (green
with dots: {glue:text}`smNGasSnaps` snapshots from $z$ = {glue:text}`smGasZFirst` to {glue:text}`shGasZLast`), both at {glue:text}`ltRMinBirth` < $R$ < {glue:text}`ltRBirth` kpc from the progenitor's centre. Black: the LMC
giant bins at the present-day stellar mass of the LMC's more metal-poor stars (a different estimator); squares
$v_{\rm rot}(4\,{\rm kpc})$, open triangles today's $\sigma_z$, an upper limit to the value at birth. (b) $v_\phi/\sigma_z$ of the same
stars against the time since their birth (Gyr), for stars born more than {glue:text}`smAgeOld` (red), {glue:text}`smAgeMid`–{glue:text}`smAgeOld` (orange) and less than {glue:text}`smAgeYoung` Gyr ago
(blue); the first point after birth is the first tracked snapshot (every {glue:text}`ltTrackStep` snapshot, a median {glue:text}`shFirstStep` Gyr after birth);
medians and 16th–84th percentiles over the galaxies; dotted: the LMC's giants with [Fe/H] < {glue:text}`ltFehA` today. (c) Coherence of the
orbital planes $C$ (1: all orbits in one plane with one sense) of the stars formed in the {glue:text}`smWindowHeat` Myr before each snapshot, against the
time since birth (Gyr), followed for {glue:text}`smCohEnd` Gyr at every snapshot in {glue:text}`ltNFull` of the progenitors, for cohorts born at
$z$ = {glue:text}`ltZSnapB`–{glue:text}`ltZSnapA` (red; median and 16th–84th percentiles over the galaxies) and late control cohorts born at $z\leq$ {glue:text}`smLateZHi` (blue squares, at birth and one
tracked snapshot later); dotted: the value for randomly oriented orbits of the median cohort size. `scripts/paper_sims_heating.py`

### The tables of the progenitors

`scripts/build_sims_heating.py` makes the three tables of {fig}`paper_sims_heating` from the main progenitors. For the star-forming
gas it measures, at a set of snapshots, the rotation, vertical dispersion and sound speed at {glue:text}`ltRMinBirth` < $R$ <
{glue:text}`ltRBirth` kpc about the spin of the gas itself. For the stars it measures each cohort at birth as in
`scripts/build_sims_birth.py`, with the progenitor's stellar mass at the time from the merger tree, and then finds the same stars, by
their particle numbers, at every later tracked snapshot and measures them in the progenitor's disc frame. For the coherence it takes
the stars formed in the {glue:text}`smWindowHeat` Myr before each snapshot and follows them at every later snapshot.

<!-- script: build_sims_heating.py show: the merger trees and cutouts of the TNG50 progenitors (scripts/fetch_tng50_progenitors.py) and the TNG50 birth files (scripts/fetch_tng50.py); about a minute in the recorded run -->

## What turns the orbits

To find what turns the orbital planes we repeat the first {glue:text}`ltDtMax` Gyr of each cohort with test particles, massless particles moved
in a given gravitational field ({fig}`paper_sims_scramble`). The {glue:text}`ltNCohorts` cohorts born at $z$ = {glue:text}`ltZSnapB`–{glue:text}`ltZSnapA` in {glue:text}`ltNCohortGal` of the {glue:text}`ltNFull`
progenitors followed at every snapshot (stars formed in the {glue:text}`ltWindow` Myr before a snapshot, median {glue:text}`smScrN` stars) are started from the
stars' positions and velocities at the birth snapshot and integrated in the gravity of their own progenitor. The gravity is a
multipole expansion, a sum of spherical harmonics $Y_{lm}$ of degree $l$ and order $m$ with radial coefficients, to $l$ = {glue:text}`ltLmax`, of all
the particles of the main subhalo within {glue:text}`ltRMaxPot` kpc of the stellar centre (made with AGAMA;
[Vasiliev 2019](https://doi.org/10.1093/mnras/sty2672)), made at every snapshot and interpolated linearly in time; the frame follows
the stellar centre, with the inertial acceleration that this adds. The particles are the Subfind members (the particles the halo
finder assigns to the subhalo) of the main subhalo, so the gravity of satellites is not included. Removing terms of the expansion
removes parts of the gravity, and the coherence $C$ of the test particles is measured as for the stars. With all terms the test
particles follow the stars: the median difference in $C$ is {glue:text}`smScrDC` (16th–84th percentiles {glue:text}`smScrDCLo` to {glue:text}`smScrDCHi`), with a correlation
coefficient of {glue:text}`smScrR` over the cohorts and snapshots. Below we quote each variant's loss of $C$ after {glue:text}`ltTimeB`–{glue:text}`ltTimeC` Gyr, summed over the
cohorts, as a fraction of the real stars' loss, with the 16th–84th percentiles of a bootstrap over the galaxies.

The orbits are turned by the non-axisymmetric gravity of the dark-matter halo ({fig}`paper_sims_scramble`a). The potential held fixed
at the birth snapshot turns them as much as the evolving one ({glue:text}`smScrE` of the real loss, {glue:text}`smScrELo`–{glue:text}`smScrEHi`), so no change of the potential
during the {glue:text}`ltDtMax` Gyr, by accretion or a merger, is needed. Its axisymmetric part about the cohort's spin ($m=0$) gives {glue:text}`smScrDZero`, and its
spherical part ($l=0$) {glue:text}`smScrB`. Making only the gas and stars spherical leaves {glue:text}`smScrGDM` ({glue:text}`smScrGDMLo`–{glue:text}`smScrGDMHi`); making only the dark matter
spherical leaves {glue:text}`smScrGBar` ({glue:text}`smScrGBarLo`–{glue:text}`smScrGBarHi`). Without the odd terms, which describe a lopsided or off-centre mass distribution, the loss is
unchanged ({glue:text}`smScrC`).

Which shape of the halo matters is tested in the frozen potential, with the halo changed in its own frame, given by the iterative
inertia tensor of the dark matter within {glue:text}`smHaloR` kpc ([Zemp et al. 2011](https://doi.org/10.1088/0067-0049/197/2/30)), whose
eigenvectors are the halo's major, intermediate and minor axes ({fig}`paper_sims_scramble`b). Rotating the halo so that its minor
axis lies along the stars' spin does not help: the loss is {glue:text}`smScrHTwo` ({glue:text}`smScrHTwoLo`–{glue:text}`smScrHTwoHi`) of the real one. Making it axisymmetric about its own
minor axis, which keeps its flattening and its tilt but removes its elongation in the plane, lowers the loss to {glue:text}`smScrHOne` overall; making
it axisymmetric about the stars' spin, which keeps only the flattening, removes it ({glue:text}`smScrDZeroFr`). How the loss divides between the two
remaining parts depends on the angle between the halo's minor axis and the stars' spin at birth. For cohorts born within {glue:text}`smMis`° of
alignment, the elongation in the plane does most of it: the mean loss falls from {glue:text}`smAlELo`–{glue:text}`smAlEHi` to {glue:text}`smAlHOneLo`–{glue:text}`smAlHOneHi` when the elongation is
removed. For cohorts born more than {glue:text}`smMis`° from alignment, the tilt of the flattened halo alone gives {glue:text}`smMisHOne`, as much as the full halo's
{glue:text}`smMisE`. Both are departures of the potential from axisymmetry about the stars' spin; the flattening alone does nothing. Rotating the
halo into alignment turns its tilt into an elongation in the plane, which turns the orbits as much.

The departure from axisymmetry about the disc is measured by $\epsilon_2=|\Phi_2|/|\Phi_0|$, the ratio of the $l=2$, $m\neq0$ terms
of the dark matter's potential, about the spin of the stars within {glue:text}`ltSetRIn` kpc, to its monopole, at {glue:text}`ltSetRZero` kpc (defined in full in the
next section). For {glue:text}`smNCohHalo` cohorts born at $z$ = {glue:text}`smZCohHi`–{glue:text}`smZCohLo` in all {glue:text}`ltNSample` progenitors (followed at every {glue:text}`ltTrackStep` snapshot), the fraction of
their coherence that they lose in the next {glue:text}`smStepLo`–{glue:text}`smStepHi` Gyr, $(C_0-C_1)/(C_0-C_{\rm random})$, where $C_0$ and $C_1$ are the coherence
at birth and one tracked snapshot later and $C_{\rm random}$ = {glue:text}`smCRandom` that of randomly oriented orbits, rises from {glue:text}`smFracBinOne` at
$\epsilon_2\approx$ {glue:text}`smEpsBinOne` to {glue:text}`smFracBinThree` at {glue:text}`smEpsBinThree` and {glue:text}`smFracBinFive` at {glue:text}`smEpsBinFive` ({fig}`paper_sims_scramble`c). The Spearman rank correlation is
{glue:text}`smRankAll`, {glue:text}`smRankZ` at fixed redshift (partial correlation) and {glue:text}`smRankGal` within galaxies (each galaxy's mean $\epsilon_2$ and mean
fraction subtracted). How quickly $\epsilon_2$
falls, and why, is the subject of the next section. Three caveats apply. The test particles do not attract each other. The expansion
is not softened, so inside {glue:text}`smForceR` kpc its force is up to {glue:text}`smForcePct` per cent stronger than the simulation's (median ratio at {glue:text}`smForceN` star
positions {glue:text}`smForceRLo`–{glue:text}`smForceR` kpc from the centre of one progenitor at $z$ = {glue:text}`smForceZ`, against a direct sum over its particles with the TNG50
softening). The early halos are nearly prolate, so their minor axis, and the angle in {fig}`paper_sims_scramble`b, are not sharply
defined.

The same happens in the two other suites, with stars formed under different models of star formation and feedback. We measured the
public LMC-mass runs of each, the {glue:text}`ltNAu` Auriga analogues and the {glue:text}`ltNFire` FIRE-2 ones ({glue:text}`ltFireNames`), as above, in {glue:text}`smWindowHeat`-Myr cohorts
followed through their high-cadence outputs (not shown). The oldest stars lose their rotation as quickly as in TNG50: those born more
than {glue:text}`ltSuOldLb` Gyr ago fall from $v_\phi/\sigma_z$ = {glue:text}`ssOldAuBirth` to {glue:text}`ssOldAuAfter` within {glue:text}`ltSuTAfter` Gyr in Auriga, and from {glue:text}`ssOldFireBirth` to {glue:text}`ssOldFireAfter`
in FIRE-2, and the orbital planes of the cohorts born at $z$ > {glue:text}`smZEarly` lose their alignment ($C$ from {glue:text}`ssCohAuBirth` to {glue:text}`ssCohAuEnd` and from
{glue:text}`ssCohFireBirth` to {glue:text}`ssCohFireEnd` within {glue:text}`smCohEnd` Gyr). The suites differ at birth. Stars born {glue:text}`smBirthLbLo`–{glue:text}`smBirthLbHi` Gyr ago have $v_\phi/\sigma_z$ = {glue:text}`ssBirthTngLo`–{glue:text}`ssBirthTngHi` at
birth in TNG50, {glue:text}`ssBirthAuLo`–{glue:text}`ssBirthAuHi` in Auriga and {glue:text}`ssBirthFireLo`–{glue:text}`ssBirthFireHi` in FIRE-2, whose dwarfs never form an ordered disc (even its cohorts
born less than {glue:text}`smFireLateLb` Gyr ago start at $C$ = {glue:text}`ssCohFireLate`), in line with the rarity of stellar discs among the dwarfs of the FIREbox
volume below 10⁹ M☉ ([Benavides et al. 2025](https://doi.org/10.1093/mnras/staf1847)). In Auriga the halo tests of
{fig}`paper_sims_scramble`b, run on its {glue:text}`ssAuN` cohorts at $z$ = {glue:text}`smAuZHi`–{glue:text}`smAuZLo`, give the TNG50 result: with the dark matter made round the loss
falls to {glue:text}`ssAuRound` of the full one, axisymmetric about the stars' spin to {glue:text}`ssAuAxi`, and with the halo's elongation in the plane removed to
{glue:text}`ssAuHOneAligned` for cohorts born within {glue:text}`smMis`° of alignment and {glue:text}`ssAuHOneTilted` for the others. What turns the orbits in FIRE-2 we did not test.

The Letter's version of this experiment follows $v_\phi/\sigma_z$ instead of $C$. On the {glue:text}`smAuN` Auriga cohorts (`scripts/auriga_scramble.py`,
`results/auriga_scramble.csv`; gravity frozen at the birth snapshot, as the full Auriga snapshots are too far apart to interpolate),
the real stars go from $v_\phi/\sigma_z$ = {glue:text}`smAuVsRealOne` at birth to {glue:text}`smAuVsRealTwo`, {glue:text}`smAuVsRealThree` and {glue:text}`smAuVsRealFour` at {glue:text}`smAuTTwo`, {glue:text}`smAuTThree` and {glue:text}`smAuTFour` Gyr; test particles in all
of the gravity go to {glue:text}`smAuVsETwo`, {glue:text}`smAuVsEThree` and {glue:text}`smAuVsEFour`, with only the dark matter non-spherical to {glue:text}`smAuVsGDMTwo`, {glue:text}`smAuVsGDMThree` and {glue:text}`smAuVsGDMFour`, and in its
part axisymmetric about the cohort's spin they keep their rotation ({glue:text}`smAuVsDZeroTwo`, {glue:text}`smAuVsDZeroThree` and {glue:text}`smAuVsDZeroFour`). In TNG50 the real stars go from
{glue:text}`ltScrStart` to {glue:text}`ltScrEnd` over the same time (medians over the galaxies of each galaxy's median, in the time bins of the Letter's figure).

<!-- script: paper_sims_scramble.py -->

**Figure (paper_sims_scramble).** What turns the orbital planes of the old stars in TNG50. (a) Which mass. Coherence $C$ of the
{glue:text}`ltNCohorts` cohorts born at $z$ = {glue:text}`ltZSnapB`–{glue:text}`ltZSnapA` in {glue:text}`ltNCohortGal` progenitors, against the time since their birth snapshot (Gyr): the real stars (black;
median over the galaxies of each galaxy's median, 16th–84th percentiles shaded grey) and test particles started from the same stars in
the multipole gravity of the progenitor, with all of it (blue), all of it fixed at the birth snapshot (orange), its part
axisymmetric about the cohort's spin (purple), the gas and stars made spherical, so that only the dark matter is non-round (red), or
the dark matter made spherical, so that only the gas and stars are non-round (green). The non-axisymmetric gravity of the dark matter
turns the orbits; the potential need not change in time. (b) Which shape. Mean loss of $C$ after {glue:text}`ltTimeB`–{glue:text}`ltTimeC` Gyr against the angle
(deg) between the halo's minor axis and the cohort's spin at birth (bars: 16th–84th percentiles of a bootstrap over the cohorts), in
the potential fixed at birth: the real stars (black), the full halo (orange), the same halo rotated to align with the stars (blue), the
halo made axisymmetric about its minor axis, keeping its flattening and tilt (red), and about the stars' spin, keeping only its
flattening (purple). Aligning the halo does not help. The elongation of the halo in the plane of the orbits turns them in aligned
halos (orange minus red), the tilt of the flattened halo in misaligned ones (red minus purple); flattening alone does nothing.
(c) The size of the asymmetry sets the loss. Fraction of the coherence lost over the next tracked snapshot,
$(C_0-C_1)/(C_0-C_{\rm random})$, with $C_{\rm random}$ = {glue:text}`smCRandom`, of {glue:text}`smNCohHalo` cohorts born at $z$ = {glue:text}`smZCohHi`–{glue:text}`smZCohLo` in {glue:text}`ltNSample` progenitors, against
the quadrupole $\epsilon_2$ at {glue:text}`ltSetRZero` kpc of the dark matter's potential, non-axisymmetric about the disc, at birth (logarithmic axis;
median and 16th–84th percentiles in bins). Below $\epsilon_2\approx$ {glue:text}`smEpsQuiet` the cohorts keep their planes; at the largest values they lose
more than half of their coherence. The early halos have the large values. `scripts/paper_sims_scramble.py`

### The test-particle experiments

`scripts/sims_scramble.py` runs the experiments of {fig}`paper_sims_scramble`a and b. For each progenitor it reads the snapshots of
the first {glue:text}`ltDtMax` Gyr after each cohort's birth, puts them in a frame centred on the stars, and takes the cohort's stars
at the birth snapshot as initial conditions. The gravity of the progenitor at each snapshot is a multipole expansion made with AGAMA,
written to a file of coefficients; a variant of the gravity is the same file with some coefficients set to zero. Expansions of all
the particles together, of the dark matter alone and of the gas and stars alone allow one kind of mass to be made spherical while
the other keeps its shape. A uniform acceleration is added to each potential so that the frame stays on the stellar centre. The
stars are then integrated as test particles in every variant, and their coherence $C$ and their $v_\phi/\sigma_z$ are measured at
the times of the snapshots, as for the real stars.

The Letter's Fig. 4 uses three versions of the gravity. In the first, all of the progenitor's gravity is kept but frozen at the birth
snapshot: if the test particles lose their rotation in it, no change of the potential in time is needed. In the second, the gas and
stars are made spherical (their expansion is kept only to $l=0$) and the dark matter keeps its full shape, so only the dark matter can
turn the orbits. In the third, all of the gravity is kept only in its terms with $m=0$, which makes it axisymmetric about the cohort's
spin, so that it exerts no torque about that axis. The second and third follow the evolution of the potential from snapshot to
snapshot. The test particles keep their rotation only in the third. A runnable version of this experiment for one cohort is on the
page of the Letter's Fig. 4.

<!-- script: sims_scramble.py show: the full cutouts of the TNG50 progenitors at every snapshot (scripts/fetch_tng50_progenitors.py) and AGAMA; a few minutes on 4 processes -->

### The same experiment in Auriga

`scripts/auriga_scramble.py` repeats the experiment of the Letter's Fig. 4 on the Auriga analogues, with the functions of
`scripts/sims_scramble.py`. The progenitor is traced back by the particle numbers of the analogue's stars today. The full Auriga
snapshots, with gas and dark matter, are too far apart to interpolate the gravity over the {glue:text}`ltDtMax` Gyr of the
experiment, so all three versions are frozen at the birth snapshot; the real stars are followed through the star-only outputs written
between the full snapshots.

<!-- script: auriga_scramble.py show: the Auriga level-3 snapshots and star-only outputs (scripts/fetch_auriga.py) and AGAMA; about a minute on 2 processes -->

### The orbital period of the early progenitors

The time axis of the Letter's Fig. 4 is also given in orbital periods at {glue:text}`ltROrb` kpc. `scripts/build_sims_vcirc.py` sums
the mass of stars, gas and dark matter within that radius of each progenitor at the birth snapshot of each early cohort and gives the
circular speed of a spherical mass, $v_c=\sqrt{GM(<r)/r}$, with $G$ the gravitational constant and $M(<r)$ the mass within $r$, and
the period $T_{\rm orb}=2\pi r/v_c$ (`results/sims_vcirc.csv`).

<!-- script: build_sims_vcirc.py show: the full cutouts of the TNG50 progenitors (scripts/fetch_tng50_progenitors.py); seconds -->

## How the halo settles

{fig}`paper_sims_settle` follows the centre of the dark-matter halo of the {glue:text}`ltNSample` progenitors from {glue:text}`ltLbEarly` to {glue:text}`ltLbLate` Gyr ago ($z$ = {glue:text}`ltZSetHi`–{glue:text}`ltZSetLo`),
at every {glue:text}`ltTrackStep` snapshot (every snapshot from $z$ = {glue:text}`ltZSnapA` to {glue:text}`ltZSnapB` for {glue:text}`ltNFull` of them; {glue:text}`ltNSettle` measurements, {glue:text}`smNPerGalLo`–{glue:text}`smNPerGalHi` per galaxy).
At each snapshot the origin is the position of the main subhalo, $\mathbf{x}_k$ are the physical positions of the particles $k$
relative to it, $\mathbf{v}_k$ their velocities and $m_k$ their masses, and the disc axis is the direction of the angular momentum
of the star particles within {glue:text}`ltSetRIn` kpc,

$$
\hat{\mathbf{e}}_z=\frac{\mathbf{L}}{|\mathbf{L}|},\qquad \mathbf{L}=\sum_{|\mathbf{x}_k|<r_{\rm in}} m_k\,\mathbf{x}_k\times(\mathbf{v}_k-\bar{\mathbf{v}}),
$$

with $r_{\rm in}$ = {glue:text}`ltSetRIn` kpc and $\bar{\mathbf{v}}$ their mass-weighted mean velocity. Only the dark matter is used below.

The shape of the halo is given by the second-moment tensor of the $N$ dark-matter particles within $r_{\rm in}$,

$$
I_{ij}=\frac{1}{N}\sum_{|\mathbf{x}_k|<r_{\rm in}} x_{k,i}\,x_{k,j},
$$

whose eigenvalues $a^2\geq b^2\geq c^2$ give the axis ratios $b/a$ (intermediate to major) and $c/a$ (minor to major), and whose
eigenvector $\hat{\mathbf{e}}_c$ of the smallest eigenvalue is the minor axis. The misalignment is the angle between the minor axis
and the disc axis,

$$
\psi=\arccos|\hat{\mathbf{e}}_c\cdot\hat{\mathbf{e}}_z|,\qquad 0\leq\psi\leq90^\circ .
$$

Unlike the iterative tensor of the previous section, this one is measured in a sphere, which biases $b/a$ and $c/a$ towards 1
([Zemp et al. 2011](https://doi.org/10.1088/0067-0049/197/2/30)); the true halo is somewhat flatter, and the trends with time are not
affected.

The asymmetry of the potential comes from a multipole expansion of the dark-matter particles within {glue:text}`ltSetRPot` kpc (AGAMA, to
$l$ = {glue:text}`ltSetLmax`, no symmetry imposed), in the frame with $\hat{\mathbf{e}}_z$ along the disc axis,

$$
\Phi(r,\theta,\phi)=\sum_{l=0}^{l_{\max}}\sum_{m=-l}^{l}\Phi_{lm}(r)\,Y_{lm}(\theta,\phi),
$$

where $r$, $\theta$ and $\phi$ are the spherical radius, polar angle from $\hat{\mathbf{e}}_z$ and azimuth, $l_{\max}$ = {glue:text}`ltSetLmax`, and $Y_{lm}$ are
real spherical harmonics, $\propto\cos m\phi$ for $m>0$ and $\propto\sin|m|\phi$ for $m<0$, normalised to unit mean square over the sphere
(so $\Phi_{00}$ is the spherical average of the potential). The quadrupole ($l=2$) has three parts. The $m=0$ term is a flattening along
the disc axis; it is axisymmetric about the disc, exerts no torque about that axis and does not turn the orbital planes
({fig}`paper_sims_scramble`b). The $m=\pm1$ terms ($\propto xz$, $yz$) appear when the halo's principal axes are tilted to the disc
axis, and the $m=\pm2$ terms ($\propto x^2-y^2$, $xy$) when the halo is elongated in the plane of the disc. Their amplitudes at
$r$ = {glue:text}`ltSetRZero` kpc, relative to the monopole, are

$$
\epsilon_{\rm tilt}=\frac{\sqrt{\Phi_{2,1}^2+\Phi_{2,-1}^2}}{|\Phi_{00}|},\qquad
\epsilon_{\rm elong}=\frac{\sqrt{\Phi_{2,2}^2+\Phi_{2,-2}^2}}{|\Phi_{00}|},\qquad
\epsilon_2=\sqrt{\epsilon_{\rm tilt}^2+\epsilon_{\rm elong}^2},
$$

which do not depend on the direction chosen for $\hat{\mathbf{e}}_x$ in the plane. $\epsilon_2$ is the quantity of
{fig}`paper_sims_scramble`c: with this normalisation it is the root-mean-square over the sphere of the non-axisymmetric quadrupole,
relative to the monopole. With all five $l=2$ terms, $m=0$ included, the same ratio is $\epsilon_{2,\rm all}$, which does not depend
on the orientation of the frame and so bounds $\epsilon_2$ about any axis. For each galaxy the values are binned in {glue:text}`ltEpoch`-Gyr steps of
look-back time and their median is taken; {fig}`paper_sims_settle` shows the median of these over the galaxies and its 16th–84th
percentiles (bins with at least {glue:text}`smMinGal` galaxies).

The halo lines up with the disc ({fig}`paper_sims_settle`a). The misalignment is $\psi\approx$ {glue:text}`smPsiEarlyLo`°–{glue:text}`smPsiEarlyHi`° until {glue:text}`smLbEarlyEnd` Gyr ago and falls
to {glue:text}`smPsiMid`° {glue:text}`smLbMid` Gyr ago and {glue:text}`smPsiLate`° {glue:text}`ltLbLate` Gyr ago; the spread between galaxies is large early (16th–84th percentiles {glue:text}`smPsiEarlyLoP`°–{glue:text}`smPsiEarlyHiP`°
{glue:text}`ltLbEarly` Gyr ago) and smaller later ({glue:text}`smPsiLateLoP`°–{glue:text}`smPsiLateHiP`° {glue:text}`ltLbLate` Gyr ago). The halo becomes rounder only slowly ({fig}`paper_sims_settle`b):
over the same {glue:text}`smSpan` Gyr $c/a$ rises from {glue:text}`smCaEarly` to {glue:text}`smCaLate` and $b/a$ from {glue:text}`smBaEarly` to {glue:text}`smBaLate`. The quadrupole falls by a factor of {glue:text}`smEpsFactor`,
$\epsilon_2$ = {glue:text}`smEpsEarly` → {glue:text}`smEpsLate` ({fig}`paper_sims_settle`c), unequally in its two parts: $\epsilon_{\rm tilt}$ = {glue:text}`smTiltEarly` → {glue:text}`smTiltLate` and
$\epsilon_{\rm elong}$ = {glue:text}`smElongEarly` → {glue:text}`smElongLate`. The tilt term is the one that follows the alignment. For a halo axisymmetric about its
minor axis, with $l=2$, $m=0$ amplitude $\epsilon_{\rm flat}$ in its own frame, a rotation by $\psi$ gives

$$
\epsilon_{\rm tilt}\propto\epsilon_{\rm flat}\,|\sin2\psi|,\qquad \epsilon_{\rm elong}\propto\epsilon_{\rm flat}\sin^2\psi ,
$$

from the $l=2$ rotation matrix. With $\epsilon_{\rm flat}$ scaling as $1-c/a$, the medians predict a fall of $\epsilon_{\rm tilt}$ by
$[\sin {glue:text}`smTwoPsiLate`^\circ/\sin {glue:text}`smTwoPsiEarly`^\circ]\times[(1-{glue:text}`smCaLate`)/(1-{glue:text}`smCaEarly`)]$ = {glue:text}`smFacSin` × {glue:text}`smFacCa` = {glue:text}`smFacPred`, as measured
({glue:text}`smTiltLateFour`/{glue:text}`smTiltEarlyThree` = {glue:text}`smFacMeas`); over the {glue:text}`ltNSettle` measurements $\epsilon_{\rm tilt}$ correlates with $(1-c/a)|\sin2\psi|$ with a
Spearman rank coefficient of {glue:text}`smRankTilt`. The elongation in the plane falls much less than the $\sin^2\psi$ of the last equation would give
(a factor {glue:text}`smSinSq` between these $\psi$): it comes mostly from the halo's own triaxiality, $b/a<1$ (rank correlation with $1-b/a$ of {glue:text}`smRankElong`),
which changes little.

The tilt therefore disappears as the halo aligns with the disc, by ≈{glue:text}`smLbTiltQuiet` Gyr ago, and the remaining asymmetry is the elongation
of the halo in the plane. The total stays above the $\epsilon_2\approx$ {glue:text}`smEpsQuiet` below which cohorts keep their orbital planes
({fig}`paper_sims_scramble`c): {glue:text}`ltLbLate` Gyr ago the median is {glue:text}`smEpsLate`, where cohorts lose {glue:text}`smPctBinTwo`–{glue:text}`smPctBinThree` per cent of the coherence they have left
in the next tracked step (medians {glue:text}`smFracBinTwo` at $\epsilon_2\approx$ {glue:text}`smEpsBinTwo` and {glue:text}`smFracBinThree` at {glue:text}`smEpsBinThree`), against {glue:text}`smFracBinFour`–{glue:text}`smFracBinFive` at the {glue:text}`smEpsBinFour`–{glue:text}`smEpsBinFive` of
cohorts born {glue:text}`smLbBinFour`–{glue:text}`smLbBinFive` Gyr ago (median look-back times of those bins). The same holds for the shape and the alignment taken one at a
time. The {glue:text}`smNCohHalo` cohorts lose a median {glue:text}`smQuietEps` of their available coherence if born at $\epsilon_2$ < {glue:text}`smEpsQuiet`, {glue:text}`smQuietPsi` if born at
$\psi$ < {glue:text}`smPsiQuiet`°, {glue:text}`smQuietCa` at $c/a$ > {glue:text}`smCaQuiet` and {glue:text}`smQuietBa` at $b/a$ > {glue:text}`smBaQuiet`, against {glue:text}`smOtherLo`–{glue:text}`smOtherHi` otherwise (dotted lines in {fig}`paper_sims_settle`). Each
level is set by the same rule: it is the edge of the first bin, in the binned cohorts, whose median loss is below {glue:text}`smRule` per cent. These
thresholds are not independent: the four quantities are correlated with each other and with time. The median halo reaches the
alignment threshold {glue:text}`smLbAlign` Gyr ago, but reaches neither of the shape thresholds ($c/a$ = {glue:text}`smCaLate`, $b/a$ = {glue:text}`smBaLate` {glue:text}`ltLbLate` Gyr ago). Stars born
early are born in a flattened, misaligned and elongated halo and lose their orbital planes; those born later are born in a halo
aligned with the disc and lose less, but not nothing, because the halo stays elongated in the plane of the disc.

<!-- script: paper_sims_settle.py -->

**Figure (paper_sims_settle).** How the centre of the dark-matter halo of the {glue:text}`ltNSample` TNG50 progenitors settles, against look-back time
(Gyr; median over the galaxies of each galaxy's median in {glue:text}`ltEpoch`-Gyr bins, 16th–84th percentiles shaded; dark matter only). (a) Angle
$\psi$ (deg) between the minor axis of the halo within {glue:text}`ltSetRIn` kpc and the spin of the stars within {glue:text}`ltSetRIn` kpc. The halo lines up with the
disc, from ≈{glue:text}`ltPsiEarly`° {glue:text}`ltLbEarly` Gyr ago to {glue:text}`smPsiLate`° {glue:text}`ltLbLate` Gyr ago. Dotted lines in all panels: where the median loss of the cohorts,
binned in that quantity at birth, falls below {glue:text}`smRule` per cent of their available coherence over the next tracked snapshot; in (a)
$\psi$ = {glue:text}`smPsiQuiet`°: cohorts born below it lose a median {glue:text}`smQuietPsi` of their available coherence, against more above. (b) Axis ratios
$b/a$ (blue) and $c/a$ (red) of the dark matter within a sphere of {glue:text}`ltSetRIn` kpc. The halo becomes rounder only slowly: $c/a$
{glue:text}`smCaEarly` → {glue:text}`smCaLate`, $b/a$ {glue:text}`smBaEarly` → {glue:text}`smBaLate`. Dotted: $c/a$ = {glue:text}`smCaQuiet` (red) and $b/a$ = {glue:text}`smBaQuiet` (blue); cohorts born above them lose a median
{glue:text}`smQuietCa` and {glue:text}`smQuietBa`. The median halo reaches neither. (c) Quadrupole of the dark matter's potential at {glue:text}`ltSetRZero` kpc relative to its
monopole, about the disc axis (logarithmic axis): the tilt of the halo to the disc ($\epsilon_{\rm tilt}$, $m=\pm1$, blue dashed), its
elongation in the plane of the disc ($\epsilon_{\rm elong}$, $m=\pm2$, red dashed) and both together, $\epsilon_2$ of
{fig}`paper_sims_scramble`c (black); the flattening along the disc axis ($m=0$), which does not turn the orbits, is excluded. Dotted:
$\epsilon_2$ = {glue:text}`smEpsQuiet`, below which cohorts lose a median {glue:text}`smQuietEps`. The tilt term falls by a factor of {glue:text}`smTiltFactorPct` as the halo aligns and reaches the
line ≈{glue:text}`smLbTiltQuiet` Gyr ago; the elongation in the plane, set by the halo's triaxiality, falls by a factor of {glue:text}`smElongFactor`, and keeps $\epsilon_2$ at
{glue:text}`smEpsLate`. `scripts/paper_sims_settle.py`

### Measuring the halo of each progenitor

`scripts/sims_halo_settle.py` makes the tables of {fig}`paper_sims_settle` and of {fig}`paper_sims_scramble`c. At each snapshot it
turns the dark matter of a progenitor into the frame of the stellar disc, reads the quadrupole of its multipole expansion at
{glue:text}`ltSetRZero` kpc and the shape tensor within {glue:text}`ltSetRIn` kpc, and gives $\epsilon_{\rm tilt}$,
$\epsilon_{\rm elong}$, $\epsilon_2$, $b/a$, $c/a$ and $\psi$ as defined above. It then takes the cohorts born at each tracked
snapshot, with their coherence at birth and one tracked snapshot later, and joins them to the state of the halo at their birth.

<!-- script: sims_halo_settle.py show: the merger trees and cutouts of the TNG50 progenitors (scripts/fetch_tng50_progenitors.py) and AGAMA; under a minute in the recorded run -->

## A round halo: self-interacting dark matter

If the asymmetry of the young halo turns the orbits, a halo that is round at its centre would let the old stars keep their rotation.
Self-interacting dark matter (SIDM; [Spergel & Steinhardt 2000](https://doi.org/10.1103/PhysRevLett.84.3760);
[Tulin & Yu 2018](https://doi.org/10.1016/j.physrep.2017.11.004)), in which dark-matter particles scatter off each other with a
cross-section per unit mass $\sigma/m$, gives such a halo: scattering makes the velocities of the dark-matter particles isotropic and
the halo round inside the radius where each particle has scattered about once
([Peter et al. 2013](https://doi.org/10.1093/mnras/sts535)). Cold dark matter (CDM) is the standard, collisionless case. We measured
this in the LMC-mass zoom Halo104 of the SIDM Concerto suite ([Nadler et al. 2025](https://doi.org/10.3847/1538-4357/adf553)), run
from the same initial conditions with CDM and with a velocity-dependent cross-section, characterised by its value at low
relative velocity, $\sigma_0/m$ = {glue:text}`ltSigmaZero` cm² g⁻¹, and the velocity scale above which it falls, $w$ = {glue:text}`ltWSidm` km/s (dark matter only, particle mass
${glue:text}`ltMPart`$ M☉), at $z$ = {glue:text}`ltZList` ({fig}`paper_sims_sidm`). The host's virial mass grows from ≈${glue:text}`smSiMvirZFour`$ M☉ at $z$ = 4 to
≈${glue:text}`smSiMvirZOneHalf`$ M☉ at $z$ = 1.5 and ≈${glue:text}`smSiMvirZZero`$ M☉ today in both runs. Its shape is measured with the
same iterative tensor, in ellipsoidal shells around its densest core, and $\epsilon_2$ from a multipole expansion of its particles.

The CDM halo is elongated at its centre at every redshift: $c/a$ = {glue:text}`smSiCaCdmLo`–{glue:text}`smSiCaCdmHi` within {glue:text}`smSiRCdm` kpc and $\epsilon_{2,\rm all}$({glue:text}`ltREps` kpc) = {glue:text}`smSiCdmEpsLo`–{glue:text}`smSiCdmEpsHi`.
The SIDM halo is round at its centre already at $z$ = 4 ($c/a$ = {glue:text}`smSiCaZFourLo`–{glue:text}`smSiCaZFourHi` within {glue:text}`smSiRoundZFour` kpc; here 'round' means that every shell out to
that radius has $c/a$ > {glue:text}`smSiRound`), and the round region grows with time, to {glue:text}`smSiRoundLo`–{glue:text}`smSiRoundHi` kpc ($c/a$ = {glue:text}`smSiCaRoundLo`–{glue:text}`smSiCaRoundHi`) from $z$ = 2 on, as
expected as more of the halo has had time to scatter (Peter et al. 2013). Its quadrupole at {glue:text}`ltREps` kpc is {glue:text}`smSiRatioLo` to {glue:text}`smSiRatioHi` times smaller than in
CDM: $\epsilon_{2,\rm all}$ = {glue:text}`smSiEpsZFour` at $z$ = 4, {glue:text}`smSiEpsZTwo` at $z$ = 2 and {glue:text}`smSiEpsLate` at $z$ = 1.5 and today. The exception is $z$ = 3, during a merger of two
cores of almost equal mass, when the SIDM halo is elongated too ($c/a\approx$ {glue:text}`smSiCaZThree`, $\epsilon_{2,\rm all}$ = {glue:text}`smSiEpsZThree`); it is round again by $z$ = 2.
The CDM values are as large as, or up to {glue:text}`smSiTngRatio` times, the TNG50 medians at the same redshifts, as expected for a halo without baryons:
in TNG50 the baryons make the inner halo rounder ([Kazantzidis et al. 2004](https://doi.org/10.1086/423992);
[Despali et al. 2019](https://doi.org/10.1093/mnras/stz273)). The SIDM halo lies below $\epsilon_2$ = {glue:text}`smEpsQuiet` from $z$ = 2 on and just
above it at $z$ = 4; in TNG50, cohorts born in halos below this value keep their orbital planes. A CDM halo of the LMC's mass is
therefore asymmetric enough at its centre to turn the orbits of its first stars, and an SIDM halo with this cross-section, outside
mergers, is not. This is one halo, simulated without baryons and without a disc; whether a disc forming in an SIDM halo keeps its old
stars rotating needs a hydrodynamical SIDM simulation.

<!-- script: paper_sims_sidm.py -->

### Measuring the Concerto halo

`scripts/sims_sidm_shapes.py` reads the cold and self-interacting runs of Halo104 at each redshift, finds the host and its densest
core, and measures the axis ratios in ellipsoidal shells and the quadrupole of a multipole expansion of its particles, as described
in the details below (`results/sims_sidm_shapes.csv`).

<!-- script: sims_sidm_shapes.py show: the Concerto snapshots, partly available from the simulation authors on request (scripts/fetch_concerto.py), AGAMA and about 20 GB of memory; about 15 min -->

## What the comparison shows

Observed as the giants were, simulated LMC-mass galaxies from three suites reproduce, in TNG50, the LMC's rotating metal-rich disc, but
none reproduces its metal-poor population. Their lowest-metallicity {glue:text}`smFirstFrac` per cent of stars, by formed mass, are
dispersion-supported ($v_{\rm rot}/\sigma_z\approx$ {glue:text}`ltTngRatioMP`, at most {glue:text}`smRatioMaxMPOne`); the LMC's are rotation-supported ($v_{\rm rot}/\sigma_z$ = {glue:text}`ltKinRatioMP`). The gap
is not a measurement effect: the same selection, errors and fit recover the analogues' own $v_{\rm rot}/\sigma_z$ (in the most
metal-poor bin to {glue:text}`smRecRatioMP`, 16th–84th percentiles {glue:text}`smRecRatioMPLo` to {glue:text}`smRecRatioMPHi`), and their particle values without any fitting are as low; nor does it
depend on whether the old population is defined by metallicity rank, by the simulations' own metallicity or by age.

Two limitations remain. The TNG50 analogues may be heated by their dark-matter particles, which are about 5 times more massive than
their star particles (4.5 × 10⁵ against 8.5 × 10⁴ M☉; [Ludlow et al. 2021](https://doi.org/10.1093/mnras/stab2770)); the {glue:text}`smDmFactor` times
better resolved Auriga and FIRE-2 zooms give the same result, but they are {glue:text}`ltNAu` + {glue:text}`ltNFire` galaxies. The analogues are not selected for the
LMC's history: its first passage, its interaction with the SMC, or a quiet early assembly. Within these limits, the simulations do not
form their earliest, most metal-poor stars in a hot spheroid: they form them rotating, like the LMC's, and then heat them, early stars
within a few hundred Myr of their birth ({fig}`paper_sims_birth` and {fig}`paper_sims_heating`). The heating is the turning of their
orbital planes by the gravity of the young dark-matter halo, which is not axisymmetric about the disc: elongated in its plane or tilted
to it ({fig}`paper_sims_scramble`); the tilt goes as the halo lines up with the disc, by ≈{glue:text}`smLbTiltQuiet` Gyr ago, but the elongation in the
plane stays ({fig}`paper_sims_settle`). In the LMC this heating was much weaker, or the old stars have since been spun up. A halo whose
centre was nearly axisymmetric when the first stars formed would explain the first possibility; an SIDM halo of the LMC's mass is round
enough at its centre from $z\approx2$ on, outside mergers, but a CDM halo is not ({fig}`paper_sims_sidm`).

## Details

### Stars at birth

`scripts/build_sims_birth.py` measures the TNG50 analogues with birth data (all stars within {glue:text}`ltRToday` kpc today) and the Auriga analogues
(in-situ stars only), in cohorts of {glue:text}`ltCohort` Gyr of age with at least {glue:text}`ltMinCohort` stars. The galaxy in which the stars formed is followed
through time from their recorded birth positions and velocities: the stars are sorted by formation time and taken in groups of
{glue:text}`ltTrackGroup`; each group gives a centre (shrinking sphere) and a velocity (median of the inner half), median-filtered over {glue:text}`ltTrackSmooth` groups and
interpolated in the cosmological scale factor $a=1/(1+z)$. Birth positions are in physical kpc relative to this track at each star's formation time. The frame has
$z$ along the angular momentum of the cohort within {glue:text}`ltRBirth` kpc; $v_\phi$ and $\sigma_z$ are mass-weighted at {glue:text}`ltRMinBirth` < $R$ < {glue:text}`ltRBirth` kpc (the inner cut
guards against errors in the track's centre), with at least {glue:text}`ltMinIn` stars; today they are measured in the $z=0$ disc frame.

### Numerical heating

Collisional heating of the stars by the more massive dark-matter particles (Ludlow et al. 2021) raises velocity dispersions; it cannot
turn orbital planes, which is what removes the rotation here. Two results of this chapter argue against it as the cause. Test
particles in a smooth multipole expansion of the progenitor's gravity, frozen at birth and free of particle noise, lose their
coherence as fast as the real stars ({fig}`paper_sims_scramble`a). And the Auriga and FIRE-2 zooms, with lighter dark-matter
particles, give the same result.

### Normalisation of the multipole coefficients

AGAMA's multipole expansion writes $\Phi(r,\theta,\phi)=\sum\Phi_{lm}(r)Y_{lm}$ with real $Y_{lm}$ of unit mean square over the sphere
($Y_{00}=1$, $Y_{l0}=\sqrt{2l+1}\,P_l(\cos\theta)$, with $P_l$ the Legendre polynomials), so $\Phi_{00}$ is the spherical average of the
potential, and the same normalisation holds for every $m$; $\epsilon_2$ is therefore the root-mean-square over the sphere of the
non-axisymmetric quadrupole relative to the monopole, and $\sum_m\Phi_{2m}^2$ does not change when the frame is rotated, so
$\epsilon_2\leq\epsilon_{2,\rm all}$. We checked this on a [Miyamoto & Nagai (1975)](https://doi.org/10.1093/pasj/27.4.533) disc sampled with
${glue:text}`smMNN`$ particles and expanded as in the scrambling experiment: at {glue:text}`smMNR` kpc, $\Phi_{00}$ equals the spherical average of the
expansion's potential and $\Phi_{20}$ equals $\sqrt5\langle\Phi P_2\rangle$, each to a fraction better than ${glue:text}`smMNNorm`$, and rotating the disc
changes $\sum_m\Phi_{2m}^2$ by a fraction smaller than ${glue:text}`smMNRot`$. In `scripts/sims_halo_settle.py` each $\Phi_{lm}(r)$ is a spline in $\ln r$ on {glue:text}`ltSetGrid` radii from
{glue:text}`ltSetRMin` to {glue:text}`ltSetRPot` kpc fitted to the particles (AGAMA Multipole, $l_{\max}$ = {glue:text}`ltSetLmax`), not a fixed radial basis.

### The SIDM measurement

`scripts/fetch_concerto.py` and `scripts/sims_sidm_shapes.py` use the CDM and SIDM runs of Halo104, dark matter only, the two finest
particle types. The host is the distinct halo of highest virial mass of the halo catalogue with no particle of the coarser types within
its virial radius $R_{\rm vir}$ (the radius within which the mean density reaches the virial overdensity of the halo finder). Its centre is the densest core: the {glue:text}`ltNPeaks` highest peaks of the smoothed particle counts within
{glue:text}`ltCoreBox` $R_{\rm vir}$, each refined by a shrinking sphere from {glue:text}`ltShrinkA` to {glue:text}`ltShrinkB` kpc, and the one with most mass within {glue:text}`ltCoreR` kpc kept (a single
shrinking sphere from $R_{\rm vir}$ lands between the two cores during the $z$ = 3 merger). The shape is the iterative reduced inertia
tensor (Zemp et al. 2011) in logarithmically spaced ellipsoidal shells from {glue:text}`ltRShellMin` kpc to $R_{\rm vir}$, shells with at least
${glue:text}`ltNShellMin`$ particles. The quadrupole is an AGAMA multipole expansion of the particles within {glue:text}`ltRKeep` $R_{\rm vir}$ ($l_{\max}$ = {glue:text}`ltSLmax`, {glue:text}`ltSGrid` radial
nodes from {glue:text}`ltSRMin` kpc), with the polar axis along the minor axis of the {glue:text}`ltFrameA`–{glue:text}`ltFrameB` kpc shell; $\epsilon_{2,\rm all}$ (all $m$) and $\epsilon_2$
($m\neq0$) are read at {glue:text}`ltREps` kpc.
