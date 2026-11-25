---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Data: the samples

Every number on this page is written by `scripts/paper_text_data.py` from the result tables and catalogues
(`results/macros_data.tex`); the code cells are the figure script `scripts/paper_maps.py`, step by step, and the scripts that
build the catalogues, shown but not run.

We use three tracers of the old population of the Large Magellanic Cloud (LMC). The first are RR Lyrae stars of type ab (RRab:
RR Lyrae pulsating in the fundamental radial mode, with asymmetric, saw-tooth light curves), old pulsating horizontal-branch stars
whose period and light-curve shape give a distance and a photometric metallicity. They come from the
fourth phase of the Optical Gravitational Lensing Experiment (OGLE-IV; [Udalski et al. 2015](https://doi.org/10.48550/arXiv.1504.05966))
and from the third data release of the *Gaia* mission (*Gaia* DR3; [Gaia Collaboration 2016](https://doi.org/10.1051/0004-6361/201629272),
[2023](https://doi.org/10.1051/0004-6361/202243940)). The other two are red giants with spectra from the 17th data release of the
Apache Point Observatory Galactic Evolution Experiment (APOGEE DR17; [Majewski et al. 2017](https://doi.org/10.3847/1538-3881/aa784d),
[Abdurro'uf et al. 2022](https://doi.org/10.3847/1538-4365/ac4414)) and from the 20th data release of the fifth Sloan Digital Sky
Survey, observed with the Baryon Oscillation Spectroscopic Survey spectrographs (SDSS-V DR20 BOSS;
[Kollmeier et al. 2026](https://doi.org/10.3847/1538-3881/ae0576), [SDSS Collaboration 2026](https://doi.org/10.48550/arXiv.2607.26149),
[Smee et al. 2013](https://doi.org/10.1088/0004-6256/146/2/32)). This chapter defines the samples; the chapter on selection
functions derives the probability that each star entered its sample, and the chapter on stellar mass converts the
selection-corrected giant counts into the stellar mass formed at each metallicity.

Metallicity here is the iron abundance [Fe/H], the logarithm of the ratio of iron to hydrogen atoms relative to the Sun's, in dex.
All positions refer to the LMC centre $(\alpha,\delta)$ = ({glue:text}`daRaZero`°, {glue:text}`daDecZero`°) in right ascension
and declination of [van der Marel & Kallivayalil (2014)](https://doi.org/10.1088/0004-637X/781/2/121), used here only as the
origin of the angular distance $\rho$ of a star from the centre; the kinematics chapter fits the kinematic centre.

## RR Lyrae

We unite the OGLE-IV collection of LMC RRab ([Udalski et al. 2015](https://doi.org/10.48550/arXiv.1504.05966);
[Soszyński et al. 2016](https://doi.org/10.48550/arXiv.1606.02727), [2019](https://doi.org/10.32023/0001-5237/69.2.1)) with the
RRab of the *Gaia* DR3 variability catalogue ([Clementini et al. 2023](https://doi.org/10.1051/0004-6361/202243964)) within
{glue:text}`daRrlQuery`° of the centre, matching the two at {glue:text}`daMatch`″ (arcsec). The union holds {glue:text}`daNUnion`
RRab, of which {glue:text}`daNBoth` are in both catalogues. Periods and light-curve parameters come from OGLE where a star has them
and from *Gaia* otherwise. OGLE-LMC-RRLYR-{glue:text}`daFirstAdded` and later identifiers were added to the OGLE collection from
*Gaia* ([Soszyński et al. 2019](https://doi.org/10.32023/0001-5237/69.2.1)); only the {glue:text}`daNOwn` stars of OGLE's own
search are independent of *Gaia*, which matters for the completeness estimate of the selection-function chapter.

**Metallicity.** Photometric [Fe/H] follows from the period $P$ and the Fourier phase $\phi_{31}=\phi_3-3\phi_1$ of the light
curve (the phase of its third harmonic relative to three times that of the first, in a Fourier series of the magnitude in time).
For stars with OGLE $I$-band light curves we transform $\phi_{31}$ to the $V$ band and apply the relation of
[Mullen et al. (2021)](https://doi.org/10.3847/1538-4357/abefd4), as in
[Martínez-Vázquez et al. (2026, their eq. A2)](https://doi.org/10.48550/arXiv.2609.27119); this places [Fe/H] on the scale of
[Crestani et al. (2021)](https://doi.org/10.3847/1538-4357/abd183). For the remaining stars we use the *Gaia* $G$-band relation of
[Muraveva et al. (2025)](https://doi.org/10.1093/mnras/stae2679), shifted by +{glue:text}`daFehShift` dex, the median difference
on the {glue:text}`daNShift` stars with both; the difference does not depend on [Fe/H] (slope of a Deming regression, which allows
for errors in both estimates, {glue:text}`daDeming` ± {glue:text}`daDemingErr`). Against the high-resolution spectroscopy of
[Gratton et al. (2004)](https://doi.org/10.1051/0004-6361:20035840), transformed to the same scale
([Zinn & West 1984](https://doi.org/10.1086/190947); [Carretta et al. 2009](https://doi.org/10.1051/0004-6361/200913003)), the
adopted values are offset by +{glue:text}`daGrattonOff` ± {glue:text}`daGrattonErr` dex ({glue:text}`daNGratton` stars; see
the chapter on RR Lyrae and BOSS calibration). Of the members defined below, {glue:text}`daNFehI` have $I$-band and
{glue:text}`daNFehG` $G$-band metallicities; {glue:text}`daNFehNone` have neither.

**Distance.** We measure the distance modulus of each star relative to the LMC centre, $\Delta\mu$ (mag; $\Delta\mu=0$ at the mean
LMC distance), from a period–Wesenheit–metallicity relation fitted to the stars at $\rho$ < {glue:text}`daPwRho`°. A Wesenheit
index is a magnitude minus a multiple of a colour, chosen so that interstellar extinction cancels. We use the OGLE index
$W_I=I-$ {glue:text}`daWICoef` $(V-I)$ where available (scatter about the relation {glue:text}`daScatI` mag,
{glue:text}`daNPwI` stars) and the *Gaia* index $W_G=G-$ {glue:text}`daWGCoef` $(G_{\rm BP}-G_{\rm RP})$
([Ripepi et al. 2019](https://doi.org/10.1051/0004-6361/201834506)) otherwise ({glue:text}`daScatG` mag); $G$, $G_{\rm BP}$ and
$G_{\rm RP}$ are the *Gaia* broad, blue and red magnitudes. Only relative distances enter this analysis.

**Members.** An RRab is a member if it lies at $\rho$ < {glue:text}`daRhoMax`°, more than {glue:text}`daSmcMask`° from the Small
Magellanic Cloud (SMC), and within $|\Delta\mu|$ < {glue:text}`daDmuWin` mag; the window is {glue:text}`daDmuRatio` times the
{glue:text}`daDmuScat`-mag scatter of $\Delta\mu$ for the *Gaia* index and leaves room for the inclined disc. The SMC radius is
where the fraction of stars with SMC-like proper motions falls to the level set by LMC stars (see the chapter on RR Lyrae and BOSS
calibration). The sample holds {glue:text}`daNMembers` members, {glue:text}`daNMembersPM` with *Gaia* proper motions. No
proper-motion cut is applied, so the kinematics remain unbiased. Stars in the $\Delta\mu$ sidebands {glue:text}`daSideLo` <
$|\Delta\mu|$ < {glue:text}`daSideHi` mag measure the foreground left in the window: {glue:text}`daContIn` per cent at $\rho$ <
{glue:text}`daRhoContIn`°, {glue:text}`daContMidLo`–{glue:text}`daContMidHi` per cent at
{glue:text}`daRhoContIn`°–{glue:text}`daRhoContMid`° and {glue:text}`daContOut` per cent at
{glue:text}`daRhoContMid`°–{glue:text}`daRhoMax`°.

### Building the RR Lyrae catalogue

`scripts/build_rrab.py` makes the catalogue `data/rrab.parquet` that every RR Lyrae figure of this site reads. It matches the
OGLE-IV collection to the *Gaia* DR3 RR Lyrae, forms their union, computes the two photometric metallicities and the adopted one,
fits the period–Wesenheit–metallicity relations on the stars at $\rho$ < {glue:text}`daPwRho`° to obtain $\Delta\mu$, and flags
the members.

<!-- script: build_rrab.py show: the OGLE-IV and Gaia DR3 RR Lyrae catalogues (scripts/fetch_ogle4_rrab.py, scripts/fetch_gaia_rrlyrae.py) -->

## Red giants

The giants come from APOGEE DR17 ([Majewski et al. 2017](https://doi.org/10.3847/1538-3881/aa784d);
[Wilson et al. 2019](https://doi.org/10.1088/1538-3873/ab0075); [Abdurro'uf et al. 2022](https://doi.org/10.3847/1538-4365/ac4414)),
a high-resolution near-infrared survey whose stellar parameters come from the APOGEE Stellar Parameters and Chemical Abundances
Pipeline (ASPCAP; [García Pérez et al. 2016](https://doi.org/10.3847/0004-6256/151/6/144)), and from SDSS-V DR20
([Kollmeier et al. 2026](https://doi.org/10.3847/1538-3881/ae0576); [SDSS Collaboration 2026](https://doi.org/10.48550/arXiv.2607.26149)),
with low-resolution optical BOSS spectra ([Smee et al. 2013](https://doi.org/10.1088/0004-6256/146/2/32)) analysed by BOSSNet, a
neural network that estimates stellar parameters from the spectra ([Sizemore et al. 2024](https://doi.org/10.3847/1538-3881/ad291d)).
In the LMC, APOGEE observed red giants in {glue:text}`daNFields` fields (LMC1–{glue:text}`daNFields`;
[Nidever et al. 2020](https://doi.org/10.3847/1538-4357/ab7305)), and SDSS-V observed red giants to $G$ = {glue:text}`daGLim`
over the whole galaxy in the Magellanic Genesis survey ([Nidever et al. 2026](https://doi.org/10.3847/1538-3881/ae7c76)). We keep
stars within {glue:text}`daGiantRho`° of the centre with heliocentric line-of-sight velocity {glue:text}`daVlosLo` < $v_{\rm los}$ <
{glue:text}`daVlosHi` km/s. A good APOGEE spectrum is not a duplicate observation, has the ASPCAP flag STAR_BAD unset and a
finite [Fe/H], with signal-to-noise ratio S/N ≥ {glue:text}`daSnrApoKin` for the kinematic samples and S/N ≥
{glue:text}`daSnrApoSel` for the selection function and the mass (selection-function chapter); a good BOSS spectrum has the
BOSSNet quality flag unset, a finite [Fe/H] and velocity, S/N ≥ {glue:text}`daSnrBoss` and a velocity error below
{glue:text}`daEvBoss` km/s. Below S/N = {glue:text}`daSnrBoss` BOSSNet [Fe/H] follows APOGEE poorly on the stars observed by
both: the slope of BOSSNet against APOGEE [Fe/H] is {glue:text}`daSlopeA` ± {glue:text}`daSlopeAErr` at S/N
{glue:text}`daSnrA`–{glue:text}`daSnrB` and {glue:text}`daSlopeB` ± {glue:text}`daSlopeBErr` at
{glue:text}`daSnrB`–{glue:text}`daSnrC`, against {glue:text}`daSlopeTop` ± {glue:text}`daSlopeTopErr` above
{glue:text}`daSnrTop`, and the calibrated values run {glue:text}`daCalHighA`–{glue:text}`daCalHighB` dex high (see the chapter on
RR Lyrae and BOSS calibration). Both samples keep giants only (effective temperature $T_{\rm eff}$ < {glue:text}`daTeffMax` K and
surface gravity $\log g$ < {glue:text}`daLoggMax`, with $g$ in cm s⁻²). For kinematics, APOGEE stars also need velocity errors
below {glue:text}`daEvApo` km/s and a visit-to-visit scatter below {glue:text}`daVscatApo` km/s; all stars need *Gaia* DR3 proper
motions with renormalised unit weight error (RUWE, a measure of how well a single-star astrometric model fits) ≤
{glue:text}`daRuwe` and a parallax $\varpi$ (with error $\sigma_\varpi$, both in mas) consistent with the LMC,
$(\varpi+$ {glue:text}`daPlxZero`$)/\sigma_\varpi$ < {glue:text}`daPlxSig`, where {glue:text}`daPlxZero` mas corrects the
*Gaia* parallax zero point ([Lindegren et al. 2021](https://doi.org/10.1051/0004-6361/202039653)).

We place BOSS on the APOGEE scale with the {glue:text}`daNOverlap` stars observed by both surveys. With
$t=(T_{\rm eff,BOSS}-$ {glue:text}`daTPivot` K$)/$ {glue:text}`daTScale` K and $s=$ {glue:text}`daSPivot` $/({\rm S/N})_{\rm BOSS}-1$,
robust fits with Huber weights (which reduce the weight of outlying residuals) give

$$
v_{\rm BOSS}-v_{\rm APOGEE}=d_0+d_1\,t,\qquad
[{\rm Fe/H}]_{\rm BOSS}=c_0+c_1\,[{\rm Fe/H}]_{\rm APOGEE}+c_2\,t+c_3\,s,
$$

with $d_0$ = {glue:text}`daVcalA` km/s, $d_1$ = −{glue:text}`daVcalB` km/s, $c_0$ = {glue:text}`daFcalA` dex, $c_1$ =
{glue:text}`daFcalB`, $c_2$ = −{glue:text}`daFcalC` dex and $c_3$ = {glue:text}`daFcalD` dex. The [Fe/H] relation is fitted with
the more precise APOGEE value as the independent variable and inverted for BOSS-only stars; this undoes the compression of the
BOSSNet scale without the attenuation of a fit in the other direction. Fitted on half of the common stars and tested on the other
half, the calibrated BOSS values differ from APOGEE by a median of {glue:text}`daHoMed` dex with a robust scatter (the median absolute
deviation scaled to the standard deviation of a Gaussian) of {glue:text}`daHoRsd` dex, and the velocities by {glue:text}`daHoDv` km/s; the median residual is
within {glue:text}`daHoBin` dex in every bin of S/N and of $T_{\rm eff}$. The calibration still leaves a bias of
−{glue:text}`daBiasLo` to −{glue:text}`daBiasHi` dex at {glue:text}`daBiasFehLo` < [Fe/H] < {glue:text}`daBiasFehHi`, which the stellar-mass chapter measures and corrects.
The chapter on RR Lyrae and BOSS calibration shows the fits on the common stars. The kinematic samples hold {glue:text}`daNKinApo`
APOGEE and {glue:text}`daNKinBoss` BOSS giants ({glue:text}`daNKinBoth` in both).

### Building the giant catalogues

Two scripts make the giant catalogues. `scripts/build_spec_giants.py` reads the public APOGEE DR17 and BOSSNet summary files and
keeps the spectra within {glue:text}`daGiantRho`° of the centre and inside the velocity range above, without quality cuts.

<!-- script: build_spec_giants.py show: the APOGEE DR17 allStarLite and SDSS-V DR20 BOSSNet summary files (scripts/fetch_giants.py) -->

`scripts/build_giants.py` adds the *Gaia* DR3 astrometry, applies the quality and giant cuts above, fits the calibration of BOSS
on APOGEE (on half of the common stars, tested on the other half, then refitted on all of them) and writes the release catalogues
`data/giants_apogee.parquet` and `data/giants_boss.parquet`, the common stars `data/giants_overlap.parquet` and the coefficients
`results/boss_calibration.json`.

<!-- script: build_giants.py show: data/spec_giants.parquet and the Gaia DR3 astrometry of its stars (scripts/build_spec_giants.py, scripts/fetch_gaia_giants.py) -->

## The samples on the sky

{fig}`paper_maps` shows the three samples on the sky. The RR Lyrae fill the inner {glue:text}`daRhoMax`° evenly; APOGEE covers its
{glue:text}`daNFields` fields; BOSS covers the whole {glue:text}`daGiantRho`° disc, with gaps where its targeting reached few
stars. The mean proper motions of all three samples are dominated by the bulk motion of the LMC and by perspective, and the
line-of-sight velocities of the giants and the relative distances of the RR Lyrae show the inclined, rotating disc.

<!-- script: paper_maps.py -->

**Figure (paper_maps).** The three samples on the sky, in an orthographic projection about the reference centre: $x$ (east,
increasing to the left) and $y$ (north), in degrees, {glue:text}`daMapPix`° pixels. Rows: RR Lyrae members, APOGEE giants, BOSS giants. Columns:
stars per pixel (logarithmic blue scale); median [Fe/H] (dex); median proper motions $\mu_{\alpha*}$ (east, including the
$\cos\delta$ factor) and $\mu_\delta$ (north), in mas/yr, raw, that is including the bulk motion and perspective; and median
relative distance modulus $\Delta\mu$ (mag; RR Lyrae, upper scale of the last colour bar) or median line-of-sight velocity
$v_{\rm los}$ (km/s; giants, lower scale of the same bar). Medians are drawn in pixels with at least {glue:text}`daMapNMin` stars. Dashed circles: the
RR Lyrae science radius ({glue:text}`daRhoMax`°, top row) and the {glue:text}`daGiantRho`° cone of the giants (lower rows); orange
arc: edge of the SMC mask. One [Fe/H] colour scale serves all rows; the RR Lyrae scale is photometric. `scripts/paper_maps.py`

## The Letter's velocity clip

In panel (c) of Fig. 1 of the Letter, stars more than {glue:text}`daClip` km/s from their population's disc model (plus
{glue:text}`daClipErr` times their own proper-motion error, for the proper-motion components) are dropped
(`scripts/paper_letter_signal.py`). The clip drops {glue:text}`daClipRr` of the {glue:text}`daClipNRr` RR Lyrae ({glue:text}`daClipRrPct`
per cent) and {glue:text}`daClipGLo`–{glue:text}`daClipGHi` per cent of the giants per [Fe/H] bin, most in the metal-poorest bin
({glue:text}`daClipMp` of {glue:text}`daClipNMp`).

## Parent catalogues of the selection functions

The selection functions of the next chapter need, for each giant survey, the catalogue it targeted from (the parent), the parent
stars it observed and those that gave a good spectrum, and an XP metallicity ([M/H]$_{\rm XP}$, from the low-resolution prism
spectra of *Gaia*) for as many parent stars as possible. Five scripts build these tables; the next chapter defines the cuts and
uses the tables.

### The BOSS parent

`scripts/build_boss_parent.py` queries *Gaia* DR3 for the stars within {glue:text}`daGiantRho`° of the centre that pass the rough
cuts of the SDSS-V carton mwm_magcloud_rgb_boss, adds the reddening of the
[Schlegel et al. (1998)](https://doi.org/10.1086/305772) map and flags the stars that pass the full carton selection.

<!-- script: build_boss_parent.py show: a live query of the Gaia DR3 archive at Gaia@AIP and the SFD98 dust map of the dustmaps package -->

### Observed stars and good BOSS spectra

`scripts/build_boss_observed.py` marks the parent stars that SDSS-V observed, from the spectra of the SDSS-V spectroscopic
summary file matched on the *Gaia* source identifier, and those with a good BOSSNet spectrum. Observed stars missing from the
BOSSNet catalogue count as failed spectra.

<!-- script: build_boss_observed.py show: the SDSS-V DR20 spAll-lite and BOSSNet summary files (scripts/fetch_targeting.py, scripts/fetch_giants.py) -->

### Carton memberships

`scripts/build_boss_cartons.py` lists every SDSS-V carton that each parent star belongs to, from the public SDSS-V target
database; the next chapter sets the fibre-priority class of a star from this list. It also keeps the XP metallicity that SDSS-V
used in its targeting, for comparison.

<!-- script: build_boss_cartons.py show: the SDSS-V DR20 target database tables (scripts/fetch_targetdb.py) -->

### The APOGEE parent

`scripts/build_apogee_parent.py` rebuilds the candidates of the APOGEE-2 fields LMC1–{glue:text}`daNFields` from the public
targeting files: the 2MASS stars inside the colour–magnitude polygons of each field, with the bright-neighbour flag, their
*Gaia* DR3 mates and the astrometric cuts of the BOSS carton, the observed flag from the APOGEE visit file, and the stellar
parameters of the observed stars.

<!-- script: build_apogee_parent.py show: the APOGEE-2 targeting files, the DR17 allVisit and allStarLite files (scripts/fetch_targeting.py, scripts/fetch_giants.py) and live Gaia DR3 queries at Gaia@AIP -->

### XP metallicities

`scripts/build_xp_andrae.py` takes [M/H]$_{\rm XP}$ of [Andrae et al. (2023)](https://doi.org/10.3847/1538-4365/acd53e) for the
stars of both parents and keeps it only for cool giants, where it is reliable. These values define the pass groups of both
selection functions and the tests of the next chapter.

<!-- script: build_xp_andrae.py show: Table 1 of Andrae et al. (2023) from Zenodo (scripts/fetch_andrae23.py) and the two parent tables above -->
