---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# The most metal-poor stars

*Third page of the kinematics chapter.* Every number on this page is written by `scripts/paper_text_letter.py` from the tables of
`scripts/kin_feh_lowest.py` (`results/macros_letter.tex`); the code cells are that script, shown but not run (about half an hour),
and `scripts/fig_kin_feh_lowest.py`, which draws the figure from its tables.

The Letter's most metal-poor bin holds the giants with iron abundance [Fe/H] (the logarithm of the iron-to-hydrogen ratio relative
to the Sun's) below {glue:text}`ltFehA`. Its ratio of rotation to vertical velocity dispersion, $v_{\rm rot}/\sigma_z$ (rotation
speed at the radius $R$ = {glue:text}`ltRSig` kpc of the disc frame over the dispersion perpendicular to the disc), is {glue:text}`ltKinRatioMP`. Do
the stars at the very bottom of that bin, with [Fe/H] below {glue:text}`ltLowFeh`, still rotate? The LMC had formed only
$\sim{glue:text}`ltLowMass`$ M☉ of stars when they were born (the formed mass below [Fe/H] = {glue:text}`ltLowFeh`, chapter 4), so
their rotation tells us whether the disc existed from the start.

## Why these stars are hard to measure

There are {glue:text}`ltLowN` giants with measured [Fe/H] below {glue:text}`ltLowFeh` in the fitted ring, almost all with SDSS-V
BOSS spectra. Their [Fe/H] comes from BOSS, calibrated to APOGEE. Among the stars observed by both surveys, only
{glue:text}`ltLowPurity` per cent of those with calibrated BOSS [Fe/H] below {glue:text}`ltLowFeh` have APOGEE [Fe/H] below it;
the others lie at APOGEE [Fe/H] of {glue:text}`ltLowInterLo` to {glue:text}`ltLowInterHi`. So a group selected by measured [Fe/H] below {glue:text}`ltLowFeh`
contains interlopers from just above it. Its purity, the fraction of its stars truly below {glue:text}`ltLowFeh`, is uncertain:
the stars in both surveys are brighter than the BOSS sample, and APOGEE itself has few stars below [Fe/H] = {glue:text}`ltLowNarrowFeh`. Milky Way halo
stars, which are metal-poor and kinematically hot, and any hot LMC component add further stars that do not share the disc's
rotation.

## How we measured it

Before running the measurement we asked two independent reviewers to criticise the plan. Both rejected the first idea, to correct
the fitted values of narrow [Fe/H] bins with a matrix of interloper fractions: the bins rotate alike, so the correction has no
leverage, and a fit of a mixture of populations does not return the average of their kinematics. The measurement therefore has
three parts.

1. **Fits of narrow bins.** The kinematic model of chapter 6 (the velocity-ellipsoid likelihood) is fitted to bins of measured
   [Fe/H] inside the Letter's lowest bin, down to [Fe/H] < {glue:text}`ltLowNarrowFeh`. Each bin has its own rotation amplitude, dispersions $\sigma_R$,
   $\sigma_\phi$, $\sigma_z$, Gaia error scale and outlier fraction; everything else (disc orientation, centre, bulk motion,
   streaming and the shape of the rotation curve) is held at the fit of the whole lowest bin, which a few hundred stars cannot
   constrain. Fitting the whole lowest bin this way returns the Letter's value. Holding these parameters is tested by refitting at
   draws of their posterior and with the joint geometry of all populations.
2. **Subsets.** The stars below {glue:text}`ltLowFeh` with the best spectra (BOSS signal-to-noise ratio above {glue:text}`ltLowSnr`, or APOGEE), and
   cool and hot giants separately, are fitted the same way.
3. **A null test.** Mock velocities are drawn for the same stars, at their positions and with their measurement errors, from the
   ellipsoid model: a fraction (the purity) are truly metal-poor stars with a chosen $v_{\rm rot}/\sigma_z$, from zero to the
   Letter's value, the rest are interlopers moving like the stars at {glue:text}`ltLowInterBinLo` to {glue:text}`ltLowInterBinHi`,
   and in half of the mocks {glue:text}`ltLowHotPct` per cent of the stars are replaced by a hot, non-rotating component with a
   dispersion of {glue:text}`ltLowHotSigma` km/s, standing in for halo stars. Each mock is
   fitted exactly as the data.

A test of the mock generator (`tests/test_kin_feh_lowest.py`) checks that a fit to a large mock returns the ratio and dispersion
it was drawn with.

## Result

The giants below [Fe/H] = {glue:text}`ltLowFeh` rotate: the two narrow bins below it have
$v_{\rm rot}/\sigma_z\approx${glue:text}`ltLowRatioLo`–{glue:text}`ltLowRatioHi` ({fig}`fig_kin_feh_lowest`a), and the
best-measured subset agrees. In the mocks, a group whose truly metal-poor stars do not rotate gives a lower value than the one
measured unless they make up less than {glue:text}`ltLowPurityMin` per cent of it ({fig}`fig_kin_feh_lowest`b), and a hot
component lowers the mock values further. For the purities suggested by the stars in both surveys, the truly metal-poor stars
must themselves have $v_{\rm rot}/\sigma_z$ well above 1.

Two points remain open. The ratio falls slightly towards the lowest [Fe/H], which could be real or a sign of a hot component. And
the cool giants rotate faster relative to their dispersion than the hot ones, for reasons we have not established. Neither changes
the conclusion that the stars formed when the LMC had $\sim{glue:text}`ltLowMass`$ M☉ of stars rotate; the Letter reports the
result with its purity caveat.

**Figure (fig_kin_feh_lowest).** (a) $v_{\rm rot}(4\,{\rm kpc})/\sigma_z$ against median measured [Fe/H]: narrow bins inside the
Letter's lowest bin, with the geometry held at that bin's fit (red; errors include the spread over the held parameters), and the
Letter's five bins, each fitted on its own (black squares). (b) The null test for measured [Fe/H] < {glue:text}`ltLowFeh`:
$v_{\rm rot}/\sigma_z$ fitted to mocks against the purity, for true ratios of the metal-poor stars from zero (blue) to the
Letter's value (red), without (filled) and with (open) a {glue:text}`ltLowHotPct` per cent hot component; grey band: the measured value (16th–84th
percentiles).

### The fits and the null test

<!-- script: kin_feh_lowest.py show: the release giants and the posterior samples of the kinematic fit (Zenodo bundle 1); about 30 min (9 NUTS runs and 256 mock fits) -->

### The figure

<!-- script: fig_kin_feh_lowest.py -->
