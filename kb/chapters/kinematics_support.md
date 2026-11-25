---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Rotational support, centre and sensitivity

*First page of the kinematics chapter; the second is 'Rotation and the circular speed'.* Every number on this page is written by
`scripts/paper_text_kin_support.py` from the result tables (`results/macros_kinsupport.tex`); the code cells are the figure scripts
themselves (`scripts/paper_kin_*.py`), step by step, with the diagnostic scripts that make some of their inputs shown but not run.

## The quantities

The old population of the Large Magellanic Cloud (LMC) is traced by two kinds of star. The red giants have proper motions from
*Gaia* DR3 ([Gaia Collaboration, Vallenari et al. 2023](https://doi.org/10.1051/0004-6361/202243940)) and line-of-sight velocities
and iron abundances from two spectroscopic surveys: APOGEE ([Majewski et al. 2017](https://doi.org/10.3847/1538-3881/aa784d); data
release 17, [Abdurro'uf et al. 2022](https://doi.org/10.3847/1538-4365/ac4414)) and the BOSS spectrograph of SDSS-V
([Kollmeier et al. 2026](https://doi.org/10.3847/1538-3881/ae0576)), whose spectra are analysed by the BOSSNet code
([Sizemore et al. 2024](https://doi.org/10.3847/1538-3881/ad291d)). The iron abundance [Fe/H] is the logarithm of the
iron-to-hydrogen ratio relative to the Sun's; the giants are divided into five bins of measured [Fe/H], with edges at
{glue:text}`ksFehA`, {glue:text}`ksFehB`, {glue:text}`ksFehC` and {glue:text}`ksFehD`. The RR Lyrae stars of the OGLE-IV survey
([Udalski et al. 2015](https://doi.org/10.48550/arXiv.1504.05966); [Soszyński et al. 2016](https://doi.org/10.48550/arXiv.1606.02727))
have *Gaia* proper motions and photometric [Fe/H], but no line-of-sight velocities. A 'group' below is one giant bin or the RR Lyrae.
Stars are selected by their angular distance $\rho$ from the LMC centre of
[van der Marel & Kallivayalil (2014)](https://doi.org/10.1088/0004-637X/781/2/121) (vdMK14 below): the default sample has
{glue:text}`ksRhoIn` < $\rho$ < {glue:text}`ksRhoOut`°.

The kinematic model, described with the velocity field in chapter 6, places each group on a thin disc seen in perspective. Its
geometry is the kinematic centre (the centre of the model disc), offset by $\Delta x$ (east) and $\Delta y$ (north) from the vdMK14 centre, in degrees; the
inclination $i$ of the disc ($i=0$ face-on); and the position angle $\theta$ (north through east) of the line of nodes, the line along
which the disc plane crosses the plane of the sky. Each star then has a cylindrical radius $R$ in the disc plane and an azimuth $\phi$
measured from the line of nodes. The mean rotation speed is

$$
v_{\rm rot}(R)=v_0\tanh(R/R_0)\,[1+s\,(R-R_\sigma)],
$$

with a level $v_0$, a turnover radius $R_0$, an outer slope $s$ and a reference radius $R_\sigma$ = {glue:text}`ksRSig` kpc. Because the
three parameters are correlated, we quote the rotation speed at $R_\sigma$, $v_{\rm rot}(4\,{\rm kpc})=v_0\tanh(4\,{\rm kpc}/R_0)$. In the symmetric disc (Level 1) the stars
move on circles; Level 2 adds non-circular streaming with a two-fold ($m=2$) pattern in azimuth. The random motions about the mean form
a velocity ellipsoid aligned with the disc, with dispersions $\sigma_R$ (radial), $\sigma_\phi$ (azimuthal) and $\sigma_z$
(vertical) at $R_\sigma$; all three change with radius by the same factor $e^{\gamma(R-R_\sigma)}$, so $\gamma$ (kpc$^{-1}$) is the
logarithmic radial gradient of the dispersions and $1-e^\gamma$ their fractional decline per kpc. Rotational support is the ratio of
the rotation speed to a dispersion, $v_{\rm rot}/\sigma_R$ or $v_{\rm rot}/\sigma_z$. The *Gaia* proper-motion errors of a group are
multiplied by one scale factor $k$, and a fraction $f$ of its stars is assigned to a broad outlier component. In stage 1 each group is
fitted on its own, with its own centre, systemic motion (the proper motion and line-of-sight velocity of the centre), orientation
and streaming; in stage 2 all groups are fitted jointly and share
these, each keeping its own rotation curve and ellipsoid. The fits sample the posterior probability of the parameters with a Markov
chain; we quote its median and its 16th–84th percentiles. Where a fit is instead a maximum of the likelihood $L$, its errors come from
the inverse Hessian (the curvature of $\ln L$), and two models of the same stars are compared by the difference of their maximum
log-likelihoods, $\Delta\ln L$. A mock is a simulated data set drawn from a known model at the positions and with the
errors of the real stars, and fitted like the data.

## Rotational support against metallicity

{fig}`paper_kin_support_ellipsoid` and {tab}`kin` give the result. From the most metal-poor to the most metal-rich giants the rotation
speed at 4 kpc rises from {glue:text}`ksVrotA` to {glue:text}`ksVrotE` km/s, and all three dispersions fall: $\sigma_R$ from
{glue:text}`ksSigRA` to {glue:text}`ksSigRE` km/s, $\sigma_\phi$ from {glue:text}`ksSigPhiA` to {glue:text}`ksSigPhiE` km/s and
$\sigma_z$ from {glue:text}`ksSigzA` to {glue:text}`ksSigzE` km/s. The rotational support rises accordingly, $v_{\rm rot}/\sigma_R$
from {glue:text}`ksRatRA` to {glue:text}`ksRatRE` and $v_{\rm rot}/\sigma_z$ from {glue:text}`ksRatzA` to {glue:text}`ksRatzE`. Most of
the change happens above [Fe/H] = {glue:text}`ksFehB`: the two metal-poor bins differ by {glue:text}`ksDVrotAB` km/s in $v_{\rm rot}$
and by at most {glue:text}`ksDSigAB` km/s in each dispersion, while from {glue:text}`ksFehB` to the metal-rich end $v_{\rm rot}$ grows
by {glue:text}`ksDVrotCE` km/s and $\sigma_R$ and $\sigma_z$ drop by {glue:text}`ksDSigRCE` and {glue:text}`ksDSigzCE` km/s. Rotation
and dispersion both change, by comparable amounts.

The ellipsoid has the same ordering of axes in every giant bin: $\sigma_R$ is the largest, and $\sigma_\phi$ and $\sigma_z$ are
smaller and nearly equal, $\sigma_\phi/\sigma_R$ = {glue:text}`ksRatPhiLo`–{glue:text}`ksRatPhiHi` and $\sigma_z/\sigma_R$ =
{glue:text}`ksRatZLo`–{glue:text}`ksRatZHi`. The ratios differ between bins by more than their errors ($\sigma_z/\sigma_R$ =
{glue:text}`ksRatZD` and {glue:text}`ksRatZE` in the two metal-rich bins, and {glue:text}`ksRatZApoE` for the APOGEE giants alone above
{glue:text}`ksFehD`), so the shape is similar, not identical, and the measured bins mix populations. The vertical support is therefore
everywhere higher than the radial one, $v_{\rm rot}/\sigma_z$ exceeding $v_{\rm rot}/\sigma_R$ by
{glue:text}`ksSuppExLo`–{glue:text}`ksSuppExHi` per cent. All dispersions decline outwards, by
{glue:text}`ksGamPctMPLo`–{glue:text}`ksGamPctMPHi` per cent per kpc in the two metal-poor bins and
{glue:text}`ksGamPctMRLo`–{glue:text}`ksGamPctMRHi` per cent per kpc above [Fe/H] = {glue:text}`ksFehB`, and the outer slope is
negative in every group ($s$ = {glue:text}`ksSlopeLo` to {glue:text}`ksSlopeHi` kpc$^{-1}$): the rotation curves of the giants peak at
$R$ = {glue:text}`ksPeakLo`–{glue:text}`ksPeakHi` kpc and decline beyond. The values depend little on the radial range of the sample:
fitted at {glue:text}`ksRhoOuter` < $\rho$ < {glue:text}`ksRhoOut`°, the giants' $\sigma_R$ and $\sigma_z$ change by at most
{glue:text}`ksDSigRzOut` km/s, $\sigma_\phi$ falls by {glue:text}`ksDSigPhiOutLo`–{glue:text}`ksDSigPhiOutHi` km/s, and
$v_{\rm rot}(4\,{\rm kpc})$ rises by {glue:text}`ksDVrotOutLo`–{glue:text}`ksDVrotOutHi` km/s, so the support ratios change by at most
{glue:text}`ksDRatOut`. We take these shifts as the systematic error of the values in {tab}`kin`.

The RR Lyrae have proper motions only, and their errors, {glue:text}`ksErrMedRrl` km/s per component (median), are larger than their dispersions. Fitted at {glue:text}`ksRhoIn` < $\rho$ <
{glue:text}`ksRhoOut`° with one error scale, the crowded inner disc, where *Gaia* underestimates its errors most, sets $k$ =
{glue:text}`ksKRrlIn` and the fit removes the excess scatter by a dispersion that falls by {glue:text}`ksGamPctRrlIn` per cent per kpc
($\gamma$ = {glue:text}`ksGamRrlIn` kpc$^{-1}$), which no other group shows; beyond {glue:text}`ksRhoOuter`° one $k$ =
{glue:text}`ksKRrlOut` describes the errors ({fig}`paper_kin_dispersion_cut`) and the gradient is $\gamma$ = {glue:text}`ksGamRrlOut`
kpc$^{-1}$, as for the giants. We therefore quote the RR Lyrae at {glue:text}`ksRhoOuter` < $\rho$ < {glue:text}`ksRhoOut`°:
$v_{\rm rot}(4\,{\rm kpc})$ = {glue:text}`ksVrotRrl` km/s, $\sigma_R$ = {glue:text}`ksSigRRrl` km/s and $\sigma_\phi$ =
{glue:text}`ksSigPhiRrl` km/s, and $v_{\rm rot}/\sigma_R$ = {glue:text}`ksRatRRrl` (16th–84th percentiles about the median). Their
$\sigma_z$, constrained by the proper motions alone, is not measured. The RR Lyrae thus have the radial dispersion and radial support
of the two metal-poor giant bins ($\sigma_R$ = {glue:text}`ksSigRMPLo`–{glue:text}`ksSigRMPHi` km/s, $v_{\rm rot}/\sigma_R$ =
{glue:text}`ksRatRMPLo`–{glue:text}`ksRatRMPHi`), close to their median [Fe/H] of {glue:text}`ksFehRrl` (16th–84th percentiles
{glue:text}`ksFehRrlLo` to {glue:text}`ksFehRrlHi`), and, compared over the same radii ({glue:text}`ksRhoOuter` < $\rho$ <
{glue:text}`ksRhoOut`°, where the two metal-poor giant bins give $v_{\rm rot}(4\,{\rm kpc})$ = {glue:text}`ksVrotOutA` and
{glue:text}`ksVrotOutB` km/s), the same rotation to within {glue:text}`ksDVrotRrlOut` km/s. Their $\sigma_\phi$ is less certain than its
error suggests: it is correlated with $k$ (correlation coefficient {glue:text}`ksCorrRrl`) and rises to about
{glue:text}`ksSigPhiRrlBand` km/s in the posterior draws where $\sigma_z$ takes values like those of the giants
({glue:text}`ksSigzBandLo`–{glue:text}`ksSigzBandHi` km/s), against {glue:text}`ksSigPhiMPLo`–{glue:text}`ksSigPhiMPHi` km/s in the
metal-poor giant bins, so we do not claim a difference in $\sigma_\phi$.

```{code-cell} python
:tags: [remove-input]
import pandas as pd
from lmcold.tex import fout_percent
# Rotational support per population (s) and the stage-1 fits: giants from the 2-9 deg run, RR Lyrae from the 'outer' run (4 < rho < 9 deg)
s = pd.read_csv('results/kin_support_ellipsoid.csv').set_index('population')
fits = pd.concat([pd.read_csv('results/kin_mcmc_data_ellipsoid.csv').query('stage == 1 and population != "RR Lyrae"'),
                  pd.read_csv('results/kin_mcmc_data_ellipsoid_outer.csv').query('stage == 1 and population == "RR Lyrae"')])
# par: one value (median, p16 or p84) of a fitted parameter; pm: median with its 16th-84th percentile offsets
par = lambda pop, n, c='median': fits[(fits.population == pop) & (fits.param == n)][c].iloc[0]
def pm(m, lo, hi, d):
    return f'{m:.{d}f} (+{hi - m:.{d}f}/−{m - lo:.{d}f})'
# One row per population: rotation at 4 kpc, dispersions, their radial gradient γ = d ln σ / dR, ratios, error scale k, outlier fraction f
rows = {}
for pop, r in s.iterrows():
    q = lambda c, d: pm(r[c], r[f'{c}_p16'], r[f'{c}_p84'], d)
    rows[pop] = {'N': int(r.n), 'v_rot(4 kpc) [km/s]': q('vrot4', 1), 'σ_R [km/s]': q('sigma_R', 1), 'σ_φ [km/s]': q('sigma_phi', 1),
                 'σ_z [km/s]': '–' if pop == 'RR Lyrae' else q('sigma_z', 1),
                 'γ [1/kpc]': pm(par(pop, 'dlnsig_dR'), par(pop, 'dlnsig_dR', 'p16'), par(pop, 'dlnsig_dR', 'p84'), 3),
                 'v_rot/σ_R': q('ratio_R', 2), 'v_rot/σ_z': '–' if pop == 'RR Lyrae' else q('ratio_z', 2),
                 'k': pm(par(pop, 'k'), par(pop, 'k', 'p16'), par(pop, 'k', 'p84'), 2),
                 'f [%]': f"{fout_percent(par(pop, 'logit_fout')):.1f}"}
# Rows from the APOGEE line-of-sight velocities alone, where measured (value ± error)
for pop, r in s[s.vrot4_apogee.notna()].iterrows():
    e = lambda c, d: f"{r[f'{c}_apogee']:.{d}f} ± {r[f'{c}_apogee_err']:.{d}f}"
    rows['APOGEE ' + pop] = {'N': int(r.n_apogee), 'v_rot(4 kpc) [km/s]': e('vrot4', 1), 'σ_R [km/s]': e('sigma_R', 1),
                             'σ_φ [km/s]': e('sigma_phi', 1), 'σ_z [km/s]': e('sigma_z', 1), 'γ [1/kpc]': '',
                             'v_rot/σ_R': e('ratio_R', 2), 'v_rot/σ_z': e('ratio_z', 2), 'k': '', 'f [%]': ''}
pd.DataFrame(rows).T
```

**Table (kin).** Rotation and velocity ellipsoid of each group fitted on its own (stage 1), at $R_\sigma$ = {glue:text}`ksRSig` kpc:
posterior median, with the 16th and 84th percentiles as offsets in parentheses (ratios from the posterior draws). Giants at
{glue:text}`ksRhoIn` < $\rho$ < {glue:text}`ksRhoOut`°, RR Lyrae at {glue:text}`ksRhoOuter` < $\rho$ < {glue:text}`ksRhoOut`°. $N$:
stars; $\gamma$: logarithmic radial gradient of the dispersions; $k$: scale of the *Gaia* proper-motion errors; $f$: outlier fraction.
The shape parameters of the rotation curve, correlated with each other and with $v_0$, are not listed: $R_0$ =
{glue:text}`ksRNullGLo`–{glue:text}`ksRNullGHi` kpc and $s$ = {glue:text}`ksSlopeGLo` to {glue:text}`ksSlopeGHi` kpc$^{-1}$ for the
giants, $R_0$ = {glue:text}`ksRNullRrl` kpc and $s$ = {glue:text}`ksSlopeRrl` kpc$^{-1}$ for the RR Lyrae. Last three rows: APOGEE giants
alone in the three bins above [Fe/H] = {glue:text}`ksApogeeFrom`, maximum-likelihood fits of $v_0$, $\sigma_R$, $\sigma_\phi$,
$\sigma_z$, $k$ and $f$ with the other parameters held at the posterior medians of the full bin (errors from the inverse Hessian,
conditional on the fixed geometry, streaming, $R_0$, $s$ and $\gamma$). The RR Lyrae have no line-of-sight velocities, so no
$\sigma_z$. `results/kin_support_ellipsoid.csv` (`scripts/paper_kin_support_ellipsoid.py`), `results/kin_mcmc_data_ellipsoid.csv`
and `results/kin_mcmc_data_ellipsoid_outer.csv` (`scripts/fit_kinematics_mcmc.py`).

<!-- script: paper_kin_support_ellipsoid.py -->

**Figure (paper_kin_support_ellipsoid).** Rotation and velocity ellipsoid of the old population against [Fe/H] (dex), at $R$ =
{glue:text}`ksRSig` kpc, from the fits of each group on its own (stage 1). The joint fit (stage 2) gives the same rotation and
dispersions to within {glue:text}`ksDStage` km/s for the giants. (a) Rotation speed $v_{\rm rot}(4\,{\rm kpc})$ (km/s), black;
horizontal bars: 16th–84th percentiles of the [Fe/H] of each group. (b) Dispersions $\sigma_R$ (blue), $\sigma_\phi$ (orange) and
$\sigma_z$ (green), km/s. (c) Rotational support $v_{\rm rot}/\sigma_R$ (blue) and $v_{\rm rot}/\sigma_z$ (green), from the posterior
draws. Circles joined by thin lines: all giants in their measured [Fe/H] bin ({glue:text}`ksRhoIn` < $\rho$ <
{glue:text}`ksRhoOut`°), at the median [Fe/H] of the bin. Open squares joined by dashed lines: APOGEE giants alone in the three bins
above [Fe/H] = {glue:text}`ksApogeeFrom`, the lines starting from the last bin without an APOGEE reference. Stars: RR Lyrae at their
median photometric [Fe/H], fitted at {glue:text}`ksRhoOuter` < $\rho$ < {glue:text}`ksRhoOut`° (proper motions only, so no $\sigma_z$
and no $v_{\rm rot}/\sigma_z$). Vertical bars: 16th–84th percentiles of the posterior (APOGEE alone: inverse-Hessian errors). The
quantities of one panel are offset slightly in [Fe/H]. `scripts/paper_kin_support_ellipsoid.py`

## Metallicity errors and the two surveys

A star's bin is set by its measured [Fe/H], and BOSS measurements scatter by about {glue:text}`ksBossScatter` dex about APOGEE's (robust
standard deviation on stars observed by both and not used in the calibration of the BOSS scale). The bins above
{glue:text}`ksFehB` are {glue:text}`ksBinWidth` dex wide or open-ended and lie near the peak of the metallicity distribution, so this scatter moves many stars
between neighbouring bins: of the BOSS giants with effective temperature $T_{\rm eff}$ < {glue:text}`ksTHot` K measured above
{glue:text}`ksFehD` and also observed by APOGEE, {glue:text}`ksLeakE` per cent lie between {glue:text}`ksFehC` and
{glue:text}`ksFehD` on the APOGEE scale. Because the
dispersions change steeply with [Fe/H] at the metal-rich end, the mixing raises the dispersions of the metal-rich bins. Fitted alone,
the APOGEE giants give $\sigma_z$ = {glue:text}`ksSigzApoC`, {glue:text}`ksSigzApoD` and {glue:text}`ksSigzApoE` km/s and $\sigma_R$ =
{glue:text}`ksSigRApoC`, {glue:text}`ksSigRApoD` and {glue:text}`ksSigRApoE` km/s in the three bins above {glue:text}`ksFehB`, against
{glue:text}`ksSigzC`, {glue:text}`ksSigzD` and {glue:text}`ksSigzE` km/s and {glue:text}`ksSigRC`, {glue:text}`ksSigRD` and
{glue:text}`ksSigRE` km/s for all giants ({tab}`kin`); the largest change is the vertical dispersion of the most metal-rich bin.

Two checks support this reading ({fig}`paper_kin_feh_mixing`). They use the sky-aligned dispersion model of chapter 6: instead
of the disc-frame ellipsoid, each group has one dispersion $\sigma_{\rm los}$ of the line-of-sight velocities and one, $\sigma_{\rm PM}$
(also written $\sigma$), of the proper motions, both constant over the disc, with a pure $\tanh$ rotation curve of level $v_0$. The
BOSS velocities are not at fault: on the {glue:text}`ksNBoth` stars observed by both surveys, the velocity difference divided by its
quoted error has a robust standard deviation of {glue:text}`ksZsdLo`–{glue:text}`ksZsdHi` in every bin of signal-to-noise ratio, *Gaia*
$G$ magnitude and $T_{\rm eff}$ (`scripts/fig_boss_velocity_errors.py`), where 1 means correct errors. And mixing the APOGEE dispersions of the bins in the proportions in
which the common stars of each BOSS bin fall into the APOGEE bins,

$$
\sigma_{\rm pred}^2=\sum_tP(t\,|\,j)\,\sigma_{{\rm APOGEE},t}^2,
$$

where $P(t\,|\,j)$ is the fraction of the common stars measured by BOSS in bin $j$ whose APOGEE [Fe/H] lies in bin $t$, predicts the
dispersions of the cool BOSS giants in the sky cells shared with APOGEE (cells of the HEALPix grid,
[Górski et al. 2005](https://doi.org/10.1086/427976), about {glue:text}`ksNsideDeg`° across, holding at least
{glue:text}`ksMinApogee` APOGEE stars of the bin): {glue:text}`ksPredE`, {glue:text}`ksPredD` and {glue:text}`ksPredC` km/s against the
measured {glue:text}`ksMeasE`, {glue:text}`ksMeasD` and {glue:text}`ksMeasC` km/s, from metal-rich to metal-poor. The prediction
neglects the differences in mean velocity between the bins and uses overlap stars brighter than the BOSS sample, which may explain the
{glue:text}`ksShortC` km/s shortfall in the third bin. The hot BOSS giants ($T_{\rm eff}$ ≥ {glue:text}`ksTHot` K), whose [Fe/H] BOSSNet overestimates, are no hotter
kinematically than the cool ones above {glue:text}`ksFehD` ({glue:text}`ksHotE` against {glue:text}`ksCoolE` km/s).

We therefore quote all giants in their measured bins as the default sample, which covers the whole metallicity range with one
selection, and the APOGEE giants alone as the reference above [Fe/H] = {glue:text}`ksFehB`, where their [Fe/H] places nearly every star
in its true bin. With them the rise of rotational support is steeper: $v_{\rm rot}/\sigma_R$ = {glue:text}`ksRatRApoC`,
{glue:text}`ksRatRApoD` and {glue:text}`ksRatRApoE` and $v_{\rm rot}/\sigma_z$ = {glue:text}`ksRatzApoC`, {glue:text}`ksRatzApoD` and
{glue:text}`ksRatzApoE` ({fig}`paper_kin_support_ellipsoid`, dashed). Below {glue:text}`ksFehB` the APOGEE samples are too small for
this test ({glue:text}`ksNApoA` stars below {glue:text}`ksFehA` in the shared cells); there the bins are wider and the dispersions
change slowly with [Fe/H], so mixing changes little. We also tried to deconvolve the metallicity errors with a mixture model in which
each BOSS star belongs to each true bin with a probability set by the response of the BOSS [Fe/H] to the true one. It recovered the
kinematics of mock data, but on the real data the dispersions of the middle bins moved with the details of the response model, so we do not use it.

<!-- script: paper_kin_feh_mixing.py -->

**Figure (paper_kin_feh_mixing).** Metallicity mixing between the giant bins, with the sky-aligned dispersion model, against the
median [Fe/H] of each measured bin (dex). (a) Line-of-sight dispersion $\sigma_{\rm los}$ and (b) rotation level $v_0$ (km/s) of the
APOGEE giants (black circles), the BOSS giants with $T_{\rm eff}$ < {glue:text}`ksTHot` K (blue squares) and with $T_{\rm eff}$ ≥
{glue:text}`ksTHot` K (red triangles) of each bin in the HEALPix cells shared with APOGEE (open symbols: the BOSS subsets on the whole
ring), refitted with the geometry, systemic motion and streaming of the bin's Level-2 fit; bars: inverse-Hessian errors; crosses: the
BOSS values predicted by mixing the APOGEE values of the bins in the proportions in which the stars observed by both surveys fall into
the APOGEE bins (blue: cool, red: hot). (c) Fraction of the BOSS giants with $T_{\rm eff}$ ≥ {glue:text}`ksTHot` K in each bin. (d)
Fraction of the stars observed by both surveys whose APOGEE [Fe/H] lies in another bin than their BOSS [Fe/H], for cool (blue squares)
and hot (red triangles) BOSS giants, with binomial errors. `scripts/paper_kin_feh_mixing.py`

### The BOSS velocity errors

The test of the BOSS line-of-sight velocity errors quoted above uses the {glue:text}`ksNBoth` giants of both release catalogues. For
each star the difference between the BOSS and APOGEE velocities is divided by the two quoted errors added in quadrature; if the BOSS
errors are right, this ratio has a robust standard deviation (1.4826 times the median absolute deviation) of 1, because the APOGEE
errors are much smaller. The script repeats the test in bins of BOSS signal-to-noise ratio, *Gaia* $G$ magnitude and BOSS
$T_{\rm eff}$ and draws a diagnostic figure; the numbers on this page are recomputed from the same stars and bins by
`scripts/paper_text_kin_support.py`.

<!-- script: fig_boss_velocity_errors.py show: the APOGEE and BOSS giant catalogues of Zenodo bundle 1; diagnostic figure, not in the paper -->

## Centre and geometry

The kinematic centre of the joint (stage-2) fit lies {glue:text}`ksCenDx`° east of the vdMK14 centre, at right ascension and
declination $(\alpha,\delta)$ = ({glue:text}`ksCenRa`°, {glue:text}`ksCenDec`°); the Level-1 joint fit puts it at
{glue:text}`ksLoneDx`° east and {glue:text}`ksLoneDy`° south, and the giant bins fitted on their own lie
{glue:text}`ksOwnDxLo`–{glue:text}`ksOwnDxHi`° east and up to {glue:text}`ksOwnDySouth`° south, the metal-poor ones furthest south
({fig}`paper_kin_centre`). The RR Lyrae fitted at {glue:text}`ksRhoOuter` < $\rho$ < {glue:text}`ksRhoOut`° put the centre at
{glue:text}`ksRrlOutDx`° east, with no stars within {glue:text}`ksRhoOuter`° to pin it down; fitted at {glue:text}`ksRhoIn` < $\rho$ <
{glue:text}`ksRhoOut`° they give {glue:text}`ksRrlInDx`°, close to the giants. The joint fit with sky-aligned dispersions puts the
centre {glue:text}`ksSkyDDx`° further east ({glue:text}`ksSkyDx`° east and {glue:text}`ksSkyDy`° south, with $i$ =
{glue:text}`ksSkyInc`°, $\theta$ = {glue:text}`ksSkyTheta`° and the $m=2$ coefficient of the azimuthal velocity $m_{2c}$ =
{glue:text}`ksSkyMtwo`). The density centres of the RR Lyrae and of the *Gaia* red giants, iterated means within apertures of
{glue:text}`ksApertures`° radius, lie {glue:text}`ksDensDxLo`–{glue:text}`ksDensDxHi`° east. The kinematic centre is thus
{glue:text}`ksDistDensLo`–{glue:text}`ksDistDensHi`° from the density centres, {glue:text}`ksDistDensKpcLo`–{glue:text}`ksDistDensKpcHi`
kpc at the LMC distance; published proper-motion centres from *Gaia* ([Gaia Collaboration, Luri et al.
2021](https://doi.org/10.1051/0004-6361/202039588)) and from *Gaia* combined with the VISTA survey of the Magellanic Clouds (VMC; e.g.
[Choi et al. 2022](https://doi.org/10.3847/1538-4357/ac4e90)) lie {glue:text}`ksPubDxLo`–{glue:text}`ksPubDxHi`° east and
{glue:text}`ksPubDyLo`–{glue:text}`ksPubDyHi`° north. We read this as the centre of a disc model fitted to a perturbed velocity field,
not as the position of the potential minimum: it moves by {glue:text}`ksMoveL`° east between the Level-1 and Level-2 joint fits and by up to {glue:text}`ksMoveSky`° with the
sky coverage of a sample (chapter 6), while no
symmetric mock fitted with proper motions and line-of-sight velocities produces an offset larger than {glue:text}`ksMockOffMax`°.

The orientation of the joint fit, $i$ = {glue:text}`ksIncJoint`° and $\theta$ = {glue:text}`ksThetaJoint`°, is likewise a parameter
of the model. Proper motions alone give $i$ = {glue:text}`ksIncPMLo`–{glue:text}`ksIncPMHi`° (maximum-likelihood fits of the giant
bins with the sky-aligned model), as in earlier proper-motion studies (Gaia Collaboration, Luri et al. 2021; Choi et al. 2022), and
adding the line-of-sight velocities lowers it to {glue:text}`ksIncBothLo`–{glue:text}`ksIncBothHi`°; the RR Lyrae fitted with their
distances as well (from the relation between period and Wesenheit magnitude, a dust-insensitive combination of their OGLE-IV $V$ and $I$
magnitudes) lie in a plane with $i\approx$ {glue:text}`ksIncDist`° and $\theta\approx$ {glue:text}`ksThetaDist`°, the plane of
their positions rather than of their motions.

{fig}`paper_kin_geometry_feh` shows how the orientation and the shape of the velocity ellipsoid change with metallicity when each group
is fitted on its own, and compares them with the joint fit. The line of nodes is the same for all giant bins, $\theta$ =
{glue:text}`ksThetaOwnLo`–{glue:text}`ksThetaOwnHi`° against {glue:text}`ksThetaJoint` ± {glue:text}`ksThetaJointErr`° in the joint
fit. The inclination is not: it rises from {glue:text}`ksIncA`° and {glue:text}`ksIncB`° in the two metal-poor bins to
{glue:text}`ksIncRichLo`–{glue:text}`ksIncRichHi`° above [Fe/H] = {glue:text}`ksFehB`, where the joint value
({glue:text}`ksIncJoint` ± {glue:text}`ksIncJointErr`°) is set by the most numerous stars; the metal-poor bins lie
{glue:text}`ksNSigLo`–{glue:text}`ksNSigHi` standard deviations below it. Whether this is a thicker or more flared metal-poor disc seen
in projection, or a different plane, our thin-disc model cannot tell from these data alone; a mock as thick as the old population is
recovered at its true inclination (chapter 8). The RR Lyrae, with proper motions only, constrain their orientation weakly ($i$ =
{glue:text}`ksIncRrl`°, $\theta$ = {glue:text}`ksThetaRrl`°). The shape of the ellipsoid does not depend on the geometry:
$\sigma_\phi/\sigma_R$ and $\sigma_z/\sigma_R$ of each giant bin agree between the two stages to within {glue:text}`ksDRatStage`, well
inside their errors (half-widths of the 16th–84th percentile ranges {glue:text}`ksRatErrLo`–{glue:text}`ksRatErrHi`). For the RR
Lyrae $\sigma_\phi/\sigma_R$ = {glue:text}`ksRatPhiRrl` on their own and {glue:text}`ksRatPhiRrlJ` with the shared geometry of the
joint fit at {glue:text}`ksRhoOuter` < $\rho$ < {glue:text}`ksRhoOut`°, lower than the {glue:text}`ksRatPhiLo`–{glue:text}`ksRatPhiHi`
of the giants, but correlated with $k$ and with the unconstrained $\sigma_z$.

<!-- script: paper_kin_geometry_feh.py -->

**Figure (paper_kin_geometry_feh).** Orientation of the disc and shape of the velocity ellipsoid against [Fe/H] (dex). (a) Inclination
$i$ (deg); (b) position angle $\theta$ of the line of nodes (deg); (c) $\sigma_\phi/\sigma_R$; (d) $\sigma_z/\sigma_R$ (at
{glue:text}`ksRSig` kpc; from paired posterior draws). Filled symbols: each group fitted on its own (stage 1; giants at
{glue:text}`ksRhoIn` < $\rho$ < {glue:text}`ksRhoOut`°, black circles joined by a thin line; RR Lyrae at {glue:text}`ksRhoOuter` <
$\rho$ < {glue:text}`ksRhoOut`°, blue star, not in (d) because their $\sigma_z$ is not measured). Grey band and line in (a) and (b): the
shared orientation of the joint fit (16th–84th percentiles and median). Open symbols in (c) and (d): each group's ellipsoid in the joint
fit (RR Lyrae: joint fit of all groups at {glue:text}`ksRhoOuter` < $\rho$ < {glue:text}`ksRhoOut`°). Vertical bars: 16th–84th
percentiles; horizontal bars: 16th–84th percentiles of the [Fe/H] of each group. `scripts/paper_kin_geometry_feh.py`

<!-- script: paper_kin_centre.py -->

**Figure (paper_kin_centre).** Centres of the LMC: offsets east (to the left) and north of the vdMK14 centre (deg, on the orthographic
projection about it). Coloured circles: kinematic centres of the Level-2 velocity-ellipsoid fits of each group on its own (stage 1;
giants at {glue:text}`ksRhoIn` < $\rho$ < {glue:text}`ksRhoOut`°, red to blue from metal-poor to metal-rich; RR Lyrae, purple, at
{glue:text}`ksRhoOuter` < $\rho$ < {glue:text}`ksRhoOut`°), with 16th–84th percentile bars. Stars: joint fits (filled: Level 2;
open: Level 1 with sky-aligned dispersions). Diamonds joined by lines: density centres of the RR Lyrae (purple) and of the *Gaia* red
giants with $G$ < {glue:text}`ksGLim` (grey) within apertures of {glue:text}`ksApertures`° (open diamond: the largest). Open grey symbols: published
kinematic centres (squares), density centres (diamonds) and density centres of old stars (triangles), labelled in the panel, among them
[van der Marel (2001)](https://doi.org/10.1086/323100) and [Cusano et al. (2021)](https://doi.org/10.1093/mnras/stab901). +: the
vdMK14 centre. `scripts/paper_kin_centre.py`

## Sensitivity to the model

The sky-aligned model ranks the groups in the same order as the velocity ellipsoid: its rotational support $v_0/\sigma_{\rm los}$
rises from {glue:text}`ksRatLosA` to {glue:text}`ksRatLosE` from the metal-poor to the metal-rich giants (APOGEE alone:
{glue:text}`ksRatLosApoE`; {fig}`paper_kin_support`).

<!-- script: paper_kin_support.py -->

**Figure (paper_kin_support).** Rotational support of the LMC giants against [Fe/H] (dex) in the model with sky-aligned dispersions,
stage-1 fits at {glue:text}`ksRhoIn` < $\rho$ < {glue:text}`ksRhoOut`°. (a) Rotation level $v_0$ of $v_{\rm rot}=v_0\tanh(R/R_0)$ (km/s).
(b) Dispersion of the line-of-sight velocities $\sigma_{\rm los}$ (filled) and of the proper motions $\sigma_{\rm PM}$ (open), km/s,
the latter from the fits at {glue:text}`ksRhoOuter` < $\rho$ < {glue:text}`ksRhoOut`°. (c) $v_0/\sigma_{\rm los}$ (filled) and
$v_0/\sigma_{\rm PM}$ (open; RR Lyrae: star), from the posterior draws. Horizontal bars: 16th–84th percentiles of the [Fe/H] of each
group. Black circles: all giants in their measured [Fe/H] bin; red open squares: APOGEE giants alone above [Fe/H] =
{glue:text}`ksApogeeFrom` (inverse-Hessian errors); blue stars: RR Lyrae (own fit; $\sigma_{\rm PM}$ from {glue:text}`ksRhoOuter` <
$\rho$ < {glue:text}`ksRhoOut`°), open star in (a): their $v_0$ in the joint fit. Vertical bars: 16th–84th percentiles.
`scripts/paper_kin_support.py`

The $m=2$ terms are not the only way to describe the residual pattern of the velocity field (chapter 6). We refitted each giant bin (maximum
likelihood, sky-aligned dispersions) with three alternatives, each recovered from mocks in which it is the only perturbation
({fig}`paper_kin_sensitivity`): (i) an inner disc ($\rho$ < {glue:text}`ksRhoTilt`°) in its own plane, inclined
{glue:text}`ksDIncTiltLo`–{glue:text}`ksDIncTiltHi`° less than the outer disc, which improves the fit of the two metal-rich bins
($\Delta\ln L$ = {glue:text}`ksDlnLTiltD` and {glue:text}`ksDlnLTiltE`; ≤ {glue:text}`ksDlnLTiltOther` in the others) and changes $v_0$
by ≤ {glue:text}`ksDVTilt` km/s; (ii) a different scale of the internal motions in the line-of-sight velocities and in the proper
motions, the line-of-sight deviations from the mean motion being multiplied by $(1+q_{\rm los})$, with $q_{\rm los}$ =
{glue:text}`ksQlosD` and {glue:text}`ksQlosE` and $\Delta\ln L$ = {glue:text}`ksDlnLQD` and {glue:text}`ksDlnLQE` in the two metal-rich
bins ($|q_{\rm los}|$ ≤ {glue:text}`ksQlosOther`, $\Delta\ln L$ ≤ {glue:text}`ksDlnLQOther` in the others), at the price of $i$ =
{glue:text}`ksIncQLo`–{glue:text}`ksIncQHi`°; (iii) a disc whose orientation changes with time
([van der Marel et al. 2002](https://doi.org/10.1086/343775)), which alone fits worse than Level 2 with sky-aligned dispersions
($\Delta\ln L$ = {glue:text}`ksTurnWorseLo` to {glue:text}`ksTurnWorseHi`) and combined with it fits better ($\Delta\ln L$ =
{glue:text}`ksTurnBetterLo` to {glue:text}`ksTurnBetterHi`) but requires rates of change of the line of nodes of
+{glue:text}`ksDthdtLo` to +{glue:text}`ksDthdtHi`° Gyr$^{-1}$: a turn of {glue:text}`ksTurnLo`–{glue:text}`ksTurnHi`° in one rotation
period at {glue:text}`ksRSig` kpc ({glue:text}`ksPeriod` Gyr, $2\pi R/v_{\rm rot}$ for the fastest-rotating giants), which we do not
consider a physical description of the disc. $\Delta\ln L$ is measured against Level 2 of the same bin; positive values favour the
alternative. Part of what model (ii) absorbs, a different amplitude of the internal motions along the line of sight and on the sky, is
what the velocity ellipsoid describes physically. The line-of-sight dispersions are the same in every model to
{glue:text}`ksDSigLos` km/s. The rotation amplitude is not: it changes by up to {glue:text}`ksDVAlt` km/s between Level 2 and the
alternatives (i)–(ii) (the metal-rich bin in model ii) and by {glue:text}`ksDVTurnLo`–{glue:text}`ksDVTurnHi` km/s in (iii) with
streaming. The ordering by rotational support is robust: $v_0/\sigma_{\rm los}$ increases monotonically from the metal-poor to the
metal-rich bins in every model except the turning disc alone, which fits worst and in which only the two most metal-poor bins swap
({glue:text}`ksTurnSwapA` and {glue:text}`ksTurnSwapB`); in the velocity-ellipsoid model both $v_{\rm rot}/\sigma_R$ and
$v_{\rm rot}/\sigma_z$ increase monotonically ({tab}`kin`). Absolute rotation speeds are conditional on the model.

With the velocity ellipsoid we repeated tests (i) and (ii) and tested the single error scale $k$ (maximum likelihood, each giant bin on
its own). The inner disc still improves the fit of the two metal-rich bins ($\Delta\ln L$ = {glue:text}`ksDlnLETiltD` and
{glue:text}`ksDlnLETiltE`; {glue:text}`ksDlnLETiltOtherLo`–{glue:text}`ksDlnLETiltOtherHi` in the others), inclined
{glue:text}`ksDIncETiltLo`–{glue:text}`ksDIncETiltHi`° less than the outer disc, and changes $v_{\rm rot}(4\,{\rm kpc})$ by at most
{glue:text}`ksDVETilt` km/s and the dispersions by at most {glue:text}`ksDSigETilt` km/s. The line-of-sight scale is no longer needed in
the bin from {glue:text}`ksFehC` to {glue:text}`ksFehD` ($q_{\rm los}$ = {glue:text}`ksQlosED`, $\Delta\ln L$ = {glue:text}`ksDlnLQED`,
against {glue:text}`ksDlnLQD` with sky-aligned dispersions): the ellipsoid describes the different amplitude of the motions along the
line of sight and on the sky. In the most metal-rich bin it still gives $\Delta\ln L$ = {glue:text}`ksDlnLQEE` with $q_{\rm los}$ =
{glue:text}`ksQlosEE`, again at the price of $i$ = {glue:text}`ksIncQEE`°, and in the two metal-poor bins it runs to its bound,
+{glue:text}`ksQBound`, for $\Delta\ln L$ = {glue:text}`ksDlnLQEALo`–{glue:text}`ksDlnLQEAHi`; in every bin the dispersions change by
at most {glue:text}`ksDSigEQ` km/s and $v_{\rm rot}(4\,{\rm kpc})$ by up to {glue:text}`ksDVEQ` km/s. Fitting the stars brighter and
fainter than the median *Gaia* $G$ magnitude of each bin separately, each half with its own $k$, rotation curve and ellipsoid and the
geometry and streaming held, gives $k$ = {glue:text}`ksKBrightLo`–{glue:text}`ksKBrightHi` for the bright halves and
{glue:text}`ksKFaintLo`–{glue:text}`ksKFaintHi` for the faint ones: *Gaia* underestimates the errors of the brighter stars more, as a
constant error floor would. The dispersions of the two halves differ by at most {glue:text}`ksDSigHalf` km/s in the three bins below
[Fe/H] = {glue:text}`ksFehC`, and are {glue:text}`ksDSigHalfMRLo`–{glue:text}`ksDSigHalfMRHi` km/s higher for the faint halves in the
two metal-rich bins, in $\sigma_z$ as much as in $\sigma_R$ and $\sigma_\phi$. Because $\sigma_z$ rests mostly on the line-of-sight
velocities, which do not depend on $k$, that difference comes from the stars, plausibly the stronger [Fe/H] mixing of the fainter BOSS
giants, and not from the single $k$ (`results/kin_ellipsoid_checks.csv`, `scripts/kin_ellipsoid_checks.py`).

The maximum-likelihood fits of the sky-aligned models (i)–(iii) are made by three diagnostic scripts, each of which writes a table
read by `scripts/paper_kin_sensitivity.py` and draws a diagnostic figure. Each script also fits mocks drawn at the positions of the
giants with {glue:text}`ksFehC` < [Fe/H] < {glue:text}`ksFehD`, to test whether one perturbation is mistaken for another.

### The tilted inner disc

`scripts/fig_kin_tilted_ring.py` fits each giant bin and the RR Lyrae with a symmetric disc (Level 1), with the $m=2$ streaming
(Level 2), with an inner disc at $\rho$ < {glue:text}`ksRhoTilt`° in its own plane (offsets of inclination and line of nodes from
the outer disc, same centre), and with both. Its mocks are drawn from the tilted-disc fit, to see whether the streaming model turns a
tilt into streaming locked to the line of nodes, and from the streaming fit, to see whether a tilt appears where there is none. Its
table `results/kin_tilted_ring.csv` gives the Level-1, Level-2 and tilted-disc-with-streaming rows of {fig}`paper_kin_sensitivity`.

<!-- script: fig_kin_tilted_ring.py show: the giant and RR Lyrae catalogues of Zenodo bundle 1 and maximum-likelihood fits of four models to every sample and two mocks; diagnostic figure, not in the paper -->

### The line-of-sight scale and the two surveys

`scripts/fig_kin_los_scale.py` fits each giant bin with the symmetric disc, with streaming, with the line-of-sight scale
$(1+q_{\rm los})$ and with both, three times: with the line-of-sight velocities of both surveys, of APOGEE alone and of BOSS alone
(all proper motions kept). The split by survey tests whether a velocity offset between the surveys, which cover different fields,
produces the streaming. Its mocks are drawn from the $q_{\rm los}$ fit and from the streaming fit. The rows with the velocities of both
surveys give model (ii) of {fig}`paper_kin_sensitivity` (`results/kin_los_scale.csv`).

<!-- script: fig_kin_los_scale.py show: the giant catalogues of Zenodo bundle 1 and maximum-likelihood fits of four models to every bin with three sets of velocities; diagnostic figure, not in the paper -->

### The turning disc

`scripts/fig_kin_turning.py` adds to the model a disc whose inclination and line of nodes change at constant rates (in ° Gyr$^{-1}$),
the same in the proper motions and the line-of-sight velocities, and fits each giant bin with it alone, with streaming, and with the
line-of-sight scale and streaming. Its mocks are drawn from the streaming fit, to see whether streaming produces a spurious turn, and
from the turning-disc fit, to see whether a turn produces spurious streaming. Its table `results/kin_turning.csv` gives model (iii) of
{fig}`paper_kin_sensitivity`.

<!-- script: fig_kin_turning.py show: the giant catalogues of Zenodo bundle 1 and maximum-likelihood fits of six models to every bin and two mocks; diagnostic figure, not in the paper -->

### Rotation and support in every model

<!-- script: paper_kin_sensitivity.py -->

**Figure (paper_kin_sensitivity).** Rotation level $v_0$ (a; km/s) and rotational support $v_0/\sigma_{\rm los}$ (b) of the giant bins
against their median [Fe/H] (dex) in the alternative disc models, maximum-likelihood fits of each bin on its own with sky-aligned
dispersions: Level 1 (grey), Level 2 (blue), an inner disc ($\rho$ < {glue:text}`ksRhoTilt`°) in its own plane with streaming
(green), a line-of-sight scale $(1+q_{\rm los})$ with streaming (orange), and a disc whose orientation changes with time, alone
(purple, dashed) and with streaming (red, dashed); the models are offset slightly in [Fe/H]. Black squares: the Level-2 posterior
medians with 16th–84th percentiles ({fig}`paper_kin_support`). `scripts/paper_kin_sensitivity.py`

## The inner edge of the sample

The proper-motion dispersion of a group is separated from the *Gaia* errors only through the error scale $k$: for a star with velocity
error $e$ the likelihood constrains mainly $\sigma_{\rm PM}^2+k^2e^2$. The giants have $e\approx$
{glue:text}`ksErrMedGLo`–{glue:text}`ksErrMedGHi` km/s and the RR Lyrae $e\approx$ {glue:text}`ksErrMedRrl` km/s per component
(medians at {glue:text}`ksRhoIn` < $\rho$ < {glue:text}`ksRhoOut`°); for the RR Lyrae at {glue:text}`ksRhoOuter` < $\rho$ <
{glue:text}`ksRhoOut`° ($e\approx$ {glue:text}`ksErrMedRrlOut` km/s) a change of $k$ by {glue:text}`ksDK` moves $\sigma_{\rm PM}$ by
{glue:text}`ksDSigKLo`–{glue:text}`ksDSigKHi` km/s. For the giants $k$
matters little; for the RR Lyrae it sets the dispersion. *Gaia* underestimates its errors more in crowded fields, so one $k$ need not
describe a whole group. {fig}`paper_kin_dispersion_cut` shows the maximum-likelihood $\sigma_{\rm PM}$ and $k$ of each group
(sky-aligned Level 2, all parameters free) as the inner edge of the sample moves from {glue:text}`ksInnerLo` to
{glue:text}`ksInnerHi`°. For the giants $\sigma_{\rm PM}$ changes by {glue:text}`ksDSigGiantLo`–{glue:text}`ksDSigGiantHi` km/s,
partly a real decline with radius. For the RR Lyrae $k$ falls from {glue:text}`ksKRrlCutHi` to {glue:text}`ksKRrlCutLo` and
$\sigma_{\rm PM}$ rises from the {glue:text}`ksSigBound` km/s lower bound of these fits to {glue:text}`ksSigRrlThree` km/s at {glue:text}`ksInnerTwo`° and
{glue:text}`ksSigRrlOutLo`–{glue:text}`ksSigRrlOutHi` km/s from {glue:text}`ksRhoOuter`° out. A mock with the positions and errors of
the RR Lyrae, $\sigma_{\rm PM}$ = {glue:text}`ksMockSigma` km/s and $k$ = {glue:text}`ksMockKIn` inside {glue:text}`ksRhoOuter`° and
{glue:text}`ksMockKOut` beyond, shows the same behaviour: one $k$ fitted from {glue:text}`ksInnerLo`° returns $\sigma_{\rm PM}$ =
{glue:text}`ksMockSigTwo` km/s, and from {glue:text}`ksRhoOuter`° out the fit recovers
{glue:text}`ksMockSigOutLo`–{glue:text}`ksMockSigOutHi` km/s and $k$ = {glue:text}`ksMockKOutFitLo`–{glue:text}`ksMockKOutFitHi`. The
velocity-ellipsoid model, whose dispersions may vary with radius, meets the same problem in another form: at {glue:text}`ksRhoIn` <
$\rho$ < {glue:text}`ksRhoOut`° it fits the RR Lyrae with a dispersion that falls by {glue:text}`ksGamPctRrlIn` per cent per kpc. We
therefore measure the RR Lyrae dispersions at {glue:text}`ksRhoOuter` < $\rho$ < {glue:text}`ksRhoOut`°.

<!-- script: paper_kin_dispersion_cut.py -->

**Figure (paper_kin_dispersion_cut).** Proper-motion dispersion $\sigma_{\rm PM}$ (top; km/s) and *Gaia* error scale $k$ (bottom)
against the inner edge of the sample (deg; outer edge {glue:text}`ksRhoOut`°), maximum-likelihood fits of the sky-aligned Level-2 model
with all parameters free; bars from the inverse Hessian with the geometry held (the RR Lyrae at {glue:text}`ksInnerLo`° sit at the
{glue:text}`ksSigBound` km/s bound and have none). Left: data, one line per group (giant bins red to blue from metal-poor to
metal-rich, RR Lyrae purple; offset slightly in the inner edge). Right: mock RR Lyrae with $\sigma_{\rm PM}$ =
{glue:text}`ksMockSigma` km/s and $k$ = {glue:text}`ksMockKIn` inside {glue:text}`ksRhoOuter`° and {glue:text}`ksMockKOut` beyond
(dashed: the true values). `scripts/paper_kin_dispersion_cut.py`

## Summary of the kinematics

1. Rotational support rises monotonically with metallicity. At {glue:text}`ksRSig` kpc, $v_{\rm rot}/\sigma_R$ rises from
   {glue:text}`ksRatRA` below [Fe/H] = {glue:text}`ksFehA` to {glue:text}`ksRatRE` above {glue:text}`ksFehD` and
   $v_{\rm rot}/\sigma_z$ from {glue:text}`ksRatzA` to {glue:text}`ksRatzE` (APOGEE alone: {glue:text}`ksRatRApoE` and
   {glue:text}`ksRatzApoE`); in total the rotation rises by {glue:text}`ksDVrotAE` km/s and $\sigma_R$ and $\sigma_z$ fall by
   {glue:text}`ksDSigRAE` and {glue:text}`ksDSigzAE` km/s, most of it above [Fe/H] = {glue:text}`ksFehB`.
2. The velocity ellipsoid has the same ordering of axes in every giant bin, $\sigma_R>\sigma_\phi\approx\sigma_z$ with
   $\sigma_\phi/\sigma_R$ and $\sigma_z/\sigma_R$ of {glue:text}`ksRatZLo`–{glue:text}`ksRatPhiHi`, and all dispersions decline
   outwards by {glue:text}`ksGamPctMPLo`–{glue:text}`ksGamPctMRHi` per cent per kpc. The RR Lyrae have the radial dispersion, rotation
   and radial support of the giants below [Fe/H] = {glue:text}`ksFehB` ($\sigma_R$ = {glue:text}`ksSigRRrl` km/s,
   $v_{\rm rot}/\sigma_R$ = {glue:text}`ksRatRRrl`).
3. The ordering holds in both surveys and in every disc model we tried except the worst-fitting one, which swaps only the two most
   metal-poor bins; the absolute rotation speed depends on the model.
4. Fitted on their own, the giant bins put the rotation centre {glue:text}`ksOwnDxRoundLo`–{glue:text}`ksOwnDxRoundHi`° east of the
   vdMK14 centre and {glue:text}`ksOwnDistDensLo`–{glue:text}`ksOwnDistDensHi`° from the density centres (joint fit:
   {glue:text}`ksCenDx`° east, {glue:text}`ksDistDensLo`–{glue:text}`ksDistDensHi`° from them), and all groups leave an $m=2$
   residual of {glue:text}`ksMTwoAmpLo`–{glue:text}`ksMTwoAmpHi` km/s in $v_t$ with a common phase about a symmetric disc.
5. Fitted on their own, all giant bins share the line of nodes ($\theta$ = {glue:text}`ksThetaOwnIntLo`–{glue:text}`ksThetaOwnIntHi`°),
   but the two bins below [Fe/H] = {glue:text}`ksFehB` are less inclined ($i$ = {glue:text}`ksIncAInt` and {glue:text}`ksIncBInt`°)
   than those above ({glue:text}`ksIncRichLo`–{glue:text}`ksIncRichHi`°; {fig}`paper_kin_geometry_feh`); the shape of the velocity
   ellipsoid is the same whether the geometry is fitted per group or shared.
