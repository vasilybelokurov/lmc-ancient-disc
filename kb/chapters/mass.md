---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Stellar mass per metallicity

Every number on this page is written by `scripts/paper_text_mass.py` from the result tables (`results/macros_mass.tex`); the code
cells are the figure scripts themselves, step by step, and the tables are read from the result files.

The Letter compares the LMC's old stars with simulated galaxies at the stellar mass the LMC had formed when those stars were born.
That comparison needs the stellar mass the LMC formed at each metallicity. Metallicity here is the iron abundance [Fe/H], the
logarithm of the ratio of iron to hydrogen atoms relative to the Sun's (so [Fe/H] = −1 is one tenth of the solar iron abundance).
This page converts the red giants with spectra into that mass. The giants come from two spectroscopic surveys:
[APOGEE DR17](https://doi.org/10.3847/1538-4365/ac4414) (the seventeenth data release of the Sloan Digital Sky Survey, SDSS, with
near-infrared spectra analysed by its pipeline ASPCAP) and the BOSS optical spectra of [SDSS-V DR20](https://doi.org/10.48550/arXiv.2607.26149),
whose stellar parameters come from the neural network BOSSNet ([Sizemore et al. 2024](https://doi.org/10.3847/1538-3881/ad291d)). Both are
placed on one scale: the BOSS [Fe/H] is calibrated to APOGEE on the stars observed by both (chapter 3). Positions are angular
distances $\rho$ from the LMC centre of [van der Marel & Kallivayalil (2014)](https://doi.org/10.1088/0004-637X/781/2/121); magnitudes
$G$ and colours $G_{\rm BP}-G_{\rm RP}$ are from [*Gaia* DR3](https://doi.org/10.1051/0004-6361/202243940), $H$ from
[2MASS](https://doi.org/10.1086/498708). M☉ is the mass of the Sun.

**Mass samples.** The masses use the quality cuts of the selection functions (chapter 2), without the kinematic cuts of the samples
used for the velocities, with the giant cut and {glue:text}`msVlosLo` < $v_{\rm los}$ < {glue:text}`msVlosHi` km/s, where
$v_{\rm los}$ is the line-of-sight velocity (calibrated velocities for BOSS). They hold {glue:text}`msNBoss` BOSS and
{glue:text}`msNApogee` APOGEE giants with positive selection weights.

We convert the weighted BOSS giants into the stellar mass formed in each [Fe/H] bin within $\rho$ = {glue:text}`msRad`°. Bins are
{glue:text}`msBinW` dex wide from {glue:text}`msFehLo` to {glue:text}`msFehHi`; the first collects all stars below
{glue:text}`msFehLo` and the last all stars above {glue:text}`msFehHi`. Three steps lead from the spectra to the mass: corrected
counts, giants per unit mass from stellar-population models, and the response of the measured BOSS [Fe/H] to the true one.

## Corrected counts

The corrected number of LMC giants in bin $k$ is

$$
N_k=\sum_{i\in k}w_i\,c_i\,\bigl(1-f_{{\rm MW},i}\bigr),
$$

where the sum runs over the BOSS giants $i$ with [Fe/H] in bin $k$. The selection weight $w_i$ is the number of stars of the BOSS
target list (the parent sample, rebuilt from *Gaia* DR3 as the SDSS-V target class, or carton, `mwm_magcloud_rgb_boss`, selected it) that
each giant with a good spectrum stands for; the coverage factor $c_i$ adds the parent stars in sky areas where too few spectra were
taken to compute a weight. Both are derived in chapter 2. $f_{{\rm MW},i}$ is the fraction of Milky Way foreground stars expected at the
position and metallicity of star $i$. It comes from a fit of the LMC disc and a Milky Way component to the velocities and proper
motions of the giants per [Fe/H] group ({fig}`fig_giant_foreground`), which gives mean fractions of
{glue:text}`msFmwPoorLo`–{glue:text}`msFmwPoorHi` per cent below [Fe/H] = −1.5 and about {glue:text}`msFmwRich` per cent above −1
over $\rho$ = {glue:text}`msRefLo`–{glue:text}`msRefHi`°. We assume a foreground of uniform surface density in each [Fe/H] bin, so
$f_{\rm MW}(\rho)$ is this mean fraction times the ratio of the mean surface density of the bin over
{glue:text}`msRefLo`–{glue:text}`msRefHi`° to its surface density at $\rho$. The metal-poor giants whose proper motions lie more than
{glue:text}`msSide` mas/yr from the local LMC motion, a sample dominated by the foreground, have a surface density that is constant
with radius within its errors ($\chi^2$ = {glue:text}`msSideChi` for {glue:text}`msSideDof` degrees of freedom, set by
{glue:text}`msSideN` stars at {glue:text}`msSideRhoLo`–{glue:text}`msSideRhoHi`°; see the checks below).

## Giants per unit mass

A stellar population of formed mass $M$, age $\tau$ and metallicity [M/H] places $\eta([{\rm M/H}],\tau)\,M$ stars inside the BOSS
selection. [M/H] is the abundance of all metals relative to the Sun's, which the stellar models take as input; $\eta$, the number of
selected giants per solar mass formed, is the efficiency of the conversion. We compute $\eta$ from the PARSEC v1.2S stellar isochrones
([Bressan et al. 2012](https://doi.org/10.1111/j.1365-2966.2012.21948.x)) with the COLIBRI tracks of the thermally pulsing asymptotic giant
branch (TP-AGB) calibrated on the LMC ([Marigo et al. 2017](https://doi.org/10.3847/1538-4357/835/1/77);
[Pastorelli et al. 2020](https://doi.org/10.1093/mnras/staa2565)), for a [Kroupa (2001)](https://doi.org/10.1046/j.1365-8711.2001.04022.x)
initial mass function (IMF) normalised to unit mass between {glue:text}`msImfLo` and {glue:text}`msImfHi` M☉. Each isochrone is placed
at the distance modulus {glue:text}`msDM` ([Pietrzyński et al. 2019](https://doi.org/10.1038/s41586-019-0999-4)) plus the offset of
the inclined disc (inclination {glue:text}`msInc`°, line of nodes at position angle {glue:text}`msTheta`°;
[van der Marel & Kallivayalil 2014](https://doi.org/10.1088/0004-637X/781/2/121)), smeared with a Gaussian depth of
{glue:text}`msDepth` mag, and reddened with the true extinction. Reddening is measured by the colour excess $E(B-V)$, the extra
$B-V$ colour that dust adds. Inside the footprint of the OGLE-IV survey ([Udalski et al. 2015](https://ui.adsabs.harvard.edu/abs/2015AcA....65....1U))
we take the colour excess $E(V-I)$ from the red-clump map of [Skowron et al. (2021)](https://doi.org/10.3847/1538-4365/abcb81) and convert
it with $E(V-I)/E(B-V)$ = {glue:text}`msEviEbv` ([Schlafly & Finkbeiner 2011](https://doi.org/10.1088/0004-637X/737/2/103), Table 6);
outside it we use the dust map of [Schlegel, Finkbeiner & Davis (1998)](https://doi.org/10.1086/305772) (SFD). The carton's polygon in
the colour–magnitude diagram (CMD) is then applied to the photometry as the carton dereddened it, with its own reddening, and the
limit $G\leq$ {glue:text}`msGLim` to the observed magnitude. The difference matters: the median true $E(B-V)$ of the parent is
{glue:text}`msEbvTrue`, while the carton removed {glue:text}`msEbvCarton`. We count the isochrone stars inside the selection by
integrating the cumulative IMF along each isochrone, in {glue:text}`msNSub` steps between tabulated points. The calculation runs
separately in {glue:text}`msNReg` regions (rings with edges at {glue:text}`msRings`°, each split into quadrants), at the median
extinction and disc offset of their parent stars; the offsets span {glue:text}`msDmuLo` to {glue:text}`msDmuHi` mag. Applying the
regional median to dusty patches underestimates the stars they lose; an infrared test ({fig}`paper_parent_extinction`) puts the
missing parent stars at {glue:text}`msMissing` per cent of the central {glue:text}`msRMax`°.

The models need [M/H], an age and a mass for each [Fe/H] bin. We convert with
[M/H] = [Fe/H] + log({glue:text}`msSalA` × 10<sup>[α/Fe]</sup> + {glue:text}`msSalB`)
([Salaris, Chieffi & Straniero 1993](https://doi.org/10.1086/173105)), where [α/Fe] is the abundance of the α elements (O, Mg, Si,
Ca, Ti) relative to iron, in solar units. For [α/Fe] we use the median ASPCAP [α/M] of the APOGEE members in the bin:
{glue:text}`msAlOne` at [Fe/H] = {glue:text}`msAlFehOne`, {glue:text}`msAlTwo` at {glue:text}`msAlFehTwo`, {glue:text}`msAlThree` at
{glue:text}`msAlFehThree` and {glue:text}`msAlFour` at {glue:text}`msAlFehFour`, held constant beyond the measured range. The giants of
one [Fe/H] bin span a range of ages, so we average $\eta$ over the distribution of formed mass in age expected for the bin. We take it
from the age–metallicity relation of the LMC field giants of [Povick et al. (2024)](https://doi.org/10.1093/mnras/stae1949): a median
[Fe/H] falling from {glue:text}`msAmrNow` at the present to {glue:text}`msAmrOld` at {glue:text}`msAmrAge` Gyr, with a Gaussian scatter
of {glue:text}`msAmrSigLo`–{glue:text}`msAmrSigHi` dex (read from their fig. 20). With a constant star-formation rate, a bin's ages
are weighted by the probability that a star of each age has the bin's [Fe/H]. {tab}`ages` gives the resulting ages for four [Fe/H]
ranges, together with three fixed windows of uniform formed mass that we keep as alternatives: A, which changes the masses below
[Fe/H] = −2, −1.5 and −1 and the total by {glue:text}`msAWinLo` to {glue:text}`msAWinHi` per cent from the reference, and B and C,
which shift the windows older and younger. The isochrone grid ends at {glue:text}`msGridAge` Gyr, so older ages (windows A and B) are
truncated there, and the grid starts at [M/H] = {glue:text}`msGridMh`. {fig}`paper_mass_method`(a) shows $\eta$ for four
metallicities: between the grid ages nearest {glue:text}`msEtaAgeLo` and {glue:text}`msEtaAgeHi` Gyr it falls by a factor of
{glue:text}`msEtaFacLo` at [M/H] = {glue:text}`msEtaMhLo` and {glue:text}`msEtaFacHi` at [M/H] = {glue:text}`msEtaMhHi`, the main
reason why the age prior matters.

```{code-cell} python
:tags: [remove-input]
import json
import pandas as pd
from lmcold.mass_conversion import AGE_RANGES
# Reference ages [Gyr] per [Fe/H] bin (16th-84th percentiles and mean) from the age-metallicity relation
a = json.load(open('results/age_weighting.json'))
rows = {'Reference (age–metallicity relation)': [f"{v['p16_p84'][0]:.1f}–{v['p16_p84'][1]:.1f} ({v['mean']:.1f})" for v in a.values()]}
# The age windows [Gyr] of lmcold.mass_conversion, one row each
rows.update({f'Window {k}' + {'A': '', 'B': ' (older)', 'C': ' (younger)'}[k]: [f'{lo:g}–{hi:g}' for lo, hi in w]
             for k, w in AGE_RANGES.items()})
pd.DataFrame(rows, index=['[Fe/H] < −1.5', '−1.5 to −1', '−1 to −0.5', '> −0.5']).T
```

**Table (ages).** Ages (Gyr) of the formed mass per [Fe/H] range: the reference weighting from the age–metallicity relation of
Povick et al. (2024), 16th–84th percentiles with the mean in parentheses (`results/age_weighting.json`, written by
`scripts/paper_mass_method.py`), and the windows of uniform formed mass of the alternatives A, B and C
(`lmcold.mass_conversion.AGE_RANGES`).

## The BOSS metallicity response

The mass at low [Fe/H] is sensitive to how BOSS measures metallicity. On the {glue:text}`msNOverlap` stars of the two mass samples
observed by both surveys, calibrated BOSS agrees with APOGEE to within {glue:text}`msBiasOk` dex in median above
[Fe/H]<sub>APOGEE</sub> = −1.5 and below −2, but is {glue:text}`msBiasLo`–{glue:text}`msBiasHi` dex too low at
−2 < [Fe/H]<sub>APOGEE</sub> < −1.5 ({fig}`paper_boss_response`). Stars of this range are therefore measured below −2: in the parent
sample common to both surveys, BOSS finds {glue:text}`msDirectRatioOne` times as many stars below −2 as APOGEE
({fig}`paper_mass_method`b, grey). A model of independent Gaussian errors does not describe BOSSNet, whose estimates are drawn towards
the bulk of its training set, so we measure the response directly. The response matrix $R_{jk}$ is the fraction of stars with APOGEE
[Fe/H] in bin $k$ that BOSS places in bin $j$, built from the stars observed by both and weighted to the joint distribution of $G$
(edges {glue:text}`msGEdges`) and BOSS signal-to-noise ratio S/N (edge {glue:text}`msSnrEdge`) of the sample being corrected,
separately above and below [Fe/H] = {glue:text}`msRespSplit`: the scatter of BOSS [Fe/H] grows as S/N falls, and the stars observed
by both are brighter and have higher S/N than the BOSS sample (median S/N {glue:text}`msSnrBoth` against {glue:text}`msSnrFaint`
for the BOSS giants at $G$ = {glue:text}`msSnrGLo`–{glue:text}`msSnrGHi`). With $\eta_k$ the efficiency and $M_k$ the
formed mass of true bin $k$, the expected BOSS counts in measured bin $j$ are then

$$
\langle N_j\rangle=\sum_k R_{jk}\,\eta_k\,M_k .
$$

Passed through $R$, the APOGEE counts of the common parent reproduce the BOSS counts ($\chi^2$ = {glue:text}`msRespChi` for
{glue:text}`msRespDof` bins with the bootstrap covariance); below [Fe/H] = −2 BOSS holds {glue:text}`msRespTwo`
(+{glue:text}`msRespTwoUp`/−{glue:text}`msRespTwoDn`) of the prediction, against {glue:text}`msDirectRatio` without the response
({fig}`paper_mass_method`b, blue).

We fit the masses $M_k\geq0$ of this equation to the BOSS counts in {glue:text}`msMeasW`-dex measured bins by maximising a Poisson
likelihood, in true bins below −2, from −2 to −1.5, from −1.5 to −1 and above −1. Narrower true bins oscillate: the response rests on
{glue:text}`msNOverlap` stars, of which {glue:text}`msNPoor` lie below −1.5. The fit changes the mass of these four ranges by factors
of {glue:text}`msFacOne`, {glue:text}`msFacTwo`, {glue:text}`msFacThree` and {glue:text}`msFacFour`. Within each range below −1, the
mass is shared among its {glue:text}`msBinW`-dex bins as the uncorrected masses are.

Above −1 the BOSS values cannot share out the mass. BOSSNet overestimates the metallicity of hot giants: on the stars observed by
both surveys with APOGEE [Fe/H] between −1 and −0.5, BOSS reads +{glue:text}`msOffWarmLo`–{glue:text}`msOffWarmHi` dex high at
effective temperature $T_{\rm eff}$ = {glue:text}`msTeffA`–{glue:text}`msTeffB` K and +{glue:text}`msOffHotLo` to
+{glue:text}`msOffHotHi` dex at {glue:text}`msTeffB`–{glue:text}`msTeffC` K, against at most {glue:text}`msOffCool` dex below
{glue:text}`msTeffA` K (chapter 3). Hot, low-S/N giants (above {glue:text}`msTeffA` K and below S/N = {glue:text}`msSnrSplit`) carry
{glue:text}`msHotWeight` per cent of the BOSS weight but are {glue:text}`msHotOverlap` per cent of the stars that define the calibration
and the response, so neither corrects them, and BOSS places {glue:text}`msHotAbove` per cent of them above [Fe/H] = −0.5. We therefore
keep the BOSS total above −1, which agrees with APOGEE in the field cores ({fig}`paper_mass_cores`), and share it among the bins above
−1 as the APOGEE giants of the same cell: each BOSS giant above −1 is distributed as the weighted [Fe/H] of the APOGEE giants above −1
with the same ring ({glue:text}`msSplitRings`°), $G$ (edges {glue:text}`msSplitG`) and dereddened $G_{\rm BP}-G_{\rm RP}$ (edges
{glue:text}`msSplitC`), or of the next coarser cell with at least {glue:text}`msNMinSplit` APOGEE giants, and converted with the
efficiency of its own region and bin.

<!-- script: boss_response.py -->

**Figure (paper_boss_response).** The BOSS metallicity response. (a) Calibrated BOSS minus APOGEE [Fe/H] (dex) against APOGEE [Fe/H]
for the {glue:text}`msNOverlap` stars of the mass samples observed by both surveys (grey points), with the median per
{glue:text}`msBinW`-dex bin of APOGEE [Fe/H] (blue); dashed: zero. (b) Corrected BOSS counts in the common parent over the APOGEE
counts passed through the response, against [Fe/H], separately for the APOGEE fields LMC1–17 (orange; targeted with a Washington +
DDO51 giant flag) and LMC18–35 (green); bars: 16th–84th percentiles of {glue:text}`msNBootResp` joint bootstraps of sky pixels and of
the stars observed by both; dashed: a ratio of 1. `scripts/boss_response.py`

### The formed-mass computation

`scripts/build_giant_mass.py` carries out the steps above for the whole BOSS sample. It measures the [α/Fe] trend, sets up the
conversion regions with their extinction and disc offset, computes the foreground fraction of each giant, counts the isochrone stars
inside the selection for every region and model alternative, and fits the masses $M_k$ through the response. It also draws the
bootstrap resamples and evaluates the model alternatives described under Uncertainties, and it computes the masses of BOSS and APOGEE
in the field cores used in the checks below. Its tables in `results/` feed every mass figure of this page and the mass per
metallicity of Letter Fig. 1b.

<!-- script: build_giant_mass.py show: the selection-function catalogues of bundle 1, APOGEE DR17 files and PARSEC grids; 162 s -->

## Present-day mass

The mass formed includes stars that have died. The present-day mass, in living stars and remnants (white dwarfs, neutron stars and
black holes), follows from the same isochrones and age priors: living stars from the current masses along the isochrone, completed
with unevolved stars between {glue:text}`msImfLo` M☉ and the lowest tabulated mass; remnants from the initial–final mass relations of
[Conroy, Gunn & White (2009)](https://doi.org/10.1088/0004-637X/699/1/486) (section 2), after
[Renzini & Ciotti (1993)](https://doi.org/10.1086/187068), with all remnants retained and returned gas excluded. The present-to-formed
ratio is {glue:text}`msPresAll` overall and {glue:text}`msPresPoor` below [Fe/H] = −1.5.

### The present-mass computation

`scripts/build_present_mass.py` computes, for every isochrone, the fractions of the formed mass now in living stars and in
remnants, averages them over age with the weights of each conversion, and multiplies the formed masses of
`scripts/build_giant_mass.py` by them. It writes `results/giant_present_mass.csv`, the table of formed and present mass per [Fe/H]
bin that {fig}`paper_mass`, {tab}`mass` and Letter Fig. 1b read.

<!-- script: build_present_mass.py show: the tables of scripts/build_giant_mass.py and the PARSEC grid; 2 s -->

## Uncertainties

We give two kinds of uncertainty. Statistical errors are the 16th–84th percentiles of {glue:text}`msNBoot` bootstrap resamples of
HEALPix ([Górski et al. 2005](https://doi.org/10.1086/427976)) sky pixels at resolution parameter nside = {glue:text}`msNside` of the parent, with the selection weights and coverage factors recomputed for each resample, and of the stars that
define the response; the conversions are fixed. Model ranges span the age weightings (reference, A, B and C); [α/Fe] from the
measured trend or fixed at {glue:text}`msAlphas` (capped at {glue:text}`msAlphaCap` above [Fe/H] = −1); zero, the reference and
twice the foreground; the systematic error of ±{glue:text}`msDEvi` mag in $E(V-I)$ of Skowron et al. (2021), which changes the total
by ±{glue:text}`msEChange` per cent; and the inclination error of ±{glue:text}`msDInc`°, which changes it by less than
{glue:text}`msIncChange` per cent.

## Result

{fig}`paper_mass` and {tab}`mass_summary` give the result ({tab}`mass` lists every bin). Within {glue:text}`msRad`° the LMC formed
({glue:text}`msTot` +{glue:text}`msTotUp`/−{glue:text}`msTotDn`) × 10⁹ M☉ (model range {glue:text}`msTotMLo`–{glue:text}`msTotMHi`
× 10⁹) and holds {glue:text}`msPres` × 10⁹ M☉ today. Of the formed mass, {glue:text}`msBelowTwo`
(+{glue:text}`msBelowTwoUp`/−{glue:text}`msBelowTwoDn`) × 10⁷ M☉ has [Fe/H] < −2 and {glue:text}`msBelowOneFive`
(+{glue:text}`msBelowOneFiveUp`/−{glue:text}`msBelowOneFiveDn`) × 10⁷ M☉ has [Fe/H] < −1.5: {glue:text}`msPctTwo` and
{glue:text}`msPctOneFive` per cent of the total; {glue:text}`msFracRich` per cent (bootstrap
{glue:text}`msFracRichLo`–{glue:text}`msFracRichHi`, model range {glue:text}`msFracRichMLo`–{glue:text}`msFracRichMHi` per cent) has
[Fe/H] > −0.5.
Without the response correction the masses below −2 and −1.5 would be {glue:text}`msNoRespTwo` × 10⁷ and
{glue:text}`msNoRespOneFive` × 10⁷ M☉. The mass rises on average by a factor of {glue:text}`msRise` per {glue:text}`msBinW` dex from
[Fe/H] = −2 to the peak in the bin {glue:text}`msPeakBin`. APOGEE checks this in {glue:text}`msNCore` non-overlapping field cores of
radius {glue:text}`msRCore`° where both surveys are at least {glue:text}`msCoreCov` per cent complete: with their own
colour–magnitude selections and conversions, the integrated BOSS/APOGEE mass ratio is {glue:text}`msCoreRatio`
(+{glue:text}`msCoreRatioUp`/−{glue:text}`msCoreRatioDn`) ({fig}`paper_mass_cores`).

{tab}`mass_summary` and {fig}`paper_mass_summary` compare the totals with published LMC masses, each as published: they differ in
initial mass function, aperture and method, and most give either the formed or the present mass. The formed masses from
colour–magnitude star-formation histories, {glue:text}`msLitHZ` × 10⁹ M☉ ([Harris & Zaritsky 2009](https://doi.org/10.1088/0004-6256/138/5/1243))
and {glue:text}`msLitMazzi` × 10⁹ M☉ ([Mazzi et al. 2021](https://doi.org/10.1093/mnras/stab2399)), lie within our bootstrap range;
the published present masses, {glue:text}`msLitPresLo`–{glue:text}`msLitPresHi` × 10⁹ M☉
([Kim et al. 1998](https://doi.org/10.1086/306030); [McConnachie 2012](https://doi.org/10.1088/0004-6256/144/1/4);
[Eskew, Zaritsky & Meidt 2012](https://doi.org/10.1088/0004-6256/143/6/139); [Skibba et al. 2012](https://doi.org/10.1088/0004-637X/761/1/42)),
lie within our bootstrap range ({glue:text}`msPresBootLo`–{glue:text}`msPresBootHi` × 10⁹ M☉) too, except
{glue:text}`msLitVdM` × 10⁹ M☉ from integrated light ([van der Marel et al. 2002](https://doi.org/10.1086/343775)), and all are at or
above our reference value; the stellar mass of Kim et al. (1998) refers to the inner 4 kpc and is quoted by
[Keim et al. (2023)](https://doi.org/10.1093/mnras/stad554).

<!-- script: paper_mass.py -->

**Figure (paper_mass).** Stellar mass of the LMC within {glue:text}`msRad`° per metallicity, from the BOSS giants. (a) Mass per [Fe/H]
bin (M☉, logarithmic) against [Fe/H]; the first bin holds all stars below −2, the last all stars above 0. (b) Mass below each [Fe/H]
(M☉); 'All' is the total. Blue: formed (initial) mass; orange: present mass in living stars and remnants. Points: reference model
(age–metallicity relation, measured [α/Fe], reference foreground); bars: 16th–84th percentiles of the bootstrap; shading: range of the
model alternatives. The cumulative axis orders mass by metallicity, not by time. (c) Distributions per dex of [Fe/H] (fraction of
the total per dex, logarithmic): our formed mass, binned as measured (blue steps; the two open end bins are drawn closed; band: bootstrap); a Gaussian kernel
density (width {glue:text}`msKdeXp` dex) of the *Gaia* XP metallicities
([Andrae, Rix & Chandra 2023](https://doi.org/10.3847/1538-4365/acd53e)) of the whole BOSS parent, converted to [Fe/H] with the [α/Fe]
trend and weighted by the inverse efficiency of each star's bin, i.e. as mass (purple dashed; its bootstrap band is narrower than the
line); and a kernel density (width {glue:text}`msKdeCl` dex) of the LMC clusters with [Fe/H] from the Ca II triplet
([Grocholski et al. 2006](https://doi.org/10.1086/507303)), by number (black; grey: bootstrap over clusters). `scripts/paper_mass.py`

<!-- script: paper_mass_summary.py -->

**Figure (paper_mass_summary).** Total stellar mass of the LMC ($10^9$ M☉, logarithmic): this work (filled symbols, top row; thin
bars: 16th–84th bootstrap percentiles; thick bars and vertical shading: range of the model alternatives) and published estimates as
published (open symbols, one row per study in order of publication). Blue triangles: formed mass; orange squares: present mass in
living stars and remnants. `scripts/paper_mass_summary.py`

```{code-cell} python
:tags: [remove-input]
import numpy as np
import pandas as pd
from lmcold.constants import LMC_MASS_LITERATURE
# Formed and present stellar masses per [Fe/H] bin, cumulative below each edge
t = pd.read_csv('results/giant_present_mass.csv')


def cell(r, key):
    """Reference value (16th–84th bootstrap percentiles) [model range], in units of 10^n Msun."""
    v, lo, hi, mlo, mhi = (r[key + s] for s in ('', '_p16', '_p84', '_model_min', '_model_max'))
    n = int(np.floor(np.log10(v)))
    return f'{v / 10**n:.2f} ({lo / 10**n:.2f}–{hi / 10**n:.2f}) [{mlo / 10**n:.2f}–{mhi / 10**n:.2f}] × 10^{n}'


# This work: all stars (last row of t) and below [Fe/H] = -1, -1.5, -2; then the literature values of lmcold.constants
rows = [('This work, all', cell(t.iloc[-1], 'cumulative'), cell(t.iloc[-1], 'present_cumulative'), 'ρ < 12°, BOSS giants, PARSEC', 'Kroupa')]
rows += [(f'This work, [Fe/H] < {e:g}', cell(t[t.feh_hi == e].iloc[0], 'cumulative'), cell(t[t.feh_hi == e].iloc[0], 'present_cumulative'), '', '')
         for e in (-1, -1.5, -2)]
rows += [(r['label'], f"{r['mass'] / 1e9:.2f} × 10^9" if r['kind'] == 'formed' else '–',
          f"{r['mass'] / 1e9:.2f} × 10^9" if r['kind'] == 'present' else '–', f"{r['coverage']}; {r['method']}", r['imf'])
         for r in LMC_MASS_LITERATURE]
pd.DataFrame(rows, columns=['', 'Formed mass (M☉)', 'Present mass (M☉)', 'Aperture, method', 'IMF']).set_index('')
```

**Table (mass_summary).** Stellar mass of the LMC. This work (`results/giant_present_mass.csv`): reference value with the 16th–84th
bootstrap percentiles in parentheses and the range of the model alternatives in brackets, within {glue:text}`msRad`°; rows below the
total give the mass below each [Fe/H]. Published estimates (`lmcold.constants.LMC_MASS_LITERATURE`) are given as published, formed or
present, with their aperture, method and initial mass function.

The comparison panel of {fig}`paper_mass` sets the distribution against two independent ones. The *Gaia* XP metallicities (estimated
from the low-resolution BP/RP spectra of *Gaia*) of the {glue:text}`msNXp` parent stars, converted to mass in the same way, give
{glue:text}`msXpOneFive` per cent of the mass below [Fe/H] = −1.5 (ours {glue:text}`msPctOneFive`), {glue:text}`msXpTwo` per cent below
−2 ({glue:text}`msPctTwo`) and {glue:text}`msXpRich` per cent above −0.5 ({glue:text}`msFracRich`), with the peak one bin lower, at
{glue:text}`msXpPeakBin`: XP metallicities of metal-rich giants run low against APOGEE at faint magnitudes, by {glue:text}`msXpOffLo`–{glue:text}`msXpOffHi`
dex for the stars observed by both at $G$ ≥ {glue:text}`msXpG` with APOGEE [Fe/H] from {glue:text}`msXpRichLo` to
{glue:text}`msXpRichHi`. The {glue:text}`msNCl` clusters of Grocholski et al. (2006) are more
metal-rich, with a median of {glue:text}`msClMed` and {glue:text}`msClRich` per cent above −0.5: the populous clusters with
spectroscopy are mostly younger than 3 Gyr, and the LMC formed almost no clusters between 4 and 11 Gyr ago
([Piatti 2025](https://doi.org/10.1093/mnras/staf1242)), when much of its field mass formed. The metallicity-structure chapter compares
our giants, by number, with the published field-star distributions in the same places: the Ca II triplet distributions of the bar
([Cole et al. 2005](https://doi.org/10.1086/428007)) and of fields 5–8° north of it ([Carrera et al. 2008](https://doi.org/10.1088/0004-6256/135/3/836))
lie {glue:text}`msMdfOffLo`–{glue:text}`msMdfOffHi` dex above BOSS, less than the {glue:text}`msMdfScale` dex by which the Ca II
triplet scale exceeds ours for the stars in common ({glue:text}`msNCarMatch` with Carrera et al. 2008, {glue:text}`msNColeMatch` with
Cole et al. 2005), and the *Gaia* XP Strömgren metallicities of [Omkumar et al. (2025)](https://doi.org/10.1051/0004-6361/202452510)
have the BOSS median but a much broader distribution.

## Checks of the mass

```{code-cell} python
:tags: [remove-input]
import pandas as pd
# Formed and present stellar masses per [Fe/H] bin, cumulative below each edge
t = pd.read_csv('results/giant_present_mass.csv')


def cell(r, key, pre=None):
    """Value r[key] (16th–84th bootstrap percentiles) [model range] in 10^6 Msun; the percentile and range columns start with pre."""
    v = r[key] / 1e6
    lo, hi, mlo, mhi = (r[(pre or key) + s] / 1e6 for s in ('_p16', '_p84', '_model_min', '_model_max'))
    f = lambda x: f'{x:.3g}' if x < 1000 else f'{x:.0f}'  # noqa: E731, three significant figures
    return f'{f(v)} ({f(lo)}–{f(hi)}) [{f(mlo)}–{f(mhi)}]'


# Row labels: the [Fe/H] bins; columns: mass in each bin and below its upper edge, formed and present
lab = ['< −2'] + [f'{lo:g} to {hi:g}'.replace('-', '−') for lo, hi in zip(t.feh_lo[1:-1], t.feh_hi[1:-1])] + ['≥ 0']
pd.DataFrame({'Formed': [cell(r, 'formed_mass', 'mass') if r.formed_mass > 0 else '0' for _, r in t.iterrows()],
              'Present': [cell(r, 'present_mass') if r.present_mass > 0 else '0' for _, r in t.iterrows()],
              'Below': [f'{x:g}' for x in t.feh_hi[:-1]] + ['all'],
              'Formed below': [cell(r, 'cumulative') for _, r in t.iterrows()],
              'Present below': [cell(r, 'present_cumulative') for _, r in t.iterrows()]}, index=lab)
```

**Table (mass).** Formed and present stellar mass of the LMC within {glue:text}`msRad`°, in $10^6$ M☉, per [Fe/H] bin and below the
upper edge of each bin ('Below'). Values: reference model with the 16th–84th bootstrap percentiles in parentheses; brackets: range of
the model alternatives. Above [Fe/H] = −1 the BOSS mass is shared among the bins as the APOGEE giants of the same cell; no APOGEE giant
of these cells lies above 0 (`results/giant_present_mass.csv`, written by `scripts/build_giant_mass.py` and
`scripts/build_present_mass.py`).

{tab}`mass` lists the formed and present mass of every [Fe/H] bin, with bootstrap percentiles and model ranges.

<!-- script: paper_mass_method.py -->

**Figure (paper_mass_method).** From giant counts to mass. (a) BOSS giants per $10^6$ M☉ formed, $\eta$ (logarithmic), against age
(Gyr, logarithmic) for a single-age population at four model metallicities (legend; orange, green, blue and purple from metal-poor to
metal-rich), one in each [Fe/H] range of {tab}`ages`. Each curve takes the PARSEC v1.2S isochrone with COLIBRI TP-AGB tracks, from the
main sequence to the end of the AGB; places it at distance modulus {glue:text}`msDM` with a Gaussian depth of {glue:text}`msDepth`
mag; reddens it by the median true $E(B-V)$ = {glue:text}`msEbvTrue` of the parent and de-reddens it by the median
{glue:text}`msEbvCarton` that the carton removed; and counts the stars inside the carton's colour–magnitude polygon with observed
$G\leq$ {glue:text}`msGLim`, integrating the Kroupa (2001) IMF, normalised to 1 M☉ formed between {glue:text}`msImfLo` and
{glue:text}`msImfHi` M☉, along the isochrone in {glue:text}`msNSub` steps between tabulated points. The curves have no disc-distance
offset; the masses use the regional values. Horizontal bars (placed at arbitrary heights, in the colour of their curve) show the ages
over which $\eta$ is averaged for the {glue:text}`msBinW`-dex [Fe/H] bin that contains each curve's metallicity. Thick bars: the
reference weighting, i.e. the distribution of formed mass in age expected for that bin from the LMC age–metallicity relation of
Povick et al. (2024) with a constant star-formation rate; the bar spans its 16th–84th percentiles. Thin bars: the fixed window A of
{tab}`ages`, an alternative. (b) The BOSS metallicity response, tested on the parent common to BOSS and APOGEE: corrected BOSS counts
divided by corrected APOGEE counts (grey) and by the APOGEE counts passed through the response measured on the stars observed by both
(blue), against [Fe/H]; dashed: a ratio of 1. Bars: 16th–84th percentiles of {glue:text}`msNBootResp` joint bootstraps of sky pixels
and of the stars observed by both. Above [Fe/H] = −1 the masses do not use this response: the BOSS mass there is shared as the APOGEE
giants of the same cell. `scripts/paper_mass_method.py`

**The BOSS response.** {fig}`paper_boss_response`(a) shows the stars observed by both surveys: the calibrated BOSS [Fe/H] is low by
{glue:text}`msBiasLo`–{glue:text}`msBiasHi` dex only between [Fe/H]<sub>APOGEE</sub> = −2 and −1.5. {fig}`paper_boss_response`(b)
repeats the closure test of {fig}`paper_mass_method`(b) by APOGEE field type. In LMC18–35, BOSS holds {glue:text}`msRespLate`
(+{glue:text}`msRespLateUp`/−{glue:text}`msRespLateDn`) of the stars predicted below [Fe/H] = −1.5; in LMC1–17 it holds
{glue:text}`msRespEarly` (+{glue:text}`msRespEarlyUp`/−{glue:text}`msRespEarlyDn`). APOGEE thus has more metal-poor stars than BOSS
where it used its unpublished giant flag from Washington and DDO51 photometry, as the test of the APOGEE selection against *Gaia* XP
metallicities in chapter 2 also suggests.

**BOSS against APOGEE.** {fig}`paper_mass_cores` compares the masses of the two surveys in {glue:text}`msNCore` non-overlapping
field cores of radius {glue:text}`msRCore`° ({glue:text}`msCoreArea` deg² in all), where APOGEE covers at least
{glue:text}`msCoreCov` per cent of its candidates to $H$ = {glue:text}`msHLim` and BOSS at least {glue:text}`msCoreCov` per cent of
its parent. Each survey uses its own colour–magnitude selection and the extinction and disc offset of each core. The integrated ratio
is {glue:text}`msCoreRatio` (+{glue:text}`msCoreRatioUp`/−{glue:text}`msCoreRatioDn`). The bins differ more: below [Fe/H] = −2 BOSS is
{glue:text}`msCoreTwo` (+{glue:text}`msCoreTwoUp`/−{glue:text}`msCoreTwoDn`) times APOGEE, from {glue:text}`msCoreTwoN` APOGEE stars,
and at {glue:text}`msCoreMidBin` it is {glue:text}`msCoreMid` (+{glue:text}`msCoreMidUp`/−{glue:text}`msCoreMidDn`), where
{glue:text}`msNCoreWD` of the {glue:text}`msNCore` cores are LMC1–17 fields, targeted with the Washington + DDO51 giant flag. Above −1
the total agrees, {glue:text}`msCoreRich`.

**Figure (paper_mass_cores).** BOSS and APOGEE in the same {glue:text}`msNCore` field cores. (a) Formed mass per [Fe/H] bin (M☉,
logarithmic) from each survey, BOSS (blue) and APOGEE (orange), each with its own colour–magnitude selection and conversion; bands:
range over the age and [α/Fe] alternatives. (b) BOSS over APOGEE (logarithmic); bars: 16th–84th percentiles of a paired bootstrap of
the cores; grey band: range of the alternatives; dashed: a ratio of 1. Bins need at least {glue:text}`msCoreMinN` stars from each survey. Above
[Fe/H] = −1 the BOSS mass is shared among the bins as the APOGEE giants of the same cell, so there only the total is an independent
test. `scripts/paper_mass.py`

**Dust.** The conversion assumes that the *Gaia* parent loses stars to extinction only as the isochrones, reddened with the median
$E(V-I)$ of each region, predict. {fig}`paper_parent_extinction` tests this within {glue:text}`msRMax`° of the centre, where the
red-clump map varies on scales of a few arcmin. In {glue:text}`msCell`-arcmin cells (HEALPix pixels) we count the parent stars,
$N_G$, and the 2MASS giants inside the APOGEE target polygon with {glue:text}`msHLo` ≤ $H$ ≤ {glue:text}`msHLim`, $N_H$, which
extinction affects much less ($A_H/A_G$ = {glue:text}`msAhAg`, where $A_H$ and $A_G$ are the extinctions in $H$ and $G$). The
isochrones predict the ratio $f=N_G/N_H$ as a function of reddening for a population of ages {glue:text}`msAgeLo`–{glue:text}`msAgeHi`
Gyr and [M/H] = {glue:text}`msMhLo` to {glue:text}`msMhHi`; the observed $\sum N_G/\sum (fN_H)$ is normalised to the cells with
$E(V-I)$ < {glue:text}`msLow`. With $f$ at the regional median, as in the conversion, the parent holds {glue:text}`msQA` of the
predicted stars at $E(V-I)$ = {glue:text}`msQALo`–{glue:text}`msQAHi`, {glue:text}`msQB` at {glue:text}`msQBLo`–{glue:text}`msQBHi`,
{glue:text}`msQC` ± {glue:text}`msQCErr` at {glue:text}`msQCLo`–{glue:text}`msQCHi` and {glue:text}`msQD` ± {glue:text}`msQDErr` at
{glue:text}`msQDLo`–{glue:text}`msQDHi`. Cells above $E(V-I)$ = {glue:text}`msQBLo` hold {glue:text}`msDustyShare` per cent of the
central parent, and the missing stars amount to {glue:text}`msMissing` per cent of it, a deficit well inside the
±{glue:text}`msDEvi` mag extinction term of the model range. With $f$ at the $E(V-I)$ of each cell the prediction overshoots the loss by
up to {glue:text}`msOvershoot` per cent: giants in front of the dust are barely dimmed, so the mean red-clump reddening of a dusty
sightline overstates the extinction of its giants. We therefore keep the regional medians. The test assumes that 2MASS is equally
complete in all cells and that the population does not change with reddening; crowding in the bar would hide part of the loss.

<!-- script: paper_parent_extinction.py -->

**Figure (paper_parent_extinction).** The *Gaia* parent in the dusty centre of the LMC. (a) Red-clump $E(V-I)$ (mag, colour bar above
the panel) of Skowron et al. (2021) in {glue:text}`msCell`-arcmin cells within {glue:text}`msRMax`° of the centre, against $x$ (east,
increasing to the left) and $y$ (north), in degrees. (b) Parent stars over 2MASS giants ({glue:text}`msHLo` ≤ $H$ ≤
{glue:text}`msHLim`), divided by the isochrone prediction $f$ and normalised to cells with $E(V-I)$ < {glue:text}`msLow`, in bins of
the cell's $E(V-I)$ (mag): $f$ at the reddening of each cell (blue) and at the median of its conversion region, as used for the
masses (orange); dashed: a ratio of 1. Bars: 16th–84th percentiles of a bootstrap over cells; bins with at least {glue:text}`msPeMinN` parent stars.
`scripts/paper_parent_extinction.py`

**Foreground.** {fig}`fig_giant_foreground` shows the foreground fractions of the corrected counts and the test of their assumption.
The carton's proper-motion window keeps only foreground stars with LMC-like motions, so the proper-motion sideband tests the radial
shape of the foreground, not its amount. At {glue:text}`msFgRingLo`–{glue:text}`msFgRingHi`° the model assigns
{glue:text}`msFgOutLo`–{glue:text}`msFgOutHi` per cent of the giants to the foreground, an extrapolation that the zero and double
amplitudes of the model range bracket.

<!-- script: fig_giant_foreground.py -->

**Figure (fig_giant_foreground).** Milky Way foreground of the BOSS giants. (a) Model foreground fraction (logarithmic) per ring of {glue:text}`msFgRingW`°
in angular distance $\rho$ (deg), for three [Fe/H] ranges: below −1.5 (orange), −1.5 to −1 (green) and above −1 (blue). (b) Surface
density (giants per deg²) of the giants with [Fe/H] < −1.5 whose proper motion lies more than {glue:text}`msSide` mas/yr from the
local LMC motion, in rings of $\rho$ (horizontal bars: ring width; vertical bars: Poisson errors); the dashed line is the weighted
mean. `scripts/fig_giant_foreground.py`
