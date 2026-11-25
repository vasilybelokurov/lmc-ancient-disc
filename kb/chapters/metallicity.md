---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Metallicity structure

Every number on this page is written by `scripts/paper_text_feh.py` from the result tables (`results/macros_feh.tex`); the code
cells are the figure scripts themselves (`scripts/paper_feh_profile.py`, `paper_feh_literature.py`, `paper_mdf_literature.py`), step
by step.

Metallicity here is the iron abundance [Fe/H], the logarithm of the ratio of iron to hydrogen atoms relative to the Sun's. Three
tracers measure it: red giants with spectra from [APOGEE DR17](https://doi.org/10.3847/1538-4365/ac4414) (near-infrared spectra
analysed by the ASPCAP pipeline) and from the BOSS optical spectra of [SDSS-V DR20](https://doi.org/10.48550/arXiv.2607.26149)
(analysed by the neural network BOSSNet, [Sizemore et al. 2024](https://doi.org/10.3847/1538-3881/ad291d), and calibrated to the
APOGEE scale; chapter 3), and RR Lyrae stars of type ab (RRab, pulsating old stars) from the
[OGLE-IV](https://ui.adsabs.harvard.edu/abs/2015AcA....65....1U) and [*Gaia* DR3](https://doi.org/10.1051/0004-6361/202243940) surveys,
whose [Fe/H] is photometric, estimated from the shape of their light curves on the scale of
[Crestani et al. (2021)](https://doi.org/10.3847/1538-4357/abd183) (chapter 3). $G$ is the *Gaia* magnitude.

With the selection functions of chapter 2 the giant samples give the metallicity distribution of the LMC at each radius, not that of
the fields the surveys chose to observe. We place every star on the disc plane of
[van der Marel & Kallivayalil (2014)](https://doi.org/10.1088/0004-637X/781/2/121) (inclination {glue:text}`feInc`°, line of nodes at
position angle {glue:text}`feTheta`°, distance of the centre $D_0$ = {glue:text}`feDZero` kpc; with the inclination of our joint
kinematic fit, {glue:text}`feIncKin`° (chapter 6), radii on the minor axis would be {glue:text}`feMinorPct` per cent smaller and those
on the line of nodes unchanged) and use its in-plane distance $R$ from the centre, in {glue:text}`feRBin`-kpc bins to
$R$ = {glue:text}`feRMax` kpc, the largest radius at which the {glue:text}`feRad`° BOSS footprint covers all azimuths. Each BOSS giant
carries its selection weight $w_i$ (the number of stars of the BOSS target list that it stands for), its coverage factor $c_i$ (which
adds the target-list stars in sky areas with too few spectra) and one minus its Milky Way foreground fraction $f_{\rm MW}$ (all from
chapters 2 and 4); each APOGEE giant carries its selection weight. The RR Lyrae are unweighted, since the union of the two surveys is
at least {glue:text}`feUnionMin` per cent complete at every radius (chapter 2). In each bin we take the weighted median of the measured [Fe/H]. For the BOSS giants we also
count the stars in three ranges of true metallicity, below {glue:text}`feTrueA`, from {glue:text}`feTrueA` to {glue:text}`feTrueB` and
above {glue:text}`feTrueB`, and divide by the area of the ring on the disc plane. These counts must be corrected for the BOSS
metallicity errors, which move stars from −2 < [Fe/H] < −1.5 to below −2 (chapter 4). The correction applies to counts, not to single
stars: in each radial bin we rebuild the response matrix (the fraction of stars of each true [Fe/H] bin that BOSS places in each
measured bin) for the $G$ and signal-to-noise distribution of the bin's giants and fit the response equation of chapter 4, with the
efficiencies set to 1, to their weighted counts in {glue:text}`feMeasW`-dex measured bins, which returns the counts in the true bins.
The medians are of the measured values; above [Fe/H] = −1.5 calibrated BOSS agrees with APOGEE to within {glue:text}`feBiasRich` dex
in median (chapter 4). Errors are the 16th–84th percentiles of {glue:text}`feNBoot` bootstrap draws of HEALPix
([Górski et al. 2005](https://doi.org/10.1086/427976)) sky pixels at resolution nside = {glue:text}`feNside` ({glue:text}`fePix`°), with
the weights fixed and the {glue:text}`feNOverlap` stars that define the response resampled in each draw ({glue:text}`feNSkip` of the
{glue:text}`feNFits` bin fits, where a resampled response had an empty column, are dropped); the coarser pixels of the mass bootstrap
would leave one or two pixels in the inner bins.

<!-- script: paper_feh_profile.py -->

**Figure (paper_feh_profile).** Metallicity and surface density against in-plane radius $R$ (kpc). (a) Weighted median [Fe/H] of
the BOSS giants ($G\leq$ {glue:text}`feGLim`, blue filled; $G$ < {glue:text}`feGBright`, blue open, dashed) and the APOGEE giants
(orange); bars: 16th–84th percentiles of the bootstrap of the median; dashed and dotted lines: published gradients over their radial
ranges, drawn through the BOSS profile at the middle of each range (slopes only); [Choudhury et al. (2016)](https://doi.org/10.1093/mnras/stv2414)
is the middle of their quoted interval. (b) The same for the RRab (green), with the three radial segments of
[Skowron et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016AcA....66..269S) (black dashed). (c) Surface density of the BOSS giants on
the disc plane (number per kpc², logarithmic) in three ranges of true [Fe/H] (filled circles: below {glue:text}`feTrueA`, dark blue;
{glue:text}`feTrueA` to {glue:text}`feTrueB`, teal; above {glue:text}`feTrueB`, yellow), corrected for the BOSS metallicity errors;
dashed: exponentials fitted at {glue:text}`feFitLo` < $R$ < {glue:text}`feFitHi` kpc, with their scale lengths $h$ in the legend; open
squares: RRab members below and above [Fe/H] = −1.5 (photometric [Fe/H]), Poisson errors, with exponentials fitted at
{glue:text}`feRrFitLo` < $R$ < {glue:text}`feRrRMax` kpc (dotted). Bars: 16th–84th percentiles of {glue:text}`feNBoot` bootstraps of
nside = {glue:text}`feNside` sky pixels and, in (c), of the stars that define the metallicity response. `scripts/paper_feh_profile.py`

{fig}`paper_feh_profile` shows the result. The median [Fe/H] of the BOSS giants falls from {glue:text}`feBossIn` at
$R$ < {glue:text}`feRBin` kpc to {glue:text}`feBossMid` at {glue:text}`feBossMidLo`–{glue:text}`feBreak` kpc and then to
{glue:text}`feBossOut` at {glue:text}`feOutLo`–{glue:text}`feRMax` kpc. The APOGEE giants follow the same shape but lie
{glue:text}`feApoOffLo`–{glue:text}`feApoOffHi` dex lower in every bin. The two samples reach different depths on the giant branch:
BOSS goes to $G$ = {glue:text}`feGLim`, while {glue:text}`feApoBright` per cent of the APOGEE weight in its fields is at $G$ < {glue:text}`feGBright`, because APOGEE's faint
selection groups are mostly uncovered and its weights have no coverage factor. Limited to $G$ < {glue:text}`feGBright`, the BOSS median
falls by {glue:text}`feBrightDropLo`–{glue:text}`feBrightDropHi` dex inside {glue:text}`feBreak` kpc and agrees with APOGEE to within
{glue:text}`feBrightAgree` dex out to {glue:text}`feAgree` kpc (open symbols). Part of the excess of the fainter BOSS giants is a
measurement bias: BOSSNet overestimates the metallicity of hot, low-S/N giants, which are common at faint $G$ (chapters 3 and 4), so
the open symbols are the reliable BOSS profile, and we take the gradients from them. Straight lines fitted to the medians of the BOSS
giants with $G$ < {glue:text}`feGBright` give {glue:text}`feSlBrightIn` ± {glue:text}`feSlBrightInErr` dex/kpc inside
{glue:text}`feBreak` kpc and {glue:text}`feSlBrightOut` ± {glue:text}`feSlBrightOutErr` dex/kpc beyond, and the APOGEE giants
{glue:text}`feSlApoIn` ± {glue:text}`feSlApoInErr` and {glue:text}`feSlApoOut` ± {glue:text}`feSlApoOutErr` dex/kpc; all BOSS giants
to $G$ = {glue:text}`feGLim` give {glue:text}`feSlBossIn` ± {glue:text}`feSlBossInErr` and {glue:text}`feSlBossOut` ±
{glue:text}`feSlBossOutErr` dex/kpc. In every sample the profile is nearly flat inside {glue:text}`feBreak` kpc and falls beyond. The
scale lengths below refer to the BOSS giants with $G\leq$ {glue:text}`feGLim`; the range above {glue:text}`feTrueB` is a total, which
BOSS and APOGEE agree on.

{fig}`paper_feh_profile`(c) shows why the median falls: the metal-poor giants are spread more widely than the metal-rich ones. Between
{glue:text}`feFitLo` and {glue:text}`feFitHi` kpc exponentials $\Sigma\propto e^{-R/h}$ fitted to the surface density $\Sigma$ of the
three ranges have scale lengths $h$ = {glue:text}`feHPoor` (+{glue:text}`feHPoorUp`/−{glue:text}`feHPoorDn`) kpc below
{glue:text}`feTrueA`, {glue:text}`feHMid` (+{glue:text}`feHMidUp`/−{glue:text}`feHMidDn`) kpc from {glue:text}`feTrueA` to
{glue:text}`feTrueB` and {glue:text}`feHRich` (+{glue:text}`feHRichUp`/−{glue:text}`feHRichDn`) kpc above {glue:text}`feTrueB`. The
metal-rich profile is not exponential: it falls by factors of {glue:text}`feFallLo`–{glue:text}`feFallHi` per kpc beyond
{glue:text}`feBreak` kpc, against {glue:text}`feFallFit` for the fit, and this steepening produces the break in the median. Inside
{glue:text}`feFitLo` kpc the density of metal-poor giants falls below that at {glue:text}`feFitLo`–{glue:text}`feSigRefHi` kpc ({glue:text}`feSigIn` and
{glue:text}`feSigInTwo` against {glue:text}`feSigRef` per kpc²); the selection function recovers {glue:text}`feRecIn` of the expected
metal-poor stars inside {glue:text}`feRecRho`° (chapter 2), and {glue:text}`feMockIn` ± {glue:text}`feMockInErr` within
{glue:text}`feMockRho`° in the mock, so this dip is at most marginally significant. The foreground matters most for the
metal-poor giants at large radii: set to zero or doubled, it changes $h$ to {glue:text}`feHPoorZero` and {glue:text}`feHPoorDouble`
kpc below {glue:text}`feTrueA`, {glue:text}`feHMidZero` and {glue:text}`feHMidDouble` kpc from {glue:text}`feTrueA` to
{glue:text}`feTrueB` and {glue:text}`feHRichZero` and {glue:text}`feHRichDouble` kpc above {glue:text}`feTrueB`; a threshold of
S/N ≥ {glue:text}`feSnrHi` instead of {glue:text}`feSnrLo` changes them by less than {glue:text}`feSnrDh` kpc (chapter 3), and the
medians by at most {glue:text}`feSnrDm` dex outside {glue:text}`feRBin` kpc.

The RR Lyrae [Fe/H] declines steadily, from {glue:text}`feRrIn` at the centre to {glue:text}`feRrOut` at {glue:text}`feOutLo`–{glue:text}`feRMax` kpc
({glue:text}`feSlRrIn` ± {glue:text}`feSlRrInErr` dex/kpc inside {glue:text}`feBreak` kpc), with a flat stretch between
{glue:text}`feFlatLo` and {glue:text}`feFlatHi` kpc (medians {glue:text}`feFlatMedLo` to {glue:text}`feFlatMedHi`). Their surface
density ({fig}`paper_feh_profile`c, open squares; to {glue:text}`feRrRMax` kpc, the largest radius at which their footprint,
$\rho$ < {glue:text}`feRhoRr`° from the centre, covers all azimuths) follows the metal-poor giants between {glue:text}`feRrFitLo` and
{glue:text}`feRrRMax` kpc, where there are {glue:text}`feRatioMean` RRab below [Fe/H] = −1.5 per BOSS giant below −1.5 in every {glue:text}`feRBin`-kpc
ring ({glue:text}`feRatioLo`–{glue:text}`feRatioHi`, scatter {glue:text}`feRatioSd`), although each sample has its own selection
function and [Fe/H] scale: exponentials have $h$ = {glue:text}`feHRrPoor` kpc for the RRab below [Fe/H] = −1.5 and
{glue:text}`feHRrRich` kpc above, against {glue:text}`feHPoor` and {glue:text}`feHMid` kpc for the giants below −1.5 and from −1.5 to
−1. Inside {glue:text}`feFitLo` kpc they part: the RRab below −1.5 rise to {glue:text}`feRrCentre` per kpc² at $R$ <
{glue:text}`feRBin` kpc while the giants below −1.5 dip to {glue:text}`feSigIn` (bootstrap {glue:text}`feSigInLo`–{glue:text}`feSigInHi`),
so there are {glue:text}`feRatioIn` RRab per metal-poor giant ({glue:text}`feRatioInLo`–{glue:text}`feRatioInHi`), and
{glue:text}`feRatioTwo` ({glue:text}`feRatioTwoLo`–{glue:text}`feRatioTwoHi`) at {glue:text}`feRBin`–{glue:text}`feFitLo` kpc. These
profiles count stars, not mass: a fixed formed mass of the old metal-poor populations places fewer giants in the selection than the
younger metal-rich ones (an efficiency $\eta\approx$ {glue:text}`feEtaPoor` × 10⁻⁴ against
{glue:text}`feEtaRichLo`–{glue:text}`feEtaRichHi` × 10⁻⁴ selected giants per M☉ formed for the bins from −2.0 to −1.5 and from −0.75
to −0.25; chapter 4, {fig}`paper_mass_method`a), so in mass the metal-poor profile would lie higher relative to the metal-rich one,
but its scale length would not change if the conversion does not vary with radius.

## Published gradients

The published gradients are drawn in {fig}`paper_feh_profile`(a, b) over their own radial ranges, through our profile at the middle
of each range, so that only the slopes are compared; {fig}`paper_feh_literature` compares them range by range. The APOGEE gradient of
[Povick et al. (2025)](https://doi.org/10.1093/mnras/staf1428), {glue:text}`feLitPovick` ± {glue:text}`feLitPovickErr` dex/kpc from
medians in 1-kpc bins out to about 10°, agrees with our APOGEE slope over the same range ({glue:text}`feOurPovickApo` ±
{glue:text}`feOurPovickApoErr`); the BOSS giants with $G$ < {glue:text}`feGBright` give a shallower {glue:text}`feOurPovickBright` ±
{glue:text}`feOurPovickBrightErr`. The slope of [Muñoz et al. (2023)](https://doi.org/10.1051/0004-6361/202347046),
{glue:text}`feLitMunoz` dex/deg out to {glue:text}`feLitMunozR`°, agrees with both ({glue:text}`feOurMunozApo` ±
{glue:text}`feOurMunozApoErr` for APOGEE, {glue:text}`feOurMunozBright` ± {glue:text}`feOurMunozBrightErr` for BOSS). The gradients
from photometric metallicities and from the ratio of carbon to M-type stars on the asymptotic giant branch are steeper:
{glue:text}`feLitGrady` ± {glue:text}`feLitGradyErr` ([Grady, Belokurov & Evans 2021](https://doi.org/10.3847/1538-4357/abd4e4)) and
{glue:text}`feLitCioni` ± {glue:text}`feLitCioniErr` dex/kpc ([Cioni 2009](https://doi.org/10.1051/0004-6361/200912138)),
{glue:text}`feSteepLo`–{glue:text}`feSteepHi` times our {glue:text}`feOurSteepLo` to {glue:text}`feOurSteepHi` dex/kpc over the same
ranges, and {glue:text}`feLitChOneLo` to {glue:text}`feLitChOneHi` dex/kpc inside {glue:text}`feLitChOneR` kpc (Choudhury et al.
2016, from the slope of the red giant branch in optical photometry), where ours are {glue:text}`feOurChOneBright` ±
{glue:text}`feOurChOneBrightErr` (BOSS) and {glue:text}`feOurChOneApo` ± {glue:text}`feOurChOneApoErr` (APOGEE). These methods respond
to the age of the population as well as its metallicity, and [Feast et al. (2010)](https://doi.org/10.1111/j.1745-3933.2010.00933.x)
attributed the outer fall of the carbon-star ratio to older stars rather than to metallicity. The near-infrared giant-branch slopes
of [Choudhury et al. (2021)](https://doi.org/10.1093/mnras/stab2446) give {glue:text}`feLitChTwo` ± {glue:text}`feLitChTwoErr`
dex/kpc inside {glue:text}`feLitChTwoR` kpc, the slope of our BOSS giants with $G$ < {glue:text}`feGBright`
({glue:text}`feOurChTwoBright` ± {glue:text}`feOurChTwoBrightErr`) and shallower than that of APOGEE ({glue:text}`feOurChTwoApo` ±
{glue:text}`feOurChTwoApoErr`). The shape of our profile, flat to about {glue:text}`feBreak` kpc and falling beyond, matches the
fields of [Carrera et al. (2008)](https://doi.org/10.1088/0004-6256/135/3/836), with [Fe/H] from the Ca II triplet (the three strong
lines of ionised calcium near 850 nm) of ≈ {glue:text}`feCarIn` out to {glue:text}`feCarInR`° and {glue:text}`feCarOut` at
{glue:text}`feCarOutR`°, and the plateau and drop near 8° in the *Gaia* XP metallicities (from the low-resolution BP/RP spectra of
*Gaia*) of [Massana et al. (2024)](https://doi.org/10.1093/mnras/stad3788); a single linear gradient does not describe it. The RR
Lyrae agree with Skowron et al. (2016) inside {glue:text}`feLitSkR` kpc ({glue:text}`feLitSk` ± {glue:text}`feLitSkErr` against our
{glue:text}`feOurSkIn` ± {glue:text}`feOurSkInErr` dex/kpc); from {glue:text}`feSkMidLo` to {glue:text}`feSkMidHi` kpc they find no
gradient, and our slope there, {glue:text}`feOurSkMid` ± {glue:text}`feOurSkMidErr` dex/kpc, is shallower but not zero.
[Cusano et al. (2021)](https://doi.org/10.1093/mnras/stab901) found no RR Lyrae gradient. From *Gaia* XP metallicities tied to APOGEE,
[Frankel et al. (2025)](https://doi.org/10.3847/1538-4357/ad9b17) found exponential scale lengths that shrink from 2.3 kpc at
[M/H] = −1.8 to 0.75 kpc at [M/H] = −0.25 ([M/H] is the abundance of all metals relative to the Sun's); ours,
{glue:text}`feHPoorOne`–{glue:text}`feHMidOne` kpc below [Fe/H] = −1 and {glue:text}`feHRichOne` kpc above, follow the same trend,
with a metal-rich range that spans −1 to above 0 and so mixes their shorter and longer scales.

<!-- script: paper_feh_literature.py -->

**Figure (paper_feh_literature).** Our metallicity gradients against published ones. (a) Giants: radial gradient d[Fe/H]/d$R$
(dex/kpc) of each published study (black squares, labelled with study and radial range; the bar of Choudhury et al. 2016 spans their
quoted interval and errors) and ours over the same in-plane range from BOSS with $G$ < {glue:text}`feGBright` (blue) and APOGEE
(orange); * marks ranges converted from degrees. (b) The same for the RRab (green) and the three segments of Skowron et al. (2016);
the cross marks their {glue:text}`feSkMidLo`–{glue:text}`feSkMidHi` kpc segment, for which they quote no gradient and no error. (c)
Median [Fe/H] against projected distance $\rho$ (deg) for BOSS with $G$ < {glue:text}`feGBright` (blue) and APOGEE (orange), with the
mean levels quoted by Carrera et al. (2008; black bars) and Massana et al. (2024; grey dashed bars) and the slope of Muñoz et al.
(2023) drawn through the BOSS profile at the middle of its range (purple dashed). `scripts/paper_feh_literature.py`

### How the published gradients are compared

For every study we fit a straight line to our medians in the {glue:text}`feRBin`-kpc bins whose centres lie inside its range, weighted by their
bootstrap errors (floor {glue:text}`feFloor` dex), and quote the formal error of the fit; ranges with fewer than
{glue:text}`feMinBins` of our bins, as beyond {glue:text}`feSkOutLo` kpc for the RR Lyrae, are not fitted. Where a study gives its
extent in degrees we convert at {glue:text}`feKpcDeg` kpc per degree ($D_0\,\pi/180$), so those ranges are approximate.
Because our profile is curved, the fitted slope depends on the range and on the weighting: unweighted, the slope of the BOSS medians
($G$ < {glue:text}`feGBright`) out to {glue:text}`feLitMunozR`° is {glue:text}`feOurMunozUnw` rather than {glue:text}`feOurMunozBright`
dex/deg. For the giants we use the BOSS giants with $G$ < {glue:text}`feGBright`, the reliable BOSS profile, and APOGEE. The published
values are from the abstracts and were checked against the published full texts; the preprint of Povick et al. (2025) quotes
−0.0380 ± 0.0022 dex/kpc, the published version {glue:text}`feLitPovick` ± {glue:text}`feLitPovickErr`, which we use. Panel (c) of
{fig}`paper_feh_literature` shows the levels that Carrera et al. (2008) and Massana et al. (2024) quote against projected distance;
they are on their own metallicity scales (Ca II triplet mean [Fe/H]; *Gaia* XP [M/H]), so only the shape compares with ours. The RR
Lyrae metallicities of Skowron et al. (2016) are on the high-dispersion scale of
[Jurcsik (1995)](https://ui.adsabs.harvard.edu/abs/1995AcA....45..653J) and ours on that of Crestani et al. (2021), which is the scale of
[Carretta et al. (2009)](https://doi.org/10.1051/0004-6361/200913003) plus {glue:text}`feCrestOff` dex. Skowron et al. (2016)
themselves convert to the Carretta scale with [Fe/H]<sub>C09</sub> = 1.001 [Fe/H]<sub>J95</sub> − 0.112, a slope of unity, so their
gradients apply unchanged on our scale; their alternative route through the scale of [Zinn & West (1984)](https://doi.org/10.1086/190947),
[Fe/H]<sub>ZW</sub> = ([Fe/H]<sub>J95</sub> − 0.88)/1.431 followed by [Fe/H]<sub>C09</sub> = 1.105 [Fe/H]<sub>ZW</sub> + 0.160, has a
slope of {glue:text}`feZwSlope` and would make their inner gradient {glue:text}`feZwInner` dex/kpc. Two further differences limit the
comparison: their radius is the three-dimensional distance from the centre, with each star's own photometric distance, rather than
our in-plane $R$; and with a different light-curve relation their whole-galaxy gradient falls from −0.019 to −0.011 dex/kpc
(Skowron et al. 2016).

## Published metallicity distributions

{fig}`paper_mdf_literature` compares, by number of giants, our [Fe/H] distributions with published ones in the same places. Ours are
the BOSS giants with their calibrated [Fe/H] as measured (the response correction of chapter 4 acts on binned counts, not on stars)
and the APOGEE giants, each with its selection weights. In the bar we take our giants within {glue:text}`feFieldR`° of each of the
{glue:text}`feNColeF` fields of [Cole et al. (2005)](https://doi.org/10.1086/428007), and in the disc those within
{glue:text}`feFieldR`° of the fields of Carrera et al. (2008) at 5, 6 and 8° from the centre (their 3° field holds {glue:text}`feNCarIn` BOSS giants), reweighted so
that each field has its share of the published sample. For Carrera et al. (2008) we compute [Fe/H] from their Ca II triplet index with
their equation (1), on the scale of [Carretta & Gratton (1997)](https://doi.org/10.1051/aas:1997116), for their radial-velocity
members.

The Ca II triplet distributions are more metal-rich than ours: in the bar the median is {glue:text}`feColeMed` (Cole et al. 2005)
against {glue:text}`feBarBoss` (BOSS) and {glue:text}`feBarApo` (APOGEE), and {glue:text}`feColeRich` per cent of the stars lie above
−0.5; in the disc fields it is {glue:text}`feCarMed` (Carrera et al. 2008) against {glue:text}`feDiscBoss` and {glue:text}`feDiscApo`,
with a broader metal-poor tail ({glue:text}`feCarPoor` per cent below −1, against {glue:text}`feDiscBossPoor` and
{glue:text}`feDiscApoPoor`). Much of the offset is the metallicity scale. For the stars that are in both samples, BOSS minus Ca II
triplet [Fe/H] is {glue:text}`feScCar` ({glue:text}`feNScCar` stars of Carrera et al. 2008, robust dispersion {glue:text}`feSdCar` dex)
and APOGEE minus Ca II triplet {glue:text}`feScApo` ({glue:text}`feNScApo` stars, {glue:text}`feSdApo` dex); only
{glue:text}`feNColeMatch` stars of Cole et al. (2005) are in BOSS ({glue:text}`feScCole`).

Over the whole LMC the selection weights move the APOGEE median from {glue:text}`feApoTgt` as targeted to {glue:text}`feApoW`, the
share above −0.5 from {glue:text}`feApoTgtRich` to {glue:text}`feApoWRich` per cent and the share below −1 from
{glue:text}`feApoTgtPoor` to {glue:text}`feApoWPoor` per cent ({fig}`paper_mdf_literature`c). The BOSS giants as measured have a median
of {glue:text}`feBossAll` and {glue:text}`feBossAllRich` per cent above −0.5, which includes the overestimated metallicities of hot,
low-S/N giants (chapter 3). Panel (d) shows the {glue:text}`feNXp` BOSS giants that also have the *Gaia* XP Strömgren metallicities of
[Omkumar et al. (2025)](https://doi.org/10.1051/0004-6361/202452510): their median, {glue:text}`feOmkMed`, matches the BOSS one,
{glue:text}`feXpBossMed`, but their distribution is much broader ({glue:text}`feOmkRich` per cent above −0.5 and {glue:text}`feOmkPoor`
per cent below −1, against {glue:text}`feXpBossRich` and {glue:text}`feXpBossPoor`), while the XP [M/H] of
[Andrae, Rix & Chandra (2023)](https://doi.org/10.3847/1538-4365/acd53e) for the same stars, converted to [Fe/H], peaks about
{glue:text}`feAndShift` dex lower (median {glue:text}`feAndMed`).

<!-- script: paper_mdf_literature.py -->

**Figure (paper_mdf_literature).** [Fe/H] distributions of LMC giants by number: fraction per dex against [Fe/H] (dex), histograms
normalised to unit area, with numbers of stars and medians in the legends above each panel. (a) Bar: Ca II triplet [Fe/H] of Cole et
al. (2005) (black) and our BOSS (blue; calibrated [Fe/H] as measured) and APOGEE (orange dashed) giants within {glue:text}`feFieldR`° of
their fields. (b) The fields of Carrera et al. (2008) at 5, 6 and 8° north of the bar, [Fe/H] from their equation (1), with BOSS and
APOGEE in the same fields. (c) All giants within {glue:text}`feRad`°: APOGEE as targeted (dotted) and with selection weights (dashed),
BOSS with weights (blue). (d) The same BOSS giants with *Gaia* XP metallicities: Strömgren [Fe/H] of Omkumar et al. (2025) (green),
[M/H] of Andrae et al. (2023) converted to [Fe/H] (purple dashed) and BOSS (blue). The Ca II triplet scale lies
{glue:text}`feScaleCaT` dex above the BOSS and APOGEE scales for stars in common. `scripts/paper_mdf_literature.py`
