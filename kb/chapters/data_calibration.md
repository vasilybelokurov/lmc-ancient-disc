---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Data: RR Lyrae and BOSS calibration

Every number on this page is written by `scripts/paper_text_data.py` from the result tables and catalogues
(`results/macros_data.tex`); the code cells are the figure scripts `scripts/paper_rrab_membership.py`, `scripts/paper_rrab_feh.py`,
`scripts/paper_boss_calibration.py` and `scripts/paper_boss_hot_bias.py`, step by step, and the two RR Lyrae diagnostics behind
them, shown but not run.

This chapter supports the samples chapter: how the RR Lyrae members are chosen, how their two photometric metallicity scales are
joined, how the BOSS spectra of the red giants are placed on the APOGEE scale, and where that calibration fails. RRab are RR Lyrae
stars pulsating in the fundamental mode; [Fe/H] is the iron abundance relative to the Sun's (dex); $\rho$ is the angular distance
from the LMC centre and $\Delta\mu$ the distance modulus of a star relative to the LMC centre (mag), both defined in the samples
chapter.

## RR Lyrae membership

{fig}`paper_rrab_membership` shows how the RR Lyrae members are selected. The SMC mask removes stars within
{glue:text}`daSmcMask`° of the centre of the Small Magellanic Cloud (SMC), the radius beyond which the fraction of RRab with
SMC-like proper motions (closer to the SMC's systemic motion than to the LMC's) falls to the level set by LMC stars. The distance
window $|\Delta\mu|$ < {glue:text}`daDmuWin` mag removes the foreground and background; the density in the sidebands
{glue:text}`daSideLo` < $|\Delta\mu|$ < {glue:text}`daSideHi` mag, assumed flat in $\Delta\mu$, gives the foreground left in the
window, at most {glue:text}`daContMax` per cent inside {glue:text}`daRhoMax`° and {glue:text}`daContBeyond` per cent at
{glue:text}`daRhoMax`–{glue:text}`daRhoBeyond`°.

<!-- script: paper_rrab_membership.py -->

**Figure (paper_rrab_membership).** RR Lyrae membership. (a) The sky to {glue:text}`daRrlQuery`°, in the orthographic projection
about the LMC centre, $x$ (east, increasing to the left) and $y$ (north) in degrees: members (blue, $|\Delta\mu|$ <
{glue:text}`daDmuWin` mag outside the SMC mask), stars in the SMC mask (orange) and stars outside the distance window (yellow);
orange circle: the SMC mask; dashed circle: the {glue:text}`daRhoMax`° science radius; grey circle: the
{glue:text}`daRrlQuery`° limit of the data. (b) Relative distance modulus $\Delta\mu$ (mag) against $\rho$ (deg) for the stars
outside the SMC mask (two-dimensional histogram, logarithmic colour scale in stars per pixel); orange lines: the window
$|\Delta\mu|$ < {glue:text}`daDmuWin` mag; dashed line: the science radius. (c) Foreground left in the window (per cent,
logarithmic axis) against $\rho$, from the $\Delta\mu$ sidebands, with horizontal bars the widths of the radial bins; inset:
fraction of window stars with SMC-like proper motions against distance from the SMC centre (deg), with the mask radius (orange
line). `scripts/paper_rrab_membership.py`

### The membership diagnostic

`scripts/fig_rrab_membership.py` is the diagnostic behind the membership limits. Per radial bin it counts the stars removed by the
SMC mask, the members and the sideband stars, and scales the sideband counts to the width of the window to estimate the
foreground left among the members. It writes these numbers to `results/rrab_membership.csv`, which panel (c) of
{fig}`paper_rrab_membership` and the text use, and draws a figure of its own that is not part of the paper.

<!-- script: fig_rrab_membership.py show: data/rrab.parquet (scripts/build_rrab.py); a diagnostic whose figure is not on this site -->

## RR Lyrae metallicities

{fig}`paper_rrab_feh` compares the two photometric metallicity scales, both derived from the period and the light-curve phase
$\phi_{31}$ (samples chapter). The $G$-band values of [Muraveva et al. (2025)](https://doi.org/10.1093/mnras/stae2679) lie
{glue:text}`daFehShift` dex below the $I$-band values of [Martínez-Vázquez et al. (2026)](https://doi.org/10.48550/arXiv.2609.27119)
with no trend with [Fe/H] (Deming slope {glue:text}`daDeming` ± {glue:text}`daDemingErr`, which allows for errors in both), so a
constant shift brings them onto one scale. On the scale of [Crestani et al. (2021)](https://doi.org/10.3847/1538-4357/abd183), the
adopted values exceed the spectroscopic [Fe/H] of [Gratton et al. (2004)](https://doi.org/10.1051/0004-6361:20035840) by
{glue:text}`daGrattonOff` ± {glue:text}`daGrattonErr` dex ({glue:text}`daNGratton` stars) and those of
[Borissova et al. (2004](https://doi.org/10.1051/0004-6361:20034494), [2006)](https://doi.org/10.1051/0004-6361:20054132) by
{glue:text}`daBorA` ± {glue:text}`daBorAErr` and {glue:text}`daBorB` ± {glue:text}`daBorBErr` dex ({glue:text}`daNBorA` and
{glue:text}`daNBorB` stars, from lower-resolution spectra). The RR Lyrae and giant [Fe/H] scales are therefore not identical; we do
not combine them in one metallicity axis.

<!-- script: paper_rrab_feh.py -->

**Figure (paper_rrab_feh).** RRab photometric metallicities. (a) $I$-band [Fe/H] (OGLE light curves;
[Mullen et al. 2021](https://doi.org/10.3847/1538-4357/abefd4) relation as in Martínez-Vázquez et al. 2026) against $G$-band [Fe/H]
(*Gaia* light curves; Muraveva et al. 2025), in dex, for the stars with both (two-dimensional histogram, logarithmic colour scale);
dotted: one-to-one line; orange: the adopted constant shift of {glue:text}`daFehShift` dex. (b) Their difference against their
mean (dex), with the running median (solid black) and 16th and 84th percentiles (dashed black), the adopted shift (orange) and the
Deming slope (quoted above the panel). (c) Both photometric values against the spectroscopy of Gratton et al. (2004) on the
Crestani et al. (2021) scale, for the stars with all three: $I$ band (blue) and $G$ band plus the shift (orange); the legend gives
the median offset (photometric minus spectroscopic) and the robust scatter, in dex; dotted: one-to-one line.
`scripts/paper_rrab_feh.py`

### The shift between the two scales

`scripts/fig_rrab_feh_transform.py` measures, on the stars with both estimates and after dropping gross outliers of their
difference, the median difference between the $I$-band and $G$-band [Fe/H] and the Deming fit with its bootstrap errors. It writes
them to `results/rrab_feh_transform.csv`, from which the text takes the shift and the slope. Its figure shows the difference
against each estimate and against their mean, since binning on one noisy estimate alone produces a slope.

<!-- script: fig_rrab_feh_transform.py show: data/rrab.parquet (scripts/build_rrab.py); a diagnostic whose figure is not on this site -->

## BOSS calibration

{fig}`paper_boss_calibration` shows the {glue:text}`daNOverlap` giants observed by both surveys that fix the two calibration
relations of the samples chapter, the velocity offset $v_{\rm BOSS}-v_{\rm APOGEE}=d_0+d_1t$ and
$[{\rm Fe/H}]_{\rm BOSS}=c_0+c_1[{\rm Fe/H}]_{\rm APOGEE}+c_2t+c_3s$, with $t$ the BOSS effective temperature and $s$ the inverse
BOSS signal-to-noise ratio, both scaled about a pivot as defined there. BOSSNet compresses the APOGEE scale by a factor of
$c_1$ = {glue:text}`daCompress`. The calibrated values follow APOGEE with no trend from [Fe/H] = {glue:text}`daBiasFehHi`
upwards and fall {glue:text}`daBiasLo`–{glue:text}`daBiasHi` dex low at {glue:text}`daBiasFehLo` < [Fe/H] <
{glue:text}`daBiasFehHi`, the bias corrected in the stellar-mass chapter. The compression depends on S/N: fitted in bins of BOSS
S/N, the slope of BOSSNet against APOGEE [Fe/H] is {glue:text}`daSlopeA` ± {glue:text}`daSlopeAErr` ({glue:text}`daNSlopeA` stars)
at {glue:text}`daSnrA`–{glue:text}`daSnrB`, {glue:text}`daSlopeB` ± {glue:text}`daSlopeBErr` ({glue:text}`daNSlopeB`) at
{glue:text}`daSnrB`–{glue:text}`daSnrC`, {glue:text}`daSlopeC` ± {glue:text}`daSlopeCErr` ({glue:text}`daNSlopeC`) at
{glue:text}`daSnrC`–{glue:text}`daSnrD`, {glue:text}`daSlopeD` ± {glue:text}`daSlopeDErr` at
{glue:text}`daSnrD`–{glue:text}`daSnrE` and {glue:text}`daSlopeFLo`–{glue:text}`daSlopeFHi` ± {glue:text}`daSlopeFErr` above
{glue:text}`daSnrF`. Below S/N = {glue:text}`daSnrBoss` the calibrated values run {glue:text}`daCalHighA` and
{glue:text}`daCalHighB` dex high, so we require S/N ≥ {glue:text}`daSnrBoss` for a good BOSS spectrum; above it the median offset
is within {glue:text}`daCalAbove` dex in every S/N bin. Rebuilding the BOSS sample with S/N ≥ {glue:text}`daSnrAlt` instead
changes the total formed mass by {glue:text}`daDTot` per cent, the mass below [Fe/H] = {glue:text}`daBiasFehHi` by
{glue:text}`daDMP` per cent and below {glue:text}`daBiasFehLo` by {glue:text}`daDVMP` per cent, the scale lengths of the
metallicity-structure chapter by at most {glue:text}`daDH` kpc and the radial median [Fe/H] profile by at most
{glue:text}`daDProf` dex outside {glue:text}`daProfR` kpc, all within the statistical errors. The velocity offset falls with
temperature as the relation for $v_{\rm BOSS}-v_{\rm APOGEE}$ describes.

<!-- script: paper_boss_calibration.py -->

**Figure (paper_boss_calibration).** BOSS on the APOGEE scale, from the {glue:text}`daNOverlap` giants observed by both surveys
(grey points). (a) Raw BOSSNet against APOGEE [Fe/H] (dex), with the [Fe/H] relation at $t=s=0$ (orange; slope $c_1$ =
{glue:text}`daCompress`) and the one-to-one line (dashed). (b) Calibrated BOSS minus APOGEE [Fe/H] (dex) against APOGEE [Fe/H].
(c) BOSS minus APOGEE velocity (km/s) against BOSS $T_{\rm eff}$ (K), with the velocity relation (orange). Blue lines and bands in
(b, c): median and 16th–84th percentiles per bin of at least {glue:text}`daCalNMin` stars. All stars shown enter the fits; held-out values are given
in the samples chapter. `scripts/paper_boss_calibration.py`

## BOSSNet metallicities of hot giants

{fig}`paper_boss_hot_bias` shows why the BOSS metallicities do not share out the mass above [Fe/H] = {glue:text}`daSplitFeh` (stellar-mass chapter).
Against APOGEE, for the stars observed by both with APOGEE [Fe/H] between {glue:text}`daRichLo` and {glue:text}`daRichHi`, BOSS is unbiased below $T_{\rm eff}$ =
{glue:text}`daTCool` K (offsets within {glue:text}`daCoolMax` dex) but reads +{glue:text}`daWarmLo`–{glue:text}`daWarmHi` dex high
at {glue:text}`daTCool`–{glue:text}`daTWarm` K and +{glue:text}`daHotLo` to +{glue:text}`daHotHi` dex at
{glue:text}`daTWarm`–{glue:text}`daTHot` K, at both low and high S/N (panel a). At fixed magnitude and radius, the faint blue BOSS
giants (dereddened $G_{\rm BP}-G_{\rm RP}$ < {glue:text}`daBlueEdge`) are {glue:text}`daBlueLow` per cent metal-rich ([Fe/H] >
{glue:text}`daRichHi`) at S/N < {glue:text}`daSnrSplit` and {glue:text}`daBlueHigh` per cent at S/N ≥ {glue:text}`daSnrSplit`; in the
reddest colour bin ({glue:text}`daRedLo`–{glue:text}`daRedHi`) the shares fall to {glue:text}`daRedLow` and {glue:text}`daRedHigh` per
cent, against {glue:text}`daApoRichLo`–{glue:text}`daApoRichHi` per cent for APOGEE at every colour
(panel b). The BOSS sample is dominated by hot, low-S/N giants, which the stars observed by both surveys barely contain (panel c),
so the calibration and the response cannot correct them; the stellar-mass chapter replaces the BOSS split above {glue:text}`daSplitFeh` with that of
APOGEE.

<!-- script: paper_boss_hot_bias.py -->

**Figure (paper_boss_hot_bias).** BOSSNet metallicities of hot giants. (a) Calibrated BOSS minus APOGEE [Fe/H] (dex) for the stars
observed by both surveys with APOGEE [Fe/H] between {glue:text}`daRichLo` and {glue:text}`daRichHi`, against BOSS $T_{\rm eff}$ (K), for S/N < {glue:text}`daSnrSplit`
(orange) and ≥ {glue:text}`daSnrSplit` (blue); points: stars; lines: medians in {glue:text}`daTBin`-K bins with at least
{glue:text}`daHotNMin` stars. (b) Weighted fraction of giants with [Fe/H] > {glue:text}`daRichHi` at {glue:text}`daHotGLo` ≤ $G$ ≤
{glue:text}`daHotGHi` and $\rho$ ≥ {glue:text}`daHotRho`°, against dereddened $G_{\rm BP}-G_{\rm RP}$ (mag): BOSS at S/N <
{glue:text}`daSnrSplit` (orange circles) and ≥ {glue:text}`daSnrSplit` (blue squares), and APOGEE (black dashed, triangles).
(c) Distribution of BOSS $T_{\rm eff}$ (fraction per {glue:text}`daTBin`-K bin) of the weighted BOSS mass sample (solid) and of
the stars observed by both surveys (dashed), each for S/N < {glue:text}`daSnrSplit` (orange) and ≥ {glue:text}`daSnrSplit` (blue).
`scripts/paper_boss_hot_bias.py`
