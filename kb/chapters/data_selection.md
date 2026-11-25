---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Data: selection functions

Every number on this page is written by `scripts/paper_text_data.py` from the result tables and catalogues
(`results/macros_data.tex`); the code cells are the figure scripts `scripts/paper_selection.py`, `scripts/paper_selection_checks.py`
and `scripts/paper_selection_gaps.py`, step by step, and the scripts that compute the weights and their tests, shown but not run.

The stellar mass of the next chapters counts giants per metallicity bin, so we need the probability that an LMC giant entered each
spectroscopic sample, and in particular whether that probability depends on metallicity. This probability, as a function of a
star's properties, is the selection function of the sample. For the RR Lyrae we measure the completeness of the union of two
independent surveys. For the giants we reproduce the targeting of each survey and weight its spectra in four steps:

1. **Parent.** We re-apply the survey's own targeting rules to the catalogue it targeted from (*Gaia* DR3 for BOSS, the Two Micron
   All Sky Survey, 2MASS, for APOGEE). The parent holds every star the survey could have observed ({tab}`cuts`).
2. **Selection groups.** We divide the parent into groups of stars the survey treated alike (sky position, magnitude, targeting
   class) and weight each good spectrum by the number of parent stars per good spectrum in its group.
3. **Pass factor.** Spectra of metal-poor giants fail more often. A metallicity known for observed and unobserved stars alike, the
   *Gaia* XP estimate (from the low-resolution blue and red prism spectra of *Gaia*), measures this, and a pass factor puts the
   failed spectra back.
4. **Coverage.** Groups with fewer than {glue:text}`daNminGood` good spectra are uncovered; a coverage factor puts their parent
   stars back.

Non-members (dwarfs and stars outside the LMC velocity range) stay in the sample until the weights are made and are removed
afterwards. The Milky Way giants that share the LMC velocities are subtracted statistically in the stellar-mass chapter. The weights
rest on five assumptions, each tested below:

- **A1** the parent contains every LMC giant inside the survey's selection;
- **A2** within a selection group, the survey chose which stars to observe without regard to their metallicity or membership;
- **A3** whether an observed star gives a good spectrum depends on its metallicity only through its magnitude, XP metallicity and
  ring;
- **A4** the parent stars of uncovered groups have the metallicities and member fraction of the covered stars in the same ring and
  magnitude bin;
- **A5** non-members pass the spectroscopic quality cuts at the same rate as members of the same pass group.

| | BOSS (carton mwm_magcloud_rgb_boss) | APOGEE-2 (LMC1–35) |
|---|---|---|
| Source catalogue | *Gaia* DR3 | 2MASS |
| Magnitude | $G$ ≤ {glue:text}`daGLim` | {glue:text}`daHBright` ≤ $H$ ≤ $H_{\rm lim}$ ({glue:text}`daHlimLo`–{glue:text}`daHlimHi`) |
| Colour | $G_{\rm BP}-G_{\rm RP}$ ≥ {glue:text}`daBpRp` | $J-K_s$ between the polygon edges at $H$ (observed) |
| Colour–magnitude | dereddened $(G,\,G_{\rm BP}-G_{\rm RP})$ inside the carton polygon | – |
| Proper motion | $\lvert\mu_{\alpha*}\rvert,\lvert\mu_\delta\rvert$ ≤ {glue:text}`daPmBox` mas/yr; within {glue:text}`daPmWin` mas/yr of ({glue:text}`daPmL`, {glue:text}`daPmB`) mas/yr (Magellanic Stream coordinates) | as BOSS (a) |
| Parallax | not ($\varpi$ > 0 and $(\varpi+$ {glue:text}`daPlxOff`$)/\sigma_\varpi$ > {glue:text}`daPlxSigC`) | as BOSS (a) |
| Sky | $\rho$ < {glue:text}`daGiantRho`° | within {glue:text}`daFieldR`° of a field centre |
| Other | finite photometry and astrometry | no neighbour ≥ {glue:text}`daNbrMag` mag brighter in $H$ within {glue:text}`daNbrSep`″ |
| Good spectrum | BOSSNet flag unset, finite [Fe/H] and $v$, S/N ≥ {glue:text}`daSnrBoss`, $\sigma_v$ < {glue:text}`daEvBoss` km/s | STAR_BAD unset, finite [Fe/H], S/N ≥ {glue:text}`daSnrApoSel` |

```{code-cell} python
:tags: [remove-input]
import json
import pandas as pd
# Counts at each selection step: BOSS (b) and APOGEE-2 (a) selection summaries, members of the mass sample (m)
b = json.load(open('results/boss_selection.json'))['summary']
a = json.load(open('results/apogee_selection.json'))['summary']
m = json.load(open('results/giant_counts.json'))
# Parent stars with a BOSS spectrum
b_obs = int(pd.read_parquet('data/boss_weights.parquet', columns=['observed']).observed.sum())
# One row per step, one column per survey
pd.DataFrame([('Parent stars', b['eligible'], a['candidates']), ('Observed', b_obs, a['observed']),
              ('Good spectra', b['good'], a['quality']), ('Parent in covered groups', b['covered'], a['covered']),
              ('Mass sample (b)', m['boss_members'], m['apogee_members'])],
             columns=['', 'BOSS', 'APOGEE-2']).set_index('')
```

**Table (cuts).** Parent catalogues and spectroscopic samples of the giant selection functions. Upper part: the cuts, those of the
surveys' own targeting re-applied to their source catalogues except where marked; $G$, $G_{\rm BP}$, $G_{\rm RP}$: *Gaia*
magnitudes; $J$, $H$, $K_s$: 2MASS magnitudes; $H_{\rm lim}$: faint limit of the field; $\mu_{\alpha*}$, $\mu_\delta$: proper
motions; $\varpi$, $\sigma_\varpi$: parallax and its error (mas); $\sigma_v$: velocity error. (a) Not an APOGEE cut: in LMC1–17 a
stand-in for the unpublished Washington+DDO51 giant flag, in LMC18–35 a foreground filter. Lower part: numbers of stars within
{glue:text}`daGiantRho`° (BOSS) and {glue:text}`daFieldR`° of the {glue:text}`daNFields` field centres (APOGEE), from
`results/boss_selection.json`, `results/apogee_selection.json` and `results/giant_counts.json` (as `scripts/paper_table_cuts.py`).
(b) Giants ($T_{\rm eff}$ < {glue:text}`daTeffMax` K, $\log g$ < {glue:text}`daLoggMax`) with {glue:text}`daVMassLo` <
$v_{\rm los}$ < {glue:text}`daVMassHi` km/s and a positive weight.

<!-- script: paper_selection.py -->

**Figure (paper_selection).** Selection functions. (a) RRab completeness against angular distance $\rho$ from the centre (deg):
*Gaia* (blue; fraction of OGLE's own RRab that *Gaia* classifies as RRab), OGLE's own search (orange; fraction of *Gaia* RRab inside
the OGLE-IV footprint found by OGLE) in {glue:text}`daSelBin`° bins with at least {glue:text}`daSelNBin` stars, binomial errors; and their union (green squares;
Lincoln–Petersen estimate in magnitude quartiles, horizontal bars the ring widths). (b, c) Share of the spectroscopic sample in each
bin of XP metallicity [M/H]$_{\rm XP}$ ([Andrae et al. 2023](https://doi.org/10.3847/1538-4365/acd53e)), divided by the share of the
parent, for the good spectra unweighted (open circles) and with the final weights (filled circles, joined): (b) BOSS (blue), parent =
the re-implemented carton in covered selection groups; (c) APOGEE (orange), parent = the covered candidates of the LMC1–35
targeting. Grey line: ratio 1. Errors: bootstrap over HEALPix pixels at resolution parameter $N_{\rm side}$ = {glue:text}`daSelNside` with the weights
fixed; bins with at least {glue:text}`daSelNMin` spectra. `scripts/paper_selection.py`

## RR Lyrae

OGLE's own search and *Gaia* found RRab independently, so their overlap measures the completeness of each. Within the OGLE-IV
footprint, the fraction of OGLE RRab that *Gaia* also classifies as RRab is the *Gaia* completeness, and the fraction of *Gaia* RRab
that OGLE found is the OGLE completeness. If the two surveys miss stars independently, the number of RRab in a region is the
Lincoln–Petersen (capture–recapture) estimate $N=n_{\rm O}n_{\rm G}/n_{\rm OG}$, from the numbers found by OGLE ($n_{\rm O}$), by
*Gaia* ($n_{\rm G}$) and by both ($n_{\rm OG}$). We make this estimate in quartiles of mean magnitude, which removes most of the bias
from stars that both surveys find hard; splitting by period or amplitude instead changes the union completeness by at most
{glue:text}`daSplitMax` per cent.

{fig}`paper_selection` (a) shows the completeness against $\rho$. *Gaia* misses {glue:text}`daGaiaMissIn` per cent of the RRab at
the crowded centre and at most {glue:text}`daGaiaMissOut` per cent beyond {glue:text}`daGaiaRhoOut`°; OGLE's own search is
{glue:text}`daOgleLo`–{glue:text}`daOgleHi` per cent complete inside {glue:text}`daOgleRho`° and falls beyond, where many stars
entered the OGLE collection only from *Gaia*. The union is {glue:text}`daUnionIn` per cent complete at $\rho$ <
{glue:text}`daUnionRhoA`°, {glue:text}`daUnionMidLo`–{glue:text}`daUnionMidHi` per cent at
{glue:text}`daUnionRhoA`°–{glue:text}`daUnionRhoB`° and {glue:text}`daUnionOutLo`–{glue:text}`daUnionOutHi` per cent beyond. In the
crowded centre this is an upper limit: stars that blending hides from both surveys are invisible to capture–recapture. The checks
below give the completeness against magnitude and on the sky.

### Capture–recapture estimates

`scripts/check_rrab_independence.py` makes the Lincoln–Petersen estimates in rings of $\rho$ inside the OGLE-IV footprint, once
for the whole ring and once summed over quartiles of mean magnitude, of period or of amplitude. The sums over magnitude quartiles
give the union completeness of {fig}`paper_selection` (a); the other two give the change quoted above.

<!-- script: check_rrab_independence.py show: data/rrab.parquet and the OGLE-IV field outlines (scripts/fetch_ogle4_fields.py); it writes results/rrab_independence.csv, read by scripts/paper_selection.py -->

## BOSS giants

**Parent.** The SDSS-V carton mwm_magcloud_rgb_boss ([Nidever et al. 2026](https://doi.org/10.3847/1538-3881/ae7c76)), a carton
being one targeting category of SDSS-V, selected its targets from *Gaia* DR3. We re-implement it from the public target-selection
code with the cuts of {tab}`cuts`: a magnitude limit, a colour limit, two proper-motion cuts, a parallax cut and a polygon in the
dereddened colour–magnitude diagram. The carton dereddens with the reddening $E(B-V)$ of the
[Schlegel et al. (1998)](https://doi.org/10.1086/305772) map, scaled by {glue:text}`daSfdScale` where $E(B-V)$ ≥
{glue:text}`daSfdEbv` within {glue:text}`daSfdRad`° of the LMC (where the map traces LMC dust emission), and $A_V$ =
{glue:text}`daAvA` × {glue:text}`daAvB` $E(B-V)$, with $A_V$ the extinction in the $V$ band. Every cut needs finite photometry and
astrometry, so a star without a five-parameter astrometric solution is not in the parent; no cut is made on RUWE. Within
{glue:text}`daGiantRho`°, {glue:text}`daNParent` stars pass, and {glue:text}`daInCarton` per cent of them are members of the
carton in the public SDSS-V target database: the re-implementation reproduces the carton. SDSS-V observed {glue:text}`daNObs` of
them and obtained a good spectrum for {glue:text}`daNGood`: BOSSNet flag unset, finite [Fe/H] and velocity, S/N ≥
{glue:text}`daSnrBoss` and a velocity error below {glue:text}`daEvBoss` km/s, the highest-S/N spectrum of each star. The giant
and velocity cuts of the samples chapter are not part of this definition; they remove non-members after weighting.

The parent is complete (A1) only if *Gaia* DR3 holds every LMC giant inside the selection with an astrometric solution that passes
the proper-motion window. Two tests bound the loss. APOGEE targeted LMC18–35 without astrometry: of its {glue:text}`daNAstroB`
good-spectrum giants there with LMC velocities, {glue:text}`daLossB` per cent fail the carton's astrometric cuts, and
{glue:text}`daLossA` per cent of the {glue:text}`daNAstroA` in the crowded fields LMC1–17. In the dusty centre the parent lacks
{glue:text}`daDustMissing` per cent of the stars that the conversion of the stellar-mass chapter expects. We do not correct for
either; both make the masses of the inner few degrees low by a few per cent.

**Selection groups.** Fibres did not go to the parent at random. Some sky areas were observed more often than others, and parent
stars that also belonged to other cartons, chiefly the distant halo K giants and the metal-poor candidates selected from *Gaia* XP
spectra, had higher priority. A selection group therefore combines the fibre-priority class of the star (the carton alone, K giant,
XP metal-poor candidate or other), a pixel of the HEALPix equal-area tessellation of the sphere
([Górski et al. 2005](https://doi.org/10.1086/427976)) at $N_{\rm side}$ = {glue:text}`daNside` ({glue:text}`daPixArea` deg²)
and a $G$ bin (edges {glue:text}`daGEdges`). A group with fewer than {glue:text}`daNminGroup` parent stars merges with its
neighbours at the next coarser level (no $G$ bins, then $N_{\rm side}$ = {glue:text}`daNsideCoarse` and
{glue:text}`daNsideLast`, then the whole class). There are {glue:text}`daNGroups` groups, and {glue:text}`daFinestShare` per cent
of the parent sits at the finest level. In a group $s$ with $N_{{\rm P},s}$ parent stars, $N_{{\rm O},s}$ observed and
$N_{{\rm G},s}$ good spectra, assumption A2 makes $N_{{\rm P},s}/N_{{\rm O},s}$ the weight of an observed star. If failures also
ignored metallicity, $N_{{\rm P},s}/N_{{\rm G},s}$ would be the weight of a good spectrum.

**Pass factor.** Failures do depend on metallicity: the weak lines of faint metal-poor giants fail more often, and more often
still in the crowded centre. We measure this with a metallicity known for observed and unobserved stars alike, the *Gaia* XP
estimate [M/H]$_{\rm XP}$ of [Andrae et al. (2023)](https://doi.org/10.3847/1538-4365/acd53e) ([M/H]: the overall metal abundance
relative to the Sun's, in dex), kept only for cool giants (XP $T_{\rm eff}$ < {glue:text}`daXpTeff` K, $\log g$ <
{glue:text}`daXpLogg`), where it is reliable ([Zhang et al. 2024](https://doi.org/10.1093/mnras/stae1887));
{glue:text}`daXpShare` per cent of the parent has one. Pass groups $c$ combine $G$ (edges {glue:text}`daPgEdges`),
[M/H]$_{\rm XP}$ (edges {glue:text}`daXpEdges`, and a bin for stars without [M/H]$_{\rm XP}$) and three rings ($\rho$ =
0–{glue:text}`daRingA`, {glue:text}`daRingA`–{glue:text}`daRingB` and {glue:text}`daRingB`–{glue:text}`daRingOut`°). The pass
factor

$$
f_c=\frac{\sum_{i\in{\rm O}\cap c}N_{{\rm P},s(i)}/N_{{\rm O},s(i)}}{\sum_{i\in{\rm G}\cap c}N_{{\rm P},s(i)}/N_{{\rm G},s(i)}}
$$

is the weighted number of observed stars (set O) over the weighted number of good spectra (set G) in the pass group, $s(i)$ being
the selection group of star $i$, and the weight of a good spectrum $i$ is

$$
w_i=\frac{N_{{\rm P},s(i)}}{N_{{\rm G},s(i)}}\,f_{c(i)}.
$$

The factor is close to unity at $G$ < {glue:text}`daPassG` and at [M/H]$_{\rm XP}$ > {glue:text}`daPassXp`. For
[M/H]$_{\rm XP}$ < {glue:text}`daXpLow` at {glue:text}`daGFaintLo` < $G$ < {glue:text}`daGFaintHi` it is {glue:text}`daPassA` ±
{glue:text}`daPassAErr`, {glue:text}`daPassB` ± {glue:text}`daPassBErr` and {glue:text}`daPassC` ± {glue:text}`daPassCErr` in the
three rings (bootstrap over $N_{\rm side}$ = {glue:text}`daNsideBoot` pixels). A factor common to all rings closes the whole sky
but not the rings: it leaves the share of stars below [M/H]$_{\rm XP}$ = {glue:text}`daCutB` at {glue:text}`daCommonIn` of the parent's within
{glue:text}`daRingIn`° and {glue:text}`daCommonOut` beyond {glue:text}`daRingB`°. The pass factor puts back every failure within
a pass group whatever its cause (A3); it cannot follow a dependence on true metallicity at fixed [M/H]$_{\rm XP}$, or on position
within a ring.

**Coverage.** Groups with at least {glue:text}`daNminGood` good spectra are covered: {glue:text}`daNCovGroups` groups holding
{glue:text}`daNCovParent` parent stars ({glue:text}`daCovShare` per cent). The other {glue:text}`daNUncov`
({glue:text}`daUncovShare` per cent), {glue:text}`daUncovIn` per cent of them within {glue:text}`daRingA`°, are in uncovered
groups. *Gaia* counts these stars, so no model of the spatial density is needed; only their metallicities are unknown. The coverage
factor

$$
c_i=\frac{N_{{\rm P},r(i)}}{N_{{\rm P,cov},r(i)}}
$$

is the ratio of all to covered parent stars in the cell $r$ of ring and $G$ bin (edges {glue:text}`daGEdges`) of star $i$; it is
{glue:text}`daCovRingLo`–{glue:text}`daCovRingHi` per ring and reaches {glue:text}`daCovMax` for $G$ < {glue:text}`daCovMaxG`
within {glue:text}`daRingA`°. It assumes A4. The holes in the BOSS maps of the samples chapter are mostly not uncovered groups but
covered groups with few spectra: the {glue:text}`daSparseN` groups where fewer than {glue:text}`daSparseFrac` per cent of the
parent gave a good spectrum hold {glue:text}`daSparseShare` per cent of the parent with {glue:text}`daSparseGood` spectra, at a
median $\rho$ of {glue:text}`daSparseRho`°, and weights reach {glue:text}`daMaxW`. These groups are unbiased under A2 and A3, and
their noise enters the bootstrap.

**Non-members.** Of the weighted good spectra, {glue:text}`daNonMem` per cent fail the giant or velocity cuts
({glue:text}`daNonMemLo`–{glue:text}`daNonMemHi` per cent per ring). Under A2 the survey observed them at the rate of the members
of their group, so removing them after weighting leaves the weighted members unbiased: the weights add up to the covered parent, and
those of the members to the covered parent less its non-members. A non-member whose spectrum fails is put back by the pass factor
onto the stars of its pass group (A5); dwarfs are {glue:text}`daDwarfs` per cent of the good spectra, which bounds the weight this
can move.

**Tests.** {fig}`paper_selection` (b) compares the share of each [M/H]$_{\rm XP}$ bin in the spectra with that in the covered
parent. The unweighted spectra over-represent the stars below [M/H]$_{\rm XP}$ = {glue:text}`daOverXp` by
{glue:text}`daOverLo`–{glue:text}`daOverHi` per cent, because the higher-priority cartons favoured them, and under-represent the
most metal-rich by {glue:text}`daUnder` per cent. With the final weights the shares agree: the fractions below
[M/H]$_{\rm XP}$ = {glue:text}`daCutA`, {glue:text}`daCutB` and {glue:text}`daCutC` are recovered to {glue:text}`daFinA` ± {glue:text}`daFinAErr`, {glue:text}`daFinB` ±
{glue:text}`daFinBErr` and {glue:text}`daFinC` ± {glue:text}`daFinCErr`, against {glue:text}`daGrpA`, {glue:text}`daGrpB` and
{glue:text}`daGrpC` with the selection-group weights alone. Within a pass group this agreement holds by construction, so the figure
tests A2 across groups, for the {glue:text}`daXpShare` per cent of stars with [M/H]$_{\rm XP}$. In rings of
0–{glue:text}`daRingIn`, {glue:text}`daRingIn`–{glue:text}`daRingA`, {glue:text}`daRingA`–{glue:text}`daRingB` and
{glue:text}`daRingB`–{glue:text}`daRingOut`°, the weighted share below [M/H]$_{\rm XP}$ = {glue:text}`daCutB` is {glue:text}`daRingFinA`,
{glue:text}`daRingFinB`, {glue:text}`daRingFinC` and {glue:text}`daRingFinD` of the parent's. The deficit within
{glue:text}`daRingIn`° comes from the sparsely sampled central groups, whose few spectra hold too few metal-poor stars. Pass factors
measured on half of the sky (even $N_{\rm side}$ = {glue:text}`daNsideHeld` pixels) and applied to the other half leave that half
short of {glue:text}`daHeldShort` of its stars below {glue:text}`daCutB`, {glue:text}`daHeldRatio` per cent of the parent fraction of
{glue:text}`daParentFrac` ({glue:text}`daHeldNoPass` without pass factors): the failure rate varies on scales finer than our rings,
and this test sets the size of that systematic. A mock on the real design, with known metallicities and failures that depend on
metallicity, recovers {glue:text}`daMockA` ± {glue:text}`daMockAErr` of the stars below [Fe/H] = {glue:text}`daMockCutA` and {glue:text}`daMockB` ±
{glue:text}`daMockBErr` below {glue:text}`daMockCutB`; per ring, with the coverage factor and failures twice as frequent within {glue:text}`daRingA`°,
it recovers {glue:text}`daFillA` ± {glue:text}`daFillAErr`, {glue:text}`daFillB` and {glue:text}`daFillC` below {glue:text}`daMockCutA` (see the
checks below): within {glue:text}`daRingA`° the S/N cut leaves many groups with few spectra. The metallicities of the uncovered
parent match those of covered stars in the same rings and $G$ bins to {glue:text}`daGapA` per cent below [M/H]$_{\rm XP}$ = {glue:text}`daCutA` and
{glue:text}`daGapB` per cent below {glue:text}`daCutB` (A4), and requiring two or five good spectra per covered group instead of three changes the
weighted members by {glue:text}`daCovTwo` and {glue:text}`daCovFive` per cent. Among the stars with APOGEE spectra, those whose BOSS
spectra the S/N cut rejects have the APOGEE metallicities of those it keeps in the same pass groups (mean difference
{glue:text}`daSnrDiff` (+{glue:text}`daSnrDiffHi`, −{glue:text}`daSnrDiffLo`) dex, {glue:text}`daSnrN` stars), as A3 requires.

### The BOSS weights

`scripts/boss_selection.py` joins the parent with its observed flags, carton memberships and XP metallicities (built in the
samples chapter), forms the selection groups and the pass groups and computes the weights $w_i$. It also runs the closure tests
of {fig}`paper_selection` (b): over the whole sky with a bootstrap over HEALPix pixels, on the held-out half of the sky and per
ring, and it records the pass factor of each pass group with its bootstrap error. It writes `data/boss_weights.parquet` and
`results/boss_selection.json`.

<!-- script: boss_selection.py show: the BOSS parent tables of the samples chapter, including data/boss_cartons.parquet and data/xp_andrae.parquet, which are not in Zenodo bundle 1 -->

## APOGEE giants

**Parent.** APOGEE-2 (the second phase of APOGEE) selected its LMC giants from 2MASS photometry
([Skrutskie et al. 2006](https://doi.org/10.1086/498708)) in a polygon of $J-K_s$ against $H$
([Zasowski et al. 2017](https://doi.org/10.3847/1538-3881/aa8df9); [Nidever et al. 2020](https://doi.org/10.3847/1538-4357/ab7305);
[Santana et al. 2021](https://doi.org/10.3847/1538-3881/ac2cbc)): bright giants ({glue:text}`daHBright` ≤ $H$ <
{glue:text}`daHFaint`) at random, and fainter ones ({glue:text}`daHFaint` ≤ $H$ ≤ $H_{\rm lim}$) brightest first. $H_{\rm lim}$
is {glue:text}`daHlimLo` ({glue:text}`daHlimMid` in LMC2 and LMC12) in LMC1–17 and {glue:text}`daHlimHi` in LMC18–35, and plate
design dropped stars with a neighbour at least {glue:text}`daNbrMag` mag brighter in $H$ within {glue:text}`daNbrSep`″. We rebuild
the candidates in each field from the public targeting files: {glue:text}`daNPoly` stars inside the polygon within
{glue:text}`daFieldR`° of a field centre, of which the bright-neighbour rule removes {glue:text}`daNNbr`. The polygons contain
{glue:text}`daInPoly` per cent of the stars APOGEE targeted in LMC1–17 and {glue:text}`daInPolyB` per cent of those in LMC18–35.
In LMC1–17 APOGEE also required a giant flag from Washington and DDO51 photometry (W+D; two filters whose combination separates
giants from dwarfs), which DR17 does not publish. In its place, and in all {glue:text}`daNFields` fields, we require the
astrometric cuts of the BOSS carton. They leave {glue:text}`daNCand` candidates; in LMC18–35 they remove {glue:text}`daAstroFail`
per cent of the candidates, nearly all of them Milky Way dwarfs, and {glue:text}`daAstroGiant` per cent of the giants with LMC
velocities.

**Weights.** Selection groups are a field, an annulus (distance from the field centre $\rho_{\rm field}$ <
{glue:text}`daZoneR`° or {glue:text}`daZoneR`–{glue:text}`daFieldR`°), the bright or faint class, $H$ in
{glue:text}`daHStep`-mag bins for the faint class, and {glue:text}`daNColour` bins of position across the colour range of the
polygon. A good spectrum has the ASPCAP flag STAR_BAD unset, a finite [Fe/H], S/N ≥ {glue:text}`daSnrApoSel` and is not a
duplicate; as for BOSS, the giant cut comes after weighting. APOGEE observed {glue:text}`daNApoObs` candidates and obtained
{glue:text}`daNApoGood` good spectra. A bright group is covered with at least {glue:text}`daCovBright` good spectra, a faint group
with at least {glue:text}`daCovFaint` and an observed fraction of at least {glue:text}`daCovFobs` (which drops the thin bins at each
field's fibre limit), and any group with at least a fraction {glue:text}`daCovHalf` of its candidates good; the covered groups hold
{glue:text}`daNApoCov` candidates. The weights follow the equation for $w_i$ above, with pass groups of $H$ (edges
{glue:text}`daPgHEdges`) and [M/H]$_{\rm XP}$: ASPCAP fails (STAR_BAD set or no [Fe/H]) for {glue:text}`daAspMP` per cent of the
bright observed stars with S/N ≥ {glue:text}`daSnrApoSel` and [M/H]$_{\rm XP}$ < {glue:text}`daAspXpLo` against {glue:text}`daAspMR` per cent of those with
[M/H]$_{\rm XP}$ > {glue:text}`daAspXpHi`. APOGEE fields are small, so we use no coverage factor across the sky; the stellar-mass
chapter uses APOGEE only in field cores. Of the weighted good spectra, {glue:text}`daVelA` per cent (LMC1–17) and
{glue:text}`daVelB` per cent (LMC18–35) fail the velocity cut.

**Tests.** Within each group class the weighted mean $J-K_s$ of the good spectra matches that of the covered candidates to
{glue:text}`daJk` mag, and the weighted fraction of stars below [Fe/H] = {glue:text}`daCutB` changes by at most {glue:text}`daFdiff` between S/N
thresholds of {glue:text}`daSnrT`. {fig}`paper_selection` (c) shows the test against [M/H]$_{\rm XP}$. In LMC18–35 the weighted share
below [M/H]$_{\rm XP}$ = {glue:text}`daCutB` is {glue:text}`daApoLateFin` of the parent's ({glue:text}`daApoLateNo` before the pass factor); in
LMC1–17 it is {glue:text}`daApoEarlyFin` ({glue:text}`daApoEarlyNo` before; field-bootstrap error {glue:text}`daApoEarlyNoErr`). The unpublished giant flag, which our astrometric stand-in
does not reproduce, is the likely cause. We do not correct for it and use APOGEE as a check of the BOSS masses, not as their source.

### The APOGEE weights

`scripts/apogee_selection.py` keeps one row per candidate, forms the selection groups, computes the weights with pass groups of
$H$ and [M/H]$_{\rm XP}$, and runs the tests quoted above: the share of the APOGEE targets inside the rebuilt polygons, the
weighted $J-K_s$ colour, the closure in [M/H]$_{\rm XP}$ of {fig}`paper_selection` (c) and the comparison of S/N thresholds. It
writes `data/apogee_weights.parquet` and `results/apogee_selection.json`.

<!-- script: apogee_selection.py show: data/apogee_parent.parquet, data/xp_andrae.parquet and the APOGEE DR17 allVisit file (scripts/build_apogee_parent.py, scripts/build_xp_andrae.py, scripts/fetch_targeting.py) -->

**Mass samples.** The masses of the stellar-mass chapter use the quality cuts of the selection functions, without the kinematic
cuts of the samples chapter, with the giant cut and {glue:text}`daVMassLo` < $v_{\rm los}$ < {glue:text}`daVMassHi` km/s
(calibrated velocities for BOSS). They hold {glue:text}`daNMassBoss` BOSS and {glue:text}`daNMassApo` APOGEE giants with positive
weights.

### Counts of the mass samples

`scripts/paper_giant_counts.py` counts the giants of the mass samples per [Fe/H] bin, without and with the weights, and compares
the weighted BOSS and APOGEE counts on a common parent, the *Gaia* stars in covered groups of both surveys, with a bootstrap over
HEALPix pixels. Its sample sizes enter {tab}`cuts`; the stellar-mass chapter shows the same comparison.

<!-- script: paper_giant_counts.py show: data/spec_giants.parquet, which is not in Zenodo bundle 1 (scripts/build_spec_giants.py); about 2 s -->

## Checks

### Completeness of the RR Lyrae, and the BOSS mock

{fig}`paper_selection_checks` adds three checks to {fig}`paper_selection`. The *Gaia* completeness of RRab falls at bright apparent
magnitudes in the inner {glue:text}`daGaiaBrightRho`°: these are blends, brighter than single RRab at the LMC distance, which
*Gaia* does not classify. On the sky, the *Gaia* completeness dips only at the centre, and OGLE's own completeness only at the edges
of its footprint. The BOSS mock keeps the real parent, selection groups, observed flags and the stars with XP metallicities,
assigns true [Fe/H] from the observed distribution, and lets spectra fail more often for faint metal-poor stars. The
selection-group weights alone lose up to {glue:text}`daGroupLoss` per cent of the stars below [Fe/H] = {glue:text}`daMockCutB`; the final weights
recover all bins within their scatter. When observation itself favours metal-poor stars by a factor of {glue:text}`daBiasFac`,
which violates the assumption of the weights, the recovered counts below {glue:text}`daMockCutA` are {glue:text}`daBiasHigh` per cent high: the test
in {fig}`paper_selection` (b) would show such a bias, and does not.

The same mock tests the coverage factor. Its truth is now every parent star, including those of uncovered groups, and a second
failure model makes metal-poor spectra fail twice as often within {glue:text}`daRingA`°. With the coverage factor,
{glue:text}`daNFill` realisations recover {glue:text}`daFillA` ± {glue:text}`daFillAErr`, {glue:text}`daFillB` ±
{glue:text}`daFillBErr` and {glue:text}`daFillC` ± {glue:text}`daFillCErr` of the stars below [Fe/H] = {glue:text}`daMockCutA` in the rings
0–{glue:text}`daRingA`, {glue:text}`daRingA`–{glue:text}`daRingB` and {glue:text}`daRingB`–{glue:text}`daRingOut`°; the low and
noisy inner value reflects the many central groups left with few spectra by the S/N cut. Two alternatives do no better. Filling each
gap from covered stars of the same region, $G$ bin and [M/H]$_{\rm XP}$ bin gives {glue:text}`daXpFillA` ±
{glue:text}`daXpFillAErr`, {glue:text}`daXpFillB` ± {glue:text}`daXpFillBErr` and {glue:text}`daXpFillC` ±
{glue:text}`daXpFillCErr`; treating the sparsely sampled groups as gaps leaves too few covered stars in the centre, so their parent
is filled from coarser cells and the ring 0–{glue:text}`daRingA`° gets {glue:text}`daSparseLo`–{glue:text}`daSparseHi` of its
stars. With the adopted coverage the total counts per ring come out at {glue:text}`daTotA`, {glue:text}`daTotB` and
{glue:text}`daTotC` in every realisation: the coarse selection groups, up to $N_{\rm side}$ = {glue:text}`daNsideLast`
({glue:text}`daPixCoarse`° pixels), straddle the rings and move weight between them, so regional masses are smeared by a few per
cent, while the total is not affected.

<!-- script: paper_selection_checks.py -->

**Figure (paper_selection_checks).** Checks of the selection functions. (a) *Gaia* completeness of OGLE's own RRab against
$I+0.6\approx G$ (mag; OGLE mean $I$ magnitude), in {glue:text}`daChkMag`-mag bins with at least {glue:text}`daSelNBin` stars, in three rings of $\rho$: 0–{glue:text}`daChkRA`° (blue),
{glue:text}`daChkRA`–{glue:text}`daChkRB`° (orange) and {glue:text}`daChkRB`–{glue:text}`daChkRC`° (green); binomial errors. (b, c) *Gaia* (b) and OGLE own-search (c) completeness in {glue:text}`daSelBin`° pixels of the
orthographic projection, $x$ (east, increasing to the left) and $y$ (north) in degrees, for pixels with at least {glue:text}`daChkNB` stars found by
both (colour scale {glue:text}`daChkVmin`–1, light = complete); grey line: OGLE-IV footprint. (d) BOSS mock on the real design: weighted counts over
true counts per bin of true [Fe/H] (dex), with the selection-group weights alone (grey) and the final weights (blue); mean and
standard deviation of {glue:text}`daNReal` realisations. `scripts/paper_selection_checks.py`, `scripts/boss_mock.py`

#### Running the BOSS mock

`scripts/boss_mock.py` runs the mock on the real BOSS design: the weight test with and without the observation bias, and the
coverage test with the two failure models. It writes `results/boss_mock.json`, from which panel (d) of
{fig}`paper_selection_checks` and the numbers above are taken.

<!-- script: boss_mock.py show: data/boss_weights.parquet (scripts/boss_selection.py); 140 mock realisations -->

### Coverage gaps and non-members

{fig}`paper_selection_gaps` shows where BOSS sampled the parent and checks assumptions A4 and A5 on the data. The fraction of the
parent with a good spectrum is high over most of the disc and lowest in the bar region and at the outer edge; the uncovered groups
lie mostly in the south-east. The parent stars of uncovered groups have the XP metallicities of the covered stars in the same rings
and $G$ bins: the fractions below [M/H]$_{\rm XP}$ = {glue:text}`daCutA`, {glue:text}`daCutB` and {glue:text}`daGapXpThree` are {glue:text}`daGapUOne`,
{glue:text}`daGapUTwo` and {glue:text}`daGapUThree` against {glue:text}`daGapCOne`, {glue:text}`daGapCTwo` and
{glue:text}`daGapCThree` (bootstrap over uncovered groups: ±{glue:text}`daGapEOne`, ±{glue:text}`daGapETwo` and
±{glue:text}`daGapEThree`). The deficit below {glue:text}`daGapXpThree` means the coverage factor puts slightly too many
metal-poor stars into the gaps, about {glue:text}`daDeficit` per cent of the most metal-poor bin. Non-members are
{glue:text}`daNonMemCellLo`–{glue:text}`daNonMemCellHi` per cent of the weighted good BOSS spectra, rising towards faint $G$, and
at most {glue:text}`daApoNonMemMax` per cent of the APOGEE ones.

<!-- script: paper_selection_gaps.py -->

**Figure (paper_selection_gaps).** Coverage of the BOSS parent. (a) Fraction of the parent with a good spectrum in HEALPix
$N_{\rm side}$ = {glue:text}`daNside` pixels (logarithmic colour scale, 10⁻³ to 1), in the orthographic projection $x$ (east,
increasing to the left), $y$ (north) in degrees; red outlines: pixels with most of their parent in uncovered groups. (b) Fraction of
the parent with [M/H]$_{\rm XP}$ below a threshold, against the threshold (dex; logarithmic $y$ axis), for the uncovered groups
(orange; band: 16th–84th percentiles of a bootstrap over groups) and for covered stars reweighted to the same ring and $G$ mix
(blue). (c) Weighted per cent of good spectra that are not LMC giants (not giants, or $v_{\rm los}$ outside
{glue:text}`daVMassLo`–{glue:text}`daVMassHi` km/s) per $G$ bin: BOSS in the rings 0–{glue:text}`daRingA`° (black), {glue:text}`daRingA`–{glue:text}`daRingB`° (blue) and {glue:text}`daRingB`–{glue:text}`daRingOut`° (orange),
and APOGEE in LMC1–17 (dashed grey line) and LMC18–35 (dotted grey line). `scripts/paper_selection_gaps.py`
