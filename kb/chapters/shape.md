---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# The shape of the old population

Every number on this page is written by `scripts/paper_text_shape.py` from the result tables (`results/macros_shape.tex`); the code
cells are the figure scripts themselves (`scripts/paper_shape_*.py`), step by step.

The kinematic model of the Letter puts every star in one thin plane. The old population is not thin: the vertical velocity
dispersion $\sigma_z$ of the most metal-poor giants is {glue:text}`shSigzMP` km/s, comparable to their rotation. Here 'old' and 'metal-poor' refer to the
iron abundance [Fe/H] (the logarithm of the iron-to-hydrogen ratio relative to the Sun's); the giants are divided into five bins of
[Fe/H], as in the Letter. Its three-dimensional shape tells us whether a thin-disc model can describe it. The same
old, metal-poor population is seen through two tracers. The giants have three-dimensional velocities; through the tensor virial
theorem they give the flattening that a steady population with these motions must have. The RR Lyrae have distances; their depth
along the line of sight gives the flattening directly, if their geometry can be modelled.

## Flattening implied by the motions

For a steady, axisymmetric population of tracer stars in a fixed potential $\Phi$, with $z$ along the symmetry axis, the tensor
virial theorem gives $2K_{ij}+W_{ij}=0$ with $K_{ij}=\frac{1}{2}\sum v_iv_j$ and $W_{ij}=-\sum x_i\,\partial\Phi/\partial x_j$,
where $x_i$ and $v_i$ ($i,j=x,y,z$) are the position and velocity
components of each tracer star and the sums run over the tracer's stars ([Binney & Tremaine 2008](https://ui.adsabs.harvard.edu/abs/2008gady.book.....B), section 4.8.3). The ratio
of its in-plane to its vertical components,

$$
\Omega_{\rm kin}\equiv\frac{\sum(v_x^2+v_y^2)}{2\sum v_z^2}=\frac{W_{xx}+W_{yy}}{2W_{zz}}\equiv\Omega_{\rm pot}(q),
$$

ties the kinematics on the left to the shape of the tracer on the right. Stars that rotate fast and move little up and down must lie
in a flattened distribution; the more vertical motion they have for their rotation, the rounder it must be. The axis ratio $q=c/a$, of the short
(vertical) axis $c$ to the long axis $a$ of the tracer's density, measures this: $q=1$ is a sphere and $q=0$ a thin disc. The old stars are a small part of the LMC's mass, so this is the theorem for
a tracer, not the self-gravitating relation between $v/\sigma$ and ellipticity. We realise each giant bin's kinematic model,
fitted to its velocities in the kinematics chapter (rotation curve, velocity ellipsoid and its radial gradient), on an oblate
spheroid with density $\propto K_0(m/h)$. Here $R$ and $z$ are the cylindrical radius and height in the disc frame,
$m^2=R^2+z^2/q^2$ is the spheroidal radius, and $K_0$ is the modified Bessel function of the second kind, chosen so that the face-on
profile is exponential, with the bin's scale length $h$ (fitted to its surface density of giants, weighted for the selection of the
SDSS-V BOSS spectra), and find the axis ratio $q$ at which the equation holds. The whole tracer is integrated, not only
the radii the fits constrain. The potential is not measured here, so we use three that bracket the LMC's circular speed $v_c$
({glue:text}`shVcLo`–{glue:text}`shVcHi` km/s at {glue:text}`shVcRIn`–{glue:text}`shVcROut` kpc). P1 is a cored logarithmic halo,
$\Phi=\tfrac12V^2\ln(R_{\rm c}^2+R^2+z^2/q_\Phi^2)$ with asymptotic circular speed $V$ = {glue:text}`shVOne` km/s, core radius $R_{\rm c}$ = {glue:text}`shAOne` kpc and
$q_\Phi=1$; P2 is the same halo flattened to $q_\Phi$ = {glue:text}`shQPhi`; P3 is a [Miyamoto & Nagai (1975)](https://doi.org/10.1093/pasj/27.4.533)
disc of mass {glue:text}`shMMN` × 10⁹ M☉, scale length $a_{\rm MN}$ = {glue:text}`shAMN` kpc and scale height $b_{\rm MN}$ = {glue:text}`shBMN` kpc in a spherical
cored logarithmic halo ($V$ = {glue:text}`shVThree` km/s, $R_{\rm c}$ = {glue:text}`shAThree` kpc). The uncertainty within each
potential comes from {glue:text}`shNDraw` joint posterior draws of the kinematic fit, each with $h$ drawn within its bootstrap range.

### Solving for the axis ratio

`scripts/shape_virial.py` does this for every giant bin and for the RR Lyrae. It takes the posterior samples of each group's
kinematic fit and its scale length $h$, realises the posterior-median model and each joint draw as a tracer, and finds $q$ in each
of P1, P2 and P3. Every case is run twice: once with the dispersions following the fitted radial gradient all the way in, and once
with them held constant inside the radius below which the fits do not constrain them. For the posterior-median model it also
records $2K/|W|$, the test of the depth of the potential described below. Its table, `results/shape_virial.csv`, feeds
{tab}`shape_q`, {fig}`paper_shape_virial` and the numbers of this section.

<!-- script: shape_virial.py show: the posterior samples of the kinematic fits (Zenodo bundle 1); one solution for q per posterior draw, group, potential and central floor -->

### The axis ratio of each giant bin

{tab}`shape_q` and {fig}`paper_shape_virial` give the result. The most metal-poor giants need $q$ = {glue:text}`shQMPOne`, {glue:text}`shQMPTwo` and {glue:text}`shQMPThree` in P1, P2 and
P3 (posterior 16th–84th percentiles ±{glue:text}`shQHalf`); $q$ falls steadily with metallicity, to {glue:text}`shQMRLo`–{glue:text}`shQMRHi` above [Fe/H] = {glue:text}`shFehMR`. Even the
oldest, most metal-poor stars therefore form a thick, rotating disc about half as thick as it is wide, not a round halo, and the
younger, more metal-rich groups are flatter still. The choice of potential moves $q$ by {glue:text}`shDqPotMP` in the most metal-poor bin, about
{glue:text}`shDqRatio` times its posterior half-width, and by up to {glue:text}`shDqPotMR` in the most metal-rich, and is the main uncertainty. It also fixes the
depth of the potential, which the ratio in the equation does not test: the trace of the theorem, $2K/|W|$ with $K=\sum_iK_{ii}$ and
$W=\sum_iW_{ii}$, which equals 1 for a tracer in equilibrium, is {glue:text}`shBalLo`–{glue:text}`shBalHi` in P1
and P2 for the bins below [Fe/H] = {glue:text}`shFehMR` ({glue:text}`shBalMR` above), but {glue:text}`shBalThreeLo`–{glue:text}`shBalThreeHi` in P3, which is too shallow for the measured
motions.

```{code-cell} python
:tags: [remove-input]
import pandas as pd
# Virial flattening q per giant bin and potential, without the constant-dispersion floor inside R_FLOOR (r_floor = 0; scripts/shape_virial.py)
v = pd.read_csv('results/shape_virial.csv')
v = v[(v.r_floor == 0) & v.population.str.startswith('giants')]
v['q (16th–84th)'] = v.apply(lambda r: f'{r.q:.2f} ({r.q_p16:.2f}–{r.q_p84:.2f})', axis=1)
# One row per population, one column per potential; last column: virial ratio 2K/|W| in each potential
t = v.pivot(index='population', columns='potential', values='q (16th–84th)').reindex(v.population.unique())
t['2K/|W| (P1, P2, P3)'] = v.groupby('population', sort=False).balance.apply(lambda b: ', '.join(f'{x:.2f}' for x in b))
t.index.name = None; t.columns.name = None
t
```

**Table (shape_q).** Axis ratio $q=c/a$ implied by the kinematics of each giant bin: posterior-median model, with the 16th–84th
percentiles of {glue:text}`shNDraw` joint draws in parentheses, in the three potentials P1, P2 and P3, and $2K/|W|$ of the median model at that
$q$ (`results/shape_virial.csv`, written by `scripts/shape_virial.py`).

<!-- script: paper_shape_virial.py -->

**Figure (paper_shape_virial).** Axis ratio $q$ implied by the motions of each giant bin, against its median [Fe/H] (dex), in the
potentials P1 (black), P2 (blue) and P3 (red); bars: 16th–84th percentiles of the posterior. The spread between the potentials, not the bars, is the real uncertainty: the
most metal-poor stars need $q\approx0.5$ and the flattening grows with metallicity. Grey: the RR Lyrae (median [Fe/H] = {glue:text}`shFehRrl`),
$q$ from their near-to-far distance gradient at the inclination of the metal-poor giants (dark: bootstrap 16th–84th percentiles;
light: a coherent ±{glue:text}`shDipole` mag distance error across the line of nodes; see {fig}`paper_shape_rrl`). `scripts/paper_shape_virial.py`

We tested the procedure on simulated analogues. For the two metal-poor bins of the {glue:text}`shNSimGal` zoom-in simulated galaxies with all particle types,
from Auriga ([Grand et al. 2017](https://doi.org/10.1093/mnras/stx071)) and FIRE-2 ([Hopkins et al. 2018](https://doi.org/10.1093/mnras/sty1690)), the same steps (the kinematic model fitted to the mock giants, realised as a $K_0$ spheroid) recover the true axis
ratio of the particles ({glue:text}`shQTrueLo`–{glue:text}`shQTrueHi`) to {glue:text}`shDqSim` (16th–84th percentiles {glue:text}`shDqSimLo` to {glue:text}`shDqSimHi`, worst {glue:text}`shDqSimMin`) when the
analogue's own potential is used. With P1, P2 or P3 instead the offsets are {glue:text}`shDqSimOne`, {glue:text}`shDqSimTwo` and {glue:text}`shDqSimThree`: the method is
accurate, and its error is the potential. Holding the dispersions constant inside {glue:text}`shRFloor` kpc, where the fits do not constrain them,
changes $q$ by at most {glue:text}`shDqFloor`. The bisymmetric ($m=2$) streaming of the fits, which the realisation omits, adds at most
{glue:text}`shStreamKE` per cent to the in-plane kinetic energy and lowers $q$ by at most {glue:text}`shDqStream` (posterior-median
models of the giant bins, all three potentials). For the RR Lyrae the theorem gives no answer: their $\sigma_z$ is not measured, and in most posterior draws no oblate
tracer has their kinematics.

## The depth of the RR Lyrae

The RR Lyrae offer two geometric clues to the same shape: how much nearer one side of the LMC is than the other, and how oval the
population looks on the sky. Each clue depends on both the tilt and the flattening, so each alone allows a range of shapes; the tilt
from the motions picks one. The RR Lyrae at angular distances {glue:text}`shRhoIn`° < ρ < {glue:text}`shRhoOut`° from the LMC centre ({glue:text}`shNRrl` stars) give both clues. We count them in {glue:text}`shNCells` sky cells
({glue:text}`shNRings` rings, {glue:text}`shNSect` sectors) and take the median distance modulus of each cell; the scatter of the moduli, and with it the poorly
known per-star noise of the distances (from the period–Wesenheit relation; the Wesenheit magnitude is a combination of $V$ and $I$
that is insensitive to dust), does not enter. For an oblate body whose symmetry axis is inclined by $i$ to the line of sight ($i=0$
face-on), with its line of nodes (where its mid-plane crosses the sky plane) at position angle $\theta$ (east of north), these cells
measure two numbers:
the axis ratio of its outline on the sky, $q_{\rm sky}^2=\cos^2i+q^2\sin^2i$, from the counts, and the gradient of its mean depth
across the line of nodes, $g=\sin i\cos i\,(1-q^2)/q_{\rm sky}^2$, from the medians (a thin disc has $g=\tan i$, a sphere $g=0$).
The RR Lyrae give $q_{\rm sky}$ = {glue:text}`shQSky` and $g$ = {glue:text}`shG`. Each number alone allows a curve of $(i,q)$, and together they would fix the
shape at $i$ = {glue:text}`shIncFree`°, $q$ = {glue:text}`shQFree`, with the line of nodes at {glue:text}`shThetaFree`°.

That solution is not the tilt of the LMC. The oblate model fits the cells poorly ($\chi^2$ = {glue:text}`shChiCounts` for the counts and {glue:text}`shChiMed`
for the medians, each over {glue:text}`shNCells` cells), and what it leaves is structured ({fig}`paper_shape_rrl_residuals`): the counts are lopsided, with an
amplitude $A_1$ of the first azimuthal harmonic ($m=1$, relative to the mean count of the ring) of {glue:text}`shAOneOutLo`–{glue:text}`shAOneOutHi` per cent at {glue:text}`shRhoOutLo`–{glue:text}`shRhoOut`° (an excess to the north-east and a deficit, with Pearson
residuals $(n-\lambda)/\sqrt\lambda$ of the observed counts $n$ about the model counts $\lambda$ down to {glue:text}`shPearsonMin`, to the west and south-west, the side on which the outer disc is truncated;
[Mackey et al. 2018](https://doi.org/10.3847/2041-8213/aac175)), and the OGLE-IV survey footprint covers at least {glue:text}`shCoverage` per cent of every ring. Above
all, no oblate body inclined by the {glue:text}`shIncOldInt`° of the motions can look as elongated as the RR Lyrae do, since
$q_{\rm sky}\geq\cos i$ = {glue:text}`shCosIncOld`. The elongation on the sky is an elongation in the plane, as found for the LMC's stars
([van der Marel 2001](https://doi.org/10.1086/323100)) and for its RR Lyrae (a triaxial ellipsoid of 6.5:4.6:3.7 kpc with
$i=22\pm4°$ and $\theta=167\pm7°$; [Cusano et al. 2021](https://doi.org/10.1093/mnras/stab901)); it also turns the density's line
of nodes ({glue:text}`shThetaFree`°) away from the kinematic one ({glue:text}`shThetaOld`°). A mock at the geometry of the metal-poor giants ($i$ = {glue:text}`shIncOld`°,
$\theta$ = {glue:text}`shThetaOld`°) with an in-plane axis ratio $b/a$ = {glue:text}`shMockBA` and $c/a$ = {glue:text}`shMockC`, fitted with the oblate model, gives
$i$ = {glue:text}`shMockInc`°, $q$ = {glue:text}`shMockQ`, $\theta$ = {glue:text}`shMockTheta`°, $q_{\rm sky}$ = {glue:text}`shMockQSky` and $g$ = {glue:text}`shMockG`, close to the data (arrow in {fig}`paper_shape_rrl`).

The depth gradient is the part that survives. Held at the inclination of the motions, the medians give $q$ = {glue:text}`shQDepth` (bootstrap
16th–84th percentiles {glue:text}`shQDepthLo`–{glue:text}`shQDepthHi`). A coherent distance error of ±{glue:text}`shDipole` mag across the line of nodes, the size of the
cell-to-cell scatter the model leaves ({glue:text}`shScatter` mag), moves this to {glue:text}`shQDipLo`–{glue:text}`shQDipHi`. The RR Lyrae therefore confirm that the old
population is thick and flattened, with $q$ below about {glue:text}`shQDipHi`, consistent with the {glue:text}`shQOldLo`–{glue:text}`shQOldHi` its motions imply, but they do
not measure $q$ better than that. {fig}`paper_shape_rrl` shows the two clues in the plane of $i$ and $q$.

### Fitting the sky cells

`scripts/shape_rrl_cells.py` makes the fits of this section. It selects the RR Lyrae with a Wesenheit distance inside the footprint,
counts them in the sky cells with the median distance modulus of each cell, and fits the inclined oblate model with all parameters
free. It then refits with the inclination held at each value of a grid, which gives the best $q$ at every inclination (the 'ridge'),
and reads the ridge at the inclination of the motions. At the inclination of the metal-poor giants it repeats the fit for bootstrap
resamples of the stars and with the coherent distance error added with either sign. Its two tables, `results/shape_rrl_cells.csv` and
`results/shape_rrl_depth.csv`, are read by the three figure scripts of this chapter.

<!-- script: shape_rrl_cells.py show: the RR Lyrae catalogue data/rrab.parquet (Zenodo bundle 1); repeated fits of the sky cells along a grid of inclinations and for bootstrap resamples -->

### Elongated mocks

`scripts/shape_rrl_triaxial.py` tests whether an elongation in the plane explains the free fit. It draws mock RR Lyrae, as many as
the data have in the footprint, from a $K_0$ body squashed in its plane to an axis ratio $b/a$ (the in-plane short axis $b$ to the
long axis $a$) and turned by an angle from the line of nodes, places them at the geometry of the metal-poor giants, and fits their
cells with the oblate model as for the data. The oblate controls ($b/a=1$) are drawn in the same way. Its table,
`results/shape_rrl_triaxial.csv`, gives the arrow of {fig}`paper_shape_rrl` and the mocks of the checks below.

<!-- script: shape_rrl_triaxial.py show: results/shape_rrl_cells.csv (scripts/shape_rrl_cells.py); oblate fits of mock RR Lyrae catalogues of the size of the data -->

### The inclination and flattening of the old population

<!-- script: paper_shape_rrl.py -->

**Figure (paper_shape_rrl).** Inclination $i$ (deg) and axis ratio $q$ of the old population. Curves: the $(i,q)$ of an inclined oblate body with the RR Lyrae's sky outline
($q_{\rm sky}$ = {glue:text}`shQSky`, blue) and near-to-far depth gradient ($g$ = {glue:text}`shG`, red); black dot: their crossing. Grey bands: the inclination
from the motions of the metal-poor giants (dark) and of the RR Lyrae proper motions (light); open circle: $q$ from the depth gradient
at the giants' inclination. Dashed: $q$ implied by the motions of the metal-poor giants (16th–84th percentiles over P1–P3). Arrow: a
mock with an in-plane elongation ($b/a$ = {glue:text}`shMockBA`, $c/a$ = {glue:text}`shMockC`) at the giants' geometry, from its true $(i,c/a)$ to its oblate
fit. If the population were round in its plane, its shape would lie where the curves cross. At the inclination of the motions,
however, no such body is as oval on the sky as the RR Lyrae (the blue curve does not reach it), so the crossing measures their
elongation in the plane, as the arrow shows, not their tilt. What survives is the red curve: at the motions' tilt the depth
gradient gives $q\approx$ {glue:text}`shQDepth`, consistent with the flattening the motions require. `scripts/paper_shape_rrl.py`

## The size of each population

The kinematic modelling also measures how far each population extends. For each giant [Fe/H] bin, `scripts/kin_vc_drift.py`
(chapter 7) counts the BOSS giants in rings of in-plane radius $R$, each star weighted for the selection of the spectra, the sky
coverage and the chance that it belongs to the Milky Way, and fits an exponential surface density $\Sigma(R)\propto e^{-R/h}$,
with scale length $h$, between {glue:text}`ltHFitLo` and {glue:text}`ltHFitHi` kpc; inside {glue:text}`ltHFitLo` kpc the bar
dominates. The RR Lyrae are counted the same way. The scale length grows from {glue:text}`ltHMR` kpc for the most metal-rich giants
to {glue:text}`ltHMP` kpc for the most metal-poor ({glue:text}`ltHRrl` kpc for the RR Lyrae): the oldest stars are the most extended
({fig}`fig_sizes`). The half-mass radius of an exponential disc is $R_e = 1.68\,h$, which gives {glue:text}`ltReMR`,
{glue:text}`ltReMP` and {glue:text}`ltReRrl` kpc. These values assume that the exponential continues inside
{glue:text}`ltHFitLo` kpc, which the data do not support for every bin: the surface density of the most metal-poor giants there
falls below its exponential, and that of the most metal-rich giants falls faster than an exponential at large radii. Whether the
inner deficit is real or a result of the selection in the crowded bar region we have not tested.

All these are present-day sizes. Heating and radial migration since the stars formed, and the LMC's interactions with the SMC and
the Milky Way, may have made the old disc larger than it was at birth.

**Figure (fig_sizes).** (a) Surface density of each giant [Fe/H] bin (circles) and of the RR Lyrae (black squares) against the
in-plane radius $R$, each normalised to its fitted exponential at $R$ = {glue:text}`ltRSig` kpc; lines: the exponentials fitted between
{glue:text}`ltHFitLo` and {glue:text}`ltHFitHi` kpc, with their scale lengths in the key; grey: the bar region, not fitted.
(b) The scale lengths (filled, with 16th–84th bootstrap percentiles) and the half-mass radii $R_e=1.68\,h$ they imply if the
exponential continued inwards (open), against the median [Fe/H] of each sample.

<!-- script: fig_sizes.py -->

## The shape and the kinematic model

A spheroid with $q\approx0.5$ and $h$ = {glue:text}`shHMP` kpc has a vertical extent comparable to the radii we fit: the rms height of its stars
is $q\sqrt3\,h\approx$ {glue:text}`shRmsMP` kpc, thicker than the {glue:text}`shHThick`-kpc disc of the tests of the kinematic model. We therefore drew a mock
with this shape ('hot spheroid': $q$ = {glue:text}`shQSph` with the {glue:text}`shRD`-kpc scale length of the other mocks, rms height {glue:text}`shRmsSph` kpc;
radial, azimuthal and vertical velocity dispersions
$(\sigma_R,\sigma_\phi,\sigma_z)$ = ({glue:text}`shSigR`, {glue:text}`shSigPhi`, {glue:text}`shSigZ`) km/s, asymmetric drift, inclination {glue:text}`shIncMock`°). With proper motions and
line-of-sight velocities the thin-disc model recovers its dispersions to within {glue:text}`shDSigMock` km/s and its inclination to {glue:text}`shDIncMock`°
(line of nodes {glue:text}`shDThetaMock`° low, as for the {glue:text}`shHThick`-kpc disc). The thickness of the old population therefore does not bias its
kinematics, and the lower inclination of the metal-poor giants ({glue:text}`shIncOldInt`° against {glue:text}`shIncRich`°) is not a projection of their
thickness. With proper motions alone the same mock gives $\sigma_z$ = {glue:text}`shSigzPM` ± {glue:text}`shSigzPMErr` km/s for {glue:text}`shSigZ` and an inclination of
{glue:text}`shIncPM`°: the proper-motion inclination of the RR Lyrae ({glue:text}`shIncRrlPM`°) is not a reliable tilt for a thick population, and we use that
of the metal-poor giants.

## Summary of the shape

1. To support their rotation and dispersions in a steady state, the oldest, most metal-poor stars ([Fe/H] < {glue:text}`shFehMP`) must form a
   thick, rotating disc with $q\approx0.5$ ({glue:text}`shQOldLo`–{glue:text}`shQOldHi` over the three potentials); the flattening grows with metallicity, to
   $q$ = {glue:text}`shQMRLo`–{glue:text}`shQMRHi` above [Fe/H] = {glue:text}`shFehMR`. The procedure recovers the shapes of simulated analogues to {glue:text}`shDqSim` in their own
   potential; the uncertainty is the LMC's potential.
2. The RR Lyrae, the same population seen through distances, agree: at the tilt of the motions their near-to-far distance gradient
   gives $q\approx$ {glue:text}`shQDepth` ({glue:text}`shQDipLo`–{glue:text}`shQDipHi` with plausible distance errors).
3. On their own the RR Lyrae suggest a tilt of {glue:text}`shIncFree`°, but this is an illusion: their oval outline on the sky comes from an
   elongation of the population in its plane, which a round-in-plane model mistakes for tilt.
4. The thin-disc kinematic model describes a population this thick without bias when line-of-sight velocities are available;
   proper motions alone underestimate its tilt.

## Checks

### The flattening of simulated analogues

The Auriga and FIRE-2 zooms contain all particle types, so their potential is known. For each, we compute the axisymmetrised
potential of all particles within {glue:text}`shRMaxSim` kpc in the disc frame of the stars ({glue:text}`shNSrc` particles drawn by mass at random azimuths, softened
gravity summed on a grid in $R$ and $|z|$; on a [Hernquist (1990)](https://doi.org/10.1086/168845) sphere this reproduces the
analytic $W_{xx}/W_{zz}$ of a tracer with $q$ = {glue:text}`shHernQ` to {glue:text}`shHernTol` per cent, `tests/test_virial.py`). For the two metal-poor bins we take the
kinematic model fitted to the analogue's mock giants, realise it on a $K_0$ spheroid with the bin's scale length, and solve the
equation in that potential, exactly as for the LMC. The true axis ratio is that of the bin's particles,
$q=(2\langle z^2\rangle/\langle R^2\rangle)^{1/2}$ (angle brackets: mean over the bin's star particles) at $R$ < {glue:text}`shCylR` kpc, $|z|$ < {glue:text}`shCylZ` kpc. Over the {glue:text}`shNSimBin` bins (true $q$ = {glue:text}`shQTrueLo`–{glue:text}`shQTrueHi`)
the recovered $q$ is low by {glue:text}`shDqSim` (16th–84th percentiles {glue:text}`shDqSimLo` to {glue:text}`shDqSimHi`; range {glue:text}`shDqSimMin` to {glue:text}`shDqSimMax`). In the fixed
potentials P1, P2 and P3 the offsets are {glue:text}`shDqSimOne` ({glue:text}`shDqSimOneLo` to {glue:text}`shDqSimOneHi`), {glue:text}`shDqSimTwo` ({glue:text}`shDqSimTwoLo` to {glue:text}`shDqSimTwoHi`) and
{glue:text}`shDqSimThree` ({glue:text}`shDqSimThreeLo` to {glue:text}`shDqSimThreeHi`). We do not test the particles themselves against the theorem: their metallicities are
mapped on the LMC scale only inside the cone of the observed footprint, so a bin is a truncated tracer.

#### The test script

`scripts/shape_virial_sims.py` runs this test. For each Auriga and FIRE-2 analogue it computes the potential of all particles in the
disc frame of the stars; for each of the two metal-poor bins it measures the true axis ratio and the scale length of the bin's
particles, weighted by the giants they would contribute to the BOSS sample, and solves for $q$ with the kinematic model fitted to
the analogue's mock giants, in the analogue's own potential and in P1, P2 and P3. Its table, `results/shape_virial_sims.csv`, gives
the numbers of this check.

<!-- script: shape_virial_sims.py show: the Auriga and FIRE-2 snapshots with all particle types (scripts/fetch_auriga.py, scripts/fetch_fire2.py; about 120 GB and 8 GB) and results/sims_observed.csv from scripts/sims_observe.py (about 13 h) -->

### The RR Lyrae cells

{fig}`paper_shape_rrl_residuals` shows what the oblate model leaves in the RR Lyrae cells, for its best fit and at the inclination of the motions.
The counts have Pearson residuals from {glue:text}`shPearsonMin` to {glue:text}`shPearsonMax` in coherent patches; their $m=1$ amplitude is {glue:text}`shAOneIn` at
{glue:text}`shRhoIn`–{glue:text}`shRhoInHi`° and {glue:text}`shAOneOutLoF`–{glue:text}`shAOneOutHiF` at {glue:text}`shRhoOutLo`–{glue:text}`shRhoOut`°, where the excess lies to the north-east (position angle
{glue:text}`shPAOneLo`–{glue:text}`shPAOneHi`°) and the deficit to the west and south-west. The medians scatter by {glue:text}`shScatter` mag about the model against their
errors of {glue:text}`shMedErr` mag. To test whether an elongation in the plane explains the oblate fit, we drew mocks at the geometry of the
metal-poor giants with $c/a$ = {glue:text}`shCsOne` (as Cusano et al. 2021) or {glue:text}`shCsTwo`, squashed in the plane to $b/a$ = {glue:text}`shBAOne` or {glue:text}`shBATwo` with the
major axis at {glue:text}`shPhiMin`–{glue:text}`shPhiMax`° from the line of nodes (in steps of {glue:text}`shPhiStep`°), and fitted them like the data. With the major axis within
{glue:text}`shPhiStep`° of the line of nodes the oblate fits move to $i$ = {glue:text}`shNearIncLo`–{glue:text}`shNearIncHi`° (the upper bound), $q$ = {glue:text}`shNearQLo`–{glue:text}`shNearQHi` and
$q_{\rm sky}$ = {glue:text}`shNearQSkyLo`–{glue:text}`shNearQSkyHi`, the fitted line of nodes following the major axis; with $b/a$ = {glue:text}`shMockBA`, $c/a$ = {glue:text}`shMockC` and
the major axis at {glue:text}`shMockPhi`° they give $i$ = {glue:text}`shMockIncOne`°, $q$ = {glue:text}`shMockQ`, $\theta$ = {glue:text}`shMockTheta`°, $q_{\rm sky}$ = {glue:text}`shMockQSkyThree`,
$g$ = {glue:text}`shMockGThree` (data: {glue:text}`shIncFreeOne`°, {glue:text}`shQFree`, {glue:text}`shThetaFree`°, {glue:text}`shQSkyThree`, {glue:text}`shGThree`). With the major axis at {glue:text}`shFarPhiLo`–{glue:text}`shFarPhiHi`° from the line of
nodes the fits stay at $i$ = {glue:text}`shFarIncLo`–{glue:text}`shFarIncHi`° with $q$ = {glue:text}`shFarQLo`–{glue:text}`shFarQHi`, except for the most elongated and thickest mock
($b/a$ = {glue:text}`shBAOne`, $c/a$ = {glue:text}`shCsOne`), which goes to the upper bound at {glue:text}`shFarPhiLo` and {glue:text}`shFarPhiHi`° as well. The oblate controls ($b/a=1$) come back at
$i$ = {glue:text}`shObIncOne` and {glue:text}`shObIncTwo`° with $q$ = {glue:text}`shObQOne` and {glue:text}`shObQTwo` for a true {glue:text}`shIncOld`° with {glue:text}`shCsOne` and {glue:text}`shCsTwo`: they slide along the curve
of constant $g$, which the counts constrain only weakly.

<!-- script: paper_shape_rrl_residuals.py -->

**Figure (paper_shape_rrl_residuals).** Residuals of the oblate model in the RR Lyrae sky cells, for the best fit (a, b) and at the
inclination of the metal-poor giants (c, d): (a, c) Pearson residuals of the counts, $(n-\lambda)/\sqrt\lambda$; (b, d) median
distance modulus minus model (mag). Solid line: the fit's line of nodes; dashed: the kinematic one. East to the left.
`scripts/paper_shape_rrl_residuals.py`
