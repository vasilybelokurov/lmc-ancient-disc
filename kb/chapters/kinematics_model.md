---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Kinematics: velocity field and disc model

Every number on this page is written by `scripts/paper_text_kin_model.py` from the result tables and the posterior samples of the
kinematic fits (`results/macros_kinmodel.tex`); the code cells are the figure scripts themselves (`scripts/paper_kin_residuals.py`,
`scripts/paper_kin_ellipsoid_mock.py`, `scripts/paper_kin_corner.py`), step by step. The fits themselves
(`scripts/fit_kinematics_mcmc.py`) take hours and are not rerun here; their results are read from `results/`. The fit script and the
scripts of the checks and mocks are shown step by step where they enter the story, without being run.

The Large Magellanic Cloud (LMC) is not a disc in equilibrium. Its outer disc is lopsided and truncated at different radii in
different directions, its bar is offset from the centre of the outer isophotes (the contours of constant surface brightness), and
the Small Magellanic Cloud (SMC) and the Milky Way perturb it today ([van der Marel 2001](https://doi.org/10.1086/323100);
[Choi et al. 2018](https://doi.org/10.3847/1538-4357/aae083); [Belokurov & Erkal 2019](https://doi.org/10.1093/mnrasl/sly178);
[Cullinane et al. 2022a](https://doi.org/10.1093/mnras/stab3350); [Cullinane et al. 2022b](https://doi.org/10.1093/mnras/stac733)).
A rigid disc model will not describe every star, and we do not try to build one. We ask how strongly each metallicity group of the
old population is supported by rotation, and we answer with the simplest disc model that describes all groups in the same way: a
thin disc whose stars have a mean rotation and a velocity ellipsoid (the three-dimensional Gaussian distribution of the stars'
random velocities about the mean) aligned with the disc. This page shows the measured velocity field, then the model and what it
leaves in the data, and, in the checks at the end, the tests of the model on mocks and the simpler model of our first fits. The
rotational support of each group, the centre and geometry, and the sensitivity to the metallicity errors and to the model are the
subject of chapter 7.

## The measured velocity field

The old population is seen through two tracers. The red giants have proper motions (angular motions on the sky, in mas/yr) from
[Gaia DR3](https://doi.org/10.1051/0004-6361/202243940) and line-of-sight velocities from the spectroscopic surveys
[APOGEE](https://doi.org/10.3847/1538-3881/aa784d) (Data Release 17; [Abdurro'uf et al. 2022](https://doi.org/10.3847/1538-4365/ac4414))
or SDSS-V BOSS ([Kollmeier et al. 2026](https://doi.org/10.3847/1538-3881/ae0576); Data Release 20,
[SDSS Collaboration 2026](https://doi.org/10.48550/arXiv.2607.26149)); a star observed by both surveys keeps its APOGEE values. The
samples are built in the data chapters. Each giant is placed in one of five bins of its measured iron abundance [Fe/H] (the
logarithm of the iron-to-hydrogen ratio relative to the Sun's): below {glue:text}`kmFehA`, {glue:text}`kmFehA` to {glue:text}`kmFehB`,
{glue:text}`kmFehB` to {glue:text}`kmFehC`, {glue:text}`kmFehC` to {glue:text}`kmFehD` and above {glue:text}`kmFehD`. The RR Lyrae
(old pulsating stars of type ab, LMC members from the OGLE-IV collection, [Soszyński et al. 2016](https://doi.org/10.48550/arXiv.1606.02727),
and the Gaia DR3 catalogue, [Clementini et al. 2023](https://doi.org/10.1051/0004-6361/202243964)) have Gaia proper motions only.
We use the stars at angular distances {glue:text}`kmRhoIn`° < ρ < {glue:text}`kmRhoOut`° from the LMC centre of
[van der Marel & Kallivayalil (2014)](https://doi.org/10.1088/0004-637X/781/2/121) (hereafter vdMK14; right ascension and declination
(α, δ) = ({glue:text}`kmRaVdm`°, {glue:text}`kmDecVdm`°)): the inner limit excludes the bar, the outer one the sparse periphery. These
samples hold {glue:text}`kmNG`, {glue:text}`kmNGb`, {glue:text}`kmNGc`, {glue:text}`kmNGd` and {glue:text}`kmNGe` giants, from
metal-poor to metal-rich, and {glue:text}`kmNRrl` RR Lyrae; the RR Lyrae kinematics we quote are measured at
{glue:text}`kmRhoOuter`° < ρ < {glue:text}`kmRhoOut`° ({glue:text}`kmNRrlOut` stars; chapter 7).

From each velocity we subtract the bulk motion of the LMC as seen at the star's position, which removes the systemic motion and its
perspective effects, and we convert the proper motions to velocities (km/s) at the distance where the star's line of sight meets the
disc plane. We split the velocity across the sky into a tangential component $v_t$, pointing 90° east of the direction away from the
centre, and a radial component $v_r$, pointing away from the centre. These velocities depend on the frame: the centre about which
$v_t$ and $v_r$ are taken, the systemic motion subtracted and the disc plane onto which the stars are placed. We draw all maps and
profiles in one frame, that of the joint fit of all groups (stage 2 of the model below, {tab}`kin_shared`), so that a difference
between two columns of {fig}`paper_kin_maps_vt_ellipsoid`–{fig}`paper_kin_maps_vlos_ellipsoid` is a difference between the stars and
not between the frames; had each group been drawn in the frame of its own fit, the centre of a giant bin would move by up to
{glue:text}`kmDCen`° and its inclination by up to {glue:text}`kmDInc`°. The numbers we quote for each group come instead from its own
fit (stage 1); the frame of vdMK14 is compared in the section on what the models leave in the data.

The top rows of {fig}`paper_kin_maps_vt_ellipsoid`, {fig}`paper_kin_maps_vr_ellipsoid` and {fig}`paper_kin_maps_vlos_ellipsoid` map
$v_t$, $v_r$ and the line-of-sight velocity $v_{\rm los}$ of each giant bin and of the RR Lyrae, and {fig}`paper_kin_data_profiles_ellipsoid`
plots the same velocities against position angle (the angle on the sky about the centre, measured from north through east). All
groups rotate in the same sense, clockwise on the sky, so $v_t<0$ (at {glue:text}`kmProfIn`° < ρ < {glue:text}`kmProfOut`° it ranges over
position angle from −{glue:text}`kmVtSlow` to −{glue:text}`kmVtFast` km/s in the harmonic fits described below; a negative residual in $v_t$ therefore means
faster rotation than the model), and the line-of-sight velocity is largest at the two ends of the line of nodes (the line where the
disc plane crosses the plane of the sky), with opposite signs, and changes sign across the minor axis (perpendicular to the line of
nodes). The metal-rich giants rotate faster than the metal-poor giants and the RR Lyrae at almost every position angle
({fig}`paper_kin_data_profiles_ellipsoid`a). We describe the variation of a velocity $v$ with position angle $\psi$ by a harmonic fit
to the stars, $v=c_0+\sum_m A_m\cos[m(\psi-\psi_m)]$, where $A_m$ is the amplitude of the $m$-th term and $\psi_m$ the position angle
of its maximum: $m=1$ is a lopsided (one-sided) pattern, $m=2$ a two-fold one. Both $v_t$ and $v_r$ carry an $m=2$ term of
{glue:text}`kmDataATwoLo`–{glue:text}`kmDataATwoHi` km/s (harmonic fits to the stars at {glue:text}`kmProfIn`° < ρ <
{glue:text}`kmProfOut`°) with the same phase in every group, as the projection of an inclined rotating disc does; about the fitted
centre the $m=1$ term of $v_t$ is only {glue:text}`kmDataAOneLo`–{glue:text}`kmDataAOneHi` km/s.

The script below computes the residuals of every star about each model, the maps, the profiles and the harmonic fits. It is run
with the argument `ellipsoid` (the velocity-ellipsoid model of the next section); without it, it draws the same figures for the
sky-aligned model of the checks.

```{code-cell} python
import sys
sys.argv = ['paper_kin_residuals.py', 'ellipsoid']   # as: python scripts/paper_kin_residuals.py ellipsoid
```

<!-- script: paper_kin_residuals.py -->

**Figure (paper_kin_maps_vt_ellipsoid).** Tangential velocity $v_t$ (km/s; 90° east of the direction away from the centre) of the
old population on the sky. Columns: giants in the five bins of measured [Fe/H], metal-poor (left) to metal-rich, and the RR Lyrae
(right). Rows: (top) data: proper motion minus the bulk motion of the LMC at the star's position, converted to km/s at the distance
of the disc plane; (middle) the joint Level-2 model with the velocity ellipsoid (model section); (bottom) significance of the
residual (data minus model) in each cell, $z=\sqrt{n}\,{\rm median}(\Delta v_i/\tilde\sigma_i)/\sqrt{\pi/2}$ for the $n$ stars of a
cell, where $\Delta v_i$ is the residual of star $i$ and $\tilde\sigma_i$ the scatter the model predicts for it in this component:
its velocity ellipsoid projected on the component plus the scaled measurement error. If the model is right, $z$ follows
approximately a unit normal distribution in every cell (the scatter $\tilde\sigma_i$ omits the outlier component, and stars with
residuals beyond {glue:text}`kmClip` km/s, plus three times their own error for the proper-motion components, are dropped). Colour
scales: right of the last column (top bar for the data and model rows, bottom bar for $z$). Centre, systemic motion and disc plane of
the joint fit ({tab}`kin_shared`); the RR Lyrae rotation curve, ellipsoid, error scale $k$ and outlier fraction $f$ are refitted in
that frame at {glue:text}`kmRhoOuter`° < ρ < {glue:text}`kmRhoOut`°, so inside {glue:text}`kmRhoOuter`°, where Gaia underestimates its
errors more, their $|z|$ is larger. Median per {glue:text}`kmCell`° cell with at least {glue:text}`kmNMin` stars, at
{glue:text}`kmRhoIn`° < ρ < {glue:text}`kmRhoOut`° from the vdMK14 centre. Axes: offsets east (increasing to the left) and north (deg)
of the fitted centre, (α, δ) = ({glue:text}`kmRaE`°, {glue:text}`kmDecE`°); +: the vdMK14 centre, the origin of ρ used to select the
stars; dashed line: the fitted line of nodes. `scripts/paper_kin_residuals.py ellipsoid`

**Figure (paper_kin_maps_vr_ellipsoid).** As {fig}`paper_kin_maps_vt_ellipsoid` for the radial component $v_r$ of the velocity
across the sky (km/s, positive away from the centre). `scripts/paper_kin_residuals.py ellipsoid`

**Figure (paper_kin_maps_vlos_ellipsoid).** As {fig}`paper_kin_maps_vt_ellipsoid` for the line-of-sight velocity $v_{\rm los}$ (km/s,
minus the bulk motion of the LMC along the star's line of sight), giants only (the RR Lyrae have no line-of-sight velocities).
`scripts/paper_kin_residuals.py ellipsoid`

**Figure (paper_kin_data_profiles_ellipsoid).** The data of {fig}`paper_kin_maps_vt_ellipsoid`–{fig}`paper_kin_maps_vlos_ellipsoid`
against position angle (deg, north through east, about the fitted centre) at {glue:text}`kmProfIn`° < ρ < {glue:text}`kmProfOut`°:
(a) $v_t$, (b) $v_r$, (c) $v_{\rm los}$ (km/s), median per {glue:text}`kmPaBin`° bin for each group (colours, legend above: giants
from red, metal-poor, to blue, metal-rich; purple, RR Lyrae). Dotted vertical lines: the line of nodes; dash-dotted: the minor axis.
`scripts/paper_kin_residuals.py ellipsoid`

## Disc model and fits

### Overview

We describe each group of stars as a thin, rotating disc seen in perspective. The model has three parts. (i) A geometry: where the
disc is, how it is oriented, and where on it each star lies. (ii) A mean velocity field: the bulk motion of the LMC through space
plus a rotation curve and an oval ($m=2$) distortion of the flow in the disc plane. (iii) The random motions about that mean: a
Gaussian velocity ellipsoid with three different dispersions along the radial, azimuthal and vertical directions of the disc. Each
star's proper motion and line-of-sight velocity are compared with the projection of this model onto its own position on the sky,
with its own measurement errors. The model is kinematic: it describes the velocities and does not use a gravitational potential.
The assumptions are listed at the end of this section, and {tab}`kin_params` lists every parameter.

### Geometry

The stars are placed on a thin disc whose centre lies at the fixed distance $D_0$ = {glue:text}`kmDZero` kpc (as in vdMK14; the
stellar-mass chapter uses the distance modulus {glue:text}`kmDistMod`, {glue:text}`kmDMod` kpc, which would scale velocities and
radii by {glue:text}`kmDPct` per cent) in the direction of the unit vector $\hat{\boldsymbol r}_0$, offset by $(\Delta x,\Delta y)$
degrees east and north of the vdMK14 centre. The plane has unit normal $\hat{\boldsymbol n}$, set by the inclination $i$ (the angle
between the normal and the line of sight; $i=0$ is face-on) and the position angle $\theta$ (north through east) of the line of
nodes, whose unit vector $\hat{\boldsymbol l}$ points to position angle $\theta$. A star seen in the direction of the unit vector
$\hat{\boldsymbol r}$ is placed where its line of sight meets the plane,

$$
D=D_0\,\frac{\hat{\boldsymbol n}\cdot\hat{\boldsymbol r}_0}{\hat{\boldsymbol n}\cdot\hat{\boldsymbol r}},\qquad
\boldsymbol x=D\hat{\boldsymbol r}-D_0\hat{\boldsymbol r}_0,\qquad R=|\boldsymbol x|,
$$

so each star has a distance $D$, a position $\boldsymbol x$ in the plane and a radius $R$ from the fitted centre. Its in-plane unit
vectors are $\hat{\boldsymbol e}_R=\boldsymbol x/R$ (radial) and $\hat{\boldsymbol e}_\phi=\hat{\boldsymbol n}\times\hat{\boldsymbol
e}_R$ (azimuthal), and its azimuth $\phi$ in the disc plane is measured from the line of nodes ($\cos\phi=\hat{\boldsymbol
e}_R\cdot\hat{\boldsymbol l}$) and increases along $\hat{\boldsymbol e}_\phi$.

### Mean velocity

The mean velocity of the stars at $(R,\phi)$ is the systemic velocity of the centre plus motion in the disc plane,

$$
\boldsymbol V=v_{{\rm los},0}\,\hat{\boldsymbol r}_0+\kappa D_0\,(\mu_{\alpha*,0}\,\hat{\boldsymbol e}_{E,0}+\mu_{\delta,0}\,
\hat{\boldsymbol e}_{N,0})+v_\phi\,\hat{\boldsymbol e}_\phi+v_R\,\hat{\boldsymbol e}_R,
$$

where $(\mu_{\alpha*,0},\mu_{\delta,0})$ is the proper motion of the centre (east, $\mu_{\alpha*}=\mu_\alpha\cos\delta$, and north;
mas/yr) and $v_{{\rm los},0}$ its line-of-sight velocity (km/s), $\hat{\boldsymbol e}_{E,0}$ and $\hat{\boldsymbol e}_{N,0}$ point east
and north at the centre, $v_\phi$ and $v_R$ are the mean azimuthal and radial velocities in the plane, and $\kappa$ =
{glue:text}`kmKappa` km/s per (mas/yr × kpc) converts a proper motion at a distance into a velocity. The model proper motion
$\boldsymbol\mu_{\rm m}$ and line-of-sight velocity $v_{\rm m}$ of a star are the projections of this one vector on the star's own
east, north and line-of-sight unit vectors $\hat{\boldsymbol e}_E$, $\hat{\boldsymbol e}_N$, $\hat{\boldsymbol r}$,

$$
\boldsymbol\mu_{\rm m}=\frac{1}{\kappa D}\bigl(\boldsymbol V\cdot\hat{\boldsymbol e}_E,\ \boldsymbol V\cdot\hat{\boldsymbol e}_N\bigr),\qquad
v_{\rm m}=\boldsymbol V\cdot\hat{\boldsymbol r},
$$

so the perspective effects of the bulk motion across the {glue:text}`kmField`° of sky the samples span, which reach tens of km/s, are
included exactly rather than removed beforehand. The rotation curve is

$$
v_{\rm rot}(R)=v_0\tanh(R/R_0)\,[1+s\,(R-R_\sigma)],\qquad R_\sigma={glue:text}`kmRSig`\ {\rm kpc}:
$$

a rise over a radial scale $R_0$ towards a level $v_0$, multiplied by a linear factor with slope $s$ (per kpc) that lets the curve
flatten or decline in the outer disc; $R_\sigma$ = {glue:text}`kmRSig` kpc is a fixed reference radius, the middle of the fitted radii.
In the symmetric disc (Level 1) the stars move on circles, $v_\phi=v_{\rm rot}(R)$ and $v_R=0$; we fitted it only with $s=0$ and the
sky-aligned dispersions of the checks below, as a reference for the residuals. Level 2, our default, adds in-plane non-circular
motion with an $m=2$ pattern, the shape of the flow in a bar-like or tidally stretched potential:

$$
\begin{aligned}
v_\phi&=v_{\rm rot}(R)\,[1+m_{2c}\cos2\phi+m_{2s}\sin2\phi],\\
v_R&=v_{\rm rot}(R)\,[v_{R,0}+r_{2c}\cos2\phi+r_{2s}\sin2\phi].
\end{aligned}
$$

The five streaming coefficients $m_{2c}$, $m_{2s}$, $v_{R,0}$, $r_{2c}$ and $r_{2s}$ are dimensionless fractions of the local
rotation speed. $m_{2c}<0$ slows the rotation along the line of nodes and speeds it up along the minor axis; $r_{2c}$ and $r_{2s}$
describe inward and outward motion with the same two-fold pattern; $v_{R,0}$ is a uniform expansion ($>0$) or contraction. Averaged
over azimuth the $m=2$ terms vanish, so $v_{\rm rot}(R)$ is the mean rotation of the axisymmetric part of the flow.

### The rotation speed we quote

The three parameters of the rotation curve are strongly correlated: a larger $v_0$ with a larger $R_0$ and a more negative $s$ gives
nearly the same curve over the radii the stars cover. In the giant groups $v_0$ ranges from {glue:text}`kmVZeroLo` to
{glue:text}`kmVZeroHi` km/s and $R_0$ from {glue:text}`kmRZeroLo` to {glue:text}`kmRZeroHi` kpc, and with $s<0$ the curves peak at
$R$ = {glue:text}`kmRPeakLo`–{glue:text}`kmRPeakHi` kpc, so $v_0$ is a level the stars never reach. We therefore quote the speed at
$R_\sigma$, where the linear factor equals one,

$$
v_{\rm rot}({glue:text}`kmRSig`\,{\rm kpc})=v_0\tanh({glue:text}`kmRSig`\,{\rm kpc}/R_0),
$$

computed for every posterior draw; its posterior half-width (half the range between the 16th and 84th percentiles) is
{glue:text}`kmVrotErrLo`–{glue:text}`kmVrotErrHi` km/s for the giants. All stars at {glue:text}`kmRhoIn`° < ρ < {glue:text}`kmRhoOut`°,
not only those near {glue:text}`kmRSig` kpc, constrain it through the shape of the curve. $v_{\rm rot}$ is the mean rotation of the
stars, not the circular speed of the potential: the mean rotation of a population lags the circular speed by its asymmetric drift,
which grows with its dispersion and which we do not model (it is the subject of the page on rotation and the circular speed).

### Velocity ellipsoid

The velocities of the stars scatter about $\boldsymbol V$ with a Gaussian distribution whose covariance, in the disc frame
$(\hat{\boldsymbol e}_R,\hat{\boldsymbol e}_\phi,\hat{\boldsymbol n})$ at the star's position, is

$$
\Sigma_{\rm disc}(R)={\rm diag}(\sigma_R^2,\sigma_\phi^2,\sigma_z^2)\,e^{2\gamma(R-R_\sigma)}.
$$

$\sigma_R$, $\sigma_\phi$ and $\sigma_z$ are the radial, azimuthal and vertical velocity dispersions at $R_\sigma$, and every
dispersion changes with radius by the same factor $e^{\gamma(R-R_\sigma)}$, with $\gamma={\rm d}\ln\sigma/{\rm d}R$ (per kpc): the
ellipsoid keeps its shape and its size changes by the factor $e^{\gamma}$ per kpc. The ellipsoid is aligned with the disc: its axes
point along $\hat{\boldsymbol e}_R$, $\hat{\boldsymbol e}_\phi$ and $\hat{\boldsymbol n}$, with no vertex deviation (a rotation of the
ellipsoid in the plane) and no tilt. To compare it with the data we rotate it onto each star's sky directions. The matrix $P_i$ with
elements $(P_i)_{ab}=\hat{\boldsymbol u}_a\cdot\hat{\boldsymbol d}_b$, where $\hat{\boldsymbol u}_a$ runs over star $i$'s (east, north,
line-of-sight) unit vectors and $\hat{\boldsymbol d}_b$ over $(\hat{\boldsymbol e}_R,\hat{\boldsymbol e}_\phi,\hat{\boldsymbol n})$ at
its position, does this. We write the data of star $i$ as one vector in km/s, $\boldsymbol y_i=(\kappa D_i\mu_{\alpha*},\kappa
D_i\mu_\delta,v_{\rm los})$, with model $\boldsymbol y_{\rm m}=(\kappa D_i\boldsymbol\mu_{\rm m},v_{\rm m})$ and covariance

$$
\Sigma_i=P_i\,\Sigma_{\rm disc}(R_i)\,P_i^{\rm T}+{\rm blockdiag}\bigl(k^2\kappa^2D_i^2C_i,\ e_i^2\bigr),
$$

the intrinsic scatter projected on the sky plus the measurement errors ($C_i$, $k$ and $e_i$ are defined below). Proper motions and
line-of-sight velocity are thus described by one tensor, and their covariance, of order $\sin i\cos i\,(\sigma_z^2-\sigma_{R,\phi}^2)$,
is kept.

### Measurement errors and contamination

$C_i$ is the 2 × 2 Gaia covariance matrix of the star's proper motion, multiplied by $k^2$: Gaia underestimates its proper-motion
errors, more so in crowded fields, and each group has one free error scale $k$. The fit separates $k$ from the dispersions because
the measurement term grows with each star's own error, while the intrinsic term is the same for bright and faint stars at one
position; the spread of errors within a group therefore fixes both. The errors span a wider range for the giants than for the RR
Lyrae, whose $k$ and dispersions are therefore much more strongly correlated: between the 5th and 95th percentiles the errors (the
mean of the two components, in km/s at the LMC distance) span a factor of {glue:text}`kmErrFacGLo`–{glue:text}`kmErrFacGHi` in each
giant bin ({glue:text}`kmErrGLo`–{glue:text}`kmErrGHi` km/s over all bins) but only {glue:text}`kmErrFacRrl` for the RR Lyrae at
{glue:text}`kmRhoOuter`° < ρ < {glue:text}`kmRhoOut`° ({glue:text}`kmErrRrlLo`–{glue:text}`kmErrRrlHi` km/s). $k$ also takes any excess proper-motion scatter that scales
with the quoted errors. The fitted values are {glue:text}`kmKGLo`–{glue:text}`kmKGHi` for the giants and {glue:text}`kmKRrlOut` for the
RR Lyrae at {glue:text}`kmRhoOuter`° < ρ < {glue:text}`kmRhoOut`° (chapter 7); fitted at {glue:text}`kmRhoIn`° < ρ <
{glue:text}`kmRhoOut`° the RR Lyrae need $k$ = {glue:text}`kmKRrl`, because their crowded inner stars have errors underestimated more.
$e_i$ is the quoted error of the line-of-sight velocity, used as it is. A broad component with fraction $f$ takes the Milky Way stars
and gross errors that remain after the cuts of the data chapters: its velocities follow the same mean with an extra width $S$ =
{glue:text}`kmSOut` mas/yr (about {glue:text}`kmSOutKms` km/s) in every component. The likelihood of star $i$ is

$$
\mathcal{L}_i=(\kappa D_i)^2\bigl[(1-f)\,\mathcal{N}(\boldsymbol y_i\,|\,\boldsymbol y_{\rm m},\Sigma_i)
+f\,\mathcal{N}(\boldsymbol y_i\,|\,\boldsymbol y_{\rm m},\Sigma_i+\kappa^2D_i^2S^2I)\bigr],
$$

where $\mathcal{N}(\boldsymbol y\,|\,\boldsymbol m,\Sigma)$ is the Gaussian density of $\boldsymbol y$ with mean $\boldsymbol m$ and
covariance $\Sigma$, $I$ the identity matrix, and the factor $(\kappa D_i)^2$ turns the density in km/s into a density in proper
motion, the measured quantity. A star without a line-of-sight velocity (all RR Lyrae) enters with the proper-motion block of
$\boldsymbol y_i$ and $\Sigma_i$ only, which is the Gaussian marginalised over the missing component. The fits are unweighted: the
selection functions of the data chapters change how many stars are observed at each position, not their velocities, so they drop out
of a likelihood conditional on the positions, provided that the selection does not depend on velocity within the cuts of the data
chapters. Priors are uniform within wide bounds ({tab}`kin_params`): $|\Delta x|,|\Delta y|\leq$ {glue:text}`kmPDx`°;
{glue:text}`kmPIncLo`° ≤ $i$ ≤ {glue:text}`kmPIncHi`°; systemic proper motion within {glue:text}`kmPPm` mas/yr and $v_{{\rm los},0}$
within {glue:text}`kmPVlos` km/s of the vdMK14 values; {glue:text}`kmPVZeroLo` ≤ $v_0$ ≤ {glue:text}`kmPVZero` km/s; {glue:text}`kmPRZeroLo` ≤ $R_0$ ≤
{glue:text}`kmPRZeroHi` kpc; $|s|\leq$ {glue:text}`kmPSlope` per kpc; streaming coefficients within ±{glue:text}`kmPStream`;
{glue:text}`kmPSigLo` ≤ $\sigma_{R,\phi,z}$ ≤ {glue:text}`kmPSigHi` km/s; $|\gamma|\leq$ {glue:text}`kmPGam` per kpc;
{glue:text}`kmPKLo` ≤ $k$ ≤ {glue:text}`kmPKHi`; ${\rm logit}\,f=\ln[f/(1-f)]$ uniform between {glue:text}`kmPLogitLo` and
{glue:text}`kmPLogitHi`.

### Free parameters

A group fitted on its own has {glue:text}`kmNPar` free parameters: the centre ($\Delta x,\Delta y$), the systemic motion
($\mu_{\alpha*,0},\mu_{\delta,0},v_{{\rm los},0}$), the orientation ($i,\theta$), the rotation curve ($v_0,R_0,s$), the five streaming
coefficients, the ellipsoid ($\sigma_R,\sigma_\phi,\sigma_z,\gamma$), $k$ and $f$. The RR Lyrae have {glue:text}`kmNParRrl`: their
proper motions depend on $v_{{\rm los},0}$ only through the perspective terms, too weakly to fit it, and we hold it at
{glue:text}`kmVlosVdm` km/s (vdMK14). {tab}`kin_params` lists them with their meaning, units and priors.

| Symbol | Meaning | Unit | Prior (uniform) | Joint fit |
|---|---|---|---|---|
| $\Delta x$, $\Delta y$ | position of the disc centre, east and north of the vdMK14 centre | deg | ±{glue:text}`kmPDx` | shared |
| $\mu_{\alpha*,0}$, $\mu_{\delta,0}$ | proper motion of the centre | mas/yr | ±{glue:text}`kmPPm` about vdMK14 | shared |
| $v_{{\rm los},0}$ | line-of-sight velocity of the centre | km/s | ±{glue:text}`kmPVlos` about vdMK14 | shared |
| $i$ | inclination of the disc plane | deg | {glue:text}`kmPIncLo`–{glue:text}`kmPIncHi` | shared |
| $\theta$ | position angle of the line of nodes, north through east | deg | {glue:text}`kmPThetaLo`–{glue:text}`kmPThetaHi` | shared |
| $v_0$ | level of the tanh rise of the rotation curve | km/s | {glue:text}`kmPVZeroLo`–{glue:text}`kmPVZero` | own |
| $R_0$ | radial scale of the rise | kpc | {glue:text}`kmPRZeroLo`–{glue:text}`kmPRZeroHi` | own |
| $s$ | outer slope of the rotation curve (linear factor about $R_\sigma$) | 1/kpc | ±{glue:text}`kmPSlope` | own |
| $m_{2c}$, $m_{2s}$ | $m=2$ modulation of the rotation, fraction of $v_{\rm rot}$ | – | ±{glue:text}`kmPStream` | shared |
| $v_{R,0}$ | uniform radial motion (expansion > 0), fraction of $v_{\rm rot}$ | – | ±{glue:text}`kmPStream` | shared |
| $r_{2c}$, $r_{2s}$ | $m=2$ radial motion, fraction of $v_{\rm rot}$ | – | ±{glue:text}`kmPStream` | shared |
| $\sigma_R$, $\sigma_\phi$, $\sigma_z$ | radial, azimuthal and vertical dispersions at $R_\sigma$ | km/s | {glue:text}`kmPSigLo`–{glue:text}`kmPSigHi` | own |
| $\gamma$ | ${\rm d}\ln\sigma/{\rm d}R$, common to the three dispersions | 1/kpc | ±{glue:text}`kmPGam` | own |
| $k$ | scale of the Gaia proper-motion errors | – | {glue:text}`kmPKLo`–{glue:text}`kmPKHi` | own |
| $f$ | fraction of stars in the broad (contaminant) component | – | logit $f$ in [{glue:text}`kmPLogitLo`, {glue:text}`kmPLogitHi`] | own |
| $v_{\rm rot}({glue:text}`kmRSig`\,{\rm kpc})$ | derived: rotation speed at $R_\sigma$, $v_0\tanh(R_\sigma/R_0)$ | km/s | – | own |

**Table (kin_params).** Parameters of the kinematic model. Prior: uniform within the range given. Joint fit: shared by all groups
in stage 2, or each group's own. A group fitted on its own (stage 1) has all {glue:text}`kmNPar` parameters free; the RR Lyrae have
{glue:text}`kmNParRrl` (no line-of-sight velocities: $v_{{\rm los},0}$ held at {glue:text}`kmVlosVdm` km/s). Fixed: the distance of
the centre $D_0$ = {glue:text}`kmDZero` kpc, the reference radius $R_\sigma$ = {glue:text}`kmRSig` kpc and the extra width of the
outlier component $S$ = {glue:text}`kmSOut` mas/yr. The vdMK14 centre and systemic motion serve only as the origin of the offsets and
of the prior ranges (`lmcold.kinfit.BOUNDS`).

### What constrains the three dispersions

At inclination $i$ the line of sight sees, at azimuth $\phi$, the dispersion

$$
\sigma_{\rm los}^2(\phi)=\sigma_z^2\cos^2i+\sin^2i\,(\sigma_R^2\sin^2\phi+\sigma_\phi^2\cos^2\phi),
$$

mostly $\sigma_z$ ($\cos^2i$ = {glue:text}`kmCosSq` at $i$ = {glue:text}`kmIncJ`°). The proper motions see mostly the in-plane
ellipsoid, with $\sigma_z$ entering only through $\sin^2i$ = {glue:text}`kmSinSq`: for a star on the line of nodes they measure
$\sigma_R$ along the direction to the centre and $\sigma_\phi$ across it, and the inclination mixes the two elsewhere. With proper
motions and line-of-sight velocities the three dispersions are therefore constrained separately; with proper motions alone (the RR
Lyrae) $\sigma_R$ and $\sigma_\phi$ are constrained but $\sigma_z$ is not. For the giants, the proper-motion errors $\epsilon$ (median
{glue:text}`kmErrMedGLo`–{glue:text}`kmErrMedGHi` km/s per component in the five bins), scaled by $k$, are
{glue:text}`kmErrKGLo`–{glue:text}`kmErrKGHi` km/s, comparable to the dispersions of the metal-rich bins: a change of $k$ by
{glue:text}`kmDK` moves the $\sigma_\phi$ of the most metal-rich bin by about {glue:text}`kmDSigPhiK` km/s
($2k\epsilon^2\Delta k\approx$ {glue:text}`kmKTerm` km²/s² against $\sigma_\phi^2\approx$ {glue:text}`kmSigPhiSq` km²/s²), so one $k$
per group is an assumption that matters, which chapter 7 tests. The RR Lyrae errors (median {glue:text}`kmErrMedRrl` km/s per
component) are about twice their dispersions, so the scatter fixes mainly $\sigma_{R,\phi}^2+k^2\epsilon^2$ and one $k$ must describe
all stars of the sample (chapter 7). The dispersions are averages over the part of the disc a group covers, with the radial dependence of the
ellipsoid above. Mean motions that the model does not contain, such as the steeper inner rise of the rotation curve and the radial
flow found in the residuals below, add to them or remain in the residuals.

### Tests

We tested the model on mocks drawn from physical discs (positions from an exponential disc in the footprint, Gaia errors of the
nearest RR Lyrae; see the checks, {fig}`paper_kin_ellipsoid_mock`). With proper motions and line-of-sight velocities it recovers
$\sigma_R$ and $\sigma_z$ to within {glue:text}`kmDSigRZ` km/s and $\sigma_\phi$ to within {glue:text}`kmDSigPhi` km/s for cold and hot,
thin and thick discs, including three whose mean rotation lags by the asymmetric drift and is therefore not of the tanh form above;
with a pure tanh curve (outer slope $s=0$) the dispersions change by less than {glue:text}`kmDTanh` km/s. With proper motions alone it
recovers $\sigma_R$ and $\sigma_\phi$ to within {glue:text}`kmPmInplane` km/s (the thick disc: $\sigma_\phi$ {glue:text}`kmPmThickPhi`
km/s too high) but overestimates $\sigma_z$ by {glue:text}`kmPmDzLo`–{glue:text}`kmPmDzHi` km/s, with errors of
{glue:text}`kmPmEzLo`–{glue:text}`kmPmEzHi` km/s. The model replaces the sky-aligned dispersions of our first fits (one dispersion for
both proper-motion directions and one along the line of sight; see the checks). It fits the data better in every group: the
log-likelihood $\ln L$ of the stars is higher by $\Delta\ln L$ = {glue:text}`kmDlnLLo`–{glue:text}`kmDlnLHi` for the giants at
{glue:text}`kmRhoIn`° < ρ < {glue:text}`kmRhoOut`° and {glue:text}`kmDlnLRrlOut` for the RR Lyrae at {glue:text}`kmRhoOuter`° < ρ <
{glue:text}`kmRhoOut`°, with {glue:text}`kmDkG` more parameters ({glue:text}`kmDkRrl` for the RR Lyrae).

### Assumptions

In summary, the model assumes that: (1) the stars of a group lie in one thin plane, without thickness or spread along the line of
sight; (2) the bulk motion of the LMC is one velocity vector; (3) the mean motion in the plane is a rotation curve of the form above
with an $m=2$ distortion, and has no lopsided ($m=1$) part, no warp or change of orientation with radius or time, and no mean vertical
motion (tilted, scaled and turning discs are tested as alternatives in chapter 7); (4) the random motions are Gaussian, with an
ellipsoid aligned with the disc whose three axes change with radius by one common factor; (5) one scale $k$ corrects the Gaia errors
of all stars of a group; (6) the remaining contaminants form one broad component; (7) the selection of the stars does not depend on
their velocities. No dynamical relation links the rotation curve to the ellipsoid (no Jeans equation, the moment of the collisionless Boltzmann
equation that ties mean motions and dispersions to the potential; no asymmetric-drift or epicyclic relation), so the model measures how the stars move, not the mass that makes them move.

### Two stages

Levels 1 and 2 describe the mean motion; the two stages describe how the groups are combined. We fit the model in two ways, which
make two different assumptions. In stage 1 each group is fitted on its own, with its own centre, systemic motion, orientation and
streaming. In stage 2 all groups are fitted together and share these ({glue:text}`kmNShared` parameters), each keeping its own
rotation curve, ellipsoid, $k$ and $f$. The two stages answer two different questions, and we use each for one of them. The rotation
and dispersions of a group (chapter 7) come from stage 1, which does not force one geometry on populations that may not share it.
The maps and profiles ({fig}`paper_kin_maps_vt_ellipsoid`–{fig}`paper_kin_data_profiles_ellipsoid` and the right column of
{fig}`paper_kin_residual_profiles_ellipsoid`) and the geometry ({tab}`kin_shared`) come from stage 2, which puts all groups in one
frame. For the RR Lyrae in the maps we keep the stage-2 frame and refit their own parameters at {glue:text}`kmRhoOuter`° < ρ <
{glue:text}`kmRhoOut`° (maximum likelihood), as for the values we quote. Chapter 7 compares the two stages group by group. Imposing
the shared geometry changes the giants' $v_{\rm rot}({glue:text}`kmRSig`\,{\rm kpc})$ by at most {glue:text}`kmDVrotStage` km/s and their dispersions by
at most {glue:text}`kmDSigStage` km/s; for the RR Lyrae, in the joint fit of all groups at {glue:text}`kmRhoOuter`° < ρ <
{glue:text}`kmRhoOut`° ($i$ = {glue:text}`kmIncOut`°, $\theta$ = {glue:text}`kmThetaOut`°), it changes $v_{\rm rot}({glue:text}`kmRSig`\,{\rm kpc})$ by
{glue:text}`kmDVrotRrl` km/s and $\sigma_R$ and $\sigma_\phi$ by {glue:text}`kmDSigRRrl` and {glue:text}`kmDSigPhiRrl` km/s, about one
standard deviation (stage-1 posterior half-widths {glue:text}`kmSigRRrlErr` and {glue:text}`kmSigPhiRrlErr` km/s).
{tab}`kin_shared` lists the stage-2 parameters with the range of the stage-1 values. The giant groups agree on the line of nodes to
within {glue:text}`kmDTheta`° but less well on $i$ ({glue:text}`kmIncGLo`–{glue:text}`kmIncGHi`°) and on the centre
({glue:text}`kmDxGLo`–{glue:text}`kmDxGHi`° east): the stage-2 errors are conditional on a shared geometry, and the stage-1 range is
the better measure of the uncertainty of each geometric parameter. The centre offset and $\mu_{\delta,0}$ are strongly correlated in
every group (shown for the sky-aligned fits in {fig}`paper_kin_corner`).

### Posteriors

We sample the posteriors with the No-U-Turn Sampler (NUTS; [Hoffman & Gelman 2014](https://jmlr.org/papers/v15/hoffman14a.html)), a
Hamiltonian Monte Carlo method, in [NumPyro](https://doi.org/10.48550/arXiv.1912.11554) ([Phan et al. 2019](https://ui.adsabs.harvard.edu/abs/2019arXiv191211554P)),
{glue:text}`kmNChain` chains of {glue:text}`kmNDraw` draws after {glue:text}`kmNWarm` warm-up steps ({glue:text}`kmNWarmOuter` for the
RR Lyrae fit at {glue:text}`kmRhoOuter`° < ρ < {glue:text}`kmRhoOut`°). Every run has a Gelman–Rubin statistic $\hat R$ (the ratio
of the spread between chains to that within them, 1 for converged chains) ≤ {glue:text}`kmRHat` and at least {glue:text}`kmNEff`
effective (independent-equivalent) draws per parameter; one stage-{glue:text}`kmDivStage` fit (giants, {glue:text}`kmDivLo` to
{glue:text}`kmDivHi`) has {glue:text}`kmDivN` divergent transitions (sampler steps that failed to follow the posterior) among its
{glue:text}`kmNTot` draws, the others none. Each optional term of the model was added with a test that recovers it from a mock
(`tests/test_kinfit.py`).

```{code-cell} python
:tags: [remove-input]
import pandas as pd
# Posterior summaries: level 1 (centre fit), level 2 (ellipsoid fit, giants) and level 2 'outer' run (RR Lyrae, 4 < rho < 9 deg)
L1 = pd.read_csv('results/kin_mcmc_data_centre.csv')
L2 = pd.read_csv('results/kin_mcmc_data_ellipsoid.csv')
RR = pd.read_csv('results/kin_mcmc_data_ellipsoid_outer.csv')
# Shared parameter: (table label, decimals of the shared value, decimals of the stage-1 range)
LABEL = {'dx': ('Δx (deg)', 3, 2), 'dy': ('Δy (deg)', 3, 2), 'pmra0': ('μ_α*,0 (mas/yr)', 4, 3),
         'pmdec0': ('μ_δ,0 (mas/yr)', 3, 3), 'vlos0': ('v_los,0 (km/s)', 1, 1), 'inc': ('i (deg)', 1, 1), 'theta': ('θ (deg)', 1, 1),
         'm2c': ('m_2c', 3, 3), 'm2s': ('m_2s', 3, 3), 'vr0': ('v_R,0', 3, 3), 'r2c': ('r_2c', 3, 3), 'r2s': ('r_2s', 3, 3)}
STREAM = ('m2c', 'm2s', 'vr0', 'r2c', 'r2s')   # streaming coefficients: no stage-1 range in the table


def shared(t, p, d):
    """Stage-2 shared value: median ± half the 16th-84th percentile range, or '–' if the model has no such parameter."""
    r = t[(t.stage == 2) & (t.population == 'shared') & (t.param == p)]
    return f"{r['median'].iloc[0]:.{d}f} ± {(r.p84.iloc[0] - r.p16.iloc[0]) / 2:.{d}f}" if len(r) else '–'


rows = []
# Stage-1 column: range of the per-bin giant medians, the RR Lyrae value in brackets
for p, (label, d, d1) in LABEL.items():
    g = L2[(L2.stage == 1) & L2.population.str.startswith('giants') & (L2.param == p)]['median']
    r = RR[(RR.stage == 1) & (RR.population == 'RR Lyrae') & (RR.param == p)]['median']
    stage1 = '' if p in STREAM else f'{g.min():.{d1}f} to {g.max():.{d1}f}' + (f' ({r.iloc[0]:.{d1}f})' if len(r) else '')
    rows.append((label, shared(L1, p, d), shared(L2, p, d), stage1))
pd.DataFrame(rows, columns=['Parameter', 'Level 1', 'Level 2', 'Level 2, stage 1']).set_index('Parameter')
```

**Table (kin_shared).** Shared parameters of the joint (stage-2) fits: posterior medians ± half the 16th–84th percentile range.
Level 1: symmetric disc with sky-aligned dispersions (checks); Level 2: the model of this section with the velocity ellipsoid. Last
column: range of the stage-1 Level-2 values of the five giant groups, each fitted on its own (RR Lyrae in brackets, fitted at
{glue:text}`kmRhoOuter`° < ρ < {glue:text}`kmRhoOut`°). $\Delta x$, $\Delta y$: offset east and north of the vdMK14 centre,
(α, δ) = ({glue:text}`kmRaVdm`°, {glue:text}`kmDecVdm`°). Streaming terms in units of $v_{\rm rot}(R)$. From
`results/kin_mcmc_data_centre.csv` (Level 1), `kin_mcmc_data_ellipsoid.csv` and `kin_mcmc_data_ellipsoid_outer.csv`, written by
`scripts/fit_kinematics_mcmc.py`.

### Running the fit

`scripts/fit_kinematics_mcmc.py` performs every fit of this section. For one model (the argument `ellipsoid` for the model above,
`stream` for the sky-aligned model of the checks, `centre` for Level 1) it reads the giants and the RR Lyrae, keeps the stars in the
radial range of the fit, and finds the maximum-likelihood parameters of each group. From there NUTS samples the posterior of each
group on its own (stage 1) and of all groups together with a shared geometry (stage 2). The likelihood is the one written above: for
each star, the model velocity projected on its own sky directions, compared with its proper motion and line-of-sight velocity
through a Gaussian whose covariance is the projected velocity ellipsoid plus the scaled measurement errors, mixed with the broad
outlier component.

The script writes the summary tables `results/kin_mcmc_data_<model>.csv` (median, 16th and 84th percentiles and convergence
diagnostics of every parameter), which the tables and macros of this page read, and the posterior draws
`data/kin_mcmc_data_<model>_samples.npz`. The draws of the runs the released scripts read (the velocity-ellipsoid and sky-aligned
models, each over the full radial range and beyond {glue:text}`kmRhoOuter`°) are in Zenodo bundle 1, so the corner plots, the
comparison of the two models and the derived rotation speeds are made from them without rerunning the fit.

<!-- script: fit_kinematics_mcmc.py show: the giant and RR Lyrae catalogues of Zenodo bundle 1; about 4 h of MCMC for the main run (data ellipsoid), about 2.5 h for its outer version -->

## What the models leave in the data

{fig}`paper_kin_residual_profiles_ellipsoid` shows the residual velocities of the stars at {glue:text}`kmProfIn`° < ρ <
{glue:text}`kmProfOut`° about three models, each with the rotation curve of every group fitted: the geometry and systemic motion of
vdMK14, the joint Level-1 fit and the joint Level-2 fit. In the frame of vdMK14 we refit each group's rotation curve, dispersions,
error scale and outlier fraction (maximum likelihood). About it the $m=1$ term dominates: {glue:text}`kmVdmVtLo`–{glue:text}`kmVdmVtHi`
km/s in $v_t$, {glue:text}`kmVdmVrLo`–{glue:text}`kmVdmVrHi` km/s in $v_r$ and {glue:text}`kmVdmVlLo`–{glue:text}`kmVdmVlHi` km/s in
$v_{\rm los}$ (ranges over the giant groups; harmonic fits to the individual stars), in every group and in the RR Lyrae. Freeing the
centre, systemic motion and orientation (Level 1) reduces it to {glue:text}`kmLOneAOneLo`–{glue:text}`kmLOneAOneHi` km/s in all three
components: the $m=1$ pattern of the data is described by a displaced rotation centre and needs no lopsided rotation curve. What
remains is an $m=2$ pattern of {glue:text}`kmLOneVtLo`–{glue:text}`kmLOneVtHi` km/s in $v_t$ and
{glue:text}`kmLOneVrLo`–{glue:text}`kmLOneVrHi` km/s in $v_r$, with the same phase in all groups (maximum of $v_t$ at position angle
{glue:text}`kmPhaseGLo`–{glue:text}`kmPhaseGHi`° in the five giant groups and {glue:text}`kmPhaseRrl`° in the RR Lyrae). The streaming
terms of Level 2 absorb it, leaving {glue:text}`kmLTwoVtLo`–{glue:text}`kmLTwoVtHi` km/s in $v_t$ and
{glue:text}`kmLTwoVrLo`–{glue:text}`kmLTwoVrHi` km/s in $v_r$ (the $m=1$ term of $v_t$ stays at
{glue:text}`kmLTwoAOneLo`–{glue:text}`kmLTwoAOneHi` km/s); in Level 2 the rotation is slowest along the line of nodes ($m_{2c}$ =
{glue:text}`kmMTwoC`).

**Figure (paper_kin_residual_profiles_ellipsoid).** Residual velocities (observed minus model, km/s) against position angle (deg) at
{glue:text}`kmProfIn`° < ρ < {glue:text}`kmProfOut`°, median per {glue:text}`kmPaBin`° bin for each group (colours as in
{fig}`paper_kin_data_profiles_ellipsoid`, legend above). Rows: tangential (a, d, g), radial (b, e, h) and line-of-sight (c, f, i)
components, with position angle and directions about each model's centre. Columns: the geometry and systemic motion of vdMK14 with
each group's fitted rotation curve (a–c); the joint Level-1 fit (d–f); the joint Level-2 fit with the velocity ellipsoid (g–i). In
each panel, the range over the giant groups of the $m=1$ and $m=2$ amplitudes $A_1$ and $A_2$ of a harmonic fit to the stars. Dotted
vertical lines: line of nodes; dash-dotted: minor axis. `scripts/paper_kin_residuals.py ellipsoid`

The bottom rows of {fig}`paper_kin_maps_vt_ellipsoid`–{fig}`paper_kin_maps_vlos_ellipsoid` show what Level 2 leaves on the sky, as
the significance $z$ of the residual in each cell. In the three metal-poor bins and the RR Lyrae the model is close to the data: the
cell values scatter with a standard deviation of {glue:text}`kmZsdPoorLo`–{glue:text}`kmZsdPoorHi` (1 for a perfect model), and at
most {glue:text}`kmZgtPoor` per cent of the cells exceed $|z|=3$, still several times the {glue:text}`kmZgtNorm` per cent expected for
unit normals (for the RR Lyrae, whose parameters are fitted beyond {glue:text}`kmRhoOuter`°, these cells lie mostly inside
{glue:text}`kmRhoOuter`°). The two metal-rich bins, with the most stars and the smallest dispersions, are not: their cell values
scatter by {glue:text}`kmZsdRichLo`–{glue:text}`kmZsdRichHi`, and {glue:text}`kmZgtRichLo`–{glue:text}`kmZgtRichHi` per cent of the
cells exceed $|z|=3$. The radial profile of the residuals ({fig}`paper_kin_residual_radial_ellipsoid`) shows where: in the four
giant bins below [Fe/H] = {glue:text}`kmFehD` the stars at ρ = {glue:text}`kmRingFastLo`–{glue:text}`kmRingFastHi`° rotate
{glue:text}`kmRadFastLo`–{glue:text}`kmRadFastHi` km/s faster than the model and those at {glue:text}`kmRhoIn`–{glue:text}`kmRingFastLo`°
up to {glue:text}`kmRadSlow` km/s slower, so the rotation curve rises more steeply than the tanh form allows; beyond {glue:text}`kmRingTanh`° the outer
slope $s$ removes the decline of {glue:text}`kmTanhRichLo`–{glue:text}`kmTanhRichHi` km/s that a pure tanh curve leaves in the three
metal-rich bins and the RR Lyrae (see the sky-aligned model in the checks), and beyond {glue:text}`kmRingAgree`° $v_t$ agrees with the model to within
{glue:text}`kmRadOuter` km/s for the giants. In $v_r$ the three bins above [Fe/H] = {glue:text}`kmFehB` move inwards by
{glue:text}`kmRadInLo`–{glue:text}`kmRadInHi` km/s inside {glue:text}`kmRhoOuter`°, and the four bins above {glue:text}`kmFehA`
outwards by up to {glue:text}`kmRadOutLo`–{glue:text}`kmRadOutHi` km/s beyond {glue:text}`kmRingOut`°, relative to the model, a radial flow that Level 2
does not contain. The RR Lyrae rotate more slowly than their model in {glue:text}`kmNRadRrl` of the {glue:text}`kmNRingRrl` rings at
{glue:text}`kmRhoOuter`° < ρ < {glue:text}`kmRhoOut`°, by {glue:text}`kmRadRrlLo`–{glue:text}`kmRadRrlHi` km/s, although their rotation
curve was fitted there; we have not traced the cause (the streaming they share with the giants in the joint frame, or the median of
residuals dominated by their large proper-motion errors), and it limits the precision of their $v_{\rm rot}$ to a few km/s.

**Figure (paper_kin_residual_radial_ellipsoid).** Median residual (observed minus the joint Level-2 model with the velocity
ellipsoid, km/s) of the tangential (a), radial (b) and line-of-sight (c) velocities in rings of angular distance ρ (deg) about the
fitted centre, one line per group (colours as in {fig}`paper_kin_data_profiles_ellipsoid`, legend above; points of different groups
shifted slightly in ρ for clarity). The RR Lyrae have their parameters refitted at {glue:text}`kmRhoOuter`° < ρ <
{glue:text}`kmRhoOut`° in the joint frame, as in {fig}`paper_kin_maps_vt_ellipsoid`. Bars: $\sqrt{\pi/2}$ × standard deviation /
$\sqrt n$, the error of the median of $n$ stars. `scripts/paper_kin_residuals.py ellipsoid`

The pattern is not an artefact of the fitting. In mocks drawn from symmetric discs about a known centre (positions from an exponential
disc in the footprint, Gaia errors of the nearest RR Lyrae, one realisation each, {glue:text}`kmNReal` for correlated errors;
`scripts/fig_kin_mock_physical.py`), with a thick disc, a hot disc with asymmetric drift, or spatially correlated Gaia errors, fits with
sky-aligned dispersions to proper motions and line-of-sight velocities put the centre within {glue:text}`kmMockCen`° of the true one
(proper motions with distances: up to {glue:text}`kmMockCenDist`° for the thick discs), and the fitted $m=2$ amplitude
$(m_{2c}^2+m_{2s}^2)^{1/2}$ stays at or below {glue:text}`kmMockMTwo`, against {glue:text}`kmDataMTwoLo`–{glue:text}`kmDataMTwoHi` in the
same fits to the data ({glue:text}`kmDataMTwoELo`–{glue:text}`kmDataMTwoEHi` with the velocity ellipsoid); its sine term lies between
{glue:text}`kmMockMsLo` and {glue:text}`kmMockMsHi`, negative in all but {glue:text}`kmMockNPos` of the {glue:text}`kmMockNReal` fits.
The fitted centre does depend on which part of the disc a sample covers: in symmetric-disc fits (maximum likelihood, sky-aligned
dispersions, proper motions and line-of-sight velocities; `scripts/fig_kin_footprint.py`) of each giant bin, the east offsets span
{glue:text}`kmFpOwnLo`–{glue:text}`kmFpOwnHi`° on the bins' own sky positions and {glue:text}`kmFpMatLo`–{glue:text}`kmFpMatHi`° when
every bin is resampled onto the positions of the most metal-poor bin; the two metal-rich bins move east by
{glue:text}`kmFpShiftLo`–{glue:text}`kmFpShiftHi`° (fit errors ≈ {glue:text}`kmFpErr`°), a mock symmetric disc by {glue:text}`kmFpMock`°.
Part of the spread of the stage-1 centres therefore comes from the different sky coverage of the bins sampling a non-axisymmetric
velocity field.

### Mocks drawn from symmetric discs

`scripts/fig_kin_mock_physical.py` makes the mocks of the previous paragraphs. Each mock places stars in three dimensions in a disc
with the vdMK14 centre and bulk motion and a known orientation: thin or thick, cold and isotropic or hot with asymmetric drift, or
cold with a spatially correlated field of proper-motion offsets added to the Gaia errors. Each star takes the Gaia proper-motion
covariance of the nearest RR Lyrae, and its velocity is projected on the sky at its own distance. The mocks are then fitted by maximum
likelihood with proper motions, line-of-sight velocities, distances and their combinations, with and without streaming. None of the
mocks has streaming, so any fitted $m=2$ term comes from the departure of the mock from the thin, cold model. The fitted parameters go
to `results/kin_mock_physical.csv`, from which the numbers above are written. The same drawing function makes the mocks of the
velocity-ellipsoid test in the checks.

<!-- script: fig_kin_mock_physical.py show: the RR Lyrae catalogue of Zenodo bundle 1; maximum-likelihood fits of every mock and data set (run time not recorded) -->

### Density centres

The kinematic centre is compared in chapter 7 with the centre of the star counts. `scripts/fig_density_centre.py` measures that
density centre for the RR Lyrae, for the *Gaia* red giants that the BOSS carton selects, and for the BOSS giants of each [Fe/H] bin
weighted for selection, coverage and Milky Way foreground: the weighted mean position of the stars within an aperture, iterated from
the vdMK14 centre until it stops moving, for several aperture radii, with errors from bootstrap resamples of the stars. It writes
`results/density_centre.csv`, read by the centre figure and the text of chapter 7.

<!-- script: fig_density_centre.py show: the RR Lyrae, BOSS parent and weighted BOSS giant catalogues of Zenodo bundle 1; bootstrap resampling (run time not recorded) -->

## Checks

### Recovery of the velocity ellipsoid from mocks

We drew mocks of {glue:text}`kmNMock` stars at {glue:text}`kmRhoIn`° < ρ < {glue:text}`kmRhoOut`° from five physical discs: positions in
three dimensions from an exponential disc of scale length $h$ = {glue:text}`kmRD` kpc (thin, or with a Gaussian height dispersion of
{glue:text}`kmHThick` kpc), or from a spheroid with density $\propto K_0(m/h)$ ($K_0$ the modified Bessel function of the second kind,
$m^2=R^2+z^2/q^2$ the spheroidal radius, $z$ the height above the plane; its face-on profile is exponential with the same $h$) of axis
ratio $q$ = {glue:text}`kmQSph` ('hot spheroid', the shape of the old population found in chapter 8; rms height {glue:text}`kmRmsSph` kpc, growing
with radius), Gaia covariances of the nearest RR Lyrae member, line-of-sight errors of {glue:text}`kmEVlos` km/s, and constant
dispersions $(\sigma_R,\sigma_\phi,\sigma_z)$ = ({glue:text}`kmSigCold`, {glue:text}`kmSigCold`, {glue:text}`kmSigCold`) km/s ('cold
thin') or ({glue:text}`kmSigHotR`, {glue:text}`kmSigHotPhi`, {glue:text}`kmSigHotZ`) km/s (the hot thin and hot thick discs and the hot
spheroid). The mean rotation is $v_c\tanh(R/R_0)$ (with $v_c$ the amplitude of the mock's circular-speed curve) for the cold disc and
one hot disc ('hot, no drift'), and $v_c\tanh(R/R_0)$ minus the asymmetric drift for the other three, so theirs is not of the fitted
form. None of the mocks has streaming or a dispersion gradient; the fits have both free. {fig}`paper_kin_ellipsoid_mock` shows the
maximum-likelihood dispersions. With proper motions and line-of-sight velocities, $\sigma_R$ is recovered to within
{glue:text}`kmDSigRLo` to {glue:text}`kmDSigRHi` km/s, $\sigma_\phi$ to within {glue:text}`kmDSigPhi` km/s (biased high) and $\sigma_z$ to
within {glue:text}`kmDSigZ` km/s, with statistical errors of {glue:text}`kmErrLo`–{glue:text}`kmErrHi` km/s; with a pure tanh curve
($s=0$) the dispersions change by less than {glue:text}`kmDTanh` km/s. With proper motions alone, $\sigma_R$ and $\sigma_\phi$ are
recovered to within {glue:text}`kmPmInplane` km/s (thick disc: $\sigma_\phi$ {glue:text}`kmPmThickPhi` km/s too high), and $\sigma_z$
comes out {glue:text}`kmPmSzLo`–{glue:text}`kmPmSzHi` km/s for true values of {glue:text}`kmPmTzLo`–{glue:text}`kmPmTzHi` km/s, with
errors of {glue:text}`kmPmEzLo`–{glue:text}`kmPmEzHi` km/s: the proper motions do not measure $\sigma_z$. The hot spheroid is recovered
as well as the discs with line-of-sight velocities (dispersions within {glue:text}`kmSphDSig` km/s, inclination {glue:text}`kmSphInc`°
for {glue:text}`kmTruthInc`°, line of nodes {glue:text}`kmSphTheta`° for {glue:text}`kmTruthTheta`°); with proper motions alone its
$\sigma_z$ comes out at {glue:text}`kmSphPmSz` ± {glue:text}`kmSphPmSzErr` km/s and its inclination at {glue:text}`kmSphPmInc`°.

#### Drawing and fitting the mocks

The mocks are drawn and fitted by `scripts/paper_kin_ellipsoid_mock.py`, which also makes {fig}`paper_kin_ellipsoid_mock` (about 20
minutes on one machine, so it is not run on this page: `python scripts/paper_kin_ellipsoid_mock.py`). It draws each mock with the
drawing function of `scripts/fig_kin_mock_physical.py` and fits it three times by maximum likelihood with the model of this page:
with proper motions and line-of-sight velocities, with the outer slope $s$ held at zero, and with proper motions alone.

<!-- script: paper_kin_ellipsoid_mock.py show: the RR Lyrae catalogue of Zenodo bundle 1; about 20 min -->

**Figure (paper_kin_ellipsoid_mock).** Recovery of the velocity ellipsoid from physical mocks: fitted $\sigma_R$ (a), $\sigma_\phi$ (b)
and $\sigma_z$ (c) (km/s) for the five mocks (horizontal axis). Filled black circles: proper motions and line-of-sight velocities, the
model of this page; grey squares: the same with a pure tanh rotation curve ($s=0$); open circles: proper motions only. Red bars: true
values. Errors from the inverse Hessian (the curvature of the likelihood at its maximum). Mocks: 'cold thin', $v_c\tanh(R/R_0)$;
'hot', thin with asymmetric drift; 'hot, no drift', thin with $v_c\tanh(R/R_0)$; 'hot thick', with asymmetric drift; 'hot spheroid'
($q$ = {glue:text}`kmQSph`), with asymmetric drift. `scripts/paper_kin_ellipsoid_mock.py`

### Robustness checks of the velocity-ellipsoid model

Chapter 7 tests three assumptions of the model on each giant bin with `scripts/kin_ellipsoid_checks.py`, by maximum likelihood
started from the fit of this page: an inner disc in a plane of its own (its inclination and line of nodes offset by free amounts
inside a fixed radius), a free scale of the internal motion along the line of sight, and one error scale $k$ per group, tested by
fitting the stars brighter and fainter than the bin's median *Gaia* $G$ magnitude separately, each half with its own $k$, rotation
curve and ellipsoid. It writes the log-likelihood gains, the geometry, the rotation speed at $R_\sigma$ and the dispersions of each
fit to `results/kin_ellipsoid_checks.csv`.

<!-- script: kin_ellipsoid_checks.py show: the giant catalogues of Zenodo bundle 1; maximum-likelihood fits of every giant bin (run time not recorded) -->

### The sky-aligned dispersion model

Our first fits used the mean motion of the model section with a pure tanh rotation curve, $v_{\rm rot}=v_0\tanh(R/R_0)$, and two
dispersions per group, constant over the disc: $\sigma$ (written $\sigma_{\rm PM}$ where it must be told apart), the same in both
proper-motion directions, and $\sigma_{\rm los}$ along the line of sight. In the star's (east, north, line-of-sight) frame the random
velocities then have covariance ${\rm diag}(\sigma^2,\sigma^2,\sigma_{\rm los}^2)$, an ellipsoid aligned with the line of sight, not
with the disc, and the likelihood factorises into a proper-motion and a line-of-sight term:

$$
\begin{aligned}
\mathcal{L}_i={}&(1-f)\,\mathcal{N}\bigl(\boldsymbol\mu_i\,|\,\boldsymbol\mu_{\rm m},\,k^2C_i+\sigma_\mu^2I\bigr)\,
\mathcal{N}\bigl(v_i\,|\,v_{\rm m},\,e_i^2+\sigma_{\rm los}^2\bigr)\\
&+f\,\mathcal{N}\bigl(\boldsymbol\mu_i\,|\,\boldsymbol\mu_{\rm m},\,k^2C_i+\sigma_\mu^2I+S^2I\bigr)\,
\mathcal{N}\bigl(v_i\,|\,v_{\rm m},\,e_i^2+\sigma_{\rm los}^2+\kappa^2D_i^2S^2\bigr),
\end{aligned}
$$

with $\boldsymbol\mu_i$ and $v_i$ the measured proper motion and line-of-sight velocity of star $i$ and $\sigma_\mu=\sigma/(\kappa D_i)$
the dispersion in proper motion at the star's distance. Neither dispersion is a component of the disc-frame ellipsoid, but each is
close to a combination of them: averaged over azimuth, $\sigma_{\rm los}^2\simeq\sigma_z^2\cos^2i+\tfrac12\sin^2i\,(\sigma_R^2+\sigma_\phi^2)$
and $\sigma^2\simeq\tfrac12(1-\tfrac12\sin^2i)(\sigma_R^2+\sigma_\phi^2)+\tfrac12\sigma_z^2\sin^2i$ (from $\sigma_{\rm los}^2(\phi)$ above
and the trace of the ellipsoid, with the central line of sight for every star). The hot thin mock with asymmetric drift, with
$(\sigma_R,\sigma_\phi,\sigma_z)$ = ({glue:text}`kmSigHotR`, {glue:text}`kmSigHotPhi`, {glue:text}`kmSigHotZ`) km/s, gives $\sigma$ =
{glue:text}`kmHotSig` and $\sigma_{\rm los}$ = {glue:text}`kmHotSigLos` km/s against {glue:text}`kmAvgSig` and {glue:text}`kmAvgSigLos`
km/s from these averages at its inclination of {glue:text}`kmTruthInc`°. Because a real ellipsoid projects differently at each
azimuth, these dispersions rank the groups but do not measure $\sigma_R$ or $\sigma_z$; the velocity-ellipsoid model fits better in
every group, by $\Delta\ln L$ = {glue:text}`kmDlnLInA`, {glue:text}`kmDlnLInB`, {glue:text}`kmDlnLInC`, {glue:text}`kmDlnLInD` and
{glue:text}`kmDlnLInE` for the giant bins and {glue:text}`kmDlnLInF` for the RR Lyrae at {glue:text}`kmRhoIn`° < ρ < {glue:text}`kmRhoOut`°,
and {glue:text}`kmDlnLOutA`, {glue:text}`kmDlnLOutB`, {glue:text}`kmDlnLOutC`, {glue:text}`kmDlnLOutD`, {glue:text}`kmDlnLOutE` and
{glue:text}`kmDlnLOutF` at {glue:text}`kmRhoOuter`° < ρ < {glue:text}`kmRhoOut`° (log-likelihood of each group's stars at the best of
{glue:text}`kmNDraws` posterior draws of each model; `results/kin_model_comparison.csv`, `scripts/kin_model_comparison.py`).

The joint Level-2 fit with sky-aligned dispersions puts the centre {glue:text}`kmSkyDx`° east and {glue:text}`kmSkyDy`° south of the
vdMK14 centre, with $i$ = {glue:text}`kmSkyInc`°, $\theta$ = {glue:text}`kmSkyTheta`° and $m_{2c}$ = {glue:text}`kmSkyMTwoC`,
{glue:text}`kmSkyDDx`° in $\Delta x$ and {glue:text}`kmSkyDTheta`° in $\theta$ from the values of {tab}`kin_shared`. {tab}`kin_sky`
gives the rotation and dispersions of each group fitted on its own (stage 1). The maps of $v_t$, $v_r$ and $v_{\rm los}$ in the frame
of the joint fit ({fig}`paper_kin_maps_vt`, {fig}`paper_kin_maps_vr` and {fig}`paper_kin_maps_vlos`), the data against position angle
({fig}`paper_kin_data_profiles`) and the residuals of the three models against position angle ({fig}`paper_kin_residual_profiles`)
show for this model what {fig}`paper_kin_maps_vt_ellipsoid`–{fig}`paper_kin_data_profiles_ellipsoid` and
{fig}`paper_kin_residual_profiles_ellipsoid` show for the velocity ellipsoid. The rotational support $v_0/\sigma_{\rm los}$ rises from {glue:text}`kmRatioLo`
to {glue:text}`kmRatioHi` from the metal-poor to the metal-rich giants (APOGEE alone: {glue:text}`kmRatioApo`), the same ordering as
with the velocity ellipsoid (chapter 7). With a pure tanh curve the radial profile of the residuals shows a decline beyond {glue:text}`kmRingTanh`°: the
three metal-rich bins and the RR Lyrae rotate {glue:text}`kmTanhRichLo`–{glue:text}`kmTanhRichHi` km/s slower than the model there (the
two metal-poor bins by {glue:text}`kmTanhPoorLo`–{glue:text}`kmTanhPoorHi` km/s; `results/kin_residual_radial.csv`, not shown), which
the outer slope $s$ of the model section removes. The sky-aligned fits sampled the posteriors with $\hat R\leq$ {glue:text}`kmRHatS`
and at least {glue:text}`kmNEffS` effective draws per parameter; the RR Lyrae fit at {glue:text}`kmRhoOuter`° < ρ <
{glue:text}`kmRhoOut`° had {glue:text}`kmDivNS` divergent transitions.

```{code-cell} python
:tags: [remove-input]
import numpy as np
import pandas as pd
# Rotational support per population (S) and the stage-1 posterior of the streaming fit (F)
S = pd.read_csv('results/kin_support.csv').set_index('population')
F = pd.read_csv('results/kin_mcmc_data_stream.csv').query('stage == 1')


def q(v, lo, hi, d):
    """Median with its 16th-84th percentile range."""
    return f'{v:.{d}f} ({lo:.{d}f}–{hi:.{d}f})'


def stage1(pop, p):
    r = F[(F.population == pop) & (F.param == p)].iloc[0]
    return r['median'], r.p16, r.p84


# One row per population: rotation amplitude v0, turnover radius R0 of v0 tanh(R/R0), dispersions, their ratios, outlier fraction f = 1/(1 + exp(-logit_fout))
rows = []
for pop, s in S.iterrows():
    rr = pop == 'RR Lyrae'
    rows.append((pop.replace('giants ', ''), f'{int(s.n):,}', q(s.v0, s.v0_p16, s.v0_p84, 1), q(*stage1(pop, 'R0'), 2),
                 '–' if rr else q(s.sigma_los, s.sigma_los_p16, s.sigma_los_p84, 1),
                 '–' if rr else q(s.ratio, s.ratio_lo, s.ratio_hi, 2), q(s.sigma, s.sigma_p16, s.sigma_p84, 1),
                 q(s.ratio_pm, s.ratio_pm_lo, s.ratio_pm_hi, 2), f'{100 / (1 + np.exp(-stage1(pop, "logit_fout")[0])):.1f}'))
# Rows from the APOGEE line-of-sight velocities alone, where measured
for pop, s in S[np.isfinite(S.v0_apogee)].iterrows():
    rows.append(('APOGEE ' + pop.replace('giants ', ''), f'{int(s.n_apogee):,}', f'{s.v0_apogee:.1f} ± {s.v0_apogee_err:.1f}', '',
                 f'{s.sigma_los_apogee:.1f} ± {s.sigma_los_apogee_err:.1f}', f'{s.ratio_apogee:.2f} ± {s.ratio_apogee_err:.2f}', '', '', ''))
pd.DataFrame(rows, columns=['Group', 'N', 'v0 (km/s)', 'R0 (kpc)', 'σ_los (km/s)', 'v0/σ_los', 'σ_PM (km/s)', 'v0/σ_PM',
                            'f (%)']).set_index('Group')
```

**Table (kin_sky).** Level-2 parameters of each group with sky-aligned dispersions, fitted on its own (stage 1) at
{glue:text}`kmRhoIn`° < ρ < {glue:text}`kmRhoOut`°: posterior median with the 16th–84th percentile range in brackets
($v_0/\sigma_{\rm los}$ from the posterior draws). $N$: stars; $f$: outlier fraction. $\sigma_{\rm PM}$ and $v_0/\sigma_{\rm PM}$: the
same fits at {glue:text}`kmRhoOuter`° < ρ < {glue:text}`kmRhoOut`° ({glue:text}`kmNOutA`, {glue:text}`kmNOutB`, {glue:text}`kmNOutC`,
{glue:text}`kmNOutD` and {glue:text}`kmNOutE` giants, {glue:text}`kmNRrlOut` RR Lyrae), where one error scale $k$ describes the Gaia
errors; the ratio from the draws of those fits, whose $v_0$ is {glue:text}`kmVZeroOutA`, {glue:text}`kmVZeroOutB`,
{glue:text}`kmVZeroOutC`, {glue:text}`kmVZeroOutD` and {glue:text}`kmVZeroOutE` km/s for the giants and {glue:text}`kmVZeroOutF` km/s for
the RR Lyrae. Last three rows: APOGEE giants alone, maximum-likelihood fits of $v_0$, $\sigma_{\rm los}$, $k$ and $f$ with the other
parameters held at the posterior medians of the full bin (errors from the inverse Hessian, conditional on the fixed geometry,
streaming and $R_0$). From `results/kin_support.csv` (`scripts/paper_kin_support.py`) and `results/kin_mcmc_data_stream.csv`.

**Figure (paper_kin_maps_vt).** As {fig}`paper_kin_maps_vt_ellipsoid` for the model with sky-aligned dispersions: frame and model of its
joint Level-2 fit at {glue:text}`kmRhoIn`° < ρ < {glue:text}`kmRhoOut`° for all groups, RR Lyrae included, and residual significance
with the scatter of the sky-aligned likelihood. Fitted centre (α, δ) = ({glue:text}`kmRaS`°, {glue:text}`kmDecS`°).
`scripts/paper_kin_residuals.py` (without argument)

**Figure (paper_kin_maps_vr).** As {fig}`paper_kin_maps_vt` for the radial component $v_r$ (km/s). `scripts/paper_kin_residuals.py`

**Figure (paper_kin_maps_vlos).** As {fig}`paper_kin_maps_vt` for the line-of-sight velocity $v_{\rm los}$ (km/s), giants only.
`scripts/paper_kin_residuals.py`

**Figure (paper_kin_data_profiles).** As {fig}`paper_kin_data_profiles_ellipsoid` in the frame of the joint Level-2 fit with
sky-aligned dispersions. `scripts/paper_kin_residuals.py`

**Figure (paper_kin_residual_profiles).** As {fig}`paper_kin_residual_profiles_ellipsoid`, with sky-aligned dispersions in all three
columns (the joint Level-2 fit in the right column, g–i). `scripts/paper_kin_residuals.py`

#### Comparing the two models

The log-likelihood gains $\Delta\ln L$ of the velocity-ellipsoid model over the sky-aligned model quoted above are computed by
`scripts/kin_model_comparison.py`. For each group and each radial range it evaluates the log likelihood of the group's stars at the
same random subset of posterior draws of its stage-1 fit under each model, read from the posterior samples, and takes the difference
of the best (and of the median) values; no fit is repeated. It writes `results/kin_model_comparison.csv`.

<!-- script: kin_model_comparison.py show: the posterior samples of the stream and ellipsoid fits and the star catalogues, all in Zenodo bundle 1 (run time not recorded) -->

### Correlations of the geometric parameters

{fig}`paper_kin_corner` shows the posteriors of the centre, the systemic proper motion north, the orientation and the $m=2$ part of
$v_\phi$ in the sky-aligned Level-2 fits of each group on its own and of all groups together. The centre offset $\Delta x$ and
$\mu_{\delta,0}$ are correlated in every group along one line: a centre further east goes with a larger northward systemic proper
motion. The script also writes a full corner plot of every stage-1 fit (`figures/fig_kin_mcmc_corner_data_stream_*.png`).

```{code-cell} python
import sys
sys.argv = ['paper_kin_corner.py', 'data_stream']   # as: python scripts/paper_kin_corner.py data_stream
```

<!-- script: paper_kin_corner.py -->

**Figure (paper_kin_corner).** Posteriors of the centre offset east and north ($\Delta x$, $\Delta y$, deg), the systemic proper
motion north ($\mu_{\delta,0}$, mas/yr), the inclination $i$ and line of nodes $\theta$ (deg) and the $m=2$ part of $v_\phi$ ($m_{2c}$,
$m_{2s}$) in the Level-2 fits with sky-aligned dispersions of each group on its own (colours, stage 1, legend top right; RR Lyrae
at {glue:text}`kmRhoIn`° < ρ < {glue:text}`kmRhoOut`°) and of all groups together (black, stage 2). Diagonal: one-dimensional
posteriors, each normalised to unit area; off-diagonal: contours enclosing {glue:text}`kmLevOne` and {glue:text}`kmLevTwo` per cent of
each two-dimensional posterior. `scripts/paper_kin_corner.py data_stream`
