---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Rotation and the circular speed

*Second page of the kinematics chapter, after 'Rotational support, centre and sensitivity'.* Every number on this page is written by `scripts/paper_text_vc_drift.py` from the tables of the
shared-$v_c$ + drift model (`results/macros_vcdrift.tex`); the code cells are `scripts/fig_kin_vc_drift.py`, step by step, and the two scripts that make its fits, shown but not run.

The rotation $v_{\rm rot}$ of the kinematic model is the mean azimuthal velocity of a population of stars, not the circular speed
$v_c$ of the potential ($v_c^2=R\,\partial\Phi/\partial R$ in the disc plane, with $\Phi$ the gravitational potential and $R$ the
cylindrical radius in the disc frame). A population with velocity dispersion rotates more slowly than the circular speed, by its asymmetric drift,
which grows with its dispersion. The LMC's red giants, divided into five bins of iron abundance [Fe/H], rotate at {glue:text}`vdVrotLo`–{glue:text}`vdVrotHi` km/s at $R$ = {glue:text}`vdRSig` kpc, faster for the cooler,
more metal-rich bins. How much of that rise is asymmetric drift, rather than a difference in the potential the populations feel?

None of the Letter's conclusions depends on the answer: the Letter compares the rotation and dispersion of the LMC's stars with those
of simulated analogues observed and fitted in the same way, so any asymmetric drift is present in both, and it does not claim a
circular speed.

## One circular speed for all populations

We fitted all five giant bins jointly with one circular-speed curve $v_c(R)$ (the form of the free rotation curves), and let each
bin's mean rotation lag it by its asymmetric drift from the radial Jeans equation for a steady axisymmetric disc,

$$
\bar v_\phi^2 = v_c^2 - \left[\sigma_\phi^2 + \sigma_R^2\left(\frac{R}{h} - 1 - 2\,\frac{d\ln\sigma}{dR}\,R\right)\right],
$$

where $\bar v_\phi$ is the bin's mean azimuthal velocity (its rotation), $\sigma_R$ and $\sigma_\phi$ are its radial and azimuthal
velocity dispersions, $\sigma$ their common radial profile, all from the bin's own fit, and $h$ is the exponential scale length of its
surface density (the radial Jeans equation of [Binney & Tremaine 2008](https://ui.adsabs.harvard.edu/abs/2008gady.book.....B), for an exponential disc
whose velocity ellipsoid is aligned with the cylindrical axes). $h$ is measured from the bin's surface density of giants at {glue:text}`vdHFitIn`–{glue:text}`vdHFitOut` kpc (`lmcold.kinfit.rotation`, `scripts/kin_vc_drift.py`). On mocks with a known curve
($v_c(4\,{\rm kpc})$ = {glue:text}`vdMockVc` km/s) the fit recovers $v_c$ to {glue:text}`vdMockDvc` km/s.

For the LMC this gives $v_c(4\,{\rm kpc})$ = {glue:text}`vdVcFour` km/s. The bins then lag the circular speed by {glue:text}`vdLagMP` km/s (the most
metal-poor) to {glue:text}`vdLagMR` km/s (the most metal-rich): most of the rise of the rotation with metallicity is the asymmetric drift of
populations whose dispersions fall. The model predicts the RR Lyrae, which it did not fit, to rotate at {glue:text}`vdRrlPred` km/s (16th–84th
percentiles {glue:text}`vdRrlPredLo`–{glue:text}`vdRrlPredHi`), against the measured {glue:text}`vdRrlMeas` ({glue:text}`vdRrlMeasLo`–{glue:text}`vdRrlMeasHi`). {fig}`fig_kin_vc_drift` shows the fitted curves and this comparison.

**Figure (fig_kin_vc_drift).** (a) The shared circular speed $v_c(R)$ (black) and the mean rotation $\bar v_\phi(R)$ of each giant [Fe/H] bin
(colours, legend above), free (solid) and in the shared-$v_c$ + drift model (dashed), against cylindrical radius $R$ (kpc).
(b) Rotation at $R$ = 4 kpc, $v_{\rm rot}(4\,{\rm kpc})$ (km/s), of the shared-$v_c$ + drift model against the free fit, one point per
bin; star: the RR Lyrae, with the rotation the model predicts for them (vertical bar) against the measured one (horizontal bar),
16th–84th percentiles; grey line: equality. `scripts/fig_kin_vc_drift.py`

### The joint fits

`scripts/kin_vc_drift.py` makes the fits behind these numbers in four steps. It measures the scale length $h$ of each giant bin from
the selection-weighted surface density of the BOSS giants, and that of the RR Lyrae, against $R$ in the disc frame of the joint
ellipsoid fit. It then fits mocks of three populations that share one known $v_c(R)$ and rotate at their exact drift means, with
the shared-$v_c$ + drift model, with the same model and $h$ free, and with free rotation curves; this is the recovery test quoted
above. It fits the five giant bins jointly with free rotation curves and with the shared-$v_c$ + drift model on the same stars, and
finally predicts the rotation of the RR Lyrae from the fitted $v_c$ and their own dispersions and $h$, posterior draw by posterior
draw. Its tables (`results/kin_vc_drift_h.csv`, `_mock.csv`, `_data.csv`, `_rrl.csv`) are read by `scripts/paper_text_vc_drift.py`
and by the figure script.

<!-- script: kin_vc_drift.py show: the release catalogues, the joint ellipsoid fit and joint maximum-likelihood fits of mocks and of the five giant bins -->

### The figures

<!-- script: fig_kin_vc_drift.py -->

## Where one circular speed is not enough

The shared model fits the giants worse than free rotation curves: the log-likelihood $\ln L$ of the data is lower by
$\Delta\ln L$ = {glue:text}`vdDlnL`, for {glue:text}`vdNPar` fewer parameters. The deficit is spread over all radii and all five bins ({glue:text}`vdDefLo` to {glue:text}`vdDefHi` per bin),
not one region or survey ({fig}`fig_kin_vc_drift_deficit`).

**Figure (fig_kin_vc_drift_deficit).** Difference of the log-likelihoods of the shared-$v_c$ + drift and the free fits, summed over the
stars of each giant [Fe/H] bin (colours) in bins of cylindrical radius $R$ (a; kpc) and of azimuth $\phi$ in the disc plane, measured
from the line of nodes along the rotation (b; deg). Negative: the shared model fits worse. `scripts/fig_kin_vc_drift.py` Comparing the binned data with the
models ({fig}`fig_kin_vc_drift_data`): in the most metal-poor bin the data at {glue:text}`vdMidLo`–{glue:text}`vdMidHi` kpc ({glue:text}`vdMidData` km/s) follow the shared model, but beyond {glue:text}`vdOutFrom` kpc they stay flat
({glue:text}`vdOutData` km/s) where the model falls to {glue:text}`vdOutDrift` km/s, because its drift term $\sigma_R^2 R/h$ grows too fast; at {glue:text}`vdInLo`–{glue:text}`vdInHi` kpc both
models miss. Against azimuth ({fig}`fig_kin_vc_drift_data_phi`) the two models leave almost the same residuals: the shared circular
speed changes the radial profile of the rotation, not its pattern around the disc.

**Figure (fig_kin_vc_drift_data).** Data against the models, one column per giant [Fe/H] bin. Top: mean azimuthal velocity $v_\phi$
(km/s) of the giants with proper motions and line-of-sight velocities in 1-kpc bins of cylindrical radius $R$ (points, error: standard
deviation over $\sqrt n$ for $n$ stars), and of the models averaged over the same stars: free (grey), shared $v_c$ + drift (dashed),
and shared $v_c$ + drift with population-dependent streaming and thickness (dotted). Bottom: data minus model for the free (open) and
the shared-$v_c$ + drift (filled) fits. `scripts/fig_kin_vc_drift.py`

**Figure (fig_kin_vc_drift_data_phi).** Data minus model of the mean azimuthal velocity $v_\phi$ (top) and radial velocity $v_R$
(bottom; km/s) of the same giants in 45° bins of azimuth $\phi$ (from the line of nodes, along the rotation) at 2 < $R$ < 9 kpc, for
the free (open) and the shared-$v_c$ + drift (filled) fits; one column per [Fe/H] bin. `scripts/fig_kin_vc_drift.py`

The deficit is not caused by the scale lengths, the tracers or the streaming:

- with each $h$ at the 16th or the 84th percentile of its bootstrap, $\Delta\ln L$ = {glue:text}`vdDlnLHLo` and {glue:text}`vdDlnLHHi`;
- on the {glue:text}`vdNMatched` giants with SDSS-V BOSS spectra that are also in the mass sample, with $h$ measured from those stars, {glue:text}`vdDlnLMatched`;
- population-dependent bisymmetric ($m=2$) streaming improves the free model by {glue:text}`vdDlnLStreamFree` and leaves the gap at {glue:text}`vdGapStream`;
- letting the circular speed a population feels fall with its vertical dispersion (a thick population feels a weaker average radial
  force) gives $v_c(4\,{\rm kpc})$ = {glue:text}`vdVcBetaLo` km/s for the most metal-poor to {glue:text}`vdVcBetaHi` km/s for the most metal-rich bin and
  $\Delta\ln L$ = {glue:text}`vdDlnLBeta` ({glue:text}`vdDlnLStreamBeta` with streaming as well).

Left free, the scale lengths fall to {glue:text}`vdHFreeLo`–{glue:text}`vdHFreeHi` kpc, about half those measured ({glue:text}`vdHMeasLo`–{glue:text}`vdHMeasHi` kpc), and
$v_c(4\,{\rm kpc})$ rises to {glue:text}`vdVcHfree` km/s. One potential with the drift of a thin, steady disc therefore describes the old
population's rotation at {glue:text}`vdMidLo`–{glue:text}`vdMidHi` kpc but not over the whole disc: the old population is thick and may not be fully in equilibrium. The
circular speed this model gives depends on these assumptions, which is why the Letter does not use it.

### The diagnostic fits

`scripts/kin_vc_drift_extensions.py` makes the fits of this section. It refits the free and the shared-$v_c$ + drift models on all
giants, and the latter again with every $h$ at the 16th and at the 84th percentile of its bootstrap; it sums the difference of the
per-star log-likelihoods of the two models in bins of $R$ and $\phi$ and per survey ({fig}`fig_kin_vc_drift_deficit`); it repeats
both fits on the BOSS giants that are also in the mass sample, with $h$ measured from those stars; and it adds the
population-dependent streaming and the thickness term, alone and together. It writes `results/kin_vc_drift_ext.csv`,
`results/kin_vc_drift_deficit.csv` and the fitted parameter vectors (`data/kin_vc_drift_ext_vectors.npz`, Zenodo bundle 1), from
which `scripts/fig_kin_vc_drift.py` draws the figures of this page.

<!-- script: kin_vc_drift_extensions.py show: joint maximum-likelihood fits of the giant bins in every model variant; about 5 h -->
