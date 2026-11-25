---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Fig. 1: the tracers, the mass they formed, and their rotation

Every number in the text and caption is written by `scripts/paper_text_letter.py` from the result tables
(`results/macros_letter.tex`); the code cells are the figure script itself (`scripts/paper_letter_signal.py`), step by step.

**What it shows.** {fig}`paper_letter_signal` shows where the old stellar tracers of the Large Magellanic Cloud (LMC) are on the sky
(a), how much stellar mass the LMC had formed below each metallicity (b), and how each population moves about the LMC's centre (c).

The tracers are of two kinds. Red giants with spectra from [APOGEE DR17](https://doi.org/10.3847/1538-4365/ac4414) and from the
BOSS spectrograph of [SDSS-V](https://doi.org/10.3847/1538-3881/ae0576) have a line-of-sight velocity $v_{\rm los}$ and an iron
abundance [Fe/H] (the logarithm of the star's iron-to-hydrogen ratio relative to the Sun's). They are divided into five [Fe/H] bins:
below {glue:text}`ltFehA`, {glue:text}`ltFehA` to {glue:text}`ltFehB`, {glue:text}`ltFehB` to {glue:text}`ltFehC`,
{glue:text}`ltFehC` to {glue:text}`ltFehD`, and above {glue:text}`ltFehD`. RR Lyrae are pulsating stars that are old and
metal-poor; the fundamental-mode (RRab) RR Lyrae of [OGLE-IV](https://doi.org/10.48550/arXiv.1606.02727) and
[*Gaia* DR3](https://doi.org/10.1051/0004-6361/202243964) have no spectra. All stars have proper motions (motions on the sky) from
[*Gaia* DR3](https://doi.org/10.1051/0004-6361/202243940) and lie at angular distances {glue:text}`ltRingIn`° < $\rho$ <
{glue:text}`ltRingOut`° from the centre of [van der Marel & Kallivayalil (2014)](https://doi.org/10.1088/0004-637X/781/2/121).

Positions and velocities are measured about the fitted centre, the centre of the LMC's disc found by the kinematic model, which is
{glue:text}`ltCentreOffset`° east of that centre. The line of nodes is the line where the disc plane crosses the plane of the sky,
and the position angle of a star is measured on the sky from north through east. In (c), the LMC's bulk motion (the proper motions
and $v_{\rm los}$ that its centre-of-mass motion gives at each star's position) is subtracted from every star. Its proper motion,
converted to a velocity at its distance in the fitted disc plane, is split into $v_r$, the component on the sky away from the fitted
centre, and $v_t$, the component perpendicular to it (negative: clockwise on the sky). In a rotating, inclined disc $v_{\rm los}$
makes one wave per turn in position angle and $v_t$ and $v_r$ each make two. The reference disc is the fitted model of the
[Fe/H] < {glue:text}`ltFehA` giants, without noise, put through the same steps. Every population shows the pattern of a rotating
disc. Panel (b) turns metallicity into the stellar mass the LMC had formed when stars of that metallicity were born.

**Starts from:** the quick route (this repository plus Zenodo bundle 1, which holds the giant and RR Lyrae catalogues).

## Making the figure, step by step

<!-- script: paper_letter_signal.py -->

**Figure (paper_letter_signal).** The LMC's stellar tracers, the mass they formed and their rotation. (a) The tracers on the sky
about the fitted centre: pixels with BOSS (grey) and APOGEE (slate) giants, giants with [Fe/H] < {glue:text}`ltFehA` (dark blue
dots), RR Lyrae density (orange contours); dashed circles: {glue:text}`ltProfIn`° and {glue:text}`ltProfOut`° from the fitted
centre; dotted: the line of nodes. (b) Stellar mass formed against [Fe/H]: cumulative (left axis; filled circle: the total;
triangles and squares: published formed and present masses) and fraction per dex (right axis; circles: corrected, open squares:
observed giants); shading: the five metallicity bins. (c) Median $v_t$, $v_r$ (all populations) and $v_{\rm los}$ (giants) against
position angle {glue:text}`ltProfIn`–{glue:text}`ltProfOut`° from the fitted centre; colours as the shading in (b), RR Lyrae
orange; black: the reference disc ($v_t$ solid, $v_r$ dotted, $v_{\rm los}$ dashed). Every population shows the velocity pattern of
a rotating disc.

## How the inputs are made

The figure script reads the release catalogues (`data/giants_apogee.parquet`, `data/giants_boss.parquet`, `data/rrab.parquet`;
built by `scripts/build_giants.py` and `scripts/build_rrab.py` from the public surveys, see `data/sources.yaml`), the stellar mass
per metallicity (`results/giant_present_mass.csv`) and the fitted disc geometry (`results/kin_mcmc_data_ellipsoid.csv`). The logged
steps upstream of the figure, nearest first:

```{code-cell} python
:tags: [remove-input]
import kbtools as kb
kb.upstream('paper_letter_signal.png')
```

Long steps, not run here:
- the kinematic fit that gives the centre and geometry: `python scripts/fit_kinematics_mcmc.py data ellipsoid` (about 4 h on one
  machine; its posterior samples are in bundle 1); chapter 6 (Kinematics: velocity field and disc model) describes the disc model
  and its likelihood and shows the fit script step by step;
- the stellar mass per metallicity: `python scripts/build_giant_mass.py` then `python scripts/build_present_mass.py` (a few
  minutes, from the selection-function catalogues of bundle 1).
