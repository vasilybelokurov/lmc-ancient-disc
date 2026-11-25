---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Fig. 5: the centre of an LMC-mass halo in cold and self-interacting dark matter

Every number in the text and caption is written by `scripts/paper_text_letter.py` from the result tables
(`results/macros_letter.tex`); the code cells are the figure script itself (`scripts/paper_sims_sidm.py`), step by step.

**What it shows.** {fig}`paper_sims_sidm` compares the centre of one halo with the mass of the Large Magellanic Cloud's (LMC's) halo,
simulated with cold dark matter (CDM) and with self-interacting dark matter (SIDM), in which dark-matter particles scatter off each
other. Both runs are of Halo104 of SIDM Concerto ([Nadler et al. 2025](https://doi.org/10.3847/1538-4357/adf553)), from the same
initial conditions and with dark matter only; the SIDM cross-section per unit mass has a normalisation $\sigma_0/m$ =
{glue:text}`ltSigmaZero` cm² g⁻¹ and falls as $v^{-4}$ with the collision speed $v$ above $w$ = {glue:text}`ltWSidm` km/s. The
halo's shape is measured in two ways. The axis ratio $c/a$ of its minor to its major axis is found in ellipsoidal shells. The
quadrupole of its gravitational potential is measured relative to the monopole: $\epsilon_{2,\rm all}=|\Phi_2|/|\Phi_0|$ includes
all of the quadrupole, also its part axisymmetric about the halo's minor axis, and $\epsilon_2$ only the part not axisymmetric
about that axis. (In Fig. 4 of the Letter $\epsilon_2$ is taken about the disc's spin; the full quadrupole is the same about any
axis and bounds the non-axisymmetric part about every axis.) With self-interactions the centre of the halo is nearly round already
at redshift $z\approx$ {glue:text}`ltSiZFirst`, with a quadrupole at or below the one at which the TNG50 cohorts keep their
rotation; in CDM it stays flattened and aspherical.

**Starts from:** the quick route. The high-redshift snapshots behind `results/sims_sidm_shapes.csv` are not public; they are
available from the simulation authors on request (`scripts/fetch_concerto.py` explains how to fetch them once access is given).

## Making the figure, step by step

<!-- script: paper_sims_sidm.py -->

**Figure (paper_sims_sidm).** The centre of an LMC-mass halo in cold (CDM) and self-interacting (SIDM) dark matter (SIDM Concerto
Halo104, dark matter only). (a) Quadrupole at {glue:text}`ltREps` kpc against redshift: CDM (black) and SIDM (red), the full
quadrupole including its axisymmetric part ($\epsilon_{2,\rm all}$, filled) and the part not axisymmetric about the halo's minor
axis ($\epsilon_2$, open); grey: TNG50. (b) Minor-to-major axis ratio $c/a$ and (c) $\epsilon_{2,\rm all}$ against radius, light
to dark from $z$ = {glue:text}`ltSiZFirst` to 0. With self-interactions the centre of the halo is round already at
$z\approx$ {glue:text}`ltSiZFirst`; in CDM it stays flattened and aspherical.

## How the inputs are made

The figure script reads the shapes of the Concerto halo (`results/sims_sidm_shapes.csv`) and of the TNG50 progenitors
(`results/sims_halo_settle.csv`). The logged steps upstream of the figure, nearest first:

```{code-cell} python
:tags: [remove-input]
import kbtools as kb
kb.upstream('paper_sims_sidm.png')
```

`python scripts/sims_sidm_shapes.py` measures the axis ratios and the quadrupole of both runs at each redshift (about 15 min; AGAMA;
about 20 GB of memory at z = 0); its code is shown step by step, not run, in the chapter 'How the simulated old stars lose their
rotation'.
