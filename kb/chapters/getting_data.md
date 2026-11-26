---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Getting the data and writing the tables

The previous chapters start from catalogues and result tables that are already on disk. This chapter shows where those files come
from and how the paper's tables and the numbers in its text are written from them. None of its scripts is run on this page: the
downloads need network access, credentials or many gigabytes of disk, and the table and text scripts are run with the rest of the
analysis (`python scripts/manifest.py run <script>`). Each script is shown cell by cell, with a note saying what it needs.

## Two routes to the data

Every input of the paper is listed in `data/sources.yaml`, in three tiers. Tier A holds the public inputs: survey catalogues,
published tables, isochrones and simulations. Each entry gives the origin and release of the file, its URL or query, the script that
fetches it, its number of rows where known, the paper it comes from and its sha256 checksum. Tier B holds the derived catalogues
(the cleaned and cross-matched samples, the selection weights and the posterior samples of the kinematic fits), each with the script
that builds it. Tier C holds results kept in `results/`. The function `lmcold.sources.path` refuses to return a file whose sha256
differs from the one recorded, so a script never reads an input that is not the one the paper was made with.

A reader can take one of two routes. The quick route downloads the tier-B catalogues as Zenodo bundle 1; with them and the tables in
`results/`, every figure, table and number of the paper can be remade. The full route fetches the tier-A inputs from the public
archives with the `scripts/fetch_*.py` scripts below and rebuilds the tier-B catalogues with the build scripts of the earlier
chapters; the rebuilt files are checked against the same checksums.

### The quick route: Zenodo bundle 1

`scripts/fetch_release.py` downloads the zip file of bundle 1 from its Zenodo record and checks it against the md5 that Zenodo
gives. Before it writes a member into `data/`, it compares the member's sha256 with the entry for that file in `data/sources.yaml`;
a member that is not listed, has another checksum or would land outside `data/` stops the installation, and nothing in `data/` is
overwritten by a file that fails its check. The first cell of every notebook of this site runs this script when the catalogues are
missing.

The table below lists the files of bundle 1; it is generated from `data/sources.yaml` and the run log, not typed. The row
`sims_observe_parts` holds the per-analogue pieces that `scripts/sims_observe.py` merges into `results/sims_observed.csv`; the
repository already holds that table, so the figures do not read the pieces.

<!-- bundle 1 table -->

<!-- script: fetch_release.py show: network access to Zenodo; bundle 1 is about 250 MB (docs/release.md) -->

### The full route: the public archives

The sections below follow the tiers of `data/sources.yaml` source by source. Each fetch script prints the sha256 of what it wrote,
to be compared with the entry of `data/sources.yaml`. Large files (survey summary catalogues, simulation snapshots) are written to
`~/data`, outside the repository; small ones go to `data/`.

## Surveys

### OGLE-IV RR Lyrae

The RR Lyrae sample starts from the OGLE-IV Collection of Variable Stars of the LMC
([Soszyński et al. 2016](https://doi.org/10.48550/arXiv.1606.02727)): the identifications and positions of all RR Lyrae, and the
$I$-band light-curve parameters of the RRab stars. `scripts/build_rrab.py` reads both.

<!-- script: fetch_ogle4_rrab.py show: network access to the OGLE web server -->

### OGLE-IV fields

The lists of OGLE-IV field centres ([Udalski et al. 2015](https://doi.org/10.48550/arXiv.1504.05966)) define the footprint of the
RR Lyrae completeness estimate (`lmcold.ogle`).

<!-- script: fetch_ogle4_fields.py show: network access to the OGLE web server; run after fetch_ogle4_rrab.py, which creates data/raw/ogle4/ -->

### *Gaia* DR3 RR Lyrae

Two queries of the *Gaia* archive ([Gaia Collaboration 2023](https://doi.org/10.1051/0004-6361/202243940)) give the *Gaia* part of
the data. The first selects the RR Lyrae of the DR3 variability catalogue
([Clementini et al. 2023](https://doi.org/10.1051/0004-6361/202243964)) in a cone about the LMC centre, with their astrometry; it is
joined with OGLE-IV in `scripts/build_rrab.py`.

<!-- script: fetch_gaia_rrlyrae.py show: network access to the Gaia archive -->

### *Gaia* DR3 astrometry of the giants

The second query fetches the astrometry and $G$ magnitude of every spectroscopic giant, uploading the source identifiers in batches;
`scripts/build_giants.py` joins it to the spectra.

<!-- script: fetch_gaia_giants.py show: network access to the Gaia archive, and data/spec_giants.parquet from scripts/build_spec_giants.py -->

### APOGEE DR17 and SDSS-V DR20 spectra

The giants' spectra come from two summary catalogues of the SDSS Science Archive Server: the APOGEE DR17 allStarLite file
([Abdurro'uf et al. 2022](https://doi.org/10.3847/1538-4365/ac4414)) and the SDSS-V DR20 Astra BOSSNet summary. Where the release
publishes a SHA-1 checksum, the download is checked against it.

<!-- script: fetch_giants.py show: network access to data.sdss.org and disk space in ~/data for two large summary catalogues -->

### Targeting records

The selection functions of the chapter on selection need the targeting records as well: for BOSS, the spAll-lite file (the carton
under which each fibre was assigned); for APOGEE, the allVisit file and the APOGEE-2 targeting files of the LMC fields.

<!-- script: fetch_targeting.py show: network access to data.sdss.org and disk space in ~/data for the targeting summaries -->

### SDSS-V targeting database

Five tables of the SDSS-V targeting database give every *Gaia* star of the BOSS parent its carton memberships and priorities, and
the XP metallicities that the SDSS-V cartons were selected on; `scripts/build_boss_cartons.py` reads them.

<!-- script: fetch_targetdb.py show: network access to data.sdss.org and disk space in ~/data for five targeting-database tables -->

### 2MASS

The infrared test of the BOSS parent sample in regions of high extinction uses the 2MASS Point Source Catalog
([Skrutskie et al. 2006](https://doi.org/10.1086/498708)) near the LMC centre, queried from the IRSA archive.

<!-- script: fetch_twomass_centre.py show: network access to the IRSA TAP service -->

## Published catalogues

### *Gaia* XP metallicities

The XP metallicities of [Andrae, Rix & Chandra (2023)](https://doi.org/10.3847/1538-4365/acd53e) give the metallicity of the stars
in the parent samples of the selection functions. Their table is downloaded from its Zenodo record
([doi:10.5281/zenodo.7945154](https://doi.org/10.5281/zenodo.7945154)); `scripts/build_xp_andrae.py` keeps the stars of the parents.

<!-- script: fetch_andrae23.py show: network access to Zenodo and disk space in ~/data for a table of about 10^8 rows -->

### Reddening map

The red-clump reddening map of the Magellanic Clouds of [Skowron et al. (2021)](https://doi.org/10.3847/1538-4365/abcb81) gives the
reddening of the BOSS parent in the test of extinction.

<!-- script: fetch_skowron21.py show: network access to CDS -->

### Spectroscopic metallicities of RR Lyrae

The photometric metallicities of the RR Lyrae are compared with spectroscopic ones: [Gratton et al. (2004)](https://doi.org/10.1051/0004-6361:20035840),
with positions from [Di Fabrizio et al. (2005)](https://doi.org/10.1051/0004-6361:20041659), and
[Borissova et al. (2004](https://doi.org/10.1051/0004-6361:20034494), [2006)](https://doi.org/10.1051/0004-6361:20054132), whose
stars are located through SIMBAD.

<!-- script: fetch_spec_rrl.py show: network access to CDS and SIMBAD -->

### Star clusters

The LMC clusters with Ca II triplet metallicities of [Grocholski et al. (2006)](https://doi.org/10.1086/507303), and the *Gaia* DR3
stars around each, are the clusters of the mass comparison in the chapter on stellar mass.

<!-- script: fetch_clusters.py show: network access to VizieR and the Gaia archive -->

### Published metallicity distributions

The metallicity distributions of field stars are compared with three published catalogues:
[Cole et al. (2005)](https://doi.org/10.1086/428007), [Carrera et al. (2008)](https://doi.org/10.1088/0004-6256/135/3/836) and
[Omkumar et al. (2025)](https://doi.org/10.1051/0004-6361/202452510), all from VizieR.

<!-- script: fetch_mdf_literature.py show: network access to the VizieR mirror at IDIA -->

### Published LMC masses

The total stellar masses of the LMC published by others are typed into `lmcold.constants.LMC_MASS_LITERATURE`, each with the section or
table it comes from. This script archives the source papers from arXiv, so that the table of masses ({tab}`mass_summary`) can check
each source against its checksum.

<!-- script: fetch_lmc_mass_literature.py show: network access to arxiv.org -->

### Isochrones

The conversion of giant counts into stellar mass uses PARSEC v1.2S isochrones with COLIBRI asymptotic-giant-branch tracks
([Bressan et al. 2012](https://doi.org/10.1111/j.1365-2966.2012.21948.x); [Marigo et al. 2017](https://doi.org/10.3847/1538-4357/835/1/77))
and the [Kroupa (2001)](https://doi.org/10.1046/j.1365-8711.2001.04022.x) initial mass function, in the *Gaia* and 2MASS passbands,
downloaded from the CMD web service.

<!-- script: fetch_isochrones.py show: network access to the CMD web service and the isochrones extra of pyproject.toml (pinned ezpadova); run once per photometric system -->

## Simulations

The simulation chapters compare the LMC with analogues from three suites and with one pair of dark-matter-only runs. The snapshot
directories are read by `lmcold.sims` from `~/data/sims`; they are too large to hash as a whole, so the Auriga and FIRE-2 files are
checked one by one against the lists in `data/checksums/`.

### TNG50 analogues

The TNG50-1 analogues ([Pillepich et al. 2019](https://doi.org/10.1093/mnras/stz2338);
[Nelson et al. 2019a](https://doi.org/10.1093/mnras/stz2306)) are every central subhalo in the stellar- and total-mass ranges of the
LMC at $z=0$. This script downloads two star-particle cutouts of each from the TNG web service: one with positions, velocities,
masses, metallicities and formation times, and one with the birth positions and velocities. It needs a registered key for the TNG
web service.

<!-- script: fetch_tng50.py show: a registered TNG API key (environment variable TNG_API_KEY); about 2.5 GB (docs/release.md) -->

### TNG50 progenitors

For the evolution of the old stars, this script follows the main progenitors of a subset of the analogues back in time: their merger
trees and cutouts of stars, gas and dark matter at earlier snapshots. It also needs the TNG key.

<!-- script: fetch_tng50_progenitors.py show: a registered TNG API key (environment variable TNG_API_KEY), and the z = 0 sample of fetch_tng50.py -->

### Auriga

The Auriga analogues are the level-3 dwarf-galaxy zooms of [Grand et al. (2024)](https://doi.org/10.1093/mnras/stae1598): their
$z=0$ snapshots, group catalogues and lists of accreted stars, and the early snapshots used for the scrambling of orbits. They are
transferred from the public Globus collection of the Auriga project.

<!-- script: fetch_auriga.py show: globus-cli logged in and Globus Connect Personal running; about 120 GB (script header) -->

### FIRE-2

The FIRE-2 analogues are the LMC-mass runs m11d, m11e and m11h of the public release
([Wetzel et al. 2023](https://doi.org/10.3847/1538-4365/acb99a)): the $z=0$ snapshot of each and the track of its host's centre
and velocity through time, downloaded over HTTPS.

<!-- script: fetch_fire2.py show: network access to the Flatiron Institute server; about 8 GB (script header) -->

### SIDM Concerto

The comparison of a dark-matter halo with and without self-interaction uses Halo104 of the SIDM Concerto suite
([Nadler et al. 2025](https://doi.org/10.3847/1538-4357/adf553)), an LMC-mass host run with cold dark matter (CDM) and with
self-interacting dark matter (SIDM) from the same initial conditions. The $z=0$ snapshots and halo catalogues of both runs are public
on Zenodo ([doi:10.5281/zenodo.14933624](https://doi.org/10.5281/zenodo.14933624)). The snapshots at $z>0$ are not public; they are
available from the simulation authors on request, as a Globus collection whose identifier the script reads from the environment
variable `CONCERTO_HIGHZ_GLOBUS`. Without it, the script downloads the $z=0$ data only.

<!-- script: fetch_concerto.py show: network access to Zenodo; for the z > 0 snapshots, the Globus collection id from the simulation authors (CONCERTO_HIGHZ_GLOBUS), globus-cli and Globus Connect Personal -->

## How the tables are written

Every table of the paper is written by a script `scripts/paper_table_*.py` from the result tables in `results/`. Each script
writes only the rows of its table, as LaTeX, to `results/table_*.tex`; the column headings, the caption and the notes stay in the
paper's source, which reads the rows with `\input`. The values are formatted by the helpers of `lmcold.tex` (for example a
posterior median with its 16th and 84th percentiles), so no cell is typed or edited by hand. A change in a result changes the table
when the script is run again.

### Ages of the mass priors

The ages of the formed mass in each [Fe/H] range under the reference weighting and the three age priors ({tab}`ages`).

<!-- script: paper_table_ages.py show: results/age_weighting.json from scripts/paper_mass_method.py -->

### Selection counts

The counts of the parent samples, observed stars, good spectra and mass samples of the two surveys ({tab}`cuts`).

<!-- script: paper_table_cuts.py show: the selection summaries in results/ and data/boss_weights.parquet (Zenodo bundle 1) -->

### Stellar mass per metallicity

The formed and present stellar mass per [Fe/H] bin ({tab}`mass`).

<!-- script: paper_table_mass.py show: results/giant_present_mass.csv -->

### Stellar mass against published estimates

The summary of this work against the published estimates ({tab}`mass_summary`).

<!-- script: paper_table_mass_summary.py show: the mass tables in results/ and the archived source papers of fetch_lmc_mass_literature.py -->

### Shared kinematic parameters

The shared parameters of the joint fits ({tab}`kin_shared`).

<!-- script: paper_table_kin_shared.py show: the posterior summaries of the kinematic fits in results/ -->

### Rotation and velocity ellipsoid

The rotation and velocity ellipsoid of each group fitted on its own ({tab}`kin`).

<!-- script: paper_table_kin.py show: results/kin_support_ellipsoid.csv and the posterior summaries of the kinematic fits -->

### Sky-aligned dispersions

The fits with sky-aligned dispersions ({tab}`kin_sky`).

<!-- script: paper_table_kin_sky.py show: results/kin_support.csv and the posterior summaries of the sky-aligned fits -->

### Axis ratios

The axis ratio implied by the kinematics of each giant bin in the three potentials ({tab}`shape_q`).

<!-- script: paper_table_shape.py show: results/shape_virial.csv from scripts/shape_virial.py -->

### Analogues that reach the LMC

The number of simulated analogues that reach the LMC's rotational support under each definition ({tab}`sims_definitions`).

<!-- script: paper_table_sims_definitions.py show: results/sims_definitions.csv from scripts/paper_sims_kinematics.py -->

## How the numbers in the text are written

No number in the text of the paper or of this site is typed by hand. Each is a LaTeX macro, such as `\daNUnion` for the number of RR
Lyrae in the union of OGLE-IV and *Gaia*, defined in a file `results/macros_*.tex`. A script `scripts/paper_text_*.py` writes each
file. Settings (a radius, a bin edge, a clipping threshold) are read from the code that uses them: module constants of a script are
read from its source without running it, defaults of a function from its signature, and constants of `lmcold` by import. Results are
computed from the tables and catalogues the figures are drawn from. The paper reads the macro files with `\input`; on this site,
`kb/make_notebooks.py` puts the value of each macro in place of every `{glue:text}` field. Where the text states something about a
result (for example that only one fit has divergent transitions, or that no analogue reaches the LMC today), the script stops if the
result no longer agrees with the wording.

### Shared helpers

`scripts/textnum.py` holds the functions the generators share: reading settings from the code, and writing numbers as the text
writes them (integers without a decimal point, approximate values to one significant figure).

<!-- script: textnum.py show: nothing beyond the repository (a module of helpers, imported by the generators) -->

### The Letter

The numbers of the Letter's account of how its figures were made (`results/macros_letter.tex`, prefix `\lt`).

<!-- script: paper_text_letter.py show: the result tables in results/, the release samples (Zenodo bundle 1) and the header of the z = 0 Concerto snapshot (fetch_concerto.py) -->

### Sampling of the kinematic fits

The sentences on the sampling of the kinematic fits (`results/text_mcmc_*.tex`), written whole from the run settings and the
convergence diagnostics.

<!-- script: paper_text_mcmc.py show: the posterior summaries of the kinematic fits in results/ -->

### The data chapters

The numbers of the chapters on the samples, the selection functions and the calibration (`results/macros_data.tex`, prefix `\da`).

<!-- script: paper_text_data.py show: the result tables in results/ and the catalogues of Zenodo bundle 1 -->

### Stellar mass

The numbers of the chapter on stellar mass per metallicity (`results/macros_mass.tex`, prefix `\ms`).

<!-- script: paper_text_mass.py show: the result tables in results/ and the catalogues of Zenodo bundle 1 -->

### Metallicity

The numbers of the chapter on the metallicity structure (`results/macros_feh.tex`, prefix `\fe`).

<!-- script: paper_text_feh.py show: the result tables in results/ and the catalogues of Zenodo bundle 1 -->

### The kinematic model

The numbers of the chapter on the velocity field and the disc model (`results/macros_kinmodel.tex`, prefix `\km`).

<!-- script: paper_text_kin_model.py show: the result tables in results/, the posterior samples and the catalogues of Zenodo bundle 1 -->

### Rotational support

The numbers of the page on rotational support, the centre and the sensitivity tests (`results/macros_kinsupport.tex`, prefix `\ks`).

<!-- script: paper_text_kin_support.py show: the result tables in results/, the posterior samples and the catalogues of Zenodo bundle 1 -->

### Rotation and the circular speed

The numbers of the section on the shared circular speed with asymmetric drift (`results/macros_vcdrift.tex`, prefix `\vd`).

<!-- script: paper_text_vc_drift.py show: the tables of the shared-v_c + drift model in results/ -->

### Shape

The numbers of the chapter on the shape of the old population (`results/macros_shape.tex`, prefix `\sh`).

<!-- script: paper_text_shape.py show: the result tables in results/ and the posterior samples of Zenodo bundle 1; about 10 s (script header) -->

### Simulations

The numbers of the two simulation chapters that no other generator writes (`results/macros_sims.tex`, prefix `\sm`).

<!-- script: paper_text_sims.py show: the result tables in results/, the Auriga, FIRE-2, TNG50 and Concerto files of the fetch scripts above, and AGAMA; the checks take about 10 s (script header) -->

### Simulated stars at birth

The numbers on the simulated stars at birth and soon after (`results/macros_sims_birth.tex`, prefixes `\sb` and `\sh`).

<!-- script: paper_text_sims_birth.py show: the simulation result tables in results/ -->

### Auriga and FIRE-2

The numbers of the paragraph on Auriga and FIRE-2 (`results/macros_sims_suites.tex`, prefix `\ss`).

<!-- script: paper_text_sims_suites.py show: results/sims_birth.csv and the tables in results/research/ -->
