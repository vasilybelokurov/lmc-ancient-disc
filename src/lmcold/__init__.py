"""lmcold: kinematics and 3D shape of the old LMC from RR Lyrae and red giants.

Importing the package enables 64-bit floats in JAX. The geometry takes differences of ~50 kpc vectors, where float32 loses precision.

Modules (src/lmcold), grouped by role:
  - reference values and inputs: constants (LMC/SMC parameters, literature values), sources (data/sources.yaml lookup with
    sha256 check), download (resumable parallel downloads);
  - sky geometry: geometry (sky bases, disc plane, LMC cylindrical coordinates; JAX), footprint (SMC mask, distance window,
    projections), perspective (perspective proper-motion field and sky-polar velocities), ogle (OGLE-IV footprint);
  - kinematics: samples (kinematic samples as arrays), kinfit (kinematic forward model and its posterior sampling);
  - RR Lyrae: rrlyrae (OGLE-IV reader, photometric [Fe/H], relative distance modulus), spectra (spectroscopic [Fe/H] from the
    literature);
  - giants and selection: giants (quality cuts, BOSS -> APOGEE calibration), magcloud (SDSS-V BOSS carton), apogee_target
    (APOGEE-2 LMC targeting), groups and selection (selection weights), extinction (true reddening), foreground (Milky Way
    foreground fraction), mass_samples (selection-weighted giant samples for the mass);
  - mass: isochrones (giants per formed mass), mass_conversion (age/metallicity averaging), surviving_mass (present/formed
    mass), mass_response (measured-vs-true [Fe/H] response);
  - 3D shape and flattening: shape (oblate-spheroid model of the RR Lyrae sky counts and median moduli), spheroid (particle
    realisations of a fitted kinematic model), virial (flattening implied by the kinematics, tensor virial theorem);
  - simulations: sims (TNG50, Auriga and FIRE-2 LMC analogues read and observed like the LMC giants);
  - plotting and tables: maps (pixel maps, weighted quantiles), harmonics (azimuthal Fourier fits), layout (paper panel grids),
    tex (LaTeX cells of the paper tables);
  - bookkeeping: checksums (per-file sha256 lists of inputs made of many files).
Units throughout: angles on the sky in degrees (radians inside lmcold.geometry), distances in kpc, proper motions in mas/yr
(pmra includes cos dec), velocities in km/s, masses in Msun, [Fe/H] in dex. Reference values live only in lmcold.constants,
each with its source.
"""
import jax

jax.config.update("jax_enable_x64", True)
