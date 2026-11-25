"""Gaia DR3 RR Lyrae (vari_rrlyrae, Clementini et al. 2023) within 20 deg of the LMC centre, with gaia_source astrometry.

Public Gaia archive, anonymous asynchronous job. Writes data/gaia_rrlyrae_lmc20.parquet.

sources.yaml entry written: gaia_rrlyrae_lmc20 (tier A; 40,970 rows). Read by scripts/build_rrab.py, which combines it with
OGLE-IV into the release RRab catalogue (rrab.parquet, release_rrab).
Output columns: from vari_rrlyrae the classification (best_classification: RRab, RRc or RRd), fundamental and first-overtone
periods pf, p1_o [d], the G-band Fourier parameters phi31_g [rad] and r21_g, peak_to_peak_g [mag], the number of clean G epochs,
the intensity-averaged G, BP, RP magnitudes and the photometric metallicity (Clementini et al. 2023) with its error; from
gaia_source the astrometry (ra, dec [deg], parallax [mas], pmra, pmdec [mas/yr] with errors and correlation), ruwe and the mean
G, BP, RP magnitudes.

Needs network access to the Gaia archive; no run time is recorded.
Run: .venv/bin/python scripts/fetch_gaia_rrlyrae.py
"""
from astroquery.gaia import Gaia

from lmcold.constants import LMC
from lmcold.sources import DATA, sha256

# %% [markdown]
# ### The query
# Gaia DR3 vari_rrlyrae joined with gaia_source on source_id, in a cone of 20 deg about the LMC centre.

# %%
# WHERE: the source lies within 20 deg of the adopted LMC centre (constants.LMC ra0, dec0 in deg); no other cut, so every
# Gaia DR3 RR Lyrae type (RRab, RRc, RRd) and the SMC side of the cone are kept and selected later in build_rrab.py.
QUERY = f"""
SELECT v.source_id, v.best_classification, v.pf, v.pf_error, v.p1_o, v.phi31_g, v.phi31_g_error, v.r21_g,
       v.peak_to_peak_g, v.num_clean_epochs_g, v.int_average_g, v.int_average_bp, v.int_average_rp,
       v.metallicity, v.metallicity_error,
       g.ra, g.dec, g.parallax, g.parallax_error, g.pmra, g.pmra_error, g.pmdec, g.pmdec_error, g.pmra_pmdec_corr,
       g.ruwe, g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag
FROM gaiadr3.vari_rrlyrae AS v JOIN gaiadr3.gaia_source AS g USING (source_id)
WHERE 1 = CONTAINS(POINT(g.ra, g.dec), CIRCLE({LMC['ra0']}, {LMC['dec0']}, 20))
"""

# %% [markdown]
# ### Run it and save
# One anonymous asynchronous job; the rows are saved as data/gaia_rrlyrae_lmc20.parquet and its sha256 printed.

# %%
# Submit the ADQL query as an asynchronous job, wait, and convert the result to a pandas DataFrame
t = Gaia.launch_job_async(QUERY).get_results().to_pandas()
# Save under data/ and print the row count and sha256 (to compare with sources.yaml)
out = DATA / "gaia_rrlyrae_lmc20.parquet"
t.to_parquet(out, index=False)
print(len(t), "rows;", sha256(out))
