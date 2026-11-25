"""Parent sample of the BOSS selection function: Gaia DR3 stars within 12 deg of the LMC centre that the SDSS-V carton
mwm_magcloud_rgb_boss selects (lmcold.magcloud). The carton's rough cuts (G <= 17.5, BP - RP >= 1, |pmra|, |pmdec| <= 3,
parallax) run in the Gaia@AIP TAP service (Gaia partner data centre, gaia.aip.de; the ESA archive times out anonymous jobs that
scan the ~10^7 sources per few deg^2 of the LMC); SFD98 E(B-V) from dustmaps (Schlegel et al. 1998); the PM and CMD cuts in Python.
Writes data/boss_parent.parquet (all rough-cut stars, with `eligible`).

sources.yaml entry written: release_boss_parent (tier B). Inputs: Gaia DR3 gaia_source (queried live; no file entry) and the SFD98
map installed with dustmaps (not in sources.yaml). Columns: the gaia_source columns of QUERY (ra, dec [deg]; pmra, pmdec and
errors [mas/yr]; parallax and error [mas]; ruwe; g, bp, rp [mag]), ebv (SFD98 E(B-V) [mag], before the carton's rescaling) and
eligible (lmcold.magcloud.eligible: the full carton selection, including the Magellanic Stream proper-motion window and the
dereddened colour-magnitude polygon). The eligible stars are the parent of the BOSS selection function (boss_observed.py,
build_boss_cartons.py, build_xp_andrae.py, boss_selection.py; figures/paper_selection.png).

Run: .venv/bin/python scripts/build_boss_parent.py
"""
from astropy.coordinates import SkyCoord
import pyvo
from dustmaps.sfd import SFDQuery

from lmcold import magcloud
from lmcold.constants import LMC
from lmcold.sources import DATA, sha256

# %% [markdown]
# ### The query
# The rough cuts of the carton, run in the Gaia archive: a superset of the stars the
# carton selects.

# %%
# WHERE (the carton's rough cuts, a superset of the eligible stars):
#   - within 12 deg of the adopted LMC centre (constants.LMC; the radius of the giant samples);
#   - G <= 17.5 (the carton's final magnitude limit) and observed BP - RP >= 1 (red giants);
#   - -3 <= pmra, pmdec <= 3 mas/yr (the carton's PM box);
#   - parallax not significant: parallax <= 0, or parallax + 0.025 mas (the carton's zero-point offset) <= 5 parallax_error.
#     Rows with NULL parallax or proper motion fail these tests and are not returned.
QUERY = f"""
SELECT source_id, ra, dec, pmra, pmdec, pmra_error, pmdec_error, pmra_pmdec_corr, parallax, parallax_error, ruwe,
       phot_g_mean_mag AS g, phot_bp_mean_mag AS bp, phot_rp_mean_mag AS rp
FROM gaiadr3.gaia_source
WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {LMC['ra0']}, {LMC['dec0']}, 12))
  AND phot_g_mean_mag <= 17.5 AND phot_bp_mean_mag - phot_rp_mean_mag >= 1
  AND pmra BETWEEN -3 AND 3 AND pmdec BETWEEN -3 AND 3
  AND (parallax <= 0 OR parallax + 0.025 <= 5 * parallax_error)
"""

# %% [markdown]
# ### Load the data
# The Gaia DR3 stars passing the rough cuts, from the Gaia@AIP TAP service, sorted by source_id.

# %%
# Asynchronous job; maxrec above the expected ~6 x 10^5 rows so the result is not truncated.
job = pyvo.dal.TAPService("https://gaia.aip.de/tap").submit_job(QUERY, maxrec=5_000_000)
job.run().wait(phases=["COMPLETED", "ERROR", "ABORTED"])
job.raise_if_error()
t = job.fetch_result().to_table().to_pandas().sort_values("source_id", ignore_index=True)  # the server's row order is not fixed

# %% [markdown]
# ### Carton selection
# The SFD98 reddening of each star and the full carton selection (`lmcold.magcloud.eligible`).

# %%
t["ebv"] = SFDQuery()(SkyCoord(t.ra.to_numpy(), t.dec.to_numpy(), unit="deg"))
t["eligible"] = magcloud.eligible(*(t[c].to_numpy(float) for c in ("ra", "dec", "pmra", "pmdec", "parallax", "parallax_error",
                                                                    "g", "bp", "rp", "ebv")))

# %% [markdown]
# ### Write the parent
# All rough-cut stars with the flag `eligible`: `data/boss_parent.parquet`.

# %%
t.to_parquet(DATA / "boss_parent.parquet", index=False)
print(f"{len(t)} rough-cut stars, {int(t.eligible.sum())} eligible;", sha256(DATA / "boss_parent.parquet"))
