"""Download the public targeting inputs of the giant selection functions into ~/data:
SDSS-V DR20 mwmTargets (carton memberships of observed targets, Astra 0.8.1, SHA-1 from the release) and spAll-lite (BOSS
spectra with the carton each fibre was assigned under, v6_2_1); APOGEE DR17 allVisit (per-visit targeting flags) and the
APOGEE-2 targeting files of the LMC fields (apogee2Field, apogee2Object_LMC1-35). Prints sha256 for data/sources.yaml.

sources.yaml entries written: sdss5_mwmtargets, sdss5_spall_lite, apogee_dr17_allvisit, apogee2_field and
apogee2_object_lmc1 ... apogee2_object_lmc35 (tier A). spAll-lite is read by build_boss_observed.py (BOSS selection
function); allVisit, apogee2Field and apogee2Object by build_apogee_parent.py (APOGEE selection function; allVisit also by
apogee_selection.py). No script in this repository reads mwmTargets at present. Only mwmTargets has a release SHA-1 to
check against.

Needs network access to data.sdss.org and room for the files in ~/data; no size or run time is recorded.
Run: .venv/bin/python scripts/fetch_targeting.py
"""
from pathlib import Path

import requests

from lmcold.download import fetch
from lmcold.sources import sha256

# %% [markdown]
# ### The files
# The release SHA-1 list of the Astra 0.8.1 summaries, then local path, URL and SHA-1 (where there is one) of each file.

# %%
SAS = "https://data.sdss.org/sas"
# Astra 0.8.1 summary checksum file: lines '<sha1>  <name>' -> {name: sha1}.
sha = {l.split()[1]: l.split()[0] for l in requests.get(f"{SAS}/dr20/spectro/astra/0.8.1/summary/spectro_astra_0.8.1_summary.sha1sum",
                                                        timeout=60).text.splitlines()}
# local path -> (URL, release SHA-1 or None)
FILES = {
    "~/data/sdss/dr20/astra/0.8.1/mwmTargets-0.8.1.fits.gz":
        (f"{SAS}/dr20/spectro/astra/0.8.1/summary/mwmTargets-0.8.1.fits.gz", sha["mwmTargets-0.8.1.fits.gz"]),
    "~/data/sdss/dr20/boss/spAll-lite-v6_2_1.fits.gz": (f"{SAS}/dr20/spectro/boss/redux/v6_2_1/summary/daily/spAll-lite-v6_2_1.fits.gz", None),
    "~/data/sdss/dr17/apogee/allVisit-dr17-synspec_rev1.fits":
        (f"{SAS}/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/allVisit-dr17-synspec_rev1.fits", None),
    "~/data/sdss/dr17/apogee/target/apogee2Field.fits": (f"{SAS}/dr17/apogee/target/apogee2Field.fits", None),
}
FILES.update({f"~/data/sdss/dr17/apogee/target/apogee2Object_LMC{i}.fits": (f"{SAS}/dr17/apogee/target/apogee2Object/apogee2Object_LMC{i}.fits", None)
              for i in range(1, 36)})

# %% [markdown]
# ### Download
# The 36 small APOGEE-2 targeting files first, then the large summary files; each sha256 is printed.

# %%
for out, (url, sha1) in sorted(FILES.items(), key=lambda kv: "apogee2" not in kv[0]):  # small targeting files first
    out = Path(out).expanduser()
    fetch(url, out, sha1)
    print(out, out.stat().st_size, "sha256", sha256(out), flush=True)
