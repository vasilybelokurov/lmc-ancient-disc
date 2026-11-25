"""Download the public giant catalogues into ~/data (large files stay outside the repository): APOGEE DR17 allStarLite
(ASPCAP; Abdurro'uf et al. 2022) and the SDSS-V DR20 Astra BOSSNet summary (astraAllStarBossNet 0.8.1, SHA-1 from the release).
Prints sha256 for data/sources.yaml.

sources.yaml entries written: apogee_dr17_allstarlite and boss_dr20_bossnet (tier A). Both are read by
scripts/build_spec_giants.py (spectroscopic giant catalogue); allStarLite also by build_apogee_parent.py and build_giant_mass.py,
the BOSSNet summary by build_boss_observed.py. The allStarLite release gives no checksum file, so only the BOSSNet summary
is checked against its release SHA-1; lmcold.download.fetch resumes an interrupted download.

Needs network access to data.sdss.org and room for both files in ~/data; no size or run time is recorded.
Run: .venv/bin/python scripts/fetch_giants.py
"""
from pathlib import Path

from lmcold.download import fetch
from lmcold.sources import sha256

# %% [markdown]
# ### The files
# Local path, URL on the SDSS Science Archive Server (SAS) and, where the release gives one, the SHA-1 to check against.

# %%
SAS = "https://data.sdss.org/sas"   # root URL of the SDSS Science Archive Server
# local path -> (URL, release SHA-1 or None)
FILES = {
    "~/data/sdss/dr17/apogee/allStarLite-dr17-synspec_rev1.fits":
        (f"{SAS}/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/allStarLite-dr17-synspec_rev1.fits", None),
    "~/data/sdss/dr20/astra/0.8.1/astraAllStarBossNet-0.8.1.fits.gz":
        (f"{SAS}/dr20/spectro/astra/0.8.1/summary/astraAllStarBossNet-0.8.1.fits.gz", "371090ec4cef023f59099926b99a81f2217cf7f8"),
}

# %% [markdown]
# ### Download
# Each file is downloaded (resuming an interrupted download), checked against its SHA-1 where given, and its sha256 printed.

# %%
for out, (url, sha1) in FILES.items():
    out = Path(out).expanduser()   # ~ -> home folder
    fetch(url, out, sha1)   # download (resumed if partial); checked against sha1 when it is not None
    # Path, size [bytes] and sha256, for data/sources.yaml.
    print(out, out.stat().st_size, "sha256", sha256(out), flush=True)
