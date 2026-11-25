"""Download the OGLE-IV field centre lists for the LMC and the Magellanic Bridge (Udalski et al. 2015) into data/raw/ogle4/.

sources.yaml entries written: ogle4_fields_lmc and ogle4_fields_mbr (tier A). The field centres define the OGLE-IV footprint
used for the RRab completeness (lmcold.ogle.field_centres, in_footprint). Note: data/raw/ogle4/ must exist (created by
fetch_ogle4_rrab.py).

Needs network access to the OGLE web server; two small text files.
Run: .venv/bin/python scripts/fetch_ogle4_fields.py
"""
import urllib.request

from lmcold.sources import DATA, sha256

# %% [markdown]
# ### Download
# The field-centre list of the LMC and of the Magellanic Bridge, each with its sha256 printed.

# %%
# (file suffix, directory on the OGLE sky pages); the Bridge directory is ogle4-MB but the file is named MBR.
for region, page in (("LMC", "ogle4-LMC"), ("MBR", "ogle4-MB")):
    out = DATA / "raw" / "ogle4" / f"fields-OGLE-IV-{region}.txt"
    with urllib.request.urlopen(f"https://ogle.astrouw.edu.pl/sky/{page}/fields-OGLE-IV-{region}.txt", timeout=30) as r:
        out.write_bytes(r.read())
    print(out.name, sha256(out))
