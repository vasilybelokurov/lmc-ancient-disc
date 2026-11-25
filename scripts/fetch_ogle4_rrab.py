"""Download the OGLE-IV LMC RR Lyrae identification and RRab parameter files (Soszynski et al. 2016) into data/raw/ogle4/.

sources.yaml entries written: ogle4_ident (identifications and positions of all LMC RR Lyrae) and ogle4_rrab (I-band light-curve
parameters of the RRab stars). Both are read by scripts/build_rrab.py (through lmcold.ogle).

Needs network access to the OGLE web server.
Run: .venv/bin/python scripts/fetch_ogle4_rrab.py
"""
import urllib.request

from lmcold.sources import DATA, sha256

# %% [markdown]
# ### Download
# Two files of the OGLE-IV Collection of Variable Stars, LMC RR Lyrae, each with its sha256 printed.

# %%
URL = "https://www.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/rrlyr/"   # OGLE-IV OCVS directory of the LMC RR Lyrae

out = DATA / "raw" / "ogle4"
out.mkdir(parents=True, exist_ok=True)
for name in ("ident.dat", "RRab.dat"):
    urllib.request.urlretrieve(URL + name, out / name)
    print(name, sha256(out / name))
