"""Download the OGLE-IV red-clump E(V-I) map of the Magellanic Clouds (Skowron et al. 2021, ApJS 252, 23) from CDS
(J/ApJS/252/23, table1.dat.gz; format in the catalogue ReadMe) into data/raw/skowron21/.

sources.yaml entry written: skowron21_table1 (81,684 rows). The map gives the reddening of the BOSS parent in the infrared test
of figures/paper_parent_extinction.png (scripts/paper_parent_extinction.py, via lmcold.extinction).

Needs network access to CDS.
Run: .venv/bin/python scripts/fetch_skowron21.py
"""
import urllib.request

from lmcold.sources import DATA, sha256

# %% [markdown]
# ### Download
# The gzipped table of the reddening map, with its sha256 printed.

# %%
for name in ("table1.dat.gz",):
    # Local path data/raw/skowron21/<name>; fetched from the CDS FTP area of J/ApJS/252/23 (NAOJ mirror), its sha256 printed.
    out = DATA / "raw" / "skowron21" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    # Download (timeout 120 s), write the bytes and print the file's sha256.
    with urllib.request.urlopen(f"http://vizier.nao.ac.jp/ftp/J/ApJS/252/23/{name}", timeout=120) as r:
        out.write_bytes(r.read())
    print(name, sha256(out))
