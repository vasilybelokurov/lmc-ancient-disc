"""Spectroscopic [Fe/H] of LMC RR Lyrae: static CDS files (data + ReadMe) into data/raw/spec/<catalogue>/.

Borissova et al. 2004 (J/A+A/423/97) and 2006 (J/A+A/460/459): positions and [Fe/H].
Gratton et al. 2004 (J/A+A/421/937): [Fe/H] by star name; positions from Di Fabrizio et al. 2005 (J/A+A/430/603, field A in
table5, field B in table6).

sources.yaml entries written: spec_borissova04(_readme), spec_borissova06(_readme), spec_gratton04(_readme),
spec_difabrizio05_A, spec_difabrizio05_B and spec_simbad_macho. The Di Fabrizio ReadMe is downloaded too but not registered.
They are read through lmcold.spectra; these spectroscopic [Fe/H] are the reference for the RRab photometric [Fe/H] scales
(figures/paper_rrab_feh.png, scripts/paper_rrab_feh.py).

Needs network access to CDS and SIMBAD.
Run: .venv/bin/python scripts/fetch_spec_rrl.py
"""
import urllib.request

from lmcold.sources import DATA, sha256

# %% [markdown]
# ### The CDS files
# The data table(s) and ReadMe of each catalogue, into data/raw/spec/<catalogue>/.

# %%
# CDS catalogue -> data files (from the VizieR mirror at NAOJ); each directory also gets its ReadMe (CDS byte-by-byte format).
FILES = {"J/A+A/423/97": ["macho.dat"], "J/A+A/460/459": ["table9.dat"], "J/A+A/421/937": ["table7.dat"],
         "J/A+A/430/603": ["table5.dat", "table6.dat"]}

for cat, names in FILES.items():
    # e.g. J/A+A/423/97 -> data/raw/spec/J_AA_423_97
    out = DATA / "raw" / "spec" / cat.replace("/", "_").replace("+", "")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("ReadMe", *names):
        with urllib.request.urlopen(f"http://vizier.nao.ac.jp/ftp/{cat}/{name}", timeout=30) as r:
            (out / name).write_bytes(r.read())
        print(out.name, name, sha256(out / name))

# %% [markdown]
# ### Positions from SIMBAD
# The stars of Borissova et al. (2004, 2006) are named, not positioned; SIMBAD gives their positions.

# %%
# Borissova et al. identify stars by MACHO name only: resolve them in SIMBAD (main_id is the OGLE-IV name for most).
from astropy.io import ascii  # noqa: E402
from astroquery.simbad import Simbad  # noqa: E402

names = []
for cat, f in (("J_AA_423_97", "macho.dat"), ("J_AA_460_459", "table9.dat")):
    d = DATA / "raw" / "spec" / cat
    t = ascii.read(d / f, readme=d / "ReadMe", format="cds")
    # Borissova+04 lists MACHO ids (queried as 'MACHO <id>'); Borissova+06 gives full names in column Name.
    names += [f"MACHO {n}" if cat == "J_AA_423_97" else str(n) for n in t["MACHO" if cat == "J_AA_423_97" else "Name"]]
Simbad.TIMEOUT = 30
Simbad.server = "simbad.harvard.edu"   # SIMBAD mirror at Harvard (the same database as simbad.cds.unistra.fr)
# Output columns: user_specified_id (the name queried), main_id, ra, dec (SIMBAD ICRS, deg).
r = Simbad.query_objects(sorted(set(names)))[["user_specified_id", "main_id", "ra", "dec"]].to_pandas()
out = DATA / "raw" / "spec" / "simbad_macho.csv"
r.to_csv(out, index=False)
print(out.name, len(r), sha256(out))
