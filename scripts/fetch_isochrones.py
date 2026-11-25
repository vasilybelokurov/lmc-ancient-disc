"""Public PARSEC v1.2S+COLIBRI grid for the BOSS/APOGEE mass conversion.

Install the pinned ezpadova revision in pyproject.toml's isochrones extra.
Request the Gaia EDR3 or 2MASS+Spitzer passbands, Kroupa IMF, zero extinction,
log age 8.6..10.13 at 0.05 steps and [M/H] -2.2..+0.4 at 0.1 steps.
--mh requests one metallicity for public/research reproducibility checks.
Per-metallicity downloads are cached so the full fetch can resume.

The grid is downloaded from the CMD web service (https://stev.oapd.inaf.it/cgi-bin/cmd) through ezpadova. Tracks: PARSEC
parsec_CAF09_v1.2S with COLIBRI parsec_CAF09_v1.2S_S_LMC_08_web for the TP-AGB; IMF: Kroupa (tab_imf/imf_kroupa_orig.dat).
The two photometric systems give the Gaia (G, BP, RP) and 2MASS (J, H, Ks) magnitudes used in the mass conversion of the
BOSS and APOGEE giants (scripts/build_giant_mass.py, build_present_mass.py, paper_mass_method.py).

sources.yaml entries written: parsec_gaiaEDR3 (--system gaiaEDR3) and parsec_2mass_spitzer (--system 2mass_spitzer), each
data/raw/parsec/<system>.parquet with 289,596 rows (all CMD columns, one row per isochrone point; MH is [M/H], logAge is
log10 of the age in yr). A JSON sidecar <system>.json records the request. Each metallicity is cached as
data/raw/parsec/<system>/<[M/H]>.parquet; with --mh only that cache file is written (for the reproducibility check of the
research download described in sources.yaml) and the combined file is not touched.

Run: .venv/bin/python scripts/fetch_isochrones.py --system gaiaEDR3 [--mh -1.0]
Requires the `isochrones` extra of pyproject.toml (pinned ezpadova).
"""
import argparse
import json

import ezpadova
import numpy as np
import pandas as pd

from lmcold.sources import DATA

# %% [markdown]
# ### Arguments
# --system: the photometric system (Gaia EDR3 or 2MASS+Spitzer); --mh: one metallicity only.

# %%
# Command-line options; the help text is the module docstring.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--system', choices=['gaiaEDR3', '2mass_spitzer'], required=True)
parser.add_argument('--mh', type=float)  # one [M/H] [dex] only (cache file, no combined grid)
args = parser.parse_args()
# Per-metallicity cache folder data/raw/parsec/<system>/.
folder = DATA / 'raw/parsec' / args.system
folder.mkdir(parents=True, exist_ok=True)
# %% [markdown]
# ### Download one metallicity at a time
# Each [M/H] of the grid is requested from the CMD service unless its cache file exists.

# %%
parts = []  # one table per [M/H]
# [M/H] grid -2.2, -2.1, ..., +0.4 (27 values), rounded to avoid floating-point labels such as -0.30000000000000004.
for mh in ([args.mh] if args.mh is not None else np.round(np.arange(-2.2, .41, .1), 2)):
    file = folder / f'{mh:+.2f}.parquet'  # cache file named by [M/H], e.g. -1.00.parquet
    # Read from the cache if present (resumable fetch), otherwise download and cache.
    if file.exists():
        t = pd.read_parquet(file)
    else:
        # One metallicity, all ages log(age/yr) = 8.6 ... 10.1 in steps of 0.05, zero extinction (A_V = 0).
        t = ezpadova.get_isochrones(logage=(8.6, 10.13, .05), MH=(float(mh), float(mh), 0.),
                                   photsys_file=args.system, track_parsec='parsec_CAF09_v1.2S',
                                   track_colibri='parsec_CAF09_v1.2S_S_LMC_08_web',
                                   imf_file='tab_imf/imf_kroupa_orig.dat', extinction_av=0.)
        t.to_parquet(file, index=False)
    # Progress: system, [M/H], number of isochrone points, and the MH values returned by the service.
    print(args.system, mh, len(t), sorted(t.MH.unique()), flush=True)
    parts.append(t)
# %% [markdown]
# ### Combine
# The full grid is written as one parquet file with a JSON record of the request.

# %%
# Only the full grid is combined into the file registered in sources.yaml.
if args.mh is None:
    # data/raw/parsec/<system>.parquet, plus a <system>.json sidecar recording the request.
    output = folder.parent / f'{args.system}.parquet'
    pd.concat(parts, ignore_index=True).to_parquet(output, index=False)
    output.with_suffix('.json').write_text(json.dumps(dict(system=args.system, logage=[8.6,10.13,.05],
        requested_mh=[-2.2,.4,.1], track='PARSEC v1.2S + COLIBRI S_LMC_08', imf='Kroupa', extinction_av=0), indent=2))
