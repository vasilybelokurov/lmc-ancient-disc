"""Download the TNG50-1 LMC analogues (paper Section 6): star-particle cutouts at z = 0 of every central subhalo with bound stellar
mass 1.5-5e9 Msun and bound total mass 5e10-5e11 Msun (Nelson et al. 2019, MNRAS 490, 3234; Pillepich et al. 2019, MNRAS 490, 3196;
API https://www.tng-project.org/api/, which needs a registered key).

Fields: STAR_FIELDS of PartType4. Each cutout goes to lmcold.sims.TNG_DIR/cutout_<id>.hdf5 (cutouts already there are kept), and the
subhalo metadata of the selected objects with a cutout (position, velocity, masses) to TNG_DIR/sample.json, which lmcold.sims reads;
sample.json is rewritten (atomically) after every new cutout, so that scripts/sims_observe.py can work on the analogues already on
disk while the rest arrive; objects the server keeps failing are left out and fetched by a rerun. A second cutout of each object
with BIRTH_FIELDS (birth position and velocity, for Fig. sims_birth) goes to lmcold.sims.TNG_BIRTH_DIR/birth_<id>.hdf5; its particles
are in the order of the first. The first
120 analogues of the paper's earlier draft were a random subset (seed 1) of the same selection.

Needs a registered TNG API key (environment variable TNG_API_KEY) and network access; the z = 0 cutouts take about 2.5 GB
(docs/release.md). No run time is recorded.
Run (the key from the macOS Keychain, never printed):
    TNG_API_KEY="$(security find-generic-password -a "$USER" -s illustris-tng-api -w)" .venv/bin/python scripts/fetch_tng50.py
"""
# %% [markdown]
# ### Settings
# The API address, the particle fields of the two cutouts, the mass ranges of the analogues and the download settings.

# %%
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from lmcold.sims import TNG_BIRTH_DIR, TNG_DIR, TNG_H

BASE = 'https://www.tng-project.org/api/TNG50-1/snapshots/99/subhalos/'   # subhalo endpoint of TNG50-1 at snapshot 99 (z = 0)
# PartType4 fields of the z = 0 cutout: positions, velocities, masses, element abundances, formation scale factor
STAR_FIELDS = 'Coordinates,Velocities,Masses,GFM_Metals,GFM_StellarFormationTime'
# Fields of the second cutout: ids, formation scale factor, birth position and velocity (Fig. sims_birth)
BIRTH_FIELDS = 'ParticleIDs,GFM_StellarFormationTime,BirthPos,BirthVel'
MSTAR = (1.5e9, 5e9)    # Msun, bound stellar mass of the analogues
MTOT = (5e10, 5e11)     # Msun, bound total (subhalo) mass
# Subhalo metadata kept in sample.json (TNG units as served by the API).
META = ('id', 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'mass', 'mass_stars', 'mass_gas', 'mass_dm', 'halfmassrad_stars',
        'sfr', 'len_stars', 'starmetallicity')
WORKERS = 3   # parallel downloads
TRIES = 8     # attempts per request
LOCK = threading.Lock()   # serialises the rewriting of sample.json between threads


# %% [markdown]
# ### Requests and downloads
# A request with retries, the rewriting of sample.json, and the download of the two cutouts and the metadata of one subhalo.

# %%
def request(url):
    """Open url with the API key header (TNG_API_KEY): up to TRIES attempts, waiting 10 s x 2^k (at most 300 s) after a failure (the
    server answers 503 when busy)."""
    # Request with the key in the 'api-key' header; each attempt has a 600-s timeout
    req = urllib.request.Request(url, headers={'api-key': os.environ['TNG_API_KEY']})
    for attempt in range(TRIES):
        try:
            return urllib.request.urlopen(req, timeout=600)
        except OSError:
            # Give up after the last attempt; otherwise back off exponentially
            if attempt == TRIES - 1:
                raise
            time.sleep(min(10 * 2 ** attempt, 300))


def write_sample():
    """sample.json of the objects in KNOWN that have a cutout, written to a temporary file and renamed."""
    with LOCK:
        # Metadata of the objects whose z = 0 cutout is on disk, sorted by subhalo id
        meta = [KNOWN[i] for i in sorted(KNOWN) if (TNG_DIR / f'cutout_{i}.hdf5').exists()]
        tmp = TNG_DIR / 'sample.json.tmp'
        # Selection (catalogue count, query URL, number downloaded) and the subhalo list; the rename is atomic
        json.dump(dict(selection=dict(count=COUNT, query=QUERY, n_downloaded=len(meta)), subhalos=meta), open(tmp, 'w'), indent=1)
        tmp.rename(TNG_DIR / 'sample.json')


def fetch(i):
    """Metadata of subhalo i, and its cutout unless already on disk (then sample.json is rewritten); None if the server keeps
    failing (rerun to retry)."""
    # A subhalo whose requests all fail is reported and skipped
    try:
        m = _fetch(i)
    except OSError as e:
        print(f'subhalo {i}: {e}', flush=True)
        return None
    # A newly downloaded object enters KNOWN and sample.json is rewritten
    if i not in KNOWN:
        KNOWN[i] = m
        write_sample()
    return m


def _fetch(i):
    """Download the star and birth cutouts of subhalo i that are not on disk; return its metadata (from KNOWN or the API)."""
    # Both cutouts; each written to a '.part' file and renamed once complete, so a partial download is never used
    for out, fields in ((TNG_DIR / f'cutout_{i}.hdf5', STAR_FIELDS), (TNG_BIRTH_DIR / f'birth_{i}.hdf5', BIRTH_FIELDS)):
        if not out.exists():
            tmp = out.with_suffix('.part')
            with request(f'{BASE}{i}/cutout.hdf5?stars={fields}') as r, open(tmp, 'wb') as f:
                f.write(r.read())
            tmp.rename(out)
    if i in KNOWN:  # already in sample.json
        return KNOWN[i]
    # Metadata from the subhalo's API page, the META keys only
    meta = json.load(request(f'{BASE}{i}/'))
    return {k: meta[k] for k in META}


if __name__ == '__main__':
    # %% [markdown]
    # ### Select the analogues
    # A search of the z = 0 subhalo catalogue: centrals (primary_flag = 1) in both mass ranges, converted to the API's units of
    # 1e10 Msun/h (TNG_H is the Hubble parameter h). Subhaloes already in sample.json are kept.

    # %%
    # Search URL: masses converted from Msun to the API's 1e10 Msun/h; centrals only; up to 2000 results
    TNG_BIRTH_DIR.mkdir(exist_ok=True)
    QUERY = q = (f'{BASE}?mass_stars__gt={MSTAR[0] * TNG_H / 1e10}&mass_stars__lt={MSTAR[1] * TNG_H / 1e10}&mass__gt={MTOT[0] * TNG_H / 1e10}'
         f'&mass__lt={MTOT[1] * TNG_H / 1e10}&primary_flag=1&limit=2000')
    # KNOWN: metadata already in sample.json, by subhalo id (empty on a first run)
    old = TNG_DIR / 'sample.json'
    KNOWN = {m['id']: m for m in json.load(open(old))['subhalos']} if old.exists() else {}
    # Run the search; ids: the selected subhalo ids, COUNT: the number the catalogue reports
    res = json.load(request(q))
    ids, COUNT = sorted(r['id'] for r in res['results']), res['count']
    print(f'{res["count"]} LMC-mass centrals; {sum((TNG_DIR / f"cutout_{i}.hdf5").exists() for i in ids)} already on disk', flush=True)

    # %% [markdown]
    # ### Download
    # The cutouts of every selected subhalo, WORKERS at a time; a rerun fetches the ones that failed.

    # %%
    # Download WORKERS subhaloes at a time; failures (None) are dropped; final rewrite of sample.json
    with ThreadPoolExecutor(WORKERS) as ex:
        meta = [m for m in ex.map(fetch, ids) if m is not None]
    write_sample()
    print(f'{len(meta)} of {len(ids)} cutouts in {TNG_DIR}' + ('' if len(meta) == len(ids) else ': rerun for the rest'))
