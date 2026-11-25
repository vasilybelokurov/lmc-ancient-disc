"""Download the main progenitors of a subset of the TNG50-1 LMC analogues through cosmic time (paper Section 6, Fig. sims_heating).

Samples (z = 0 subhalo ids, fixed here so the figure can be rebuilt):
- SAMPLE: 60 analogues, 11 drawn at random (numpy seed 7) from each quintile of today's v_phi/sigma_z of the stars older than 10 Gyr
  within 10 kpc, plus the five closest to the LMC in Fig. sims_kinematics (708636, 721624, 745872, 746429, 698969).
- FULL: 20 of them for every-snapshot cutouts: 708636, 746429 and 609461, and 17 chosen to span the fraction of birth
  v_phi/sigma_z kept by their oldest cohorts after the first resolved snapshot (0.01-0.88).
Products (all from https://www.tng-project.org/api/TNG50-1/, which needs a registered key), each file kept if already on disk:
- sims.TNG_TREE_DIR/sublink_<id>.hdf5: the SubLink tree (sublink/full.hdf5; the main branch is read by lmcold.sims.main_branch);
- sims.TNG_TRACK_DIR/<id>/snap_<s>.hdf5: stars (TRACK_FIELDS) of the main progenitor at every third snapshot TRACK_SNAPS;
- sims.TNG_GAS_DIR/<id>/gas_<s>.hdf5: gas (GAS_FIELDS) of the main progenitor at the full snapshots GAS_SNAPS;
- sims.TNG_FULL_DIR/<id>/snap_<s>.hdf5: stars, gas and dark matter (FULL_QUERY) of the main progenitor of the FULL sample at every
  snapshot FULL_SNAPS (z = 4.0-1.5);
- sims.TNG_DM_DIR/<id>/snap_<s>.hdf5: dark-matter positions (DM_QUERY) where there is no full cutout, for Fig. sims_settle: the FULL
  sample at DM_SNAPS_FULL (z = 1.4-1.0; gas positions and masses were fetched with them and are not used), the others at the track
  snapshots up to 51 (DM_SNAPS).

Needs a registered TNG API key (environment variable TNG_API_KEY) and network access; no size or run time is recorded.
Run (the key from the macOS Keychain, never printed):
    TNG_API_KEY="$(security find-generic-password -a "$USER" -s illustris-tng-api -w)" .venv/bin/python scripts/fetch_tng50_progenitors.py
"""
# %% [markdown]
# ### Settings
# The two samples of z = 0 subhalo ids, the snapshots of each product and the particle fields requested.

# %%
import os
from concurrent.futures import ThreadPoolExecutor

from fetch_tng50 import request
from lmcold import sims

API = 'https://www.tng-project.org/api/TNG50-1/snapshots/'   # base URL of the TNG50-1 snapshot API
# z = 0 subhalo ids of the 60 analogues (module docstring: SAMPLE).
SAMPLE = ('691655 702198 696385 714398 670355 650101 709065 717017 719882 673019 674596 663033 665506 682820 688075 670104 697949 '
          '662101 677785 679800 693067 744886 746475 710426 691358 651520 757614 710982 647687 678193 685995 651449 627683 661260 '
          '645213 721469 737577 609461 706042 657308 699432 676067 661855 737735 720707 667858 672870 727582 689767 652044 627888 '
          '687031 706344 713155 700338 708636 721624 745872 746429 698969').split()
# The 20 analogues of SAMPLE with every-snapshot cutouts (module docstring: FULL).
FULL = ('609461 627683 657308 662101 670104 676067 687031 691358 693067 697949 702198 708636 709065 713155 720707 721469 721624 '
        '744886 746429 757614').split()
TRACK_SNAPS = tuple(range(21, 100, 3))   # every third snapshot, 21-99 (99 is z = 0)
GAS_SNAPS = (21, 25, 33, 40, 50, 59)   # snapshots of the gas cutouts (full snapshots of TNG)
FULL_SNAPS = tuple(range(21, 41))   # every snapshot, z = 4.0-1.5
# Particle fields requested in each cutout (TNG field names; GFM_StellarFormationTime is the scale factor at birth).
TRACK_FIELDS = 'stars=ParticleIDs,Coordinates,Velocities,Masses,GFM_StellarFormationTime'
GAS_FIELDS = 'gas=Coordinates,Velocities,Masses,StarFormationRate,InternalEnergy'
FULL_QUERY = (TRACK_FIELDS + '&gas=ParticleIDs,Coordinates,Velocities,Masses,StarFormationRate,Density,InternalEnergy'
              '&dm=Coordinates,Velocities')
# Dark-matter snapshots: track snapshots up to 51 for most analogues; 42-51 (z = 1.4-1.0) for the FULL sample.
DM_SNAPS, DM_SNAPS_FULL = tuple(range(21, 52, 3)), (42, 45, 48, 51)
# Dark-matter queries; for FULL also gas positions and masses (fetched, not used).
DM_QUERY, DM_QUERY_FULL = 'dm=Coordinates', 'dm=Coordinates&gas=Coordinates,Masses'
WORKERS = 3  # more parallel requests make the server return errors


# %% [markdown]
# ### The files to fetch
# The SubLink tree of each analogue gives the subhalo id of its main progenitor at each snapshot, from which the cutout
# addresses are made.

# %%
def save(url, out):
    """Write the response of url to out (via a temporary file), unless out exists."""
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    # Download to <name>.part and rename, so that an interrupted file is never taken as complete.
    tmp = out.with_suffix('.part')
    with request(url) as r, open(tmp, 'wb') as f:
        f.write(r.read())
    os.replace(tmp, out)


def jobs():
    """(url, path) of every file: trees first (the cutouts need the main branch's subhalo id at each snapshot)."""
    # The SubLink tree of each analogue, from its z = 0 subhalo (snapshot 99); downloaded here, before any cutout.
    for name in SAMPLE:
        save(f'{API}99/subhalos/{name}/sublink/full.hdf5', sims.TNG_TREE_DIR / f'sublink_{name}.hdf5')
    for name in SAMPLE:
        # Snapshot -> subhalo id of the main progenitor at that snapshot.
        sub = dict(zip(*(lambda b: (b['SnapNum'], b['SubfindID']))(sims.main_branch(name))))
        # Star tracks and gas cutouts, at the snapshots where the main branch exists.
        yield from ((f'{API}{s}/subhalos/{sub[s]}/cutout.hdf5?{TRACK_FIELDS}', sims.TNG_TRACK_DIR / name / f'snap_{s:03d}.hdf5')
                    for s in TRACK_SNAPS if s in sub)
        yield from ((f'{API}{s}/subhalos/{sub[s]}/cutout.hdf5?{GAS_FIELDS}', sims.TNG_GAS_DIR / name / f'gas_{s:03d}.hdf5')
                    for s in GAS_SNAPS if s in sub)
        # Stars, gas and dark matter at every snapshot 21-40, for the FULL sample only.
        if name in FULL:
            yield from ((f'{API}{s}/subhalos/{sub[s]}/cutout.hdf5?{FULL_QUERY}', sims.TNG_FULL_DIR / name / f'snap_{s:03d}.hdf5')
                        for s in FULL_SNAPS if s in sub)
        # Dark-matter positions where there is no full cutout.
        snaps, query = (DM_SNAPS_FULL, DM_QUERY_FULL) if name in FULL else (DM_SNAPS, DM_QUERY)
        yield from ((f'{API}{s}/subhalos/{sub[s]}/cutout.hdf5?{query}', sims.TNG_DM_DIR / name / f'snap_{s:03d}.hdf5')
                    for s in snaps if s in sub)


if __name__ == '__main__':
    # %% [markdown]
    # ### Download
    # Every file not yet on disk, WORKERS at a time.

    # %%
    todo = [j for j in jobs() if not j[1].exists()]   # (url, path) of the files still missing
    print(f'{len(todo)} files to fetch', flush=True)
    # WORKERS downloads at a time.
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(lambda j: save(*j), todo))
    print('done', flush=True)
