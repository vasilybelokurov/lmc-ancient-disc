"""Download the FIRE-2 public-release files the paper uses (Wetzel et al. 2023, ApJS 265, 44): for the LMC-mass runs m11d, m11e and
m11h at 7100 Msun (lmcold.sims.FIRE), the z = 0 snapshot (output/snapshot_600.hdf5) and the host's centre and velocity through time
(track/host_coordinates.hdf5), read by lmcold.sims for Figs sims_kinematics and sims_birth (scripts/sims_observe.py).

Source: the public release over HTTPS (BASE, the Flatiron mirror listed at https://flathub.flatironinstitute.org/fire). Files already
present are not downloaded again; a download goes to a .part file renamed when complete. All files are then checked against the
per-file sha256 of data/checksums/sims_fire2.sha256 (sources.yaml: sims_fire2), written from the files the paper was made with.
Run: .venv/bin/python scripts/fetch_fire2.py   (about 8 GB)"""
# %% [markdown]
# ### Settings
# The public-release address, the two files per run, and the checksum list.

# %%
import os

import requests

from lmcold import sims
from lmcold.checksums import check_files
from lmcold.sources import DATA

BASE = 'https://users.flatironinstitute.org/~mgrudic/fire2_public_release/core'  # root of the public release over HTTPS
FILES = ('output/snapshot_600.hdf5', 'track/host_coordinates.hdf5')  # the two files per run, relative to the run's folder
CHECKSUMS = DATA / 'checksums/sims_fire2.sha256'  # per-file sha256 of the files the paper used


# %% [markdown]
# ### Download helper

# %%
def download(url, out):
    """Stream `url` to the path `out` in blocks of 16 MiB through a .part file, renamed when the download is complete."""
    # create the run's folder if needed; download into <name>.part so an interrupted download never looks complete
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + '.part')
    with requests.get(url, stream=True, timeout=60) as r, open(part, 'wb') as f:
        r.raise_for_status()
        # blocks of 1 << 24 bytes = 16 MiB
        for b in r.iter_content(1 << 24):
            f.write(b)
    # rename the .part file only after the last block is written
    os.replace(part, out)


if __name__ == '__main__':
    # %% [markdown]
    # ### Download the missing files
    # For each run (lmcold.sims.FIRE: m11d, m11e, m11h at 7100 Msun), the files not yet on disk.

    # %%
    # sims.FIRE: run name -> local folder of that run
    for d in sims.FIRE.values():
        for f in FILES:
            # skip files already on disk
            if not (d / f).exists():
                download(f'{BASE}/{d.name}/{f}', d / f)
                print('downloaded', d / f)

    # %% [markdown]
    # ### Check the files
    # Every file is compared with its sha256 in data/checksums/sims_fire2.sha256.

    # %%
    # all six files (3 runs x 2 files), downloaded now or before
    files = [d / f for d in sims.FIRE.values() for f in FILES]
    # check_files compares each file's sha256 with the list; prints the number of files checked
    print(check_files(files, sims.SIMS, CHECKSUMS), 'files checked against', CHECKSUMS.relative_to(DATA.parent))
