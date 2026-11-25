"""Download SIDM Concerto Halo104, an LMC-mass host run in CDM and GroupSIDM from the same initial conditions (Nadler, Kong, Yang & Yu
2025, ApJ 991; arXiv:2503.10748), for Fig. sims_sidm (scripts/sims_sidm_shapes.py).

- z = 0 (snapshot 235): the public Zenodo record 10.5281/zenodo.14933624 (ZENODO: particles and Rockstar+consistent-trees catalogues of
  both runs, and the GroupSIDM parametric model), each tar checked against its Zenodo md5 and unpacked in CONCERTO
  (-> central/groups/carnegie_poc/enadler/zoomins/Halo104/{output_32K, output_GroupSIDM_32K}).
- z = 4, 3, 2, 1.5 (snapshots HIGHZ): not public; available from the simulation authors on request, as a Globus collection whose id
  is given in the environment variable CONCERTO_HIGHZ_GLOBUS. Transferred with globus-cli (checksums verified) to HIGHZ_DIR (the
  local Globus collection must allow writes there), then each Gadget format-1 snapshot file is cut to its header and position block
  (the shapes need positions only; masses come from the header), with the block markers checked. Skipped if the variable is unset.
Run: .venv/bin/python scripts/fetch_concerto.py   (globus-cli logged in and Globus Connect Personal running for the z > 0 part)"""
# %% [markdown]
# ### Settings
# Local directories, the two runs (CDM and GroupSIDM), the Zenodo tar files with their md5, and the snapshots at z > 0.

# %%
import hashlib
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import requests

from lmcold import sims

CONCERTO = sims.SIMS / 'concerto'   # local root of the Concerto data (lmcold.sims.SIMS, in ~/data)
HIGHZ_DIR = CONCERTO / 'highz'      # destination of the z > 0 snapshots and hlists
Z0_DIR = CONCERTO / 'central/groups/carnegie_poc/enadler/zoomins/Halo104'   # as unpacked from the Zenodo tars
RUNS = ('output_32K', 'output_GroupSIDM_32K')   # run directories: CDM, GroupSIDM
SIDM_SIGMA0_M, SIDM_W = 147.0, 120.0   # cm^2/g, km/s: the GroupSIDM cross-section of the LMC suite (Nadler et al. 2025, ApJ 991, 69)
ZENODO = 'https://zenodo.org/api/records/14933624/files'   # Zenodo API files endpoint of the z = 0 record
# Zenodo tar file -> its md5 as listed on the record.
TARS = {'LMC_Halo104_CDM_RCT.tar': 'b5bf67f0b317e0f3c5422833ac71a41a', 'LMC_Halo104_GroupSIDM_RCT.tar': '33070b3ae839673e7b9f9c7aeffa0082',
        'LMC_Halo104_GroupSIDM_parametric.tar': 'c956fe941b068921cb633a991842b483',
        'LMC_Halo104_CDM_particles.tar': 'bd2b3208ffeadc765167049d01a9d5fc',
        'LMC_Halo104_GroupSIDM_particles.tar': '37d87d2ab67dd2b5d8576e14fabfa68d'}
GLOBUS = os.environ.get('CONCERTO_HIGHZ_GLOBUS')   # private collection of the z > 0 snapshots (on request)
# Snapshots at z = 4, 3, 2, 1.5 (16 files each, k = 0..15) and the scale factor a = 1/(1+z) in the name of each Rockstar hlist.
HIGHZ = {109: '0.20066', 126: '0.24922', 149: '0.33413', 163: '0.39942'}   # snapshot: scale factor of its Rockstar hlist


# %% [markdown]
# ### Helpers
# The md5 of a file, the download of the public z = 0 data, the cut of a snapshot file to its positions, and the transfer of
# the z > 0 snapshots.

# %%
def md5(path):
    """Hexadecimal md5 of the file at `path`, read in blocks of 16 MiB."""
    h = hashlib.md5()
    # Read 1 << 24 bytes (16 MiB) at a time until an empty block (end of file).
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def zenodo():
    """Download the five Zenodo tars into CONCERTO (via a .part file), check each against its md5 and unpack it; nothing is done if
    the z = 0 snapshot of both runs is already unpacked."""
    if all((Z0_DIR / r / 'snapshot_235.0').exists() for r in RUNS):
        return
    CONCERTO.mkdir(parents=True, exist_ok=True)
    for name, sum_ in TARS.items():
        out = CONCERTO / name
        # Stream the tar in 16 MiB blocks to name.part, renamed only when complete (an interrupted download leaves no tar).
        if not out.exists():
            with requests.get(f'{ZENODO}/{name}/content', stream=True, timeout=60) as r, open(out.with_suffix('.part'), 'wb') as f:
                r.raise_for_status()
                for b in r.iter_content(1 << 24):
                    f.write(b)
            os.replace(out.with_suffix('.part'), out)
        # Stop on an md5 mismatch; otherwise unpack (filter='data' refuses absolute paths and links leaving CONCERTO).
        if md5(out) != sum_:
            raise SystemExit(f'md5 mismatch: {out}')
        with tarfile.open(out) as t:
            t.extractall(CONCERTO, filter='data')


def strip(path):
    """Cut a Gadget format-1 file to its header and position block, checking the Fortran record markers.

    The file starts with a 256-byte header record (4-byte markers before and after), then the position record of 3 float32 per
    particle (12 bytes); the six particle counts are the first 24 bytes of the header. The file is rewritten in place.
    """
    b = path.read_bytes()
    # Leading marker of the header record must be its length, 256 bytes.
    assert np.frombuffer(b[0:4], 'i4')[0] == 256
    o = 4 + 256 + 4                                            # byte offset of the position record's leading marker
    npos = np.frombuffer(b[o:o + 4], 'i4')[0]                  # length of the position record [bytes]
    assert npos == 12 * np.frombuffer(b[4:28], 'i4').sum()     # = 12 bytes x total of the six particle counts
    # Header record + position record with both markers; the trailing marker must repeat the length.
    keep = b[:o + 4 + npos + 4]
    assert np.frombuffer(keep[-4:], 'i4')[0] == npos
    # Write to a .tmp file, then rename over the original.
    path.with_suffix(path.suffix + '.tmp').write_bytes(keep)
    os.replace(path.with_suffix(path.suffix + '.tmp'), path)


def highz():
    """Transfer the z > 0 snapshot files and Rockstar hlists of both runs from the Globus collection CONCERTO_HIGHZ_GLOBUS (given
    by the simulation authors on request) to HIGHZ_DIR and cut each snapshot file to its positions; prints a note and does
    nothing if the variable is unset."""
    # (remote path, local path) of the 16 files of each snapshot and of each hlist, both runs; only missing files are fetched.
    jobs = [(f'/{r}/snapshot_{s}.{k}', HIGHZ_DIR / r / f'snapshot_{s}.{k}') for r in RUNS for s in HIGHZ for k in range(16)]
    jobs += [(f'/{r}/rockstar/hlists/hlist_{a}.list', HIGHZ_DIR / r / 'rockstar/hlists' / f'hlist_{a}.list') for r in RUNS for a in HIGHZ.values()]
    todo = [(src, dst) for src, dst in jobs if not dst.exists()]
    if todo and GLOBUS is None:
        print('z > 0 snapshots: not public; set CONCERTO_HIGHZ_GLOBUS to the collection id given by the simulation authors')
    elif todo:
        # Id of the local Globus collection, a batch file of 'src dst' lines, one transfer task with checksum verification.
        local = subprocess.run(['globus', 'endpoint', 'local-id'], capture_output=True, text=True, check=True).stdout.strip()
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            f.writelines(f'{src} {dst}\n' for src, dst in todo)
        task = subprocess.run(['globus', 'transfer', GLOBUS, local, '--batch', f.name, '--verify-checksum', '--label', 'Concerto Halo104 z1.5-4',
                               '--jmespath', 'task_id', '-F', 'unix'], capture_output=True, text=True, check=True).stdout.strip()
        # Wait for the task (polled every 120 s), then cut each transferred snapshot file to its positions.
        subprocess.run(['globus', 'task', 'wait', task, '--polling-interval', '120'], check=True)
        for _, dst in todo:
            if 'snapshot_' in dst.name:
                strip(dst)


if __name__ == '__main__':
    # %% [markdown]
    # ### Download
    # The public z = 0 data from Zenodo, then the z > 0 snapshots if the collection id is set.

    # %%
    zenodo()
    highz()
