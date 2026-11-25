"""Download the Auriga files the paper uses (Grand et al. 2024, MNRAS 532, 1814): for the five level-3 dwarf zooms halo_2, 8, 9, 10
and 11 (lmcold.sims.AURIGA), the z = 0 snapshot (snapdir_251), its group catalogue (groups_251: the centre and velocity of the main
subhalo) and the lists of accreted stars (lists/accretedstardata/halo_<n>: in-situ / ex-situ), read by lmcold.sims for Figs
sims_kinematics and sims_birth (scripts/sims_observe.py, build_sims_birth.py); and, for scripts/auriga_scramble.py (Letter, Section 3),
the early full snapshots EARLY_SNAPS (snapdir_050, 060, 070, 090) and the star-only high-cadence outputs EARLY_OUTS
(snapdir_highfreqstars_280-1200, every 20th).

Source: the public Globus collection 'The Auriga Simulations' (GLOBUS; https://wwwmpa.mpa-garching.mpg.de/auriga/data.html), with
globus-cli (checksums verified by Globus) into the same layout under lmcold.sims.SIMS; directories already present are not
transferred again. All files used are then checked against the per-file sha256 of data/checksums/sims_auriga_l3.sha256
(sources.yaml: sims_auriga_l3), written from the files the paper was made with.
Run: .venv/bin/python scripts/fetch_auriga.py   (globus-cli logged in and Globus Connect Personal running, allowed to write under
SIMS; about 120 GB)"""
# %% [markdown]
# ### Settings
# The Globus collection, the remote directory of the level-3 dwarf zooms, and the snapshot directories to transfer.

# %%
import glob
import subprocess
import tempfile

from lmcold import sims
from lmcold.checksums import check_files
from lmcold.sources import DATA

GLOBUS = '02a2dbb8-f64d-4440-bafe-44b60b964501'   # The Auriga Simulations (Globus collection id)
REMOTE = '/level3/Halos_1e11msol'   # level-3 dwarf zooms (haloes of about 1e11 Msun) in that collection
EARLY_SNAPS = (50, 60, 70, 90)        # scripts/auriga_scramble.SNAPS
EARLY_OUTS = range(280, 1201, 20)     # scripts/auriga_scramble.OUTS
DIRS = ('snapdir_251', 'groups_251') + tuple(f'snapdir_{s:03d}' for s in EARLY_SNAPS) + tuple(
    f'snapdir_highfreqstars_{i:03d}' for i in EARLY_OUTS)
# Per directory of each halo: the z = 0 snapshot and its group catalogue, the early full snapshots and the star-only outputs.
CHECKSUMS = DATA / 'checksums/sims_auriga_l3.sha256'   # lines 'sha256  path', paths relative to lmcold.sims.SIMS


# %% [markdown]
# ### The files used
# The snapshot, group and accreted-star files of one halo that the analysis reads; only these are checked.

# %%


def used(d):
    """The files of halo directory d that lmcold.sims and scripts/auriga_scramble.py read.

    Parameters
    ----------
    d : pathlib.Path
        Local directory of one halo (a value of lmcold.sims.AURIGA).

    Returns
    -------
    list of str
        Paths of the HDF5 files: z = 0 snapshot chunks, subhalo catalogue chunks, accreted-star lists, early snapshots and
        star-only outputs.
    """
    # z = 0 snapshot chunks and subhalo-catalogue chunks of snapshot 251.
    return (glob.glob(str(d / 'snapdir_251/snapshot_251.*.hdf5')) + glob.glob(str(d / 'groups_251/fof_subhalo_tab_251.*.hdf5'))
            # Accreted-star lists, stored next to the halo directories under lists/accretedstardata/<halo>.
            + glob.glob(str(d.parent / 'lists/accretedstardata' / d.name / '*.hdf5'))
            # Early full snapshots (EARLY_SNAPS), then the star-only high-cadence outputs (EARLY_OUTS).
            + [f for s in EARLY_SNAPS for f in glob.glob(str(d / f'snapdir_{s:03d}/snapshot_{s:03d}.*.hdf5'))]
            + [f for i in EARLY_OUTS for f in glob.glob(str(d / f'snapdir_highfreqstars_{i:03d}/snapshot_highfreqstars_{i:03d}.*.hdf5'))])


if __name__ == '__main__':
    # %% [markdown]
    # ### Transfer the missing directories
    # One Globus batch transfer (remote path -> local path) of every directory not yet on disk, with checksums verified by
    # Globus; the script waits for the transfer to finish, polling every 120 s.

    # %%
    # (remote, local) pairs: every directory in DIRS of every halo, plus each halo's accreted-star list directory.
    jobs = [(f'{REMOTE}/{d.name}/{s}', d / s) for d in sims.AURIGA.values() for s in DIRS]
    jobs += [(f'{REMOTE}/lists/accretedstardata/{d.name}', d.parent / 'lists/accretedstardata' / d.name) for d in sims.AURIGA.values()]
    # Only directories not yet on disk are transferred.
    todo = [(src, dst) for src, dst in jobs if not dst.exists()]
    if todo:
        # Id of the local Globus Connect Personal endpoint (the destination).
        local = subprocess.run(['globus', 'endpoint', 'local-id'], capture_output=True, text=True, check=True).stdout.strip()
        # Batch file: one recursive transfer per line, 'source destination'.
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            f.writelines(f'--recursive {src} {dst}\n' for src, dst in todo)
        # Submit the batch with checksum verification and keep the task id; then block until the task is done.
        task = subprocess.run(['globus', 'transfer', GLOBUS, local, '--batch', f.name, '--verify-checksum', '--label', 'Auriga L3 dwarfs z0',
                               '--jmespath', 'task_id', '-F', 'unix'], capture_output=True, text=True, check=True).stdout.strip()
        subprocess.run(['globus', 'task', 'wait', task, '--polling-interval', '120'], check=True)

    # %% [markdown]
    # ### Check the files
    # Every file used is compared with its sha256 in data/checksums/sims_auriga_l3.sha256; check_files returns the number checked.

    # %%
    # Every used file of the five haloes, checked against the stored sha256 (paths relative to SIMS).
    files = [f for d in sims.AURIGA.values() for f in used(d)]
    print(check_files(files, sims.SIMS, CHECKSUMS), 'files checked against', CHECKSUMS.relative_to(DATA.parent))
