"""Download Zenodo bundle 1 (scripts/build_bundle.py) and install it in data/, so that every figure, table and number of the paper
can be remade (the quick route of README.md).

The zip is downloaded from the Zenodo record RECORD (its md5 checked against the record) or, when it is set, from URL; then every
member is checked against the sha256 of its file in data/sources.yaml before it is written: a member that is not listed, has another checksum, or would land
outside data/ stops the installation, and a file already present with the recorded checksum is kept. Nothing in data/ is
overwritten by a file that fails its check.

Bundle 1 holds the tier-B catalogues and the posterior samples of the kinematic fits, about 250 MB (docs/release.md).
Run: .venv/bin/python scripts/fetch_release.py [--sandbox] [RECORD]   (--sandbox: the Zenodo sandbox, for testing the release)
"""
# %% [markdown]
# ### Settings
# The Zenodo record of bundle 1 (set at release), an optional direct link, and the name of the zip file.

# %%
import hashlib
import sys
import tempfile
import zipfile
from pathlib import Path

import requests
import yaml

from lmcold.sources import DATA, resolve, sha256

RECORD = 24447285   # Zenodo record id of bundle 1, set at release
URL = None   # a direct link to the bundle, used instead of RECORD (a test build of the release: an asset of the test repository)
BUNDLE = 'lmc-old-population-bundle1.zip'  # file name of the bundle in the Zenodo record


# %% [markdown]
# ### The recorded checksums
# The sha256 of every file under data/ that data/sources.yaml lists, against which each member of the bundle is checked.

# %%
def recorded():
    """{data/-relative path: sha256} of the files under data/ listed in data/sources.yaml: an entry's own sha256, or, for an entry made
    of many files ('per file: data/checksums/<list>'), the lines 'sha256  path' of that list (paths relative to the entry's file)."""
    out = {}
    # every entry of data/sources.yaml; p: its file (absolute), h: its sha256 field as a string
    for e in yaml.safe_load((DATA / 'sources.yaml').read_text()).values():
        p, h = resolve(str(e['file'])).resolve(), str(e.get('sha256'))
        # skip entries whose file lies outside data/ (e.g. the public tables kept elsewhere)
        if not p.is_relative_to(DATA.resolve()):
            continue
        # a single file: its 64-character hex sha256
        if len(h) == 64:
            out[str(p.relative_to(DATA.resolve()))] = h
        # many files: read the checksum list named in the field ('per file: data/checksums/<list>')
        elif h.startswith('per file: data/checksums/'):
            for line in (DATA.parent / h.split()[2]).read_text().splitlines():
                # each line: 'sha256  path' (two spaces), path relative to the entry's folder
                sha, rel = line.split('  ', 1)
                out[str((p / rel).relative_to(DATA.resolve()))] = sha
    return out


# %% [markdown]
# ### Install the bundle
# Each member is extracted to a temporary directory, checked, and only then moved into data/.

# %%
def install(zip_path, dest=DATA, sums=None):
    """Check every member of zip_path against sums (default: recorded()) and write the members to dest; returns the paths written."""
    # sums: {data/-relative path: sha256}
    sums = recorded() if sums is None else sums
    written = []
    with zipfile.ZipFile(zip_path) as z, tempfile.TemporaryDirectory() as tmp:
        # every member of the zip, by its path inside the archive (relative to data/)
        for name in z.namelist():
            # refuse members that would land outside dest (e.g. '../' paths)
            target = (dest / name).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ValueError(f'{name}: outside {dest}')
            # refuse members that sources.yaml does not list
            if name not in sums:
                raise ValueError(f'{name}: not listed in data/sources.yaml')
            # extract to the temporary directory and check before anything is written to dest
            staged = Path(z.extract(name, tmp))
            if sha256(staged) != sums[name]:
                raise ValueError(f'{name}: sha256 differs from data/sources.yaml')
            # a file already in place with the recorded checksum is kept
            if target.exists() and sha256(target) == sums[name]:
                continue
            # move the checked member into place
            target.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(target)
            written.append(target)
    return written


# %% [markdown]
# ### Download the bundle
# From a direct link, or from the Zenodo record with the md5 that Zenodo gives checked.

# %%
def fetch(url):
    """Path of the bundle downloaded from `url` (install() then checks every member against data/sources.yaml)."""
    # written to the system temporary folder
    out = Path(tempfile.gettempdir()) / BUNDLE
    with requests.get(url, stream=True, timeout=60) as r, open(out, 'wb') as fo:
        r.raise_for_status()
        # blocks of 1 << 20 bytes = 1 MiB
        for block in r.iter_content(1 << 20):
            fo.write(block)
    return out


def download(record, sandbox):
    """Path of the bundle downloaded from Zenodo record `record` (md5 checked against the record)."""
    # Zenodo records API; the sandbox has its own host
    api = f"https://{'sandbox.' if sandbox else ''}zenodo.org/api/records/{record}"
    # the bundle's entry in the record's file list (its download link and 'md5:<hex>' checksum)
    f = next(f for f in requests.get(api, timeout=60).json()['files'] if f['key'] == BUNDLE)
    out = Path(tempfile.gettempdir()) / BUNDLE
    # md5 accumulated while streaming, compared with the record afterwards
    md5 = hashlib.md5()
    with requests.get(f['links']['self'], stream=True, timeout=60) as r, open(out, 'wb') as fo:
        r.raise_for_status()
        for block in r.iter_content(1 << 20):
            fo.write(block); md5.update(block)
    if 'md5:' + md5.hexdigest() != f['checksum']:
        raise ValueError('download md5 differs from the Zenodo record')
    return out


if __name__ == '__main__':
    # %% [markdown]
    # ### Run
    # Download from the record given on the command line (or RECORD, or URL), then install.

    # %%
    # command-line arguments other than the --sandbox flag; the first, if any, is the record id
    args = [a for a in sys.argv[1:] if a != '--sandbox']
    record = args[0] if args else RECORD
    # neither a record nor a direct link: stop with a message
    if record is None and URL is None:
        sys.exit('give the Zenodo record id of bundle 1')
    # download from Zenodo if a record is known, else from URL; then check and install every member
    path = download(record, '--sandbox' in sys.argv) if record else fetch(URL)
    print(f'{len(install(path))} files written to {DATA}')
