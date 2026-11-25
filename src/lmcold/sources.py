"""Input files listed in data/sources.yaml: name -> path, checked against the recorded sha256.

data/sources.yaml is the single source of truth for the inputs (origin, query or URL, fetch script, rows, sha256, reference).
Scripts read every input through path(name), so a changed or corrupted file stops the run instead of silently changing a result.
ROOT is the repository root and DATA its data/ directory.
"""
import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def sha256(path):
    """Hex SHA-256 of the file at path, read in 1 MiB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve(file):
    """A sources.yaml file entry as a path: '~/...' (large inputs in ~/data) or relative to data/.

    Parameters
    ----------
    file : str
        The `file` field of an entry.

    Returns
    -------
    pathlib.Path
    """
    p = Path(file).expanduser()
    return p if p.is_absolute() else DATA / p


def path(name):
    """Path of input `name`; raises if the file differs from the one recorded in sources.yaml.

    Parameters
    ----------
    name : str
        Top-level key in data/sources.yaml (e.g. 'skowron21_table1').

    Returns
    -------
    pathlib.Path

    Raises
    ------
    KeyError
        If name is not in sources.yaml.
    ValueError
        If the file's sha256 differs from the recorded one.
    """
    entry = yaml.safe_load(open(DATA / "sources.yaml"))[name]
    p = resolve(entry["file"])
    if sha256(p) != entry["sha256"]:
        raise ValueError(f"{p} does not match the sha256 recorded in data/sources.yaml")
    return p
