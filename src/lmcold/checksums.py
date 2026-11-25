"""Per-file sha256 lists (data/checksums/) for inputs made of many files, such as simulation snapshots (data/sources.yaml points to
the list): check_files records the reference list once and checks the files against it afterwards.

Nothing is modelled. The assumption is that the first call is made on the files the paper used: that call writes the list,
and every later call fails if a file is missing, extra or has a different sha256.
"""
from pathlib import Path

from lmcold.sources import sha256


def check_files(files, base, listfile):
    """Check files (paths under base) against listfile, lines 'sha256  path relative to base' (data/checksums/; sources.yaml points
    to it for inputs made of many files). If listfile does not exist it is written from the files, which become the reference: run
    once on the files the paper was made with. Raises on a missing file or a mismatch; returns the number of files checked."""
    rel = sorted(Path(f).relative_to(base).as_posix() for f in files)
    if not Path(listfile).exists():
        Path(listfile).parent.mkdir(parents=True, exist_ok=True)
        Path(listfile).write_text("".join(f"{sha256(Path(base) / r)}  {r}\n" for r in rel))
        return len(rel)
    ref = dict(line.split("  ")[::-1] for line in Path(listfile).read_text().splitlines())
    if sorted(ref) != rel:
        raise SystemExit(f"{listfile}: file list differs (missing {sorted(set(ref) - set(rel))}, extra {sorted(set(rel) - set(ref))})")
    bad = [r for r in rel if sha256(Path(base) / r) != ref[r]]
    if bad:
        raise SystemExit(f"sha256 mismatch against {listfile}: {bad}")
    return len(rel)
