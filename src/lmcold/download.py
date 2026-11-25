"""Downloads from throttled archives (the SDSS SAS gives ~0.2 MB/s per connection): byte ranges over parallel connections,
kept as chunks until the file is complete, so an interrupted run resumes.

Used by the fetch scripts of large tier-A inputs (scripts/fetch_giants.py, fetch_targetdb.py, fetch_targeting.py,
fetch_andrae23.py). Chunks of `out` are written to the directory `<out>.parts/` as files 00000, 00001, ... and removed once
`out` has been assembled.
"""
import hashlib
from concurrent.futures import ThreadPoolExecutor

import requests

CHUNK = 32 << 20  # bytes per range request (32 MiB)


def _chunk(url, part, lo, hi):
    """Fetch bytes lo..hi (inclusive) of url into the file part, unless part already holds them; up to 5 attempts.

    Parameters
    ----------
    url : str
    part : pathlib.Path
        Chunk file.
    lo, hi : int
        First and last byte (HTTP Range convention, inclusive).

    Raises
    ------
    RuntimeError
        If no attempt returned the full range with status 206 (Partial Content).
    """
    if part.exists() and part.stat().st_size == hi - lo + 1:
        return
    for _ in range(5):
        try:
            r = requests.get(url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
            if r.status_code == 206 and len(r.content) == hi - lo + 1:
                part.write_bytes(r.content)
                return
        except requests.RequestException:
            pass
    raise RuntimeError(f"{url}: bytes {lo}-{hi} failed")


def fetch(url, out, sha1=None, workers=8):
    """Download url to out (a Path) unless it is already complete; check the SHA-1 if given.

    Parameters
    ----------
    url : str
        Address; the server must report content-length and honour byte ranges.
    out : pathlib.Path
        Destination; a file of the server's size is taken as complete and not downloaded again.
    sha1 : str, optional
        Expected hex SHA-1 of the file (e.g. as published by the SDSS SAS); checked even if nothing was downloaded.
    workers : int
        Parallel connections.

    Raises
    ------
    AssertionError
        If the SHA-1 does not match.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    size = int(requests.head(url, timeout=60).headers["content-length"])
    if not (out.exists() and out.stat().st_size == size):
        parts = out.parent / (out.name + ".parts")
        parts.mkdir(exist_ok=True)
        # (chunk file, first byte, last byte) for consecutive CHUNK-sized ranges; the last one ends at size - 1.
        ranges = [(parts / f"{i:05d}", lo, min(lo + CHUNK, size) - 1) for i, lo in enumerate(range(0, size, CHUNK))]
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(lambda a: _chunk(url, *a), ranges))
        with open(out, "wb") as f:
            for part, _, _ in ranges:
                f.write(part.read_bytes())
        for part, _, _ in ranges:
            part.unlink()
        parts.rmdir()
    if sha1:
        h = hashlib.sha1()
        with open(out, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
        assert h.hexdigest() == sha1, f"{out}: sha1 mismatch"
