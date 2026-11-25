"""lmcold.checksums.check_files: the first call records the reference sha256 list, later calls pass on the same files and stop on a
changed, missing or extra file."""
import pytest

from lmcold.checksums import check_files


def test_check_files(tmp_path):
    base, lst = tmp_path / 'data', tmp_path / 'sums.sha256'
    files = [base / 'a/x.bin', base / 'y.bin']
    for i, f in enumerate(files):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(bytes([i]) * 100)
    assert check_files(files, base, lst) == 2 and lst.exists()   # records
    assert check_files(files, base, lst) == 2                    # verifies
    files[1].write_bytes(b'changed')
    with pytest.raises(SystemExit, match='mismatch'):
        check_files(files, base, lst)
    with pytest.raises(SystemExit, match='file list differs'):
        check_files(files[:1], base, lst)
