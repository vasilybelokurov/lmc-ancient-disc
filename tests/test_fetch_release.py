"""scripts/fetch_release.install: members are checked before anything is written; a bad member stops the installation."""
import hashlib
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fetch_release import install  # noqa: E402


def _zip(path, members):
    with zipfile.ZipFile(path, 'w') as z:
        for name, data in members.items():
            z.writestr(name, data)
    return path


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def test_installs_checked_members_and_keeps_good_files(tmp_path):
    dest = tmp_path / 'data'
    (dest / 'raw').mkdir(parents=True)
    (dest / 'raw/kept.txt').write_bytes(b'kept')
    z = _zip(tmp_path / 'b.zip', {'a.csv': b'1,2\n', 'raw/kept.txt': b'kept'})
    written = install(z, dest, {'a.csv': _sha(b'1,2\n'), 'raw/kept.txt': _sha(b'kept')})
    assert [p.name for p in written] == ['a.csv'] and (dest / 'a.csv').read_bytes() == b'1,2\n'


@pytest.mark.parametrize('members, sums, msg', [
    ({'a.csv': b'corrupt'}, {'a.csv': _sha(b'good')}, 'sha256'),
    ({'b.csv': b'x'}, {'a.csv': _sha(b'x')}, 'not listed'),
    ({'../evil.csv': b'x'}, {'../evil.csv': _sha(b'x')}, 'outside'),
])
def test_bad_member_stops_without_writing(tmp_path, members, sums, msg):
    dest = tmp_path / 'data'
    dest.mkdir()
    with pytest.raises(ValueError, match=msg):
        install(_zip(tmp_path / 'b.zip', members), dest, sums)
    assert not any(dest.iterdir())


def test_fetch_downloads_the_bundle_from_a_direct_link(monkeypatch, tmp_path):
    import fetch_release

    class Reply:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def raise_for_status(self): pass
        def iter_content(self, n): yield b'zip bytes'
    seen = []
    monkeypatch.setattr(fetch_release.requests, 'get', lambda url, **kw: seen.append(url) or Reply())
    monkeypatch.setattr(fetch_release.tempfile, 'gettempdir', lambda: str(tmp_path))
    path = fetch_release.fetch('https://example.org/b.zip')
    assert seen == ['https://example.org/b.zip'] and path.read_bytes() == b'zip bytes'
