"""kb/kbtools.upstream lists the logged steps upstream of a figure with the results/ and data/ files they read, not the entries of
data/sources.yaml that the run log also records as inputs."""
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip('IPython')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kb'))
import kbtools  # noqa: E402


def test_upstream_omits_sources_entries(tmp_path, monkeypatch):
    (tmp_path / 'results').mkdir()
    man = {'a.py': {'inputs': {'data/x.parquet': 'h', 'data/sources.yaml#x': 'h'}, 'outputs': {'results/a.csv': 'h'}},
           'b.py': {'inputs': {'results/a.csv': 'h', 'data/sources.yaml#y': 'h'}, 'outputs': {'figures/b.png': 'h'}}}
    (tmp_path / 'results/manifest.yaml').write_text(yaml.safe_dump(man))
    shown = []
    monkeypatch.setattr(kbtools, 'ROOT', tmp_path)
    monkeypatch.setattr(kbtools, 'display', lambda m: shown.append(m.data))
    kbtools.upstream('b.png')
    rows = shown[0].splitlines()[2:]
    assert rows == ['| `python scripts/b.py` | `results/a.csv` | `figures/b.png` |',
                    '| `python scripts/a.py` | `data/x.parquet` | `results/a.csv` |']
