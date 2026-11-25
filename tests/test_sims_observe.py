"""scripts/sims_observe.py merge: the per-analogue outputs of the workers are joined into results/sims_observed.csv only when every
analogue is present with all its [Fe/H] bins, in the order of lmcold.sims.names()."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import sims_observe as so

NAMES = [('TNG50', '11'), ('Auriga', 'Au6')]


def setup(tmp_path, monkeypatch, names):
    monkeypatch.setattr(so, 'PARTS', tmp_path / 'parts')
    monkeypatch.setattr(so, 'OUT', tmp_path / 'sims_observed.csv')
    monkeypatch.setattr(so.sims, 'names', lambda: NAMES)
    so.PARTS.mkdir()
    for suite, name in names:
        pd.DataFrame(dict(suite=suite, name=name, population=range(len(so.samples.FEH_BINS)), vrot4=1.0,
                          fit_warnings=0)).to_csv(so.part(suite, name), index=False)


def test_merge_joins_all_analogues_in_order(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch, NAMES[::-1])
    so.merge()
    t = pd.read_csv(so.OUT, dtype={'name': str})
    assert list(dict.fromkeys(zip(t.suite, t.name))) == NAMES and len(t) == 2 * len(so.samples.FEH_BINS)


def test_merge_refuses_a_missing_analogue(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch, NAMES[:1])
    with pytest.raises(SystemExit):
        so.merge()
    assert not so.OUT.exists()


def test_merge_refuses_an_incomplete_analogue(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch, NAMES)
    p = so.part(*NAMES[1])
    pd.read_csv(p).iloc[:-1].to_csv(p, index=False)
    with pytest.raises(SystemExit):
        so.merge()
