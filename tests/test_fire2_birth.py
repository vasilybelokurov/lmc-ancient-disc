"""scripts/fire2_birth.py reproduces results/research/fire_early_birth.csv, the FIRE-2 birth table made in the earlier analysis.
Needs the FIRE-2 files (scripts/fetch_fire2.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from lmcold import sims  # noqa: E402
from lmcold.sources import ROOT  # noqa: E402


@pytest.mark.skipif(not all((d / 'track/host_coordinates.hdf5').exists() for d in sims.FIRE.values()), reason='FIRE-2 files absent')
def test_reproduces_committed_table():
    from fire2_birth import cohorts
    ref = pd.read_csv(ROOT / 'results/research/fire_early_birth.csv')
    out = pd.DataFrame([r for name in sims.FIRE for r in cohorts(name)])
    assert list(out.columns) == list(ref.columns) and len(out) == len(ref)
    assert (out.halo.values == ref.halo.values).all() and (out.n.values == ref.n.values).all()
    for col in ('width', 'lb', 'vphi', 'sz'):
        assert np.allclose(out[col], ref[col], atol=1e-4), col   # the table is written to 4 decimals
    assert np.allclose(out.tilt, ref.tilt, atol=1e-3)   # deg; arccos near 0 deg amplifies rounding (max 1.5e-4 found)
