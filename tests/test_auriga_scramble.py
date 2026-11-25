"""scripts/auriga_scramble.py reproduces its committed table: the cohort of Au2 born before full snapshot 50, recomputed, gives
the rows of results/auriga_scramble.csv (cohort size, misalignment, v_phi/sigma_z and coherence of the real stars and of every
variant of the gravity). Needs the Auriga snapshots (scripts/fetch_auriga.py). The cohort is computed in a fresh Python process:
AGAMA's orbit integration crashes (macOS, segmentation fault) in a process that has already imported healpy, as pytest does when it
collects the other test modules."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from lmcold import sims  # noqa: E402
from lmcold.sources import ROOT  # noqa: E402


@pytest.mark.skipif(not (sims.AURIGA['Au2'] / 'snapdir_050').exists(), reason='Auriga snapshots absent')
def test_reproduces_committed_rows_for_au2_050(tmp_path):
    ref = pd.read_csv(ROOT / 'results/auriga_scramble.csv', dtype={'halo': str}).query("halo == 'Au2' and s_birth == 50")
    out_csv = tmp_path / 'cohort.csv'
    code = ("import sys, pandas as pd; sys.path.insert(0, 'scripts'); from auriga_scramble import cohort; "
            f"pd.DataFrame(cohort('Au2', 50)).to_csv('{out_csv}', index=False)")
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, check=True)
    out = pd.read_csv(out_csv, dtype={'halo': str})
    assert len(out) == len(ref) and out.n0.iloc[0] == ref.n0.iloc[0]
    key = ['variant', 't']
    a, b = out.sort_values(key).reset_index(drop=True), ref.sort_values(key).reset_index(drop=True)
    assert (a.variant == b.variant).all()
    for col in ('t', 'misalign', 'vs_meas', 'C_meas', 'vs_sim', 'C_sim'):
        assert np.allclose(a[col].astype(float), b[col].astype(float), atol=1e-3, equal_nan=True), col
