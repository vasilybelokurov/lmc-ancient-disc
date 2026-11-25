"""Tests of scripts/sims_birth_radii.py (why the simulated early discs are measured at 0.5-3 kpc)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from lmcold import sims  # noqa: E402
from sims_birth_radii import INNER, OUTER, in_annulus, radii  # noqa: E402


def test_in_annulus_uses_cylindrical_radius_and_height():
    P = np.array([[1.0, 0.0, 0.0],     # R = 1: inner
                  [0.3, 0.0, 0.0],     # R = 0.3: inside the inner edge
                  [4.0, 0.0, 0.0],     # R = 4: outer
                  [2.0, 0.0, 3.5],     # R = 2 but |z| = 3.5 > 3: not inner, but inside OUTER's |z| < 5 at R = 2 < 3: not outer
                  [0.0, 4.5, -4.0]])   # R = 4.5, |z| = 4 < 5: outer
    assert in_annulus(P, INNER).tolist() == [True, False, False, False, False]
    assert in_annulus(P, OUTER).tolist() == [False, False, True, False, True]


@pytest.mark.skipif(not (sims.AURIGA['Au2'] / 'snapdir_251').exists(), reason='Auriga snapshots absent')
def test_one_analogue_is_consistent():
    row = radii(('Auriga', 'Au2'))
    assert row['coh'] >= row['coh_inner'] >= 0 and row['coh'] >= row['coh_outer'] >= 0
    assert 0 <= row['f_born_inner'] <= 1 and 0 <= row['f_born_outer'] <= 1
    assert row['f_born_inner'] + row['f_born_outer'] <= 1 + 1e-9
