"""results/sims_vcirc.csv: the orbital period at R_ORB is 2 pi R_ORB / v_c, with 1 kpc/(km/s) = KPC_GYR Gyr."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd

from lmcold.constants import KPC_GYR

ROOT = Path(__file__).resolve().parents[1]


def test_orbital_period_in_gyr():
    tree = ast.parse((ROOT / 'scripts/build_sims_vcirc.py').read_text())
    r_orb = next(n.value.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'R_ORB')
    t = pd.read_csv(ROOT / 'results/sims_vcirc.csv')
    assert np.allclose(t.t_orb, 2 * np.pi * r_orb / t.vc * KPC_GYR, rtol=1e-3)
