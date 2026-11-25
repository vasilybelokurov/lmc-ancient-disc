import numpy as np
import pandas as pd
import pytest

from lmcold import ogle, sources


def test_rectangle_edges():
    f = pd.DataFrame(dict(ra=[80.0], dec=[0.0]))
    ra = np.array([80.0, 80.0 + 0.9 * ogle.HALF_WIDTH, 80.0 + 1.1 * ogle.HALF_WIDTH, 80.0])
    dec = np.array([0.0, 0.0, 0.0, 1.1 * ogle.HALF_HEIGHT])
    assert ogle.in_footprint(ra, dec, f).tolist() == [True, True, False, False]


@pytest.mark.skipif(not (sources.DATA / "raw/ogle4/fields-OGLE-IV-LMC.txt").exists(), reason="OGLE field list absent (Zenodo bundle 1)")
def test_field_list():
    f = ogle.field_centres()
    assert len(f) == 269 + 143
    assert ((abs(f.ra - 15 * 5.3166667) < 1e-4) & (f.dec == -73.0)).sum() == 1  # LMC500
