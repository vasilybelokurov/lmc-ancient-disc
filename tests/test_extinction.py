import numpy as np
import pandas as pd

from lmcold.extinction import EVI_PER_EBV, true_ebv


def test_rc_map_inside_its_boxes_sfd_outside():
    m = pd.DataFrame(dict(ra=[80.0], dec=[-69.0], box=[6.0], evi=[0.124]))  # one 6-arcmin box
    ra = np.array([80.0, 80.0 + 2.9 / 60 / np.cos(np.radians(69.0)), 80.0, 85.0])
    dec = np.array([-69.0, -69.0, -69.0 + 3.1 / 60, -69.0])
    e = true_ebv(ra, dec, np.full(4, 0.5), m)
    np.testing.assert_allclose(e, [0.124 / EVI_PER_EBV, 0.124 / EVI_PER_EBV, 0.5, 0.5])
    np.testing.assert_allclose(EVI_PER_EBV, 2.742 - 1.505)
