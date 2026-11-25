import numpy as np

from lmcold import footprint
from lmcold.constants import LMC, SMC


def test_angular_distance_lmc_smc():
    """LMC (vdMK14 centre) to SMC centre is ~20.6 deg; a point to itself is 0."""
    d = footprint.angular_distance(LMC["ra0"], LMC["dec0"], SMC["ra0"], SMC["dec0"])
    assert 20.4 < d < 20.8
    assert footprint.angular_distance(10.0, -70.0, 10.0, -70.0) < 1e-6


def test_smc_mask():
    ra = np.array([SMC["ra0"], LMC["ra0"]])
    dec = np.array([SMC["dec0"], LMC["dec0"]])
    assert footprint.smc_masked(ra, dec).tolist() == [True, False]


def test_small_circle_radius():
    ra, dec = footprint.small_circle(SMC["ra0"], SMC["dec0"], 13.0)
    np.testing.assert_allclose(footprint.angular_distance(ra, dec, SMC["ra0"], SMC["dec0"]), 13.0, atol=1e-9)


def test_lmc_xy_centre_and_east():
    x, y = footprint.lmc_xy(np.array([LMC["ra0"], LMC["ra0"] + 1.0]), np.array([LMC["dec0"]] * 2))
    assert abs(x[0]) < 1e-9 and abs(y[0]) < 1e-9 and x[1] > 0  # increasing RA is east


def test_disc_circle_exact_points():
    """Points 0 and 180 (phi = 0, pi) lie on the line of nodes at distance sqrt(D0^2 + R^2): rho = arctan(R / D0). Point 270
    (phi = 3 pi / 2) is on the far side: rho = arctan(R cos i / (D0 + R sin i))."""
    R, inc, D0 = 4.0, 34.0, 50.1
    ra, dec = footprint.disc_circle(R, inc, 139.1, D0)
    rho = footprint.angular_distance(ra, dec, LMC["ra0"], LMC["dec0"])
    np.testing.assert_allclose(rho[[0, 180]], np.rad2deg(np.arctan(R / D0)), rtol=1e-9)
    i = np.deg2rad(inc)
    np.testing.assert_allclose(rho[270], np.rad2deg(np.arctan(R * np.cos(i) / (D0 + R * np.sin(i)))), rtol=1e-9)
