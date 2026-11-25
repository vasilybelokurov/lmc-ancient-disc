"""lmcold.shape: K_0, the mock geometry, the sky-cell fit and the projected shape."""
import numpy as np
from scipy.special import k0 as k0_scipy

from lmcold import shape

TRUTH = np.array([30.0, 160.0, 0.6, 1.7, 0.0, 0.05, 0.11, 0.10, 0.09, 0.09, 0.08, 0.08])  # inc, theta, q, h, delta, f, floors


def test_k0_matches_scipy():
    x = np.geomspace(1e-3, 60, 2000)
    assert np.max(np.abs(np.asarray(shape.k0(x)) / k0_scipy(x) - 1)) < 1e-6


def test_mock_is_tilted_toward_the_near_side(rng):
    """Stars on the near side of the line of nodes (PA theta - 90) have smaller moduli than those on the far side."""
    from lmcold.constants import LMC
    ra, dec, dmu = shape.mock(TRUTH, 20000, rng)
    pa = np.rad2deg(np.arctan2(np.sin(np.deg2rad(ra - LMC['ra0'])), np.cos(np.deg2rad(LMC['dec0'])) * np.tan(np.deg2rad(dec))
                               - np.sin(np.deg2rad(LMC['dec0'])) * np.cos(np.deg2rad(ra - LMC['ra0'])))) % 360
    near = np.abs((pa - (TRUTH[1] - 90)) % 360 - 180) > 150
    far = np.abs((pa - (TRUTH[1] + 90)) % 360 - 180) > 150
    assert np.median(dmu[near]) < np.median(dmu[far]) - 0.05


def test_cell_fit_recovers_q_at_the_true_inclination_without_the_floor(rng):
    """Counts and median moduli in sky cells (no use of the modulus scatter): with the inclination held at its true value, q is
    recovered within 0.1, for a uniform floor and for one rising toward the centre (the scatter, and so the floor, does not enter)."""
    for sf in (None, lambda rho: 0.08 + 0.06 * np.exp(-(rho - 2) / 1.0)):
        stats = shape.cell_stats(*shape.mock(TRUTH, 15000, rng, sigma_f=sf))
        x, _ = shape.fit_cells(stats, fixed={'inc': TRUTH[0]})
        assert abs(x[2] - TRUTH[2]) < 0.1


def test_projected_matches_a_tilted_spheroid(rng):
    """projected(inc, q): for points of a Gaussian spheroid (axes 1, 1, q) tilted by inc, the sky axis ratio (sd across / sd along
    the line of nodes) and the slope of the mean depth against the sky offset across the line of nodes."""
    for inc, q in ((20.0, 0.1), (31.6, 0.56), (48.0, 0.69)):
        X = rng.normal(size=(400000, 3)) * [1, 1, q]
        i = np.deg2rad(inc)
        y = X[:, 1] * np.cos(i) - X[:, 2] * np.sin(i)  # sky offset across the line of nodes
        s = X[:, 1] * np.sin(i) + X[:, 2] * np.cos(i)  # depth
        q_sky, g = shape.projected(inc, q)
        assert abs(q_sky - y.std() / X[:, 0].std()) < 0.005
        assert abs(g - np.cov(s, y)[0, 1] / y.var()) < 0.005
