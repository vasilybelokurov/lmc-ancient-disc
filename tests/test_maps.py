import numpy as np

from lmcold.maps import pixel_map, robust_var


def test_pixel_map_median():
    edges = np.array([0.0, 1.0, 2.0])
    x = np.array([0.5, 0.5, 0.5, 1.5, 0.5])
    y = np.array([0.5, 0.5, 0.5, 0.5, 1.5])
    v = np.array([1.0, 2.0, 9.0, 4.0, np.nan])
    m = pixel_map(x, y, v, edges, nmin=2)
    assert m[0, 0] == 2.0  # median of three
    assert np.isnan(m[1, 0]) and np.isnan(m[0, 1]) and np.isnan(m[1, 1])  # one star; NaN only; empty


def test_robust_var_gaussian():
    v = np.random.default_rng(4).normal(0, 3.0, 200000)
    np.testing.assert_allclose(robust_var(v), 9.0, rtol=0.02)
    m = pixel_map(np.full(v.size, 0.5), np.full(v.size, 0.5), v, np.array([0.0, 1.0]), stat=robust_var)
    np.testing.assert_allclose(m[0, 0], 9.0, rtol=0.02)


def test_wquantile_equal_weights_is_the_median():
    from lmcold.maps import wquantile
    assert wquantile(np.array([3., 1., 2.]), np.ones(3), .5) == 2.


def test_wquantile_does_not_depend_on_the_order_of_tied_values():
    """Equal values with unequal weights: the result must not depend on the order of the rows (the default argsort left their order
    to the platform; the weight next to the following distinct value moved the interpolated quantile)."""
    from lmcold.maps import wquantile
    x = np.array([0., 1., 1., 1., 2., 3.])
    w = np.array([1., 5., 1., 3., 2., 1.])
    ref = wquantile(x, w, [.3, .5, .7])
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.permutation(len(x))
        np.testing.assert_allclose(wquantile(x[p], w[p], [.3, .5, .7]), ref, rtol=0, atol=1e-12)
