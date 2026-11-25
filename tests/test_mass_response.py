import numpy as np
from lmcold.mass_response import response, recover_mass


def test_no_error_conversion_and_recovery():
    edges = [-3., -2., -1., 0.]
    r = response(edges, [-np.inf, -2., -1., np.inf], [1.], [0.])
    np.testing.assert_allclose(r, np.eye(3))
    eta = np.array([.0001, .0002, .0004])
    truth = np.array([1e7, 2e7, 1e8])
    count = r @ (eta * truth)
    np.testing.assert_allclose(recover_mass(count, r, eta), truth)


def test_noisy_response_conserves_counts_and_recovers_known_mass():
    r = response([-3., -2., -1., 0.], [-np.inf, -2.5, -2, -1.5, -1, -.5, np.inf],
                 [.6, .4], [.1, .3], tail_fraction=.05)
    np.testing.assert_allclose(r.sum(axis=0), 1)
    assert np.all(r >= 0)
    eta = np.array([.0001, .0002, .0004])
    truth = np.array([1e7, 2e7, 1e8])
    count = r @ (eta * truth)
    np.testing.assert_allclose(recover_mass(count, r, eta), truth, rtol=1e-9)
    naive = np.array([count[:2].sum(), count[2:4].sum(), count[4:].sum()]) / eta
    assert abs(naive[0] / truth[0] - 1) > .1


def test_sparse_subbins_do_not_bias_no_error_mass():
    # Same known metallicity bin, uneven counts in its subbins. An estimator
    # must preserve their total; observed-variance least squares downweights
    # the populated subbin and biases the inferred mass downward.
    r = response([-3., -2., -1.], [-np.inf, -2.5, -2., -1.5, np.inf], [1.], [0.])
    counts = np.array([0., 20., 100., 110.])
    eta = np.array([.0001, .0002])
    np.testing.assert_allclose(recover_mass(counts, r, eta), [20/.0001, 210/.0002], rtol=1e-5)


def test_empirical_response_columns_are_weighted_histograms_and_borrow_when_sparse():
    from lmcold.mass_response import empirical_response
    true_edges, meas_edges = [-3., -2., -1., 0.], [-np.inf, -2., -1., np.inf]
    t = np.array([-1.5] * 30 + [-.5] * 30 + [-2.5])
    m = np.array([-2.2] * 10 + [-1.5] * 20 + [-.5] * 30 + [-2.4])
    w = np.array([2.] * 10 + [1.] * 20 + [1.] * 31)
    r = empirical_response(t, m, w, true_edges, meas_edges, nmin=15)
    np.testing.assert_allclose(r.sum(axis=0), 1)
    np.testing.assert_allclose(r[:, 1], [.5, .5, 0.])  # weights 2 x 10 against 1 x 20
    # bin -3..-2 has one star: it borrows bin -2..-1, shifted down by one bin width
    assert r[0, 0] > .45


def test_empirical_response_recovers_tail_with_shrinking_errors():
    """Measured values shrink towards -0.6 and scatter; a response built from an independent calibration subset recovers the
    tail count that the measured bins overstate."""
    from lmcold.mass_response import empirical_response
    rng = np.random.default_rng(4)
    true = [-3., -2., -1.75, -1.5, -1.25, -1., -.75, -.5, -.25, 0., .5]
    meas = [-np.inf, *np.arange(-2.75, .4, .125), np.inf]  # finer than the true bins: a square inversion oscillates
    def draw(n):
        f = np.concatenate([rng.normal(-.6, .25, n), rng.uniform(-2.9, -1., n // 30)])
        f = f[(f > -3) & (f < .5)]
        return f, -.6 + .85 * (f + .6) + np.interp(f, [-2.2, -1.8, -1.4], [0, -.2, 0]) + rng.normal(0, .1 + .1 * (f < -1.5), len(f))
    f, m = draw(200000)
    fc, mc = draw(60000)
    r = empirical_response(fc, mc, np.ones(len(fc)), true, meas, nmin=15)
    counts = np.histogram(m, meas)[0].astype(float)
    truth = np.histogram(f, true)[0]
    fit = recover_mass(counts, r, np.full(len(true) - 1, 1e-4)) * 1e-4
    assert abs(fit[0] / truth[0] - 1) < .06 and abs(fit[:3].sum() / truth[:3].sum() - 1) < .02
    assert abs(np.sum(m < -2) / truth[0] - 1) > .15
