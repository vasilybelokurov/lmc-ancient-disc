"""lmcold.spheroid: the realisation has the requested shape and profile, and the velocities the requested rotation and dispersions."""
import numpy as np
import pytest

from lmcold import kinfit, spheroid


def vector(v0, sigma):
    """kinfit vector with a tanh curve of amplitude v0 (R0 2 kpc) and an isotropic, radially constant ellipsoid sigma."""
    p = kinfit.start()
    p[[kinfit.PARAMS.index(n) for n in ('v0', 'R0', 'sigma_R', 'sigma_phi', 'sigma_z', 'dlnsig_dR')]] = [v0, 2.0, sigma, sigma, sigma, 0.0]
    return p


def test_spheroid_has_the_requested_axis_ratio_and_profile():
    """The spheroid's second moments give q, and face-on its profile is exponential with scale length h."""
    X = spheroid.spheroid(1.5, 0.4, 200000, np.random.default_rng(1))
    assert abs(spheroid.shape(X, np.ones(len(X))) - 0.4) < 0.02
    R = np.hypot(X[:, 0], X[:, 1])
    sigma = np.histogram(R, np.arange(2, 9))[0] / np.diff(np.pi * np.arange(2, 9) ** 2)
    assert abs(-1 / np.polyfit(np.arange(2.5, 8.5), np.log(sigma), 1)[0] - 1.5) < 0.05


def test_velocities_have_the_model_rotation_and_dispersion():
    """At R ~ 4 kpc the mean v_phi is kinfit.rotation and the standard deviation of v_z is sigma_z."""
    rng = np.random.default_rng(2)
    p = vector(60, 20)
    X = spheroid.spheroid(1.5, 0.4, 200000, rng)
    V = spheroid.velocities(p, X, rng)
    R = np.hypot(X[:, 0], X[:, 1])
    k = np.abs(R - 4) < 0.25
    vphi = (X[k, 0] * V[k, 1] - X[k, 1] * V[k, 0]) / R[k]
    assert abs(vphi.mean() - float(np.mean(kinfit.rotation(p, R[k])))) < 1.0
    assert abs(V[k, 2].std() - 20) < 0.5


def test_spheroid_rejects_non_positive_h():
    """h <= 0 (a rising profile) is refused."""
    with pytest.raises(ValueError):
        spheroid.spheroid(-7.0, 0.6, 1000, np.random.default_rng(4))
