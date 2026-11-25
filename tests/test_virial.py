"""lmcold.virial: the potential side W_xx / W_zz and the flattening implied by a kinematic model."""
import numpy as np

from lmcold import kinfit, spheroid, virial


def vector(v0, sR, sphi, sz):
    """kinfit vector: tanh curve of amplitude v0 (R0 2 kpc), radially constant ellipsoid."""
    p = kinfit.start()
    p[[kinfit.PARAMS.index(n) for n in ('v0', 'R0', 'sigma_R', 'sigma_phi', 'sigma_z', 'dlnsig_dR')]] = [v0, 2.0, sR, sphi, sz, 0.0]
    return p


def harmonic(x):
    """Gradient of Phi = 1/2 w^2 r^2 (w = 20 km/s/kpc)."""
    return 400.0 * x


def test_harmonic_potential_gives_inverse_q_squared(rng):
    """W_ii = -w^2 Sum x_i^2, so W_xx / W_zz = <x^2> / <z^2> = 1 / q^2 for the oblate tracer ."""
    for q in (0.3, 0.6, 1.0):
        X = spheroid.spheroid(0.7, q, 300000, rng)
        assert abs(virial.omega_pot(X, harmonic) * q ** 2 - 1) < 0.02


def test_round_tracer_in_spherical_potential_and_flattened_potential(rng):
    X = spheroid.spheroid(1.5, 1.0, 300000, rng)
    assert abs(virial.omega_pot(X, virial.POTENTIALS['P1']) - 1) < 0.02
    assert virial.omega_pot(X, virial.POTENTIALS['P2']) < 0.95  # a flattened potential pulls harder vertically


def test_gradients_match_finite_differences(rng):
    x = rng.normal(0, 3, (200, 3))
    for phi, grad in ((lambda y: virial.phi_log(y, 92.0, 2.0, 0.85), lambda y: virial.grad_log(y, 92.0, 2.0, 0.85)),
                      (lambda y: virial.phi_mn(y, 2.7e9, 1.6, 0.3), lambda y: virial.grad_mn(y, 2.7e9, 1.6, 0.3))):
        for k in range(3):
            d = np.zeros(3)
            d[k] = 1e-5
            assert np.allclose(grad(x)[:, k], (phi(x + d) - phi(x - d)) / 2e-5, rtol=1e-5, atol=1e-6)


def test_flattening_in_harmonic_potential_is_inverse_sqrt_of_omega_kin(rng):
    """In the harmonic potential Omega_pot = 1/q^2, so the implied q = Omega_kin^(-1/2); a non-rotating isotropic model is round and
    rotation flattens."""
    for p in (vector(0, 30, 30, 30), vector(40, 30, 25, 20)):
        q, om = virial.flattening(p, 0.7, harmonic, rng)
        assert abs(q - om ** -0.5) < 0.02
    assert virial.flattening(vector(0, 30, 30, 30), 1.5, virial.POTENTIALS['P1'], rng)[0] > 0.97
    assert virial.flattening(vector(60, 30, 30, 30), 1.5, virial.POTENTIALS['P1'], rng)[0] < 0.75


def test_kinematics_rounder_than_any_oblate_tracer_give_nan(rng):
    """A tracer hotter vertically than in the plane (Omega_kin < Omega_pot(q = 1)) has no oblate solution: q is NaN, not clipped to 1."""
    q, om = virial.flattening(vector(0, 30, 30, 45), 1.5, virial.POTENTIALS['P1'], rng)
    assert om < 1 and np.isnan(q)


def test_grid_potential_matches_a_hernquist_sphere(rng):
    """For 200,000 particles of a Hernquist (1990) sphere (M 1e10 Msun, a 2 kpc): the gradient matches g = G M / (r + a)^2 at 1-15 kpc
    (median error < 1 per cent; single points to 6 per cent, shot noise of the particles), and Omega_pot of a flattened tracer (q 0.5) agrees with the analytic
    potential's within 1 per cent."""
    M, a, n = 1e10, 2.0, 200000
    u = rng.uniform(0, 1, n)
    r = a * np.sqrt(u) / (1 - np.sqrt(u))  # inverse of M(<r) / M = r^2 / (r + a)^2
    d = rng.normal(size=(n, 3))
    grad = virial.grid_potential(r[:, None] * d / np.linalg.norm(d, axis=1)[:, None], np.full(n, M / n), rng)
    x = rng.normal(size=(500, 3))
    s = rng.uniform(1, 15, 500)
    x = s[:, None] * x / np.linalg.norm(x, axis=1)[:, None]
    err = np.abs(np.linalg.norm(grad(x), axis=1) / (virial.G * M / (s + a) ** 2) - 1)
    assert np.median(err) < 0.01 and err.max() < 0.06
    T = spheroid.spheroid(1.9, 0.5, 200000, rng)
    hernquist = lambda y: virial.G * M / (np.linalg.norm(y, axis=1) + a) ** 2 / np.linalg.norm(y, axis=1) * y.T
    assert abs(virial.omega_pot(T, grad) / virial.omega_pot(T, lambda y: hernquist(y).T) - 1) < 0.01


def test_balance_is_one_for_circular_orbits_in_a_log_potential(rng):
    """Stars on circular orbits in the plane of Phi = 1/2 V^2 ln(R^2 + a^2) at a = 0 have v^2 = V^2 = x . grad Phi: 2K / |W| = 1."""
    X = np.c_[rng.normal(size=(1000, 2)), np.zeros(1000)]
    R = np.hypot(X[:, 0], X[:, 1])
    V = 92.0 * np.c_[-X[:, 1] / R, X[:, 0] / R, np.zeros(1000)]
    assert abs(virial.balance(X, V, lambda x: virial.grad_log(x, 92.0, 0.0)) - 1) < 1e-9
