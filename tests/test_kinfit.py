import numpy as np

from lmcold import kinfit, perspective
from lmcold.constants import LMC


def _sky(n, seed):
    rng = np.random.default_rng(seed)
    rho, pa = np.deg2rad(rng.uniform(2, 8, n)), rng.uniform(0, 2 * np.pi, n)
    from lmcold.geometry import sky_basis
    r0, eE, eN = (np.asarray(v) for v in sky_basis(np.deg2rad(LMC["ra0"]), np.deg2rad(LMC["dec0"])))
    u = np.cos(rho)[:, None] * r0 + np.sin(rho)[:, None] * (np.sin(pa)[:, None] * eE + np.cos(pa)[:, None] * eN)
    return np.rad2deg(np.arctan2(u[:, 1], u[:, 0])) % 360, np.rad2deg(np.arcsin(u[:, 2]))


def test_model_without_rotation_is_the_perspective_field():
    ra, dec = _sky(200, 1)
    p = kinfit.start()
    p[6] = 0.0
    d = kinfit.data(ra, dec, np.zeros(200), np.zeros(200), np.ones(200), np.ones(200), np.zeros(200))
    mra, mde, _, D = (np.asarray(v) for v in kinfit.model(p, d["rhat"], d["ea"], d["ed"]))
    pra, pde = perspective.perspective_pm(ra, dec, perspective.disc_distance(ra, dec))
    np.testing.assert_allclose([mra, mde], [pra, pde], atol=1e-10)


def test_recovers_a_centre_offset_from_a_mock():
    """Noiseless-centre mock with Gaia-like errors: the centre offset, bulk motion and rotation are recovered within ~3 sigma."""
    n = 20000
    ra, dec = _sky(n, 2)
    truth = kinfit.start()
    truth[[0, 1, 6, 7, 8]] = [0.3, 0.25, 60.0, 2.0, 40.0]
    truth[2:4] += [0.02, -0.03]
    d0 = kinfit.data(ra, dec, np.zeros(n), np.zeros(n), np.full(n, 0.15), np.full(n, 0.15), np.zeros(n))
    mra, mde, _, D = (np.asarray(v) for v in kinfit.model(truth, d0["rhat"], d0["ea"], d0["ed"]))
    rng = np.random.default_rng(3)
    s = np.hypot(0.15, 40.0 / (4.740470463533348 * D))
    d = kinfit.data(ra, dec, mra + rng.normal(0, s), mde + rng.normal(0, s), np.full(n, 0.15), np.full(n, 0.15), np.zeros(n))
    p, err, _ = kinfit.fit(d, free=("dx", "dy", "pmra0", "pmdec0", "v0", "R0", "sigma", "logit_fout"))
    for i in (0, 1, 2, 3, 6, 8):
        assert abs(p[i] - truth[i]) < 3 * err[i] + 1e-3, (kinfit.PARAMS[i], p[i], truth[i], err[i])


def test_fit_respects_bounds():
    """With few, error-dominated stars sigma can run away; it must stay within BOUNDS (k must stay positive)."""
    n = 300
    ra, dec = _sky(n, 4)
    rng = np.random.default_rng(5)
    d = kinfit.data(ra, dec, rng.normal(1.9, 2.0, n), rng.normal(0.3, 2.0, n), np.full(n, 2.0), np.full(n, 2.0), np.zeros(n))
    p, _, _ = kinfit.fit(d)
    for name in ("sigma", "k", "R0"):
        lo, hi = kinfit.BOUNDS[name]
        assert lo <= p[kinfit.PARAMS.index(name)] <= hi


def test_residuals_of_a_noiseless_mock_are_zero():
    ra, dec = _sky(100, 6)
    p = kinfit.start()
    p[[0, 1]] = [0.4, -0.2]
    d = kinfit.data(ra, dec, np.zeros(100), np.zeros(100), np.ones(100), np.ones(100), np.zeros(100))
    mra, mde, _, _ = (np.asarray(v) for v in kinfit.model(p, d["rhat"], d["ea"], d["ed"]))
    pa, vt, vr = kinfit.residual_polar(p, dict(ra=ra, dec=dec, pmra=mra, pmdec=mde))
    np.testing.assert_allclose([vt, vr], 0, atol=1e-8)
    assert LMC["ra0"] == 79.88  # the adopted centre is restored


def test_sample_two_populations_share_the_centre():
    """Two mock populations with one offset centre and bulk motion but different rotation and dispersion: the joint NUTS posterior
    contains the shared truth (no lopsided part) and each population's own v0 and sigma within 3 posterior standard deviations;
    without line-of-sight velocities vlos0 stays fixed."""
    truth = kinfit.start()
    truth[[0, 1]] = [0.4, -0.2]
    own = [np.array([70.0, 2.0, 25.0, 1.0, -8.0]), np.array([40.0, 2.0, 50.0, 1.0, -8.0])]
    ds, p0s = [], []
    for i, o in enumerate(own):
        n = 3000
        ra, dec = _sky(n, 10 + i)
        p = truth.copy()
        p[6:11] = o
        d0 = kinfit.data(ra, dec, np.zeros(n), np.zeros(n), np.full(n, 0.1), np.full(n, 0.1), np.zeros(n))
        mra, mde, _, D = (np.asarray(v) for v in kinfit.model(p, d0["rhat"], d0["ea"], d0["ed"]))
        s = np.hypot(0.1, o[2] / (4.740470463533348 * D))
        rng = np.random.default_rng(20 + i)
        ds.append(kinfit.data(ra, dec, mra + rng.normal(0, s), mde + rng.normal(0, s), np.full(n, 0.1), np.full(n, 0.1), np.zeros(n)))
        p0s.append(kinfit.start())
    # vslope held: with an offset centre a radial slope of the rotation curve trades off against the lopsided part m1s, and no
    # model of the paper frees both (fit_kinematics_mcmc.FIXED holds vslope).
    m = kinfit.sample(ds, p0s, fixed=("vslope",), num_warmup=300, num_samples=300, num_chains=1)
    post = m.get_samples()
    assert "vlos0" not in post
    assert m.get_extra_fields()["diverging"].shape == (300,)  # divergent transitions are recorded, one flag per draw
    checks = {n: truth[kinfit.PARAMS.index(n)] for n in kinfit.SHARED if n != "vlos0"}
    checks.update({"v0_0": 70.0, "v0_1": 40.0, "sigma_0": 25.0, "sigma_1": 50.0})
    for name, value in checks.items():
        x = np.asarray(post[name])
        assert abs(x.mean() - value) < 3 * x.std(), (name, x.mean(), x.std(), value)


def test_line_of_sight_and_lopsided_rotation():
    """A star at the centre moves at vlos0 along the line of sight; on the line of nodes (phi = 0) the lopsided term multiplies the
    rotation by 1 + m1c; stars without a line-of-sight velocity add nothing to the likelihood."""
    ra, dec = _sky(50, 7)
    ra, dec = np.r_[LMC["ra0"], ra], np.r_[LMC["dec0"], dec]
    d = kinfit.data(ra, dec, np.zeros(51), np.zeros(51), np.ones(51), np.ones(51), np.zeros(51))
    p = kinfit.start()
    assert abs(float(kinfit.model(p, d["rhat"], d["ea"], d["ed"])[2][0]) - LMC["vlos0"]) < 1e-8
    p_lop, p_fast = p.copy(), p.copy()
    p_lop[kinfit.PARAMS.index("m1c")] = 0.3
    p_fast[kinfit.PARAMS.index("v0")] *= 1.3
    from lmcold import geometry as geo
    r0, _, _, line, n = geo.symmetry_axis(*kinfit.centre(0., 0.), np.deg2rad(LMC["inc"]), np.deg2rad(LMC["theta"]))
    star = np.cos(np.deg2rad(3.)) * np.asarray(r0) + np.sin(np.deg2rad(3.)) * np.asarray(line)  # on the line of nodes, 3 deg out
    one = np.ones(1)
    s = kinfit.data(np.rad2deg(np.arctan2(star[1], star[0])) * one, np.rad2deg(np.arcsin(star[2])) * one, 0 * one, 0 * one, one, one,
                    0 * one)
    np.testing.assert_allclose([np.asarray(x) for x in kinfit.model(p_lop, s["rhat"], s["ea"], s["ed"])],
                               [np.asarray(x) for x in kinfit.model(p_fast, s["rhat"], s["ea"], s["ed"])], rtol=1e-10)
    d_los = kinfit.data(ra, dec, np.zeros(51), np.zeros(51), np.ones(51), np.ones(51), np.zeros(51),
                        np.full(51, np.nan), np.ones(51))
    assert float(kinfit.neg_loglike(p, d_los)) == float(kinfit.neg_loglike(p, d))


def test_distances_constrain_the_tilt():
    """Relative distance moduli of stars on a disc tilted differently from the adopted one (i = 20, theta = 160) recover that
    tilt when the proper motions carry no information (errors 100 mas/yr); stars without a modulus add nothing."""
    n = 5000
    ra, dec = _sky(n, 8)
    truth = kinfit.start()
    truth[[4, 5, 15, 16]] = [20.0, 160.0, 0.02, 0.08]
    big = np.full(n, 100.0)
    d0 = kinfit.data(ra, dec, np.zeros(n), np.zeros(n), big, big, np.zeros(n))
    D = np.asarray(kinfit.model(truth, d0["rhat"], d0["ea"], d0["ed"])[3])
    mu = truth[15] + 5 * np.log10(D / LMC["D0"]) + np.random.default_rng(9).normal(0, truth[16], n)
    d = kinfit.data(ra, dec, np.zeros(n), np.zeros(n), big, big, np.zeros(n), mu=mu)
    p, err, _ = kinfit.fit(d, free=("inc", "theta", "dmu0", "sigma_mu"))
    for name in ("inc", "theta", "dmu0", "sigma_mu"):
        i = kinfit.PARAMS.index(name)
        assert abs(p[i] - truth[i]) < 3 * err[i], (name, p[i], err[i], truth[i])
    d_nan = kinfit.data(ra, dec, np.zeros(n), np.zeros(n), big, big, np.zeros(n), mu=np.full(n, np.nan))
    assert float(kinfit.neg_loglike(truth, d_nan)) == float(kinfit.neg_loglike(truth, d0))


def test_stars_without_proper_motions():
    """Stars with NaN proper motions contribute only their other terms: with no line-of-sight velocity or modulus either, the
    likelihood is exactly 1 (-lnL = 0) for any parameters."""
    ra, dec = _sky(30, 12)
    nan = np.full(30, np.nan)
    d = kinfit.data(ra, dec, nan, nan, nan, nan, nan)
    p = kinfit.start()
    p[[0, 4]] = [0.3, 40.0]
    assert abs(float(kinfit.neg_loglike(p, d))) < 1e-12


def test_errors_are_nan_at_a_bound():
    """A mock with no outliers and no intrinsic dispersion drives logit_fout and sigma to their lower bounds: their errors must be
    NaN (the inverse Hessian is not a valid error there), and the errors of the other free parameters finite and positive."""
    n = 3000
    ra, dec = _sky(n, 7)
    truth = kinfit.start()
    truth[8] = 0.0  # no dispersion: sigma runs to its lower bound
    d0 = kinfit.data(ra, dec, np.zeros(n), np.zeros(n), np.full(n, 0.1), np.full(n, 0.1), np.zeros(n))
    mra, mde, _, _ = (np.asarray(v) for v in kinfit.model(truth, d0["rhat"], d0["ea"], d0["ed"]))
    rng = np.random.default_rng(8)
    d = kinfit.data(ra, dec, mra + rng.normal(0, 0.1, n), mde + rng.normal(0, 0.1, n), np.full(n, 0.1), np.full(n, 0.1), np.zeros(n))
    free = ("dx", "dy", "pmra0", "pmdec0", "v0", "sigma", "logit_fout")
    p, err, _ = kinfit.fit(d, free=free)
    for name in ("sigma", "logit_fout"):
        i = kinfit.PARAMS.index(name)
        assert p[i] == kinfit.BOUNDS[name][0], (name, p[i])
        assert np.isnan(err[i]), (name, err[i])
    for name in ("dx", "dy", "pmra0", "pmdec0", "v0"):
        assert np.isfinite(err[kinfit.PARAMS.index(name)]) and err[kinfit.PARAMS.index(name)] > 0, name


def test_recovers_elliptical_streaming():
    """Mock disc with m = 2 azimuthal and radial streaming and a mean radial motion (proper motions and velocities): the streaming
    parameters m2c, m2s, vr0, r2c, r2s are recovered within 3 sigma, and the inclination too."""
    n = 20000
    ra, dec = _sky(n, 9)
    truth = kinfit.start()
    truth[[6, 7, 8]] = [70.0, 2.0, 30.0]
    names = ("m2c", "m2s", "vr0", "r2c", "r2s")
    for name, v in zip(names, (-0.08, 0.04, 0.03, 0.05, -0.06)):
        truth[kinfit.PARAMS.index(name)] = v
    d0 = kinfit.data(ra, dec, np.zeros(n), np.zeros(n), np.full(n, 0.05), np.full(n, 0.05), np.zeros(n))
    mra, mde, mlos, D = (np.asarray(v) for v in kinfit.model(truth, d0["rhat"], d0["ea"], d0["ed"]))
    rng = np.random.default_rng(10)
    s = np.hypot(0.05, 30.0 / (4.740470463533348 * D))
    d = kinfit.data(ra, dec, mra + rng.normal(0, s), mde + rng.normal(0, s), np.full(n, 0.05), np.full(n, 0.05), np.zeros(n),
                    mlos + rng.normal(0, np.hypot(1.0, 25.0), n), np.ones(n))
    free = ("pmra0", "pmdec0", "inc", "theta", "v0", "R0", "sigma", "vlos0", "sigma_los") + names
    p, err, _ = kinfit.fit(d, free=free)
    for name in names + ("inc",):
        i = kinfit.PARAMS.index(name)
        assert abs(p[i] - truth[i]) < 3 * err[i], (name, p[i], truth[i], err[i])


def test_recovers_a_tilted_inner_ring():
    """Mock disc whose inner ring (rho < kinfit.RHO_TILT from the adopted centre) is tilted by (dinc, dtheta) relative to the
    outer disc (proper motions and velocities): dinc, dtheta, inc and theta are recovered within 3 sigma; with dinc = dtheta = 0
    the model is the single-plane one."""
    n = 20000
    ra, dec = _sky(n, 11)
    truth = kinfit.start()
    truth[[6, 7, 8]] = [70.0, 2.0, 30.0]
    d0 = kinfit.data(ra, dec, np.zeros(n), np.zeros(n), np.full(n, 0.05), np.full(n, 0.05), np.zeros(n))
    flat = [np.asarray(v) for v in kinfit.model(truth, d0["rhat"], d0["ea"], d0["ed"])]
    truth[[kinfit.PARAMS.index("dinc"), kinfit.PARAMS.index("dtheta")]] = [8.0, -15.0]
    mra, mde, mlos, D = (np.asarray(v) for v in kinfit.model(truth, d0["rhat"], d0["ea"], d0["ed"]))
    inner = np.rad2deg(np.arccos(np.asarray(d0["rhat"]) @ np.asarray(kinfit.geo.sky_basis(*np.deg2rad([LMC["ra0"], LMC["dec0"]]))[0]))) < kinfit.RHO_TILT
    np.testing.assert_allclose(mra[~inner], flat[0][~inner], atol=1e-10)  # the outer disc is unchanged by the tilt
    assert np.abs(mra[inner] - flat[0][inner]).max() > 1e-3
    rng = np.random.default_rng(12)
    s = np.hypot(0.05, 30.0 / (4.740470463533348 * D))
    d = kinfit.data(ra, dec, mra + rng.normal(0, s), mde + rng.normal(0, s), np.full(n, 0.05), np.full(n, 0.05), np.zeros(n),
                    mlos + rng.normal(0, np.hypot(1.0, 25.0), n), np.ones(n))
    free = ("pmra0", "pmdec0", "inc", "theta", "v0", "R0", "sigma", "vlos0", "sigma_los") + kinfit.TILT
    p, err, _ = kinfit.fit(d, free=free)
    for name in kinfit.TILT + ("inc", "theta"):
        i = kinfit.PARAMS.index(name)
        assert abs(p[i] - truth[i]) < 3 * err[i], (name, p[i], truth[i], err[i])


def test_fit_converges_to_the_minimum():
    """fit() stops at the minimum, not on the way to it: from two starts (start() and i +10, theta -20 deg), the Newton
    decrement g^T H^-1 g / 2 over the interior free parameters, the -lnL still to gain, is below 1e-4. (With the L-BFGS-B
    defaults it was 1-5e-3 here and 2.1 on a flat proper-motion-only likelihood, scripts/fig_kin_mock_physical.py 'hot thick'.)"""
    import jax
    import jax.numpy as jnp
    n = 20000
    ra, dec = _sky(n, 2)
    truth = kinfit.start()
    truth[[6, 7, 8]] = [70.0, 2.0, 40.0]
    d0 = kinfit.data(ra, dec, np.zeros(n), np.zeros(n), np.full(n, 0.05), np.full(n, 0.05), np.zeros(n))
    mra, mde, _, D = (np.asarray(v) for v in kinfit.model(truth, d0["rhat"], d0["ea"], d0["ed"]))
    rng = np.random.default_rng(3)
    s = np.hypot(0.05, 40.0 / (4.740470463533348 * D))
    d = kinfit.data(ra, dec, mra + rng.normal(0, s), mde + rng.normal(0, s), np.full(n, 0.05), np.full(n, 0.05), np.zeros(n))
    idx = np.array([kinfit.PARAMS.index(x) for x in kinfit.BASE])
    p1 = truth.copy()
    p1[4], p1[5] = truth[4] + 10, truth[5] - 20
    for p0 in (None, p1):
        p, err, _ = kinfit.fit(d, p0=p0)
        f = lambda q: kinfit.neg_loglike(jnp.asarray(p).at[idx].set(q), d)  # noqa: E731
        g, H = np.asarray(jax.grad(f)(p[idx])), np.asarray(jax.hessian(f)(p[idx]))
        ok = np.isfinite(err[idx])  # interior parameters (a parameter at a bound has a NaN error)
        assert 0.5 * g[ok] @ np.linalg.solve(H[np.ix_(ok, ok)], g[ok]) < 1e-4


def test_recovers_a_los_scale():
    """Mock in which the internal (rotation) part of the line-of-sight velocities is (1 + qlos) times that implied by the proper
    motions, qlos = 0.15 (a proper-motion / line-of-sight scale mismatch, e.g. a distance-scale error): qlos, inc and theta are
    recovered within 3 sigma, and the proper motions do not depend on qlos."""
    n = 20000
    ra, dec = _sky(n, 13)
    truth = kinfit.start()
    truth[[6, 7, 8]] = [70.0, 2.0, 30.0]
    d0 = kinfit.data(ra, dec, np.zeros(n), np.zeros(n), np.full(n, 0.05), np.full(n, 0.05), np.zeros(n))
    pm0 = np.asarray(kinfit.model(truth, d0["rhat"], d0["ea"], d0["ed"])[0])
    truth[kinfit.PARAMS.index("qlos")] = 0.15
    mra, mde, mlos, D = (np.asarray(v) for v in kinfit.model(truth, d0["rhat"], d0["ea"], d0["ed"]))
    np.testing.assert_allclose(mra, pm0, atol=1e-12)
    rng = np.random.default_rng(14)
    s = np.hypot(0.05, 30.0 / (4.740470463533348 * D))
    d = kinfit.data(ra, dec, mra + rng.normal(0, s), mde + rng.normal(0, s), np.full(n, 0.05), np.full(n, 0.05), np.zeros(n),
                    mlos + rng.normal(0, np.hypot(1.0, 25.0), n), np.ones(n))
    free = ("pmra0", "pmdec0", "inc", "theta", "v0", "R0", "sigma", "vlos0", "sigma_los", "qlos")
    p, err, _ = kinfit.fit(d, free=free)
    for name in ("qlos", "inc", "theta"):
        i = kinfit.PARAMS.index(name)
        assert abs(p[i] - truth[i]) < 3 * err[i], (name, p[i], truth[i], err[i])


def test_changing_orientation_is_the_motion_of_the_plane():
    """With di/dt and dtheta/dt [deg/Gyr] non-zero, every star gains the velocity of its point of the disc plane as the plane turns:
    the change of the model equals the finite-difference velocity of a point with fixed in-plane coordinates (X along the line of
    nodes, Y along n x line) when inc and theta advance for 1e-4 Gyr, converted with 1 kpc/Gyr = 0.977792 km/s (astropy), seen
    equally in the proper motions and the line-of-sight velocities (qlos does not scale it)."""
    from lmcold import geometry as geo
    n = 500
    ra, dec = _sky(n, 15)
    d = kinfit.data(ra, dec, np.zeros(n), np.zeros(n), np.full(n, 0.05), np.full(n, 0.05), np.zeros(n))
    p = kinfit.start()
    p[kinfit.PARAMS.index("qlos")] = 0.3
    base = [np.asarray(v) for v in kinfit.model(p, d["rhat"], d["ea"], d["ed"])]
    rate = np.array([-100.0, 40.0])  # di/dt, dtheta/dt [deg/Gyr]
    p[[kinfit.PARAMS.index("didt"), kinfit.PARAMS.index("dthdt")]] = rate
    turned = [np.asarray(v) for v in kinfit.model(p, d["rhat"], d["ea"], d["ed"])]
    ra0, dec0 = np.deg2rad(LMC["ra0"]), np.deg2rad(LMC["dec0"])

    def axes(inc, theta):
        r0, _, _, line, nrm = (np.asarray(v) for v in geo.symmetry_axis(ra0, dec0, np.deg2rad(inc), np.deg2rad(theta)))
        return r0, line, np.cross(nrm, line)

    r0, line, perp = axes(LMC["inc"], LMC["theta"])
    x = base[3][:, None] * np.asarray(d["rhat"]) - LMC["D0"] * r0  # star position relative to the centre [kpc]
    dt = 1e-4
    _, line1, perp1 = axes(LMC["inc"] + rate[0] * dt, LMC["theta"] + rate[1] * dt)
    v = ((x @ line)[:, None] * (line1 - line) + (x @ perp)[:, None] * (perp1 - perp)) / dt * 0.977792  # km/s
    k = 4.740470463533348 * base[3]
    np.testing.assert_allclose(turned[0] - base[0], np.sum(v * np.asarray(d["ea"]), 1) / k, atol=2e-4)
    np.testing.assert_allclose(turned[1] - base[1], np.sum(v * np.asarray(d["ed"]), 1) / k, atol=2e-4)
    np.testing.assert_allclose(turned[2] - base[2], np.sum(v * np.asarray(d["rhat"]), 1), atol=0.05)
    np.testing.assert_allclose(turned[3], base[3])


def test_recovers_a_changing_orientation():
    """Mock with di/dt = -100 and dtheta/dt = 40 deg/Gyr (the scale of van der Marel et al. 2002's di/dt = -103 +- 61), proper
    motions and line-of-sight velocities: both rates, inc and theta are recovered within 3 sigma."""
    n = 20000
    ra, dec = _sky(n, 16)
    truth = kinfit.start()
    truth[[6, 7, 8]] = [70.0, 2.0, 30.0]
    truth[[kinfit.PARAMS.index("didt"), kinfit.PARAMS.index("dthdt")]] = [-100.0, 40.0]
    d0 = kinfit.data(ra, dec, np.zeros(n), np.zeros(n), np.full(n, 0.05), np.full(n, 0.05), np.zeros(n))
    mra, mde, mlos, D = (np.asarray(v) for v in kinfit.model(truth, d0["rhat"], d0["ea"], d0["ed"]))
    rng = np.random.default_rng(17)
    s = np.hypot(0.05, 30.0 / (4.740470463533348 * D))
    d = kinfit.data(ra, dec, mra + rng.normal(0, s), mde + rng.normal(0, s), np.full(n, 0.05), np.full(n, 0.05), np.zeros(n),
                    mlos + rng.normal(0, np.hypot(1.0, 25.0), n), np.ones(n))
    free = ("pmra0", "pmdec0", "inc", "theta", "v0", "R0", "sigma", "vlos0", "sigma_los", "didt", "dthdt")
    p, err, _ = kinfit.fit(d, free=free)
    for name in ("didt", "dthdt", "inc", "theta"):
        i = kinfit.PARAMS.index(name)
        assert abs(p[i] - truth[i]) < 3 * err[i], (name, p[i], truth[i], err[i])


def test_fit_warns_when_the_optimiser_does_not_converge(monkeypatch):
    """When L-BFGS-B stops without convergence (forced here by one iteration), fit() warns instead of returning silently."""
    import pytest
    from scipy.optimize import minimize
    monkeypatch.setattr(kinfit, "minimize", lambda *a, **k: minimize(*a, **(k | dict(options=dict(maxiter=1)))))
    ra, dec = _sky(300, 18)
    d = kinfit.data(ra, dec, np.full(300, LMC["pmra0"]), np.full(300, LMC["pmdec0"]), np.full(300, 0.1), np.full(300, 0.1), np.zeros(300))
    with pytest.warns(RuntimeWarning, match="did not converge"):
        kinfit.fit(d)



def _ellipsoid_mock(n, seed, truth, epm=0.05, elos=2.0):
    """Stars at 2-8 deg with Gaia-like errors epm [mas/yr] and line-of-sight errors elos [km/s], velocities drawn from the model
    mean plus the disc-frame ellipsoid of truth (sigma_R, sigma_phi, sigma_z), plus the errors."""
    ra, dec = _sky(n, seed)
    d0 = kinfit.data(ra, dec, np.zeros(n), np.zeros(n), np.full(n, epm), np.full(n, epm), np.zeros(n), np.zeros(n), np.full(n, elos))
    mra, mde, mlos, D, _, eR, ephi, nrm = (np.asarray(v) for v in kinfit._model(truth, d0["rhat"], d0["ea"], d0["ed"]))
    rng = np.random.default_rng(seed + 100)
    s = truth[27:30]
    v = rng.normal(size=(n, 3)) * s  # random velocity in (e_R, e_phi, n) [km/s]
    V = v[:, :1] * eR + v[:, 1:2] * ephi + v[:, 2:] * nrm
    k = 4.740470463533348 * D
    pmra = mra + np.sum(V * np.asarray(d0["ea"]), 1) / k + rng.normal(0, epm, n)
    pmdec = mde + np.sum(V * np.asarray(d0["ed"]), 1) / k + rng.normal(0, epm, n)
    vlos = mlos + np.sum(V * np.asarray(d0["rhat"]), 1) + rng.normal(0, elos, n)
    return ra, dec, pmra, pmdec, vlos


def test_ellipsoid_isotropic_limit_is_the_default_likelihood():
    """sigma_R = sigma_phi = sigma_z = sigma = sigma_los: the ellipsoid likelihood equals the default one star by star, for stars
    with and without line-of-sight velocities and moduli, with streaming and outliers."""
    n = 400
    ra, dec = _sky(n, 7)
    rng = np.random.default_rng(8)
    vlos = np.where(rng.random(n) < 0.5, rng.normal(262, 30, n), np.nan)
    mu = np.where(rng.random(n) < 0.3, rng.normal(0, 0.1, n), np.nan)
    d = kinfit.data(ra, dec, rng.normal(1.86, 0.3, n), rng.normal(0.4, 0.3, n), rng.uniform(0.02, 0.3, n),
                    rng.uniform(0.02, 0.3, n), rng.uniform(-0.5, 0.5, n), vlos, rng.uniform(0.5, 5, n), mu)
    p = kinfit.start()
    for name, v in dict(dx=0.5, sigma=30.0, sigma_los=30.0, sigma_R=30.0, sigma_phi=30.0, sigma_z=30.0, k=1.3, m2c=-0.1,
                        r2s=-0.05, logit_fout=-3.0).items():
        p[kinfit.PARAMS.index(name)] = v
    np.testing.assert_allclose(kinfit.loglike_stars(p, d, ellipsoid=True), kinfit.loglike_stars(p, d), rtol=0, atol=1e-8)


def test_ellipsoid_axes_on_the_line_of_nodes():
    """A star on the line of nodes sees e_phi along the line of sight with weight sin i and n with cos i (e_R across it)."""
    from lmcold.geometry import sky_basis, symmetry_axis
    p = kinfit.start()
    r0, _, _, line, _ = (np.asarray(v) for v in symmetry_axis(np.deg2rad(LMC["ra0"]), np.deg2rad(LMC["dec0"]), np.deg2rad(LMC["inc"]),
                                                               np.deg2rad(LMC["theta"])))
    u = LMC["D0"] * r0 + 1.0 * line  # 1 kpc from the centre along the line of nodes
    u /= np.linalg.norm(u)
    ra, dec = np.rad2deg(np.arctan2(u[1], u[0])), np.rad2deg(np.arcsin(u[2]))
    d = kinfit.data([ra], [dec], [0.0], [0.0], [1.0], [1.0], [0.0])
    _, _, _, _, _, eR, ephi, nrm = (np.asarray(v)[0] for v in kinfit._model(p, d["rhat"], d["ea"], d["ed"]))
    rhat = np.asarray(d["rhat"])[0]
    i = np.deg2rad(LMC["inc"])
    np.testing.assert_allclose([abs(rhat @ eR), abs(rhat @ ephi), abs(rhat @ nrm)], [0, np.sin(i), np.cos(i)], atol=0.02)


def test_ellipsoid_recovers_an_anisotropic_mock():
    """PM + LOS mock with (sigma_R, sigma_phi, sigma_z) = (35, 25, 20) km/s: recovered within 3 sigma; the default model cannot
    describe it (its sigma_los and sigma are averages)."""
    truth = kinfit.start()
    truth[[6, 7]] = [60.0, 2.0]
    truth[27:31] = [35.0, 25.0, 20.0, 0.0]
    truth[10] = -10.0  # no outliers
    ra, dec, pmra, pmdec, vlos = _ellipsoid_mock(20000, 9, truth)
    n = len(ra)
    d = kinfit.data(ra, dec, pmra, pmdec, np.full(n, 0.05), np.full(n, 0.05), np.zeros(n), vlos, np.full(n, 2.0))
    free = ("v0", "R0", "k", "sigma_R", "sigma_phi", "sigma_z")
    p, err, _ = kinfit.fit(d, free=free, p0=truth, ellipsoid=True)
    for name in free:
        j = kinfit.PARAMS.index(name)
        assert abs(p[j] - truth[j]) < 3 * err[j], (name, p[j], truth[j], err[j])


def test_rotation_is_the_drift_mean_with_h_and_the_circular_curve_without():
    """kinfit.rotation: with h > 0 the asymmetric-drift mean of the radial Jeans equation (aligned ellipsoid, exponential surface
    density of scale length h, dispersions scaled by exp(gamma (R - R_SIG))); with h = 0 the circular-speed curve itself; and the
    circular curve again when the dispersions vanish."""
    R = np.linspace(1.5, 9.0, 31)
    p = kinfit.start()
    p[[kinfit.PARAMS.index(n) for n in ("v0", "R0", "vslope", "sigma_R", "sigma_phi", "dlnsig_dR")]] = [85.0, 2.5, -0.02, 38.0, 31.0, -0.05]
    vc = 85.0 * np.tanh(R / 2.5) * (1 - 0.02 * (R - kinfit.R_SIG))
    assert np.allclose(np.asarray(kinfit.rotation(p, R)), vc)  # h = 0: no drift
    p[kinfit.PARAMS.index("h")] = 1.8
    sR, sphi = 38.0 * np.exp(-0.05 * (R - kinfit.R_SIG)), 31.0 * np.exp(-0.05 * (R - kinfit.R_SIG))
    expected = np.sqrt(vc**2 - (sphi**2 + sR**2 * (R / 1.8 - 1 + 0.1 * R)))  # BT08 eq. 4.228, -2 gamma R = +0.1 R
    assert np.allclose(np.asarray(kinfit.rotation(p, R)), expected, atol=0.05)  # the smooth floor adds < 0.05 km/s here
    p[[kinfit.PARAMS.index(n) for n in ("sigma_R", "sigma_phi")]] = 1e-3
    assert np.allclose(np.asarray(kinfit.rotation(p, R)), vc, atol=0.05)  # cold limit
