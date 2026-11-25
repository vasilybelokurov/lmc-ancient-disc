import numpy as np

from lmcold import perspective
from lmcold.constants import KAPPA, LMC


def test_perspective_at_the_centre_is_the_systemic_motion():
    pa, pd = perspective.perspective_pm(np.array([LMC["ra0"]]), np.array([LMC["dec0"]]), np.array([LMC["D0"]]))
    np.testing.assert_allclose([pa[0], pd[0]], [LMC["pmra0"], LMC["pmdec0"]], atol=1e-10)


def test_systemic_velocity_line_of_sight_component():
    V = perspective.systemic_velocity()
    r0 = np.array([np.cos(np.deg2rad(LMC["dec0"])) * np.cos(np.deg2rad(LMC["ra0"])),
                   np.cos(np.deg2rad(LMC["dec0"])) * np.sin(np.deg2rad(LMC["ra0"])), np.sin(np.deg2rad(LMC["dec0"]))])
    np.testing.assert_allclose(V @ r0, LMC["vlos0"], atol=1e-9)
    np.testing.assert_allclose(np.sqrt(V @ V - (V @ r0) ** 2), KAPPA * LMC["D0"] * np.hypot(LMC["pmra0"], LMC["pmdec0"]), rtol=1e-12)


def test_sky_polar_basis_directions():
    """A star due north of the centre has radial = north and tangential = east; due east: radial = east, tangential = south."""
    ra = np.array([LMC["ra0"], LMC["ra0"] + 2.0 / np.cos(np.deg2rad(LMC["dec0"]))])
    dec = np.array([LMC["dec0"] + 2.0, LMC["dec0"]])
    (rE, rN), (tE, tN) = perspective.sky_polar_basis(ra, dec)
    np.testing.assert_allclose([rE[0], rN[0], tE[0], tN[0]], [0, 1, 1, 0], atol=1e-9)
    np.testing.assert_allclose([rE[1], rN[1], tE[1], tN[1]], [1, 0, 0, -1], atol=0.05)  # small-angle: RA offsets are not great circles


def test_disc_distance_on_the_line_of_nodes():
    """Along the line of nodes the disc distance is D0 / cos(rho)."""
    from lmcold.footprint import disc_circle, angular_distance
    ra, dec = disc_circle(3.0, LMC["inc"], LMC["theta"], LMC["D0"])
    rho = np.deg2rad(angular_distance(ra[[0, 180]], dec[[0, 180]], LMC["ra0"], LMC["dec0"]))
    np.testing.assert_allclose(perspective.disc_distance(ra[[0, 180]], dec[[0, 180]]), LMC["D0"] / np.cos(rho), rtol=1e-9)


def test_perspective_velocity_at_the_centre():
    vE, vN = perspective.perspective_velocity(np.array([LMC["ra0"]]), np.array([LMC["dec0"]]))
    np.testing.assert_allclose([vE[0], vN[0]], [KAPPA * LMC["D0"] * LMC["pmra0"], KAPPA * LMC["D0"] * LMC["pmdec0"]], rtol=1e-12)


def test_perspective_effect_is_a_contraction():
    """Radial = -vlos0 sin(rho) and tangential = 0, up to second-order terms bounded by V_t rho^2 / 2 (V_t: transverse speed)."""
    from lmcold.footprint import angular_distance
    rng = np.random.default_rng(5)
    ra = LMC["ra0"] + rng.uniform(-12, 12, 400)
    dec = LMC["dec0"] + rng.uniform(-6, 6, 400)
    rho = angular_distance(ra, dec, LMC["ra0"], LMC["dec0"])
    keep = (rho > 0.5) & (rho < 6)
    vr, vt = perspective.perspective_effect(ra[keep], dec[keep])
    bound = 0.5 * KAPPA * LMC["D0"] * np.hypot(LMC["pmra0"], LMC["pmdec0"]) * np.deg2rad(rho[keep]) ** 2 + 0.1
    assert np.all(np.abs(vr + LMC["vlos0"] * np.sin(np.deg2rad(rho[keep]))) < bound)
    assert np.all(np.abs(vt) < bound)


def test_rotation_velocity_inclination_signature():
    """Beyond R0 the speed is v0. On the minor axis (perpendicular to the line of nodes) the velocity lies along the line of nodes
    and is seen in full; on the line of nodes it is foreshortened by cos(i) (up to perspective terms of order R / D0)."""
    from lmcold.footprint import disc_circle
    ra, dec = disc_circle(6.0, LMC["inc"], LMC["theta"], LMC["D0"])
    vE, vN = perspective.rotation_velocity(ra, dec)
    speed = np.hypot(vE, vN)
    np.testing.assert_allclose(speed[[90, 270]], LMC["v0"], rtol=0.03)  # minor axis
    np.testing.assert_allclose(speed[[0, 180]], LMC["v0"] * np.cos(np.deg2rad(LMC["inc"])), rtol=0.03)  # line of nodes


def test_internal_velocity_of_a_star_moving_with_the_systemic_velocity_is_zero():
    rng = np.random.default_rng(7)
    ra, dec = LMC["ra0"] + rng.uniform(-10, 10, 50), LMC["dec0"] + rng.uniform(-5, 5, 50)
    pa, pd = perspective.perspective_pm(ra, dec, perspective.disc_distance(ra, dec))
    vr, vt = perspective.internal_velocity(ra, dec, pa, pd)
    np.testing.assert_allclose([vr, vt], 0, atol=1e-8)


def test_disc_radius_is_zero_at_the_centre_and_grows_outwards():
    from lmcold.perspective import disc_radius
    R = disc_radius(np.full(3, LMC["ra0"]), LMC["dec0"] + np.array([0., 2., 5.]))
    assert R[0] < 1e-6 and 0 < R[1] < R[2]
