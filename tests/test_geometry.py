"""Tests for sky bases, the symmetry-axis convention and cylindrical coordinates."""
import numpy as np

from lmcold import geometry as geo
from conftest import radec_from_offset


def test_sky_basis_orthonormal_right_handed(rng):
    ra = rng.uniform(0, 2 * np.pi, 1000)
    dec = np.arcsin(rng.uniform(-0.999, 0.999, 1000))
    r, ea, ed = (np.asarray(v) for v in geo.sky_basis(ra, dec))
    for v in (r, ea, ed):
        np.testing.assert_allclose(np.linalg.norm(v, axis=-1), 1.0, atol=1e-14)
    for a, b in ((r, ea), (r, ed), (ea, ed)):
        np.testing.assert_allclose(np.sum(a * b, axis=-1), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.cross(ea, ed), r, atol=1e-14)


def test_basis_directions_are_increasing_ra_dec():
    """A finite-difference step in ra (dec) moves along e_alpha (e_delta)."""
    ra, dec, h = 1.3, -1.2, 1e-7
    r, ea, ed = (np.asarray(v) for v in geo.sky_basis(ra, dec))
    dra = (np.asarray(geo.sky_basis(ra + h, dec)[0]) - r) / (h * np.cos(dec))
    ddec = (np.asarray(geo.sky_basis(ra, dec + h)[0]) - r) / h
    np.testing.assert_allclose(dra, ea, atol=1e-6)
    np.testing.assert_allclose(ddec, ed, atol=1e-6)


def test_axis_is_unit_and_inclined(lit):
    r0, eE0, eN0, line, n = (np.asarray(v) for v in geo.symmetry_axis(
        lit["ra0"], lit["dec0"], lit["inc"], lit["theta"]))
    assert np.isclose(np.linalg.norm(n), 1.0)
    assert np.isclose(n @ r0, np.cos(lit["inc"]))
    assert np.isclose(n @ line, 0.0, atol=1e-15)
    # The line of nodes has PA theta.
    assert np.isclose(np.mod(np.arctan2(line @ eE0, line @ eN0), 2 * np.pi), lit["theta"])


def test_near_side_at_theta_minus_90(lit):
    """vdMK14 Fig. 2: near side at PA Theta-90, far side at Theta+90; line of nodes at D0."""
    r0, _, _, _, n = geo.symmetry_axis(lit["ra0"], lit["dec0"], lit["inc"], lit["theta"])
    rho = np.deg2rad(4.0)
    pas = lit["theta"] + np.deg2rad([-90.0, 90.0, 0.0, 180.0])
    ra, dec = radec_from_offset(lit["ra0"], lit["dec0"], rho, pas)
    rhat = geo.sky_basis(ra, dec)[0]
    D = np.asarray(geo.plane_distance(rhat, r0, lit["D0"], n))
    near, far, node1, node2 = D
    assert near < lit["D0"] < far
    # Exact results: n.rhat = cos i cos rho +/- sin i sin rho (near/far), or cos i cos rho (nodes).
    ci, si = np.cos(lit["inc"]), np.sin(lit["inc"])
    np.testing.assert_allclose([node1, node2], lit["D0"] / np.cos(rho), rtol=1e-12)
    np.testing.assert_allclose(near, lit["D0"] * ci / (ci * np.cos(rho) + si * np.sin(rho)), rtol=1e-12)
    np.testing.assert_allclose(far, lit["D0"] * ci / (ci * np.cos(rho) - si * np.sin(rho)), rtol=1e-12)
    # NE side is nearer (vdMC01): for Theta = 139.1, the near side is at PA 49.1.
    assert 0 < np.rad2deg(lit["theta"] - np.pi / 2) < 90


def test_cylindrical_basis_on_plane(lit, rng):
    r0, _, _, _, n = geo.symmetry_axis(lit["ra0"], lit["dec0"], lit["inc"], lit["theta"])
    ra, dec = radec_from_offset(lit["ra0"], lit["dec0"],
                                np.deg2rad(rng.uniform(0.2, 8, 500)),
                                rng.uniform(0, 2 * np.pi, 500))
    rhat = geo.sky_basis(ra, dec)[0]
    D = geo.plane_distance(rhat, r0, lit["D0"], n)
    R, z, eR, ephi = (np.asarray(v) for v in geo.lmc_cylindrical_basis(rhat, D, r0, lit["D0"], n))
    np.testing.assert_allclose(z, 0.0, atol=1e-10)
    x = np.asarray(D)[:, None] * np.asarray(rhat) - lit["D0"] * np.asarray(r0)
    np.testing.assert_allclose(R, np.linalg.norm(x, axis=-1), rtol=1e-12)
    np.testing.assert_allclose(np.sum(eR * ephi, axis=-1), 0, atol=1e-13)
    np.testing.assert_allclose(ephi @ np.asarray(n), 0, atol=1e-13)
    np.testing.assert_allclose(np.linalg.norm(ephi, axis=-1), 1, atol=1e-13)
    # (eR, ephi, n) is right-handed.
    np.testing.assert_allclose(np.cross(eR, ephi), np.broadcast_to(n, eR.shape), atol=1e-13)


def test_cylindrical_off_plane_height(lit):
    """A star behind the plane along the centre line of sight has z = dD cos(i) and R = dD sin(i)."""
    r0, _, _, _, n = geo.symmetry_axis(lit["ra0"], lit["dec0"], lit["inc"], lit["theta"])
    dD = 3.0
    R, z, _, _ = geo.lmc_cylindrical_basis(np.asarray(r0)[None], np.array([lit["D0"] + dD]),
                                           r0, lit["D0"], n)
    assert np.isclose(float(z[0]), dD * np.cos(lit["inc"]))
    assert np.isclose(float(R[0]), dD * np.sin(lit["inc"]))


def test_angular_separation_pa(lit):
    rho_in = np.deg2rad([1.0, 5.0, 10.0])
    pa_in = np.deg2rad([10.0, 200.0, 330.0])
    ra, dec = radec_from_offset(lit["ra0"], lit["dec0"], rho_in, pa_in)
    rho, pa = geo.angular_separation_pa(ra, dec, lit["ra0"], lit["dec0"])
    np.testing.assert_allclose(rho, rho_in, atol=1e-12)
    np.testing.assert_allclose(pa, pa_in, atol=1e-12)



def test_disc_distance_modulus_offset():
    """Zero at the centre; the near side (PA theta - 90) is brighter; matches the inclined-plane formula."""
    from lmcold.constants import LMC
    from lmcold.geometry import disc_dmu
    from lmcold.footprint import small_circle
    geo = [np.radians(LMC[k]) for k in ("inc", "theta", "ra0", "dec0")]
    np.testing.assert_allclose(disc_dmu(np.radians(LMC["ra0"]), np.radians(LMC["dec0"]), *geo), 0, atol=1e-6)
    ra, dec = small_circle(LMC["ra0"], LMC["dec0"], 6.0, n=361)
    d = np.asarray(disc_dmu(np.radians(ra), np.radians(dec), *geo))
    pa = np.radians(np.arange(361) * 360 / 360)  # small_circle starts at PA 0 and runs N through E
    i, th, r = np.radians(LMC["inc"]), np.radians(LMC["theta"]), np.radians(6.0)
    expect = 5 * np.log10(np.cos(i) / (np.cos(i) * np.cos(r) - np.sin(i) * np.sin(r) * np.sin(pa - th)))
    np.testing.assert_allclose(d, expect, atol=2e-3)
    assert d[int(round(LMC["theta"] - 90))] < -0.1
