"""The perspective proper-motion field of the LMC and the sky-polar basis about its centre. Inputs in degrees, kpc, mas/yr, km/s.

Perspective: if every star moved with the LMC systemic velocity V_sys, its proper motion would be the component of V_sys in its own
tangent plane divided by kappa * D. That field depends only on the centre, distance and systemic motion (lmcold.constants.LMC) and
on the star's distance, not on any internal motion. Stars are placed on the adopted disc plane (inclination and line of nodes).

Sky-polar basis: at each star, the unit vectors in its tangent plane pointing away from the LMC centre (radial) and 90 deg further
in position angle, N through E (tangential). Proper motions are turned into velocities with v [km/s] = KAPPA x mu [mas/yr] x D
[kpc]. All parameters come from lmcold.constants.LMC (vdMK14). Used by lmcold.kinfit and the proper-motion figures
(scripts/paper_pm_maps.py), and for the in-plane radius of the profiles (scripts/paper_feh_profile.py).
"""
import numpy as np

from . import geometry as geo
from .constants import KAPPA, LMC


def _centre():
    """Unit vector to the LMC centre, east and north vectors there, and the disc normal (lmcold.geometry.symmetry_axis)."""
    r0, eE0, eN0, _, n = (np.asarray(v) for v in geo.symmetry_axis(np.deg2rad(LMC["ra0"]), np.deg2rad(LMC["dec0"]),
                                                                    np.deg2rad(LMC["inc"]), np.deg2rad(LMC["theta"])))
    return r0, eE0, eN0, n


def disc_distance(ra, dec):
    """Distance [kpc] at which each line of sight meets the adopted LMC disc plane.

    Parameters
    ----------
    ra, dec : array_like
        Positions [deg].

    Returns
    -------
    ndarray
        Distance [kpc].
    """
    r0, _, _, n = _centre()
    rhat = np.asarray(geo.sky_basis(np.deg2rad(ra), np.deg2rad(dec))[0])
    return np.asarray(geo.plane_distance(rhat, r0, LMC["D0"], n))


def disc_radius(ra, dec):
    """In-plane distance [kpc] from the LMC centre of stars placed on the adopted disc plane.

    Parameters
    ----------
    ra, dec : array_like
        Positions [deg].

    Returns
    -------
    ndarray
        Cylindrical radius R [kpc] in the disc plane.
    """
    r0, _, _, n = _centre()
    rhat = np.asarray(geo.sky_basis(np.deg2rad(ra), np.deg2rad(dec))[0])
    return np.asarray(geo.lmc_cylindrical_basis(rhat, disc_distance(ra, dec), r0, LMC["D0"], n)[0])


def systemic_velocity():
    """3D systemic velocity of the LMC centre of mass [km/s], ICRS Cartesian.

    Returns
    -------
    ndarray, shape (3,)
        vlos0 along the line of sight to the centre plus the transverse velocity KAPPA D0 (pmra0 e_E + pmdec0 e_N).
    """
    r0, eE0, eN0, _ = _centre()
    return LMC["vlos0"] * r0 + KAPPA * LMC["D0"] * (LMC["pmra0"] * eE0 + LMC["pmdec0"] * eN0)


def perspective_velocity(ra, dec):
    """(v_east, v_north) [km/s]: the LMC systemic velocity projected on each star's sky directions; independent of distance.

    Parameters
    ----------
    ra, dec : array_like
        Positions [deg].

    Returns
    -------
    v_east, v_north : ndarray
        Components [km/s] along the star's local east and north unit vectors.
    """
    _, ea, ed = (np.asarray(v) for v in geo.sky_basis(np.deg2rad(ra), np.deg2rad(dec)))
    V = systemic_velocity()
    return ea @ V, ed @ V


def perspective_pm(ra, dec, D):
    """(pmra, pmdec) [mas/yr] of stars at (ra, dec) [deg] and distance D [kpc] moving with the LMC systemic velocity.

    Parameters
    ----------
    ra, dec : array_like
        Positions [deg].
    D : float or array_like
        Distance [kpc].

    Returns
    -------
    pmra, pmdec : ndarray
        Proper motions [mas/yr], pmra including cos(dec).
    """
    vE, vN = perspective_velocity(ra, dec)
    return vE / (KAPPA * D), vN / (KAPPA * D)


def sky_polar_basis(ra, dec):
    """At each star, the (east, north) components of the unit vector pointing away from the LMC centre (radial) and of the one
    90 deg east of it (tangential). Returns ((rE, rN), (tE, tN)).

    Parameters
    ----------
    ra, dec : array_like, shape (N,)
        Positions [deg] (1-D arrays: the code indexes [:, None]).

    Returns
    -------
    (rE, rN), (tE, tN) : tuple of tuples of ndarray, shape (N,)
        Components of the unit radial and tangential vectors along the star's local east and north.
    """
    rhat, ea, ed = (np.asarray(v) for v in geo.sky_basis(np.deg2rad(ra), np.deg2rad(dec)))
    r0 = np.asarray(geo.sky_basis(np.deg2rad(LMC["ra0"]), np.deg2rad(LMC["dec0"]))[0])
    out = -(r0 - (rhat @ r0)[:, None] * rhat)  # away from the centre, in the star's tangent plane
    out /= np.linalg.norm(out, axis=1)[:, None]
    rE, rN = np.sum(out * ea, axis=1), np.sum(out * ed, axis=1)
    return (rE, rN), (rN, -rE)  # tangential: radial turned by +90 deg in position angle (north through east)


def perspective_effect(ra, dec):
    """(radial, tangential) [km/s] perspective effect: the projected systemic velocity in each star's sky-polar basis, minus the
    centre's transverse velocity as a uniform vector on the flat tangent plane, in the plane's radial and tangential directions.
    Subtracting it this way removes the bulk motion without picking up the rotation of the local east/north directions across the
    field. To first order it is a pure contraction, radial = -vlos0 sin(rho), because the LMC recedes from us.

    Parameters
    ----------
    ra, dec : array_like, shape (N,)
        Positions [deg]; undefined at the centre itself (r = 0).

    Returns
    -------
    radial, tangential : ndarray, shape (N,)
        Velocities [km/s].
    """
    # x, y: orthographic coordinates about the centre (lmcold.footprint.lmc_xy); (x, y) / r is the radial direction on the
    # flat tangent plane and (y, -x) / r the tangential one, in (east, north) components.
    from .footprint import lmc_xy
    x, y = lmc_xy(ra, dec)
    r = np.hypot(x, y)
    vE, vN = perspective_velocity(ra, dec)
    vE0, vN0 = KAPPA * LMC["D0"] * LMC["pmra0"], KAPPA * LMC["D0"] * LMC["pmdec0"]  # centre's transverse velocity [km/s]
    (rE, rN), (tE, tN) = sky_polar_basis(ra, dec)
    return vE * rE + vN * rN - (vE0 * x + vN0 * y) / r, vE * tE + vN * tN - (vE0 * y - vN0 * x) / r


def rotation_velocity(ra, dec):
    """(v_east, v_north) [km/s] of a thin disc in the adopted LMC plane rotating with the adopted rotation curve (rising linearly
    to R0, flat at v0 beyond; lmcold.constants.LMC), for stars on that plane. Positive v_phi along n x e_R appears clockwise.

    Parameters
    ----------
    ra, dec : array_like, shape (N,)
        Positions [deg].

    Returns
    -------
    v_east, v_north : ndarray, shape (N,)
        Sky-plane components [km/s] of the rotation velocity (its line-of-sight part is dropped).
    """
    r0, _, _, n = _centre()
    rhat, ea, ed = (np.asarray(v) for v in geo.sky_basis(np.deg2rad(ra), np.deg2rad(dec)))
    R, _, _, ephi = (np.asarray(v) for v in geo.lmc_cylindrical_basis(rhat, disc_distance(ra, dec), r0, LMC["D0"], n))
    V = (LMC["v0"] * np.minimum(R / LMC["R0"], 1.0))[:, None] * ephi
    return np.sum(V * ea, axis=1), np.sum(V * ed, axis=1)


def internal_velocity(ra, dec, pmra, pmdec):
    """(radial, tangential) [km/s] internal velocity in the sky-polar basis: the observed proper motion as a velocity at the disc
    distance, minus the bulk motion (the centre's transverse velocity as a uniform vector on the flat tangent plane) and minus
    the perspective effect.

    Parameters
    ----------
    ra, dec : array_like, shape (N,)
        Positions [deg].
    pmra, pmdec : array_like, shape (N,)
        Observed proper motions [mas/yr].

    Returns
    -------
    vr, vt : ndarray, shape (N,)
        Radial and tangential internal velocities [km/s] in the sky-polar basis.
    """
    from .footprint import lmc_xy
    x, y = lmc_xy(ra, dec)
    r = np.hypot(x, y)
    kD = KAPPA * disc_distance(ra, dec)  # km/s per mas/yr at the disc distance of each star
    vE0, vN0 = KAPPA * LMC["D0"] * LMC["pmra0"], KAPPA * LMC["D0"] * LMC["pmdec0"]
    (rE, rN), (tE, tN) = sky_polar_basis(ra, dec)
    pr, pt = perspective_effect(ra, dec)
    vr = kD * (pmra * rE + pmdec * rN) - (vE0 * x + vN0 * y) / r - pr
    vt = kD * (pmra * tE + pmdec * tN) - (vE0 * y - vN0 * x) / r - pt
    return vr, vt
