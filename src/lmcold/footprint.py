"""Membership limits for the RR Lyrae: a sky mask around the SMC and a distance window.

SMC mask: one circle of radius 13 deg about the SMC centre (lmcold.constants.SMC). Beyond it, the fraction of RRab with SMC-like proper motion falls to the
floor set by LMC stars (scripts/fig_rrab_membership.py); along the LMC-SMC axis it reaches 7.6 deg from the LMC centre.
Distance window: |dmu| < 0.6 mag, about 3 times the LMC distance-modulus scatter, which leaves room for the inclined disc.
Science radius: rho < 10 deg from the LMC centre, where the foreground left in the window is <= 2.3% (6% at 10-12 deg, 20-77% beyond).

Also: great-circle distance, small circles and disc circles on the sky for figures, and the orthographic (x east, y north)
projection about the LMC centre used in maps and in lmcold.perspective. Centres from lmcold.constants. Used by the RR Lyrae
builds (scripts/build_rrab.py) and most figure scripts.
"""
import numpy as np

from .constants import LMC, SMC

SMC_MASK_RADIUS = 13.0  # deg, radius of the SMC mask about lmcold.constants.SMC (see module docstring)
DMU_WINDOW = 0.6  # mag, half-width of the window in RR Lyrae dmu (lmcold.rrlyrae.relative_modulus, fitted on the LMC core)
RHO_MAX = 10.0  # deg, science radius about the LMC centre


def angular_distance(ra, dec, ra0, dec0):
    """Great-circle distance in degrees; all inputs in degrees.

    Parameters
    ----------
    ra, dec : array_like
        Positions [deg].
    ra0, dec0 : float or array_like
        Reference position(s) [deg].

    Returns
    -------
    ndarray
        Angular distance [deg], from the spherical law of cosines (clipped against rounding).
    """
    ra, dec, ra0, dec0 = map(np.deg2rad, (ra, dec, ra0, dec0))
    c = np.sin(dec) * np.sin(dec0) + np.cos(dec) * np.cos(dec0) * np.cos(ra - ra0)
    return np.rad2deg(np.arccos(np.clip(c, -1, 1)))


def smc_masked(ra, dec):
    """True for positions (deg) inside the SMC mask.

    Parameters
    ----------
    ra, dec : array_like
        Positions [deg].

    Returns
    -------
    ndarray of bool
        Angular distance from the SMC centre < SMC_MASK_RADIUS.
    """
    return angular_distance(ra, dec, SMC["ra0"], SMC["dec0"]) < SMC_MASK_RADIUS


def small_circle(ra0, dec0, radius, n=361):
    """(ra, dec) in degrees of n points at angular distance `radius` (deg) from (ra0, dec0).

    Parameters
    ----------
    ra0, dec0 : float
        Centre [deg].
    radius : float
        Angular radius [deg].
    n : int
        Number of points, at position angles 0..2 pi (N through E); first and last coincide, so the curve is closed.

    Returns
    -------
    ra, dec : ndarray, shape (n,)
        Degrees, ra in [0, 360).
    """
    # p: position angle; spherical-triangle formulae for the point at distance d and PA p from (a0, d0).
    p, d, a0, d0 = np.linspace(0, 2 * np.pi, n), np.deg2rad(radius), np.deg2rad(ra0), np.deg2rad(dec0)
    dec = np.arcsin(np.sin(d0) * np.cos(d) + np.cos(d0) * np.sin(d) * np.cos(p))
    ra = a0 + np.arctan2(np.sin(p) * np.sin(d) * np.cos(d0), np.cos(d) - np.sin(d0) * np.sin(dec))
    return np.rad2deg(ra) % 360, np.rad2deg(dec)


def lmc_xy(ra, dec):
    """Orthographic projection (x east, y north) about the LMC centre, in degrees (x = sin(rho) sin(PA), in deg units).

    Parameters
    ----------
    ra, dec : array_like
        Positions [deg].

    Returns
    -------
    x, y : ndarray
        sin(rho) sin(PA) and sin(rho) cos(PA), converted from radians to degrees (so x ~ rho sin(PA) for small rho), with rho
        and PA (N through E) about lmcold.constants.LMC (ra0, dec0).
    """
    from .geometry import tangent_plane_coords
    x, y = tangent_plane_coords(np.deg2rad(ra), np.deg2rad(dec), np.deg2rad(LMC["ra0"]), np.deg2rad(LMC["dec0"]))
    return np.rad2deg(np.asarray(x)), np.rad2deg(np.asarray(y))


def disc_circle(R, inc, theta, D0, n=361):
    """(ra, dec) in degrees of points at in-plane radius R [kpc] in a thin disc centred on the LMC (inclination inc and line of
    nodes theta in degrees, distance D0 in kpc; convention of lmcold.geometry).

    Parameters
    ----------
    R : float
        In-plane radius [kpc].
    inc, theta : float
        Inclination and line-of-nodes position angle [deg].
    D0 : float
        Distance of the centre [kpc].
    n : int
        Number of points along the circle (closed: first and last coincide).

    Returns
    -------
    ra, dec : ndarray, shape (n,)
        Degrees, ra in [0, 360).
    """
    from .geometry import symmetry_axis
    r0, _, _, line, nrm = (np.asarray(v) for v in symmetry_axis(np.deg2rad(LMC["ra0"]), np.deg2rad(LMC["dec0"]),
                                                                np.deg2rad(inc), np.deg2rad(theta)))
    # Heliocentric Cartesian points [kpc], shape (n, 3): centre D0 r0 plus R along the line of nodes rotated by phi in the plane
    # (n x line lies in the plane, perpendicular to the line of nodes); then back to (ra, dec).
    phi = np.linspace(0, 2 * np.pi, n)[:, None]
    X = D0 * r0 + R * (np.cos(phi) * line + np.sin(phi) * np.cross(nrm, line))
    return np.rad2deg(np.arctan2(X[:, 1], X[:, 0])) % 360, np.rad2deg(np.arcsin(X[:, 2] / np.linalg.norm(X, axis=1)))
