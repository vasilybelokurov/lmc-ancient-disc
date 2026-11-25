"""Sky bases, LMC symmetry axis and LMC-centred cylindrical coordinates.

All angles are in radians and distances in kpc. The functions are written with
jax.numpy, so they are jit/grad-compatible, and they also accept NumPy arrays.

Axis convention (checked in tests/test_geometry.py)
---------------------------------------------------
Theta is the position angle of the line of nodes, measured from north through
east. i is the inclination (0 = face-on). The unit normal is

    n = cos(i) r0 + sin(i) u_near,    u_near = -cos(Theta) e_E + sin(Theta) e_N

where u_near is the tangent unit vector at PA Theta - 90 deg. The normal
therefore tilts towards the near side, which is at PA Theta - 90 deg, as in
van der Marel & Kallivayalil (2014, Fig. 2). Because n . r0 = cos(i) > 0, n
points away from the observer: rotation with positive v_phi along
e_phi = n x e_R has its angular momentum along n and appears clockwise on the sky.

Vectors are ICRS Cartesian (x towards RA = 0, Dec = 0; z towards the north celestial pole) with the observer at the origin. Used
by lmcold.kinfit (the kinematic model), lmcold.perspective, lmcold.footprint, lmcold.ogle and the mass scripts (disc_dmu).
"""
from __future__ import annotations

import jax.numpy as jnp


def sky_basis(ra, dec):
    """Unit line-of-sight vector and local east/north tangent vectors.

    Parameters
    ----------
    ra, dec : array_like
        Equatorial coordinates in radians.

    Returns
    -------
    rhat, e_alpha, e_delta : ndarray, shape (..., 3)
        ICRS Cartesian unit vectors. e_alpha x e_delta = rhat.
    """
    ra = jnp.asarray(ra, dtype=float)
    dec = jnp.asarray(dec, dtype=float)
    ca, sa = jnp.cos(ra), jnp.sin(ra)
    cd, sd = jnp.cos(dec), jnp.sin(dec)
    rhat = jnp.stack([cd * ca, cd * sa, sd], axis=-1)
    # e_alpha = d rhat / d ra / cos(dec), e_delta = d rhat / d dec: directions of increasing RA (east) and Dec (north).
    e_alpha = jnp.stack([-sa, ca, jnp.zeros_like(ra)], axis=-1)
    e_delta = jnp.stack([-ca * sd, -sa * sd, cd], axis=-1)
    return rhat, e_alpha, e_delta


def tangent_direction(eE, eN, pa):
    """Unit tangent vector at position angle `pa` (N through E, radians).

    Parameters
    ----------
    eE, eN : ndarray, shape (..., 3)
        Local east and north unit vectors (sky_basis).
    pa : float or array_like
        Position angle [rad], 0 = north, pi/2 = east.

    Returns
    -------
    ndarray, shape (..., 3)
        sin(pa) eE + cos(pa) eN.
    """
    return jnp.sin(pa) * eE + jnp.cos(pa) * eN


def symmetry_axis(ra0, dec0, inc, theta):
    """Centre basis, line-of-nodes vector and symmetry-axis unit normal.

    Parameters
    ----------
    ra0, dec0 : float
        Centre in radians.
    inc : float
        Inclination in radians (0 = face-on).
    theta : float
        Line-of-nodes position angle in radians, N through E.

    Returns
    -------
    r0, eE0, eN0, line, n : ndarray, shape (3,)
        `line` is the line-of-nodes unit vector at PA theta. `n` is the unit
        normal, tilted towards the near side (PA theta - 90 deg).
    """
    r0, eE0, eN0 = sky_basis(ra0, dec0)
    line = tangent_direction(eE0, eN0, theta)
    u_near = tangent_direction(eE0, eN0, theta - jnp.pi / 2)  # sky direction of the near side
    n = jnp.cos(inc) * r0 + jnp.sin(inc) * u_near
    return r0, eE0, eN0, line, n


def tangent_plane_coords(ra, dec, ra0, dec0):
    """Orthographic tangent-plane coordinates (x_E, x_N) about (ra0, dec0).

    x_E = rhat . e_E0 and x_N = rhat . e_N0 are dimensionless, i.e. sin(rho)
    times the unit vector towards PA phi. Used for plotting and for angular
    positions; it is not a projection used in the model.

    Parameters
    ----------
    ra, dec : array_like
        Positions [rad].
    ra0, dec0 : float
        Projection centre [rad].

    Returns
    -------
    x_E, x_N : ndarray
        Dimensionless (multiply by 180/pi for degrees near the centre). Points on the far hemisphere are not flagged.
    """
    rhat, _, _ = sky_basis(ra, dec)
    _, eE0, eN0 = sky_basis(ra0, dec0)
    return jnp.sum(rhat * eE0, axis=-1), jnp.sum(rhat * eN0, axis=-1)


def angular_separation_pa(ra, dec, ra0, dec0):
    """Angular distance rho and position angle phi (N through E) from the centre, in radians.

    Parameters
    ----------
    ra, dec : array_like
        Positions [rad].
    ra0, dec0 : float
        Centre [rad].

    Returns
    -------
    rho : ndarray
        Angular distance [rad], from atan2(sin rho, cos rho), accurate at small and large rho.
    phi : ndarray
        Position angle of the star seen from the centre [rad], in [0, 2 pi), 0 = north, pi/2 = east.
    """
    rhat, _, _ = sky_basis(ra, dec)
    r0, _, _ = sky_basis(ra0, dec0)
    xE, xN = tangent_plane_coords(ra, dec, ra0, dec0)
    cosrho = jnp.sum(rhat * r0, axis=-1)
    rho = jnp.arctan2(jnp.hypot(xE, xN), cosrho)
    phi = jnp.mod(jnp.arctan2(xE, xN), 2 * jnp.pi)
    return rho, phi


def plane_distance(rhat, r0, D0, n):
    """Distance at which the line of sight rhat meets the plane through D0*r0 with normal n.

    Parameters
    ----------
    rhat : ndarray, shape (..., 3)
        Unit line-of-sight vectors.
    r0 : ndarray, shape (3,)
        Unit vector to the centre.
    D0 : float
        Centre distance (kpc, or any unit; the result is in the same unit).
    n : ndarray, shape (3,)
        Plane normal.

    Returns
    -------
    ndarray, shape (...)
        d with (d rhat - D0 r0) . n = 0, i.e. d = D0 (n . r0) / (n . rhat).
    """
    return D0 * jnp.sum(n * r0, axis=-1) / jnp.sum(n * rhat, axis=-1)


def lmc_cylindrical_basis(rhat, distance, r0, D0, n, R_floor=1.0e-8):
    """LMC-centred cylindrical coordinates without forcing stars onto a plane.

    Parameters
    ----------
    rhat : ndarray, shape (N, 3)
        Unit line-of-sight vectors of the stars.
    distance : ndarray, shape (N,)
        Heliocentric distances (kpc).
    r0 : ndarray, shape (3,)
        Unit vector to the centre.
    D0 : float
        Centre distance (kpc).
    n : ndarray, shape (3,)
        Symmetry-axis unit normal.
    R_floor : float
        Numerical floor on R (kpc) that keeps e_R defined at the centre.

    Returns
    -------
    R, z : ndarray, shape (N,)
        Cylindrical radius and height (kpc).
    eR, ephi : ndarray, shape (N, 3)
        Unit radial and azimuthal vectors, with ephi = n x eR.
    """
    x = jnp.asarray(distance)[..., None] * rhat  # heliocentric position [kpc], shape (N, 3)
    dx = x - D0 * r0
    z = jnp.sum(dx * n, axis=-1)  # height along n; n . r0 > 0, so z > 0 is behind the plane as seen from the Sun
    Rvec = dx - z[..., None] * n
    R = jnp.linalg.norm(Rvec, axis=-1)
    eR = Rvec / jnp.maximum(R, R_floor)[..., None]
    ephi = jnp.cross(jnp.broadcast_to(n, eR.shape), eR)
    return R, z, eR, ephi



def disc_dmu(ra, dec, inc, theta, ra0, dec0):
    """Distance-modulus offset [mag], relative to the centre (ra0, dec0), of the disc plane (inc, theta) along (ra, dec).
    All angles in radians.

    Parameters
    ----------
    ra, dec : array_like
        Lines of sight [rad].
    inc, theta : float
        Inclination and line-of-nodes position angle [rad].
    ra0, dec0 : float
        Centre [rad].

    Returns
    -------
    ndarray
        5 log10(d / D0), with d the distance at which the line of sight meets the plane: positive on the far side.
    """
    r0, _, _, _, n = symmetry_axis(ra0, dec0, inc, theta)
    return 5 * jnp.log10(plane_distance(sky_basis(ra, dec)[0], r0, 1.0, n))
