"""Axisymmetric particle realisations of a fitted kinematic model (paper Section 5: the rotation-implied flattening, lmcold.virial;
the shape model, lmcold.shape) and the axis ratio of a particle system.

spheroid(): positions of an oblate spheroid of axis ratio q whose face-on surface density is exponential with scale length h
(density proportional to K_0(m / h), m^2 = R^2 + z^2 / q^2, the deprojection of an exponential; edge-on its isophotes are ellipses
of ellipticity 1 - q, and <z^2> / <x^2> = q^2, the definition of shape()).
velocities(): at given positions, the mean rotation kinfit.rotation(p, R) of a fitted kinematic model (lmcold.kinfit vector p) and
Gaussian velocities with its disc-frame ellipsoid sigma_i exp(gamma (R - R_SIG)) (gamma = dlnsig_dR, R_SIG = lmcold.kinfit.R_SIG).
Streaming and outliers are left out. Positions X (N, 3) are in the disc frame [kpc], z along the symmetry axis; velocities in km/s.
"""
import numpy as np

from . import kinfit

CYL = (10.0, 5.0)  # kpc, the cylinder R < 10, |z| < 5 used by shape() (the LMC giants reach R ~ 9 kpc)


def shape(X, w):
    """Axis ratio c/a = sqrt(<z^2> / <x^2>), <x^2> = <R^2> / 2, weighted by w, of the disc-frame particles inside CYL.

    X : (N, 3) disc-frame positions [kpc]; w : (N,) weights (e.g. masses). Returns a float."""
    k = (np.hypot(X[:, 0], X[:, 1]) < CYL[0]) & (np.abs(X[:, 2]) < CYL[1])
    return float(np.sqrt(np.average(X[k, 2] ** 2, weights=w[k]) / (0.5 * np.average(X[k, 0] ** 2 + X[k, 1] ** 2, weights=w[k]))))


def spheroid(h, q, n, rng):
    """n positions (n, 3) [kpc] of the oblate spheroid of the module docstring (density K_0(m / h), axis ratio q), not truncated.

    h [kpc] > 0; q: axis ratio; n: int; rng: numpy.random.Generator. Spherical radii r are drawn from r^2 K_0(r / h) on a grid
    out to 30 h, directions isotropically, then z is multiplied by q (density constant on m = const)."""
    from scipy.special import k0
    if not h > 0:
        raise ValueError(f'scale length h = {h} kpc: the model needs a declining profile')
    grid = np.linspace(1e-4, 30, 30000) * h  # radii [kpc], out to 30 scale lengths
    cdf = np.cumsum(grid ** 2 * k0(grid / h))  # spherical radius r with density K_0(r / h)
    r = np.interp(rng.uniform(0, cdf[-1], n), cdf, grid)
    u = rng.normal(size=(n, 3))  # isotropic directions after normalisation
    return r[:, None] * u / np.linalg.norm(u, axis=1)[:, None] * [1, 1, q]


def velocities(p, X, rng, r_floor=0.0):
    """Disc-frame velocities (N, 3) [km/s] of the kinematic model p (kinfit vector) at positions X: mean rotation kinfit.rotation(p, R)
    and Gaussian velocities with the ellipsoid sigma_i exp(gamma (max(R, r_floor) - R_SIG)) (r_floor [kpc]: the dispersions are held
    constant inside it instead of following the exponential gradient inward, beyond the radii the fits constrain).

    p : lmcold.kinfit parameter vector (33,); X : (N, 3) disc-frame positions [kpc]; rng : numpy.random.Generator. Returns
    (N, 3) Cartesian velocities in the disc frame, positive v_phi counter-clockwise about +z."""
    R = np.hypot(X[:, 0], X[:, 1])
    s = np.exp(p[kinfit.PARAMS.index('dlnsig_dR')] * (np.maximum(R, r_floor) - kinfit.R_SIG))
    sR, sphi, sz = (p[kinfit.PARAMS.index(k)] * s for k in ('sigma_R', 'sigma_phi', 'sigma_z'))
    vR, vphi = rng.normal(0, sR), rng.normal(np.asarray(kinfit.rotation(p, R)), sphi)
    c, si = X[:, 0] / R, X[:, 1] / R  # cos and sin of the azimuth
    return np.c_[vR * c - vphi * si, vR * si + vphi * c, rng.normal(0, sz)]
