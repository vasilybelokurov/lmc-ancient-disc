"""Forward model of stellar proper motions in the LMC, for a joint fit of the centre, bulk motion, orientation and rotation.

Each star is placed on the disc plane (centre, inclination i, line of nodes theta, distance D0). Its 3D velocity is the systemic
velocity (line-of-sight vlos0 and proper motion (pmra0, pmdec0) at the centre) plus rotation along e_phi = n x e_R (clockwise on
the sky for v0 > 0), v_phi = vrot [1 + m1c cos(phi) + m1s sin(phi) + m2c cos(2 phi) + m2s sin(2 phi)] with vrot = v0 tanh(R / R0):
circular, with an optional lopsided (m = 1) part and m = 2 part; plus an optional in-plane radial velocity along e_R,
v_R = vrot [vr0 + r2c cos(2 phi) + r2s sin(2 phi)] (mean expansion or contraction and m = 2 elliptical streaming; STREAM).
phi is the azimuth in the disc plane from the line of nodes (PA theta), increasing along e_phi. The model proper motion is that
velocity projected on the star's own sky directions divided by kappa D, and the model line-of-sight velocity its projection on the
line of sight, so perspective is exact. Likelihood per star: a mixture of the model, with proper-motion covariance
k^2 C_gaia + (sigma / (kappa D))^2 I and line-of-sight variance e_vlos^2 + sigma_los^2, and a broad outlier term (same mean,
s_out = 1 mas/yr added in quadrature to the proper motions and kappa D s_out to the line-of-sight velocity). Stars with a relative
distance modulus dmu (RR Lyrae) add a distance term: model dmu0 + 5 log10(D / D0), with D the distance at which the star's line
of sight meets the disc plane, and variance sigma_mu^2 (one total scatter: period-luminosity scatter, photometric errors and the
depth of the population); in the outlier term dmu is uniform over the membership window |dmu| < DMU_WINDOW
(lmcold.footprint). The distances thus measure the tilt of the plane the stars lie in, and the fit assumes it is the plane they
rotate in. Stars without a proper motion, a line-of-sight velocity or a modulus contribute only the terms they have.
Parameters (in order, PARAMS): centre offset dx (east), dy (north) on the tangent plane of the adopted centre [deg]; pmra0,
pmdec0 [mas/yr]; inc, theta [deg]; v0 [km/s]; R0 [kpc]; sigma [km/s]; k; logit of the outlier fraction; vlos0 [km/s]; sigma_los
[km/s]; m1c, m1s; dmu0 [mag], zero point of the moduli; sigma_mu [mag]; m2c, m2s, vr0, r2c, r2s (STREAM, fractions of vrot);
dinc, dtheta [deg] (TILT): tilted-ring model, the inner ring (stars within RHO_TILT of the adopted centre) lies in a plane through
the same centre with inclination inc + dinc and line of nodes theta + dtheta; zero (one plane) unless fitted; qlos: the internal
(rotation and streaming) part of the line-of-sight velocity is (1 + qlos) times that of the same 3D velocity as seen in the
proper motions, a proper-motion / line-of-sight scale mismatch (e.g. a distance-scale error: proper motions give internal
velocities proportional to the adopted D0), zero unless fitted; didt, dthdt [deg/Gyr] (TURN): rates of change of the
inclination and of the line of nodes (precession and nutation of the disc, van der Marel et al. 2002, AJ 124, 2639); each star
then also moves with its point of the turning plane (fixed in-plane coordinates along the line of nodes and n x line), a velocity
seen equally in the proper motions and the line-of-sight velocities; zero unless fitted; sigma_R, sigma_phi, sigma_z [km/s] and
dlnsig_dR [1/kpc] (ELLIPSOID): the disc-frame velocity ellipsoid of the ellipsoid likelihood (below), used only with
ellipsoid=True; vslope [1/kpc]: the rotation curve is vrot = v0 tanh(R / R0) [1 + vslope (R - R_SIG)], a linear rise or
decline beyond the turnover (asymmetric drift and a falling outer curve make the tanh shape too simple), zero unless fitted;
h [kpc]: exponential scale length of the population's surface density. With h > 0 that curve is the circular speed v_c(R) and the
mean rotation is its asymmetric-drift (radial Jeans) prediction (rotation()); h = 0 (default) keeps the free rotation curve.
D0 is fixed.
Ellipsoid likelihood (ellipsoid=True in loglike_stars, neg_loglike, fit, sample): instead of the sky-aligned dispersions sigma (both
proper-motion directions) and sigma_los, the random velocities have the diagonal tensor diag(sigma_R^2, sigma_phi^2, sigma_z^2) in
the star's cylindrical basis (e_R, e_phi, n), each scaled by exp(dlnsig_dR (R - R_SIG)), projected on its (east, north, line of
sight) directions: one 3D Gaussian per star in km/s for the proper motions (times kappa D) and the line-of-sight velocity together,
with covariance Q diag(...) Q^T + blockdiag(k^2 (kappa D)^2 C_gaia, e_vlos^2), Q the 3x3 matrix of dot products of the two bases,
and the Jacobian (kappa D)^2 of the proper-motion change of variables, so that the likelihood stays a density in mas/yr and km/s
and, for sigma_R = sigma_phi = sigma_z = sigma = sigma_los, equals the default one (tests/test_kinfit.py). Components a star lacks
are marginalised (rows and columns dropped). BASE: the proper-motion model with the adopted vlos0, no lopsided part, no streaming and no distance term.
fit() finds the maximum likelihood; sample() draws the posterior with NUTS (numpyro), flat priors within BOUNDS, for one population
or for several that share the centre, bulk motion, orientation, lopsided part and streaming (SHARED) and each have their own rotation,
dispersions, error scale and outlier fraction.

Conventions: angles on the sky in degrees in the parameter vector, radians inside the geometry; proper motions in mas/yr with
pmra = mu_alpha* (already multiplied by cos dec, as in Gaia); velocities in km/s; distances in kpc; v [km/s] = KAPPA mu [mas/yr]
D [kpc] with KAPPA = 4.74047 (lmcold.constants). The adopted centre, distance D0, orientation and systemic motion are those of
lmcold.constants.LMC. Geometry (sky bases, disc normal, in-plane coordinates) is in lmcold.geometry; its axis convention is
checked in tests/test_geometry.py. Used by scripts/fit_centre.py (maximum likelihood), scripts/fit_kinematics_mcmc.py (posterior),
scripts/fig_kin_degeneracy.py (mocks) and the centre figures; tested in tests/test_kinfit.py.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from . import geometry as geo
from .constants import KAPPA, KPC_GYR, LMC
from .footprint import DMU_WINDOW

# Order of the parameter vector p used everywhere in this module (model() reads p by index).
PARAMS = ("dx", "dy", "pmra0", "pmdec0", "inc", "theta", "v0", "R0", "sigma", "k", "logit_fout", "vlos0", "sigma_los", "m1c", "m1s", "dmu0", "sigma_mu",
          "m2c", "m2s", "vr0", "r2c", "r2s", "dinc", "dtheta", "qlos", "didt", "dthdt", "sigma_R", "sigma_phi", "sigma_z", "dlnsig_dR",
          "vslope", "h")
# The streaming parameters (in-plane non-circular motion): m = 2 modulation of v_phi and mean plus m = 2 radial velocity.
STREAM = PARAMS[17:22]
# Disc-frame velocity ellipsoid of the ellipsoid likelihood, and the radius [kpc] at which sigma_R, sigma_phi, sigma_z are quoted
# when dlnsig_dR is fitted (an analysis choice: the middle of the fitted radii).
ELLIPSOID = PARAMS[27:31]
R_SIG = 4.0  # kpc; reference radius of the dispersions and of vslope (analysis choice, see comment above)
# Tilted-ring parameters: inclination and line-of-nodes offsets [deg] of the inner ring from the outer disc.
TILT = PARAMS[22:24]
# Rates of change of the disc orientation [deg/Gyr]: inclination and line of nodes.
TURN = PARAMS[25:27]
# Radius [deg] on the sky from the adopted centre that splits the inner ring from the outer disc: the middle of the fitted range
# 2-9 deg (scripts/fit_kinematics_mcmc.py RING), an analysis choice; the bar region, tilted relative to the outer disc in red
# clump distances (Choi et al. 2018, ApJ 866, 90) and vertical velocities (Jimenez-Arranz et al. 2025, A&A 698, A88), lies inside.
RHO_TILT = 4.5
# The proper-motion-only model: the first 11 parameters (vlos0 at the adopted value, no lopsided part). fit()'s default.
BASE = PARAMS[:11]
S_OUT = 1.0  # mas/yr, width added in quadrature for the outlier component (about 240 km/s at the LMC distance)
# Parameters common to all populations in a joint fit (sample() with several data sets); the rest are per population.
SHARED = ("dx", "dy", "pmra0", "pmdec0", "inc", "theta", "vlos0", "m1c", "m1s") + STREAM + TILT + ("qlos",) + TURN
# Box bounds for the optimiser and flat (uniform) priors for sample(): wide physical ranges, centred on the adopted values for the
# bulk motion (+-2 mas/yr) and vlos0 (+-30 km/s). Units as in PARAMS.
BOUNDS = dict(dx=(-3, 3), dy=(-3, 3), pmra0=(LMC["pmra0"] - 2, LMC["pmra0"] + 2), pmdec0=(LMC["pmdec0"] - 2, LMC["pmdec0"] + 2), inc=(5, 80), theta=(0, 360), v0=(0, 200),
              R0=(0.1, 10), sigma=(5, 150), k=(0.5, 3), logit_fout=(-10, 0), vlos0=(LMC["vlos0"] - 30, LMC["vlos0"] + 30),
              sigma_los=(3, 150), m1c=(-0.5, 0.5), m1s=(-0.5, 0.5), dmu0=(-0.3, 0.3), sigma_mu=(0.02, 0.5),
              m2c=(-0.5, 0.5), m2s=(-0.5, 0.5), vr0=(-0.5, 0.5), r2c=(-0.5, 0.5), r2s=(-0.5, 0.5), dinc=(-30, 30), dtheta=(-90, 90), qlos=(-0.5, 0.5),
              didt=(-1000, 1000), dthdt=(-1000, 1000), sigma_R=(3, 150), sigma_phi=(3, 150), sigma_z=(3, 150),
              dlnsig_dR=(-1, 1), vslope=(-0.3, 0.3), h=(0.3, 10))  # physical ranges for the optimiser
# Smoothing speed [km/s] of the square root of the asymmetric-drift mean: vbar = sqrt(0.5 (u2 + sqrt(u2^2 + DRIFT_EPS^4))), smooth
# where u2 = vbar^2 < 0 (no valid equilibrium; callers count such stars) and within 0.01 km/s of sqrt(u2) for u2 > (40 km/s)^2.
DRIFT_EPS = 5.0


def centre(dx, dy):
    """(ra0, dec0) [rad] of the point offset by (dx east, dy north) [deg] on the tangent plane of the adopted centre.

    Parameters
    ----------
    dx, dy : float
        Offsets [deg] along the local east and north unit vectors at the adopted centre (lmcold.constants.LMC ra0, dec0).

    Returns
    -------
    ra0, dec0 : float
        The offset centre [rad]: the tangent-plane point r0 + dx e_E + dy e_N normalised back onto the unit sphere.
    """
    r0, eE0, eN0 = geo.sky_basis(jnp.deg2rad(LMC["ra0"]), jnp.deg2rad(LMC["dec0"]))
    u = r0 + jnp.deg2rad(dx) * eE0 + jnp.deg2rad(dy) * eN0
    u = u / jnp.linalg.norm(u)
    return jnp.arctan2(u[1], u[0]), jnp.arcsin(u[2])


def model(p, rhat, ea, ed):
    """Model (pmra, pmdec) [mas/yr], line-of-sight velocity [km/s] and disc distance D [kpc] for parameter vector p and star sky
    bases: the first four outputs of _model (see there).

    p : (33,) in PARAMS order; rhat, ea, ed : (N, 3) unit line-of-sight, east and north vectors (data()). Returns four arrays (N,).
    """
    return _model(p, rhat, ea, ed)[:4]


def rotation(p, R):
    """Mean rotation speed [km/s] at in-plane radius R [kpc]: the curve vc(R) = v0 tanh(R / R0) [1 + vslope (R - R_SIG)] or, with
    h = p['h'] > 0, the asymmetric-drift mean of a population with that circular speed, from the radial Jeans equation for a
    steady axisymmetric disc with an ellipsoid aligned with (e_R, e_phi, n) (vertical term zero; Binney & Tremaine 2008, eq. 4.228),
    exponential surface density of scale length h and the dispersions of intrinsic_cov, sigma_i exp(dlnsig_dR (R - R_SIG)):

        vbar^2 = vc^2 - [sigma_phi(R)^2 + sigma_R(R)^2 (R / h - 1 - 2 dlnsig_dR R)].

    The square root is smoothed below DRIFT_EPS^2 (a negative vbar^2 has no equilibrium; callers count such stars)."""
    vc = p[6] * jnp.tanh(R / p[7]) * (1 + p[31] * (R - R_SIG))
    drift = p[32] > 0
    h = jnp.where(drift, p[32], 1.0)  # a safe value when the drift is off, so that its (unused) branch stays finite
    scale = jnp.exp(2 * p[30] * (R - R_SIG))
    u2 = vc**2 - scale * (p[28] ** 2 + p[27] ** 2 * (R / h - 1 - 2 * p[30] * R))
    return jnp.where(drift, jnp.sqrt(0.5 * (u2 + jnp.sqrt(u2**2 + DRIFT_EPS**4))), vc)


def _model(p, rhat, ea, ed):
    """Model (pmra, pmdec) [mas/yr], line-of-sight velocity [km/s] and disc distance D [kpc] for parameter vector p and star sky
    bases, and the star's in-plane radius R [kpc] and cylindrical unit vectors eR, ephi, n (for the ellipsoid likelihood).

    Parameters
    ----------
    p : array_like, shape (33,)
        Parameter vector in the order of PARAMS.
    rhat, ea, ed : array, shape (N, 3)
        Unit line-of-sight, east and north vectors of each star (from data()).

    Returns
    -------
    pmra, pmdec : array, shape (N,)
        Model proper motions [mas/yr], pmra = mu_alpha cos(dec).
    vlos : array, shape (N,)
        Model line-of-sight velocity [km/s].
    D : array, shape (N,)
        Distance [kpc] at which the star's line of sight meets the disc plane.
    R : array, shape (N,)
        In-plane radius [kpc].
    eR, ephi, n : array, shape (N, 3)
        Radial, azimuthal and normal unit vectors of the star's plane.
    """
    ra0, dec0 = centre(p[0], p[1])
    # Centre basis (r0 line of sight, eE0, eN0), line-of-nodes unit vector and disc normal n for inclination p[4], nodes p[5].
    r0, eE0, eN0, line, n = geo.symmetry_axis(ra0, dec0, jnp.deg2rad(p[4]), jnp.deg2rad(p[5]))
    # Tilted ring: stars within RHO_TILT of the adopted centre (a fixed split, so the likelihood stays smooth in the centre) take
    # the inner plane, inclination inc + dinc and nodes theta + dtheta, through the same centre; per-star line and normal.
    _, _, _, line_in, n_in = geo.symmetry_axis(ra0, dec0, jnp.deg2rad(p[4] + p[22]), jnp.deg2rad(p[5] + p[23]))
    inner = (rhat @ geo.sky_basis(jnp.deg2rad(LMC["ra0"]), jnp.deg2rad(LMC["dec0"]))[0] > jnp.cos(jnp.deg2rad(RHO_TILT)))[:, None]
    # Turning plane: time derivatives [1/Gyr] of the in-plane axes (line of nodes, n x line) of each plane when the inclination and
    # the nodes change at didt, dthdt [deg/Gyr] (forward-mode derivative of geo.symmetry_axis).
    def axes(inc, theta):
        _, _, _, ln, nn = geo.symmetry_axis(ra0, dec0, inc, theta)
        return ln, jnp.cross(nn, ln)
    rate = (jnp.deg2rad(p[25]), jnp.deg2rad(p[26]))
    dline, dperp = jax.jvp(axes, (jnp.deg2rad(p[4]), jnp.deg2rad(p[5])), rate)[1]
    dline_in, dperp_in = jax.jvp(axes, (jnp.deg2rad(p[4] + p[22]), jnp.deg2rad(p[5] + p[23])), rate)[1]
    line, n = jnp.where(inner, line_in, line), jnp.where(inner, n_in, n)
    dline, dperp = jnp.where(inner, dline_in, dline), jnp.where(inner, dperp_in, dperp)
    D = geo.plane_distance(rhat, r0, LMC["D0"], n)  # each star is put on the disc plane through the centre at distance D0
    # In-plane radius R [kpc] and the radial and azimuthal unit vectors of each star.
    R, _, eR, ephi = geo.lmc_cylindrical_basis(rhat, D, r0, LMC["D0"], n)
    # Azimuthal speed [km/s]: tanh rotation curve times the lopsided factor. cos(phi) = eR . line (phi = 0 on the line of
    # nodes) and sin(phi) = eR . (n x line), so phi increases along e_phi = n x e_R.
    # Streaming: v_phi also carries an m = 2 factor (m2c cos 2phi + m2s sin 2phi), and there is an in-plane radial velocity
    # v_R = vrot (vr0 + r2c cos 2phi + r2s sin 2phi), vrot = rotation(), positive outwards: elliptical (bar- or tide-driven)
    # streaming and a mean expansion or contraction. All five are zero unless fitted.
    cphi, sphi = jnp.sum(eR * line, axis=1), jnp.sum(eR * jnp.cross(n, line), axis=1)
    c2, s2 = cphi**2 - sphi**2, 2 * cphi * sphi
    vrot = rotation(p, R)
    vphi = vrot * (1 + p[13] * cphi + p[14] * sphi + p[17] * c2 + p[18] * s2)
    vR = vrot * (p[19] + p[20] * c2 + p[21] * s2)
    # 3D velocity [km/s] of every star: systemic line-of-sight velocity along r0, systemic transverse velocity (bulk proper
    # motion at the centre converted with D0), and rotation. The same systemic vector projected on each star's own basis gives
    # the perspective terms.
    Vsys = p[11] * r0 + KAPPA * LMC["D0"] * (p[2] * eE0 + p[3] * eN0)
    # Turning plane: a star at x = D rhat - D0 r0 [kpc] from the centre, with in-plane coordinates X = x . line, Y = x . (n x line)
    # held fixed, moves at X dline/dt + Y d(n x line)/dt [kpc/Gyr]; to km/s with KPC_GYR. Added to Vsys (not scaled by qlos).
    x = D[:, None] * rhat - LMC["D0"] * r0
    perp = jnp.cross(n, line)
    Vsys = Vsys + KPC_GYR * (jnp.sum(x * line, axis=1)[:, None] * dline + jnp.sum(x * perp, axis=1)[:, None] * dperp)
    Vint = vphi[:, None] * ephi + vR[:, None] * eR  # internal motion
    V = Vsys + Vint
    # Project on the star's east, north and line-of-sight vectors; transverse components to mas/yr with the star's own distance.
    # The line of sight sees the internal motion scaled by 1 + qlos (zero unless fitted).
    return (jnp.sum(V * ea, axis=1) / (KAPPA * D), jnp.sum(V * ed, axis=1) / (KAPPA * D),
            jnp.sum((Vsys + (1 + p[24]) * Vint) * rhat, axis=1), D, R, eR, ephi, jnp.broadcast_to(n, eR.shape))


def _logn2(dx, dy, a, b, c):
    """log N(0, [[a, c], [c, b]]) at (dx, dy).

    Parameters
    ----------
    dx, dy : array
        Residuals (here in mas/yr).
    a, b, c : array
        Variances of dx and dy and their covariance (same units squared).

    Returns
    -------
    array
        Log density of the bivariate normal, written out for speed (no matrix inverse).
    """
    det = a * b - c**2
    return -0.5 * (b * dx**2 - 2 * c * dx * dy + a * dy**2) / det - 0.5 * jnp.log(det) - jnp.log(2 * jnp.pi)


def _logn1(x, v):
    """log N(0, v) at x: one-dimensional normal log density with variance v (here km^2/s^2)."""
    return -0.5 * x**2 / v - 0.5 * jnp.log(2 * jnp.pi * v)


def neg_loglike(p, d, ellipsoid=False):
    """Negative log likelihood: -sum of loglike_stars(p, d, ellipsoid); d from data(). p : (33,) in PARAMS order; returns a scalar."""
    return -jnp.sum(loglike_stars(p, d, ellipsoid))


def _logn3(r, C):
    """log N(0, C) at r for a stack of 3-vectors r (N, 3) and covariances C (N, 3, 3)."""
    _, logdet = jnp.linalg.slogdet(C)
    return -0.5 * jnp.sum(r * jnp.linalg.solve(C, r[..., None])[..., 0], axis=-1) - 0.5 * logdet - 1.5 * jnp.log(2 * jnp.pi)


def intrinsic_cov(p, d):
    """Intrinsic velocity covariance of the disc-frame ellipsoid at each star of d (from data()), in its (east, north, line of
    sight) basis: shape (N, 3, 3) [km^2/s^2]; no measurement errors.

    C = Q diag(sigma_R^2, sigma_phi^2, sigma_z^2) Q^T with each sigma scaled by exp(dlnsig_dR (R - R_SIG)) (module docstring).
    """
    _, _, _, _, R, eR, ephi, n = _model(p, d["rhat"], d["ea"], d["ed"])
    # Q[i, a, b]: sky direction a (east, north, line of sight) dotted with disc direction b (e_R, e_phi, n).
    Q = jnp.stack([d["ea"], d["ed"], d["rhat"]], -2) @ jnp.stack([eR, ephi, n], -1)
    s2 = (jnp.stack([p[27], p[28], p[29]]) * jnp.exp(p[30] * (R - R_SIG))[:, None]) ** 2  # (N, 3) [km^2/s^2]
    return jnp.einsum("nab,nb,ncb->nac", Q, s2, Q)


def _loglike_ellipsoid(p, d):
    """The proper-motion and line-of-sight terms of the ellipsoid likelihood (module docstring) for the model and outlier
    components: (ln L_model, ln L_outlier) per star, without the mixture weights and the distance term."""
    mra, mde, mlos, D = model(p, d["rhat"], d["ea"], d["ed"])
    kD = KAPPA * D
    m = jnp.stack([d["pm"], d["pm"], d["los"]], -1)  # 1 for a component the star has, 0 otherwise
    r = m * jnp.stack([kD * (d["pmra"] - mra), kD * (d["pmdec"] - mde), d["vlos"] - mlos], -1)  # residuals [km/s]
    C = intrinsic_cov(p, d).at[:, :2, :2].add((p[9] * kD)[:, None, None] ** 2 * d["cov"]).at[:, 2, 2].add(d["evlos"] ** 2)
    mm = m[:, :, None] * m[:, None, :]
    eye = jnp.eye(3)
    C = mm * C + (1 - m)[:, :, None] * eye  # a missing component: unit variance, zero covariance, zero residual
    # Missing components contribute log N(0 | 0, 1); remove it. Jacobian (kappa D)^2: density per mas/yr, not per km/s.
    corr = 0.5 * jnp.log(2 * jnp.pi) * jnp.sum(1 - m, -1) + 2 * d["pm"] * jnp.log(kD)
    out = ((kD * S_OUT) ** 2)[:, None, None] * m[:, :, None] * eye  # outlier width added to every component the star has
    return _logn3(r, C) + corr, _logn3(r, C + out) + corr


def loglike_stars(p, d, ellipsoid=False):
    """Log likelihood of each star; d from data(); with ellipsoid, the disc-frame ellipsoid likelihood (module docstring).

    Parameters
    ----------
    p : array_like, shape (33,)
        Parameter vector in the order of PARAMS.
    d : dict
        Star arrays from data().
    ellipsoid : bool
        Use the disc-frame ellipsoid (sigma_R, sigma_phi, sigma_z, dlnsig_dR) instead of sigma and sigma_los.

    Returns
    -------
    array, shape (N,)
        ln[(1 - f) L_model + f L_outlier] per star, each L the product of the proper-motion term and, for stars with a
        line-of-sight velocity, the line-of-sight term and, for stars with a distance modulus, the distance term.
    """
    f = jax.nn.sigmoid(p[10])  # outlier fraction from its logit
    mra, mde, mlos, D = model(p, d["rhat"], d["ea"], d["ed"])
    if ellipsoid:
        l_in, l_out = _loglike_ellipsoid(p, d)
        l_in, l_out = l_in + jnp.log1p(-f), l_out + jnp.log(f)
    else:
        l_in, l_out = _loglike_sky(p, d, mra, mde, mlos, D, f)
    # Distance term: relative modulus [mag] about dmu0 + 5 log10(D / D0), variance sigma_mu^2; outliers uniform over the window
    # |dmu| < DMU_WINDOW that selected the members. d["dist"] is 1 for stars with a modulus and 0 otherwise.
    r = d["mu"] - p[15] - 5 * jnp.log10(D / LMC["D0"])
    l_in = l_in + d["dist"] * _logn1(r, p[16] ** 2)
    l_out = l_out - d["dist"] * jnp.log(2 * DMU_WINDOW)
    return jnp.logaddexp(l_in, l_out)


def _loglike_sky(p, d, mra, mde, mlos, D, f):
    """The proper-motion and line-of-sight terms of the default likelihood (sky-aligned dispersions sigma, sigma_los), with the
    mixture weights: (ln[(1 - f) L_model], ln[f L_outlier]) per star, without the distance term."""
    dx, dy = d["pmra"] - mra, d["pmdec"] - mde  # proper-motion residuals [mas/yr]
    s2 = (p[8] / (KAPPA * D)) ** 2  # intrinsic dispersion sigma [km/s] as a proper-motion variance at the star's distance
    # Total covariance: Gaia covariance scaled by k^2 (error-scale factor) plus the isotropic intrinsic dispersion.
    a, b, c = p[9] ** 2 * d["cov"][:, 0, 0] + s2, p[9] ** 2 * d["cov"][:, 1, 1] + s2, p[9] ** 2 * d["cov"][:, 0, 1]
    # d["pm"] is 1 for stars with a proper motion and 0 otherwise (their proper-motion term is dropped).
    l_in = d["pm"] * _logn2(dx, dy, a, b, c) + jnp.log1p(-f)
    l_out = d["pm"] * _logn2(dx, dy, a + S_OUT**2, b + S_OUT**2, c) + jnp.log(f)
    # Line-of-sight term: residual [km/s] with variance e_vlos^2 + sigma_los^2; the outlier term adds (kappa D S_OUT)^2.
    # d["los"] is 1 for stars with a velocity and 0 otherwise, so stars without one contribute their proper motions only.
    r, v = d["vlos"] - mlos, d["evlos"] ** 2 + p[12] ** 2
    l_in = l_in + d["los"] * _logn1(r, v)
    l_out = l_out + d["los"] * _logn1(r, v + (KAPPA * D * S_OUT) ** 2)
    return l_in, l_out


def data(ra, dec, pmra, pmdec, pmra_error, pmdec_error, pmra_pmdec_corr, vlos=None, evlos=None, mu=None):
    """Arrays for neg_loglike from catalogue columns (degrees, mas/yr, km/s); stars with vlos NaN (or no vlos given) use their
    proper motions only.

    Parameters
    ----------
    ra, dec : array_like, shape (N,)
        Positions [deg].
    pmra, pmdec, pmra_error, pmdec_error : array_like, shape (N,)
        Gaia proper motions and their errors [mas/yr] (pmra includes cos dec); NaN pmra where a star has none (or is to be
        fitted without it).
    pmra_pmdec_corr : array_like, shape (N,)
        Gaia correlation coefficient of the two proper-motion components.
    vlos, evlos : array_like, shape (N,), optional
        Line-of-sight velocity and its error [km/s]; NaN where a star has none.
    mu : array_like, shape (N,), optional
        Relative distance modulus [mag] (0 at the mean LMC distance); NaN where a star has none.

    Returns
    -------
    dict
        rhat, ea, ed (N, 3) unit vectors; pmra, pmdec (N,); cov (N, 2, 2) [mas^2/yr^2]; vlos, evlos (N,) [km/s] with placeholders
        0 and 1 where missing; los (N,) 1.0 where a velocity exists, else 0.0; mu (N,) [mag], 0 where missing, and dist (N,) 1.0
        where a modulus exists, else 0.0.
    """
    rhat, ea, ed = (jnp.asarray(v) for v in geo.sky_basis(np.deg2rad(ra), np.deg2rad(dec)))
    pm = np.isfinite(pmra)
    # Placeholders where the proper motion is missing (masked out by `pm`): zero motion, unit errors, no correlation.
    pmra, pmdec = np.where(pm, pmra, 0.0), np.where(pm, pmdec, 0.0)
    pmra_error, pmdec_error = np.where(pm, pmra_error, 1.0), np.where(pm, pmdec_error, 1.0)
    c = np.where(pm, pmra_pmdec_corr, 0.0) * pmra_error * pmdec_error  # covariance of the two proper-motion components
    cov = np.stack([np.stack([pmra_error**2, c], -1), np.stack([c, pmdec_error**2], -1)], -2)
    vlos = np.full(len(pmra), np.nan) if vlos is None else np.asarray(vlos, float)
    los = np.isfinite(vlos)
    evlos = np.where(los, 1.0 if evlos is None else evlos, 1.0)  # placeholder 1 km/s where missing (masked out by `los`)
    mu = np.full(len(pmra), np.nan) if mu is None else np.asarray(mu, float)
    dist = np.isfinite(mu)
    return dict(rhat=rhat, ea=ea, ed=ed, pmra=jnp.asarray(pmra), pmdec=jnp.asarray(pmdec), cov=jnp.asarray(cov), pm=jnp.asarray(pm, float),
                vlos=jnp.asarray(np.where(los, vlos, 0.0)), evlos=jnp.asarray(evlos), los=jnp.asarray(los, float),
                mu=jnp.asarray(np.where(dist, mu, 0.0)), dist=jnp.asarray(dist, float))


def start():
    """Starting point: the adopted LMC parameters, no offset, v0 60 km/s, R0 2 kpc, sigma 40 km/s, k 1, outliers 5%,
    sigma_los 25 km/s, not lopsided, moduli zero point 0 and scatter 0.1 mag, no streaming, one plane (no tilted ring), no line-of-sight scale mismatch, a fixed orientation;
    ellipsoid sigma_R = sigma_phi = 40, sigma_z = 25 km/s, constant with radius; tanh rotation curve (vslope 0); no drift (h 0).

    Returns
    -------
    ndarray, shape (33,)
        Parameter vector in the order of PARAMS.
    """
    return np.array([0.0, 0.0, LMC["pmra0"], LMC["pmdec0"], LMC["inc"], LMC["theta"], 60.0, 2.0, 40.0, 1.0, np.log(0.05 / 0.95),
                     LMC["vlos0"], 25.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 40.0, 25.0, 0.0, 0.0,
                     0.0])


def fit(d, free=BASE, p0=None, ellipsoid=False):
    """Maximum likelihood over the `free` parameters (others held at p0), within BOUNDS. Returns (p, errors from the inverse
    Hessian, -lnL); errors are NaN for parameters at a bound or when the Hessian is singular.

    Parameters
    ----------
    d : dict
        Star arrays from data().
    free : tuple of str
        Names (from PARAMS) to optimise; default BASE.
    p0 : array_like, shape (33,), optional
        Starting vector and values of the held parameters; default start().
    ellipsoid : bool
        Use the ellipsoid likelihood (loglike_stars).

    Returns
    -------
    p : ndarray, shape (33,)
        Best-fitting vector.
    err : ndarray, shape (33,)
        Square roots of the diagonal of the inverse Hessian of -lnL over the free parameters that are not at a bound (NaN for
        held ones and for those at a bound; NaN for all when that Hessian is not positive definite or its condition number
        is >= 1e12).
    nll : float
        -lnL at the optimum.

    Notes
    -----
    L-BFGS-B with exact JAX gradients and tight tolerances (ftol 1e-15, gtol 1e-6); a RuntimeWarning when it stops without
    convergence. A local optimiser: the result can depend on
    the start, and a parameter can stop at a bound.
    """
    p0 = start() if p0 is None else np.asarray(p0, float)
    idx = np.array([PARAMS.index(n) for n in free])
    full = lambda q: jnp.asarray(p0).at[idx].set(q)  # noqa: E731  (free values q inserted into the full vector)
    x, e, nll = _optimise(lambda q: neg_loglike(full(q), d, ellipsoid), p0[idx], [BOUNDS[n] for n in free])
    err = np.full(len(PARAMS), np.nan)
    err[idx] = e
    return np.array(full(x)), err, nll  # a writable copy (np.asarray of a JAX array is read-only)


def _optimise(nll, x0, bounds):
    """Minimise nll (a JAX function of the free values) from x0 within bounds; returns (x, errors, nll at x).

    L-BFGS-B with exact JAX gradients and tolerances tighter than its defaults, which stop on flat likelihoods before the minimum
    (by up to 2 in -lnL on a proper-motion-only fit of a hot thick mock; tests/test_kinfit.py test_fit_converges_to_the_minimum);
    a RuntimeWarning when it stops without convergence. Errors: square roots of the diagonal of the inverse Hessian over the
    values not at a bound (within 1e-6 of the range; those get NaN), i.e. with the bounded ones held fixed; NaN for all when that
    Hessian is not positive definite or its condition number is >= 1e12.
    """
    f = jax.jit(nll)
    g = jax.jit(jax.grad(f))
    r = minimize(lambda q: float(f(q)), x0, jac=lambda q: np.asarray(g(q)), method="L-BFGS-B", bounds=bounds,
                 options=dict(ftol=1e-15, gtol=1e-6, maxiter=20000, maxfun=50000))
    if not r.success:  # e.g. iteration limit or a failed line search: the result may not be the minimum
        warnings.warn(f"kinfit.fit did not converge: {r.message} (|grad| {np.abs(r.jac).max():.2g})", RuntimeWarning)
    err = np.full(len(x0), np.nan)
    lo, hi = np.array(bounds).T
    inside = np.minimum(r.x - lo, hi - r.x) > 1e-6 * (hi - lo)
    H = np.asarray(jax.hessian(f)(r.x))[np.ix_(inside, inside)]
    if inside.any() and np.all(np.linalg.eigvalsh(H) > 0) and np.linalg.cond(H) < 1e12:
        err[inside] = np.sqrt(np.diag(np.linalg.inv(H)))
    return r.x, err, float(r.fun)


def sample(ds, p0s, fixed=(), num_warmup=500, num_samples=1000, num_chains=4, seed=0, ellipsoid=False, shared=SHARED):
    """NUTS posterior for the populations ds (list of data()) sharing `shared` (default SHARED); p0s: starting vectors (e.g. from fit()), the shared
    part taken from p0s[0]. Parameters in `fixed` stay at their p0s values, as do sigma_los of a population without line-of-sight
    velocities, dmu0 and sigma_mu of a population without distance moduli, and vlos0 when no population has them. Returns the numpyro MCMC object: samples named as PARAMS, per-population
    ones suffixed _<index>. Parallel chains need numpyro.set_host_device_count(num_chains) before JAX starts.

    Parameters
    ----------
    ds : list of dict
        One data() dictionary per population.
    p0s : list of array_like, shape (33,) each
        Starting vectors (also the values of held parameters).
    fixed : tuple of str
        Names held at their p0s values (e.g. ('m1c', 'm1s') for a symmetric disc, ('dx', 'dy') for the adopted centre).
    num_warmup, num_samples, num_chains : int
        NUTS warm-up steps, kept draws per chain and number of chains (run in parallel).
    seed : int
        JAX random seed.
    ellipsoid : bool
        Use the ellipsoid likelihood; sigma and sigma_los are then held, otherwise ELLIPSOID is held.
    shared : tuple of str
        Parameters common to all populations; SHARED plus ('v0', 'R0', 'vslope') shares one circular-speed curve when h > 0.
        The scale lengths h are always held at their p0s values (measured from the densities, not the kinematics).

    Returns
    -------
    numpyro.infer.MCMC
        Samples: shared parameters under their own names, per-population ones as '<name>_<index>'.

    Notes
    -----
    Priors are uniform within BOUNDS. The likelihood is the sum of neg_loglike over populations. NUTS uses a dense mass matrix
    (correlations such as the centre-bulk motion trade-off are strong) and starts from p0s.
    """
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_value

    has_los = [bool(np.any(np.asarray(d["los"]))) for d in ds]
    has_mu = [bool(np.any(np.asarray(d["dist"]))) for d in ds]
    held = set(fixed) | (set() if any(has_los) else {"vlos0"})  # vlos0 is unconstrained without line-of-sight velocities
    held |= {"sigma", "sigma_los"} if ellipsoid else set(ELLIPSOID)  # the other likelihood's dispersions do not enter
    held |= {"h"}
    sites = {}  # (population, name) -> site name, or None when held at p0
    for i in range(len(ds)):
        for n in PARAMS:
            hold = n in held or (n == "sigma_los" and not has_los[i]) or (n in ("dmu0", "sigma_mu") and not has_mu[i])
            sites[i, n] = None if hold else (n if n in shared else f"{n}_{i}")

    def model():
        # Draw each free site once: shared sites from population 0's entries, per-population sites from every population.
        drawn = {s: numpyro.sample(s, dist.Uniform(*BOUNDS[n])) for (i, n), s in sites.items() if s is not None and (i == 0 or n not in shared)}
        nll = 0.0
        for i, d in enumerate(ds):
            # Assemble population i's full vector: drawn values, or the held p0 values (shared ones from p0s[0]).
            q = [p0s[0 if n in shared else i][j] if sites[i, n] is None else drawn[sites[i, n]] for j, n in enumerate(PARAMS)]
            nll = nll + neg_loglike(jnp.stack([jnp.asarray(x, float) for x in q]), d, ellipsoid)
        numpyro.factor("loglike", -nll)

    # Start each free site at its p0 value, moved just inside the bounds (a value on a bound has zero prior density).
    init = {s: float(np.clip(p0s[0 if n in shared else i][PARAMS.index(n)], BOUNDS[n][0] + 1e-3, BOUNDS[n][1] - 1e-3))
            for (i, n), s in sites.items() if s is not None}
    mcmc = MCMC(NUTS(model, dense_mass=True, init_strategy=init_to_value(values=init)), num_warmup=num_warmup,
                num_samples=num_samples, num_chains=num_chains, chain_method="parallel", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    return mcmc


def residual_polar(p, cat):
    """Position angle [deg] about the fitted centre and the residual (observed - model) tangential and radial velocities [km/s]
    in the sky-polar basis about that centre, for catalogue dict cat (ra, dec, pmra, pmdec in degrees and mas/yr).

    Parameters
    ----------
    p : array_like, shape (33,)
        Parameter vector (its centre offset defines the polar basis).
    cat : dict
        ra, dec [deg], pmra, pmdec [mas/yr] arrays.

    Returns
    -------
    pa : ndarray
        Position angle about the fitted centre [deg], north through east, in [0, 360).
    vt, vr : ndarray
        Residual velocity [km/s] along the sky-polar tangential and radial directions about the fitted centre, converted from
        mas/yr with each star's disc distance.
    """
    from .footprint import lmc_xy
    from .perspective import sky_polar_basis
    # Unit errors and zero correlation: only the sky bases are needed from data() here.
    d = data(cat["ra"], cat["dec"], cat["pmra"], cat["pmdec"], np.ones(len(cat["ra"])), np.ones(len(cat["ra"])),
             np.zeros(len(cat["ra"])))
    mra, mde, _, D = (np.asarray(v) for v in model(p, d["rhat"], d["ea"], d["ed"]))
    ra0, dec0 = (np.rad2deg(float(v)) for v in centre(p[0], p[1]))
    # sky_polar_basis and lmc_xy work about lmcold.constants.LMC: point it at the fitted centre temporarily, then restore it.
    saved = dict(LMC)
    LMC.update(ra0=ra0 % 360, dec0=dec0)
    try:
        (rE, rN), (tE, tN) = sky_polar_basis(cat["ra"], cat["dec"])
        x, y = lmc_xy(cat["ra"], cat["dec"])
    finally:
        LMC.update(saved)
    dvE, dvN = KAPPA * D * (cat["pmra"] - mra), KAPPA * D * (cat["pmdec"] - mde)  # residuals east and north [km/s]
    return np.rad2deg(np.arctan2(x, y)) % 360, dvE * tE + dvN * tN, dvE * rE + dvN * rN
