"""Three-dimensional shape of the RR Lyrae from their sky positions and distance moduli (paper Section 5), summarised in sky cells.

Model: an oblate spheroid with density nu proportional to K_0(m / h), m^2 = R^2 + z^2 / q^2 (lmcold.spheroid.spheroid: its face-on
profile is exponential with scale length h, its edge-on ellipticity 1 - q), centred on the LMC centre (lmcold.constants.LMC) at
distance D0, its symmetry axis at inclination inc and line of nodes theta (lmcold.geometry.symmetry_axis). Along the line of sight r_hat
the stars at relative modulus u = 5 log10(D / D0) have density n(r_hat, u) proportional to D^3 nu(x(D r_hat)) (los_density;
D^2 dD dOmega = (ln 10 / 5) D^3 du dOmega).

Sky cells (cell_stats, cell_model, cell_loglike, fit_cells): the RR Lyrae inside the footprint annulus RHO_MIN < rho < RHO_MAX
(OGLE-IV coverage complete there; the crowded centre excluded) and the membership window |dmu| < DMU_WINDOW are counted in CELL_RHO
rings x CELL_NPA sectors, and the median modulus of each cell is taken. The counts give the projected density (its outline on the sky,
q_sky), the medians the near-to-far tilt (the mean-depth gradient g; projected() gives both for an oblate body). The scatter of the
moduli about the median, and so the per-star noise floor, does not enter. Not modelled: in-plane ellipticity, radially varying
orientation, a lopsided density, the bar (paper Section 5 and its appendix: the oblate model fits the cells poorly, and an in-plane
elongation of the old population explains its preferred inclination).

Parameters of the cell fit (CELL_PARAMS): inc, theta [deg], q, h [kpc], delta [mag]; flat priors within BOUNDS. K_0 in JAX: k0()
(Abramowitz & Stegun 1964, 9.8.5-6, relative error < 2e-7). mock() draws stars from the model for tests (place() of spheroid.spheroid
or given positions, with a noise floor per FLOOR_EDGES annulus, PARAMS order: inc, theta, q, h, delta, f, sigma_1 ... sigma_k; a
fraction f of outliers uniform over the footprint and the window).
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import i0

from . import geometry as geo
from .constants import LMC
from .footprint import DMU_WINDOW, angular_distance

FLOOR_EDGES = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0)  # deg, annuli with their own noise floor (narrow where crowding changes fastest)
RHO_MIN, RHO_MAX = FLOOR_EDGES[0], FLOOR_EDGES[-1]  # deg, inner and outer radius of the footprint annulus
# Parameter order of mock() and place(): inc, theta [deg], q, h [kpc], delta [mag], outlier fraction f, noise floors [mag].
PARAMS = ('inc', 'theta', 'q', 'h', 'delta', 'f') + tuple(f'sigma_{k + 1}' for k in range(len(FLOOR_EDGES) - 1))
# Box bounds (flat priors) in the units of PARAMS.
BOUNDS = dict(inc=(0.0, 70.0), theta=(90.0, 230.0), q=(0.1, 1.2), h=(0.5, 4.0), delta=(-0.3, 0.3), f=(0.0, 0.5),
              **{f'sigma_{k + 1}': (0.01, 0.3) for k in range(len(FLOOR_EDGES) - 1)})
U = np.linspace(-0.9, 0.9, 121)  # relative modulus grid [mag] along each line of sight (window 0.6 plus noise)
NSKY = (60, 180)  # sky quadrature: points in cos(rho) x position angle


def k0(x):
    """Modified Bessel function K_0(x), x > 0 (Abramowitz & Stegun 9.8.5 for x <= 2, 9.8.6 above).

    Parameters
    ----------
    x : array
        Argument (dimensionless), any shape.

    Returns
    -------
    array
        K_0(x), same shape. Both branches are evaluated (jnp.where), so x is clipped to >= 2 in the large-x branch.
    """
    t = (x / 2) ** 2  # A&S 9.8.5 polynomial variable (x / 2)^2
    small = -jnp.log(x / 2) * i0(x) + (-0.57721566 + t * (0.42278420 + t * (0.23069756 + t * (0.03488590 + t * (0.00262698
                                                       + t * (0.00010750 + t * 0.0000074))))))
    y = 2 / jnp.maximum(x, 2.0)  # A&S 9.8.6 variable 2 / x
    large = jnp.exp(-x) / jnp.sqrt(x) * (1.25331414 + y * (-0.07832358 + y * (0.02189568 + y * (-0.01062446 + y * (0.00587872
                                                              + y * (-0.00251540 + y * 0.00053208))))))
    return jnp.where(x <= 2, small, large)


def frame(inc, theta):
    """Centre direction r0, line of nodes, in-plane perpendicular and symmetry axis n (unit vectors) for inc, theta [deg].

    Returns four ICRS unit vectors, shape (3,) each, for the adopted centre (lmcold.constants.LMC); the perpendicular is n x line.
    """
    r0, _, _, line, n = geo.symmetry_axis(jnp.deg2rad(LMC['ra0']), jnp.deg2rad(LMC['dec0']), jnp.deg2rad(inc), jnp.deg2rad(theta))
    return r0, line, jnp.cross(n, line), n


def rhat(ra, dec):
    """ICRS unit vectors (N, 3) of sky positions [deg] (as lmcold.geometry.sky_basis)."""
    return np.asarray(geo.sky_basis(np.deg2rad(np.asarray(ra, float)), np.deg2rad(np.asarray(dec, float)))[0])


def rho(r):
    """Angular distance [deg] of unit vectors r (N, 3) from the LMC centre."""
    r0 = np.asarray(geo.sky_basis(np.deg2rad(LMC['ra0']), np.deg2rad(LMC['dec0']))[0])
    return np.rad2deg(np.arccos(np.clip(np.asarray(r) @ r0, -1, 1)))


def annulus(r):
    """Index of the FLOOR_EDGES annulus of unit vectors r (N, 3)."""
    return np.clip(np.searchsorted(FLOOR_EDGES, rho(r)) - 1, 0, len(FLOOR_EDGES) - 2)


def footprint():
    """Sky quadrature of the annulus: unit vectors (NSKY[0] x NSKY[1], 3) at the midpoints of equal steps in cos(rho) and position
    angle (each of equal solid angle), their annulus indices, and the annulus solid angle [sr]."""
    c1, c2 = np.cos(np.deg2rad(RHO_MIN)), np.cos(np.deg2rad(RHO_MAX))
    cr = c2 + (np.arange(NSKY[0]) + 0.5) / NSKY[0] * (c1 - c2)
    pa = (np.arange(NSKY[1]) + 0.5) / NSKY[1] * 2 * np.pi
    cr, pa = (a.ravel() for a in np.meshgrid(cr, pa, indexing='ij'))
    r0, eE, eN = (np.asarray(v) for v in geo.sky_basis(np.deg2rad(LMC['ra0']), np.deg2rad(LMC['dec0'])))
    r = cr[:, None] * r0 + np.sqrt(1 - cr ** 2)[:, None] * (np.cos(pa)[:, None] * eN + np.sin(pa)[:, None] * eE)
    return r, annulus(r), 2 * np.pi * (c1 - c2)


def los_density(p, r):
    """n(r_hat, u) up to a constant on the grid U for unit vectors r (N, 3): D^3 K_0(m / h), shape (N, len(U))."""
    inc, theta, q, h = p[0], p[1], p[2], p[3]
    r0, _, _, n = frame(inc, theta)
    D = LMC['D0'] * 10 ** (U / 5)  # distances [kpc] of the modulus grid U
    X = D[None, :, None] * r[:, None, :] - LMC['D0'] * r0  # positions about the centre [kpc]
    z = X @ n  # height above the symmetry plane [kpc]
    R2 = jnp.sum(X ** 2, axis=2) - z ** 2  # squared in-plane radius [kpc^2]
    return D ** 3 * k0(jnp.sqrt(R2 + (z / q) ** 2) / h)


CELL_RHO = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0)  # deg, radial edges of the sky cells of cell_stats()
CELL_NPA = 24  # position-angle sectors of the sky cells (15 deg)
CELL_PARAMS = ('inc', 'theta', 'q', 'h', 'delta')  # parameters of the cell fit: deg, deg, -, kpc, mag
CELL_BOUNDS = {n: BOUNDS[n] for n in CELL_PARAMS}


def cell_index(r):
    """Sky-cell index (radial ring x position-angle sector, CELL_RHO x CELL_NPA) of unit vectors r (N, 3); -1 outside the rings."""
    r0, eE, eN = (np.asarray(v) for v in geo.sky_basis(np.deg2rad(LMC['ra0']), np.deg2rad(LMC['dec0'])))
    pa = np.arctan2(r @ eE, r @ eN) % (2 * np.pi)  # position angle [rad], north through east
    ring = np.searchsorted(CELL_RHO, rho(r)) - 1
    ok = (ring >= 0) & (ring < len(CELL_RHO) - 1)
    return np.where(ok, ring * CELL_NPA + (pa / (2 * np.pi) * CELL_NPA).astype(int), -1)


def cell_stats(ra, dec, dmu):
    """Per sky cell: number of stars, median modulus and its standard error (1.2533 x 1.4826 MAD / sqrt(N)); cells with fewer
    than 5 stars get no median (NaN). Only these enter cell_loglike(): the scatter of the moduli about the median, and so the noise
    floor, does not (it only sets the measured error of the median).

    Parameters
    ----------
    ra, dec : array_like, shape (N,)
        Positions [deg].
    dmu : array_like, shape (N,)
        Relative distance moduli [mag].

    Returns
    -------
    n, med, se : ndarray, shape ((len(CELL_RHO) - 1) * CELL_NPA,)
        Star counts, median moduli [mag] and their standard errors [mag] per cell.
    """
    k = cell_index(rhat(ra, dec))
    ncell = (len(CELL_RHO) - 1) * CELL_NPA
    n, med, se = np.zeros(ncell), np.full(ncell, np.nan), np.full(ncell, np.nan)
    for c in range(ncell):
        d = np.asarray(dmu)[k == c]
        n[c] = len(d)
        if len(d) >= 5:
            med[c] = np.median(d)
            # 1.4826 MAD: robust Gaussian sigma; 1.2533 = sqrt(pi / 2): standard error of a median relative to a mean.
            se[c] = 1.2533 * 1.4826 * np.median(np.abs(d - med[c])) / np.sqrt(len(d))
    return n, med, se


def cell_model(p, n, sky):
    """Model of the cell statistics for parameters p (CELL_PARAMS order) and observed counts n: expected counts lam (normalised to the
    observed total) and mean modulus Mc of each cell (the line-of-sight density's mean, plus delta). sky = (unit vectors, cell index)
    of the footprint quadrature (footprint() points, which have equal solid angle)."""
    rs, ks = sky
    dens = los_density(p, rs)  # (points, U)
    # Sc: integral of the density over each cell (sum over its equal-area points and the modulus grid); Mc: its mean modulus.
    Sc = jax.ops.segment_sum(jnp.sum(dens, axis=1), ks, num_segments=len(n))
    Mc = jax.ops.segment_sum(jnp.sum(dens * U, axis=1), ks, num_segments=len(n)) / Sc + p[4]
    return jnp.sum(n) * Sc / jnp.sum(Sc), Mc


def cell_loglike(p, stats, sky):
    """Log likelihood of the cell statistics for parameters p (CELL_PARAMS order): Poisson counts about cell_model's lam (the shape of
    the projected density: the elongation on the sky) and Gaussian medians about its Mc (noise symmetric about the true modulus does
    not move it: the near-to-far tilt)."""
    n, med, se = stats
    lam, Mc = cell_model(p, n, sky)
    ok = np.isfinite(med)  # cells with at least 5 stars (cell_stats)
    return jnp.sum(n * jnp.log(lam) - lam) - 0.5 * jnp.sum(((med[ok] - Mc[ok]) / se[ok]) ** 2)


def cell_sky():
    """(unit vectors, cell index) of the footprint quadrature points inside the cells: the sky argument of cell_model."""
    rs, _, _ = footprint()
    ks = cell_index(rs)
    return rs[ks >= 0], ks[ks >= 0]


def fit_cells(stats, fixed=None, start=(30.0, 160.0, 0.6, 1.7, 0.0)):
    """Maximum of cell_loglike (flat priors within CELL_BOUNDS) by L-BFGS-B on parameters scaled to the bounds; parameters
    named in fixed (dict, e.g. {'inc': 30, 'q': 0.6}) are held. Returns (parameters in CELL_PARAMS order, log likelihood).

    Parameters
    ----------
    stats : tuple of ndarray
        (n, med, se) from cell_stats().
    fixed : dict, optional
        Held parameters and their values (units of CELL_PARAMS).
    start : sequence of 5 floats
        Starting values: inc, theta [deg], q, h [kpc], delta [mag].

    Returns
    -------
    p : ndarray, shape (5,)
        Best-fitting parameters in CELL_PARAMS order.
    loglike : float
        cell_loglike at p.

    Notes
    -----
    A local optimiser; the result can depend on start.
    """
    from scipy.optimize import minimize
    sky = cell_sky()
    lo, hi = (np.array([CELL_BOUNDS[n][k] for n in CELL_PARAMS]) for k in (0, 1))
    fixed = fixed or {}
    free = np.array([k for k, n in enumerate(CELL_PARAMS) if n not in fixed])
    x0 = np.array(start, float)  # a copy
    for n, v in fixed.items():
        x0[CELL_PARAMS.index(n)] = v

    key = (id(stats), tuple(free))
    if key not in _CELL_FN:  # compiled once per data set and free set; the held values are an argument
        def full(u, x):
            return x.at[free].set(lo[free] + u * (hi[free] - lo[free]))
        _CELL_FN[key] = (full, jax.jit(jax.value_and_grad(lambda u, x: -cell_loglike(list(full(u, x)), stats, sky))))
    full, f = _CELL_FN[key]
    xj = jnp.asarray(x0)
    res = minimize(lambda u: tuple(np.asarray(v, float) for v in f(jnp.asarray(u), xj)),
                   np.clip((x0[free] - lo[free]) / (hi[free] - lo[free]), 0, 1), jac=True, method='L-BFGS-B',
                   bounds=[(0, 1)] * len(free), options=dict(maxiter=5000, ftol=1e-15, gtol=1e-6, maxls=50))
    return np.asarray(full(jnp.asarray(res.x), xj)), -float(res.fun)


_CELL_FN = {}  # compiled objectives of fit_cells


def projected(inc, q):
    """The two numbers the sky cells measure for an oblate spheroid of axis ratio q tilted by inc [deg] (density constant on
    similar spheroids): the axis ratio of its outline on the sky, q_sky^2 = cos^2 i + q^2 sin^2 i (from the counts), and the
    gradient of the mean depth with sky offset across the line of nodes, g = sin i cos i (1 - q^2) / q_sky^2 (from the median
    moduli; along each line of sight the density is symmetric about the point closest to the centre). A thin disc (q = 0) has
    q_sky = cos i and g = tan i; a sphere has q_sky = 1 and g = 0. Returns (q_sky, g).

    Parameters
    ----------
    inc : float or array
        Inclination [deg].
    q : float or array
        Intrinsic axis ratio (vertical / in-plane).

    Returns
    -------
    q_sky : float or array
        Projected axis ratio.
    g : float or array
        Mean-depth gradient, in kpc of depth per kpc of sky offset (dimensionless).
    """
    c, s = np.cos(np.deg2rad(inc)), np.sin(np.deg2rad(inc))
    q2 = c ** 2 + q ** 2 * s ** 2
    return np.sqrt(q2), s * c * (1 - q ** 2) / q2


def place(X, p, rng, sigma_f=None):
    """(ra, dec, dmu) [deg, deg, mag] of disc-frame positions X (N, 3) [kpc] observed with parameters p (PARAMS order): moduli with the
    zero point and Gaussian noise with the floor of each star's annulus (or the per-star array sigma_f). No outliers."""
    inc, theta, delta = p[0], p[1], p[4]
    r0, line, perp, nrm = (np.asarray(v) for v in frame(inc, theta))
    # Heliocentric position [kpc]: centre plus the disc-frame coordinates along line, perp and the symmetry axis.
    pos = LMC['D0'] * r0 + X[:, :1] * line + X[:, 1:2] * perp + X[:, 2:3] * nrm
    D = np.linalg.norm(pos, axis=1)
    ra = np.rad2deg(np.arctan2(pos[:, 1], pos[:, 0])) % 360
    dec = np.rad2deg(np.arcsin(pos[:, 2] / D))
    if sigma_f is None:
        sigma_f = np.asarray(p[6:])[annulus(rhat(ra, dec))]
    return ra, dec, 5 * np.log10(D / LMC['D0']) + delta + rng.normal(0, 1, len(D)) * sigma_f


def in_footprint(ra, dec, dmu):
    """Mask of stars inside the footprint annulus and the window.

    ra, dec [deg] and dmu [mag], shape (N,); returns a bool array (N,): RHO_MIN < rho < RHO_MAX and |dmu| < DMU_WINDOW.
    """
    rho_ = angular_distance(ra, dec, LMC['ra0'], LMC['dec0'])
    return (rho_ > RHO_MIN) & (rho_ < RHO_MAX) & (np.abs(dmu) < DMU_WINDOW)


def outliers(n, rng):
    """(ra, dec, dmu) of n outliers uniform over the footprint and the window.

    n : int; rng : numpy.random.Generator. Returns three arrays (n,) [deg, deg, mag], uniform in solid angle (uniform in cos rho
    and position angle) over the annulus and uniform in dmu over (-DMU_WINDOW, DMU_WINDOW).
    """
    c1, c2 = np.cos(np.deg2rad(RHO_MIN)), np.cos(np.deg2rad(RHO_MAX))
    cr, pa = rng.uniform(c2, c1, n), rng.uniform(0, 2 * np.pi, n)
    r0, eE, eN = (np.asarray(v) for v in geo.sky_basis(np.deg2rad(LMC['ra0']), np.deg2rad(LMC['dec0'])))
    r = cr[:, None] * r0 + np.sqrt(1 - cr ** 2)[:, None] * (np.cos(pa)[:, None] * eN + np.sin(pa)[:, None] * eE)
    return np.rad2deg(np.arctan2(r[:, 1], r[:, 0])) % 360, np.rad2deg(np.arcsin(r[:, 2])), rng.uniform(-DMU_WINDOW, DMU_WINDOW, n)


def mock(p, n, rng, X=None, sigma_f=None):
    """(ra, dec, dmu) of n selected stars: a fraction p[f] of outliers (outliers()), the rest place() of positions X (default:
    spheroid.spheroid with p's h and q) that fall in the footprint and the window; sigma_f, if given, is a function of rho [deg].

    Parameters
    ----------
    p : sequence
        Parameters in PARAMS order.
    n : int
        Number of selected stars wanted.
    rng : numpy.random.Generator
    X : ndarray, shape (M, 3), optional
        Disc-frame positions [kpc]; default 20 n draws of spheroid(h, q).
    sigma_f : callable, optional
        Noise floor [mag] as a function of rho [deg]; default the per-annulus floors of p.

    Returns
    -------
    ra, dec, dmu : ndarray
        Positions [deg] and moduli [mag]: at most n stars (fewer if not enough of X falls in the footprint).
    """
    from .spheroid import spheroid
    n_out = rng.binomial(n, p[5])
    X = spheroid(p[3], p[2], 20 * n, rng) if X is None else X
    ra, dec, _ = place(X, p, rng, 0.0)  # noiseless placement, only to get rho for sigma_f
    sf = None if sigma_f is None else sigma_f(angular_distance(ra, dec, LMC['ra0'], LMC['dec0']))
    ra, dec, dmu = place(X, p, rng, sf)
    k = np.flatnonzero(in_footprint(ra, dec, dmu))[:n - n_out]
    o = outliers(n_out, rng)
    return np.r_[ra[k], o[0]], np.r_[dec[k], o[1]], np.r_[dmu[k], o[2]]
