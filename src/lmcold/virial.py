"""Flattening of a tracer population implied by its kinematics: the tensor virial theorem for a tracer in an external potential (paper
Section 5).

For a steady, axisymmetric tracer with z along the symmetry axis,
    2 K_ij + W_ij = 0,   K_ij = 1/2 Sum v_i v_j,   W_ij = -Sum x_i dPhi/dx_j   (sums over the tracer's stars),
so the ratio of the in-plane to the vertical components,
    Omega_kin = (Sum v_x^2 + Sum v_y^2) / (2 Sum v_z^2)  =  Omega_pot(q) = (W_xx + W_yy) / (2 W_zz),
ties the kinematics (left: mean rotation and dispersions) to the shape of the tracer in the potential (right; q = c/a). This is the
virial theorem of a tracer in a fixed potential (Binney & Tremaine 2008, Sect. 4.8.3,
https://ui.adsabs.harvard.edu/abs/2008gady.book.....B), not the self-gravitating (v/sigma, eps) relation: the LMC's old stars are a
small part of its mass. Steady state and axisymmetry are assumed.

flattening(): both sides from one realisation of a fitted kinematic model (lmcold.spheroid.spheroid and .velocities: oblate spheroid with the
population's exponential scale length h, mean rotation kinfit.rotation, ellipsoid with its radial gradient). The velocities depend on R
only, and the face-on profile of the realisation does not depend on q, so Omega_kin is computed once (q = 1); Omega_pot is computed on a
grid of q by squashing the same positions in z, and q follows by interpolation.

POTENTIALS (the range is the systematic error of q; values chosen to bracket the LMC circular speed, 65-90 km/s at 2-8 kpc):
  P1 cored logarithmic halo, V = 92 km/s, a = 2 kpc, spherical: Phi = 1/2 V^2 ln(R^2 + z^2 / q_Phi^2 + a^2);
  P2 the same flattened, q_Phi = 0.85;
  P3 Miyamoto & Nagai (1975) disc, M = 2.7e9 Msun, a = 1.6, b = 0.3 kpc (Phi = -G M / sqrt(R^2 + (a + sqrt(z^2 + b^2))^2)), plus a cored
     logarithmic halo V = 80 km/s, a = 3 kpc.
Units: kpc, km/s, Msun (G from lmcold.constants).
"""
import numpy as np

from . import spheroid
from .constants import G

QGRID = np.linspace(0.1, 1.2, 45)  # axis ratios at which Omega_pot is evaluated (past 1: a vertically elongated tracer is reported, not clipped)
NREAL = 300000  # particles per realisation


def phi_log(x, V, a, qp=1.0):
    """Cored logarithmic potential [(km/s)^2] at x (N, 3) [kpc]: 1/2 V^2 ln(R^2 + z^2 / qp^2 + a^2).

    V [km/s]: asymptotic circular speed; a [kpc]: core radius; qp: flattening of the potential (1: spherical). Returns shape (N,).
    """
    return 0.5 * V ** 2 * np.log(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2 / qp ** 2 + a ** 2)


def grad_log(x, V, a, qp=1.0):
    """Gradient of phi_log [(km/s)^2 / kpc] at x (N, 3) [kpc]; same parameters; returns shape (N, 3)."""
    g = V ** 2 / (x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2 / qp ** 2 + a ** 2)
    return np.c_[g * x[:, 0], g * x[:, 1], g * x[:, 2] / qp ** 2]


def phi_mn(x, M, a, b):
    """Miyamoto-Nagai potential [(km/s)^2] at x (N, 3) [kpc] (Miyamoto & Nagai 1975, PASJ 27, 533):
    -G M / sqrt(R^2 + (a + sqrt(z^2 + b^2))^2), M [Msun], scale length a and scale height b [kpc]. Returns shape (N,)."""
    return -G * M / np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + (a + np.sqrt(x[:, 2] ** 2 + b ** 2)) ** 2)


def grad_mn(x, M, a, b):
    """Gradient of phi_mn [(km/s)^2 / kpc] at x (N, 3) [kpc]; same parameters; returns shape (N, 3)."""
    zb = np.sqrt(x[:, 2] ** 2 + b ** 2)
    f = G * M / (x[:, 0] ** 2 + x[:, 1] ** 2 + (a + zb) ** 2) ** 1.5
    return np.c_[f * x[:, 0], f * x[:, 1], f * x[:, 2] * (a + zb) / zb]


# Gradient functions x (N, 3) [kpc] -> grad Phi (N, 3) [(km/s)^2 / kpc] of the three potentials of the module docstring.
POTENTIALS = {'P1': lambda x: grad_log(x, 92.0, 2.0),
              'P2': lambda x: grad_log(x, 92.0, 2.0, 0.85),
              'P3': lambda x: grad_mn(x, 2.7e9, 1.6, 0.3) + grad_log(x, 80.0, 3.0)}


GRID_R = np.r_[0.0, np.geomspace(0.05, 60.0, 90)]  # kpc, R and |z| nodes of grid_potential
NSRC, SOFT = 200000, 0.1  # grid_potential: source particles (subsampled), softening [kpc]


def grid_potential(X, m, rng):
    """Gradient function (as POTENTIALS) of the axisymmetrised potential of particles X (N, 3) [kpc, z the symmetry axis] with masses m
    [Msun]: NSRC particles drawn with probability m (each carrying M / NSRC), each turned to a random azimuth (axisymmetrising the
    mass), and the softened gravity G Sum m d / (d^2 + SOFT^2)^(3/2) summed at the nodes (R, 0, |z|) of GRID_R x GRID_R; interpolated
    linearly in (R, |z|) (positions beyond the grid take the edge value)."""
    from scipy.interpolate import RegularGridInterpolator
    src = X[rng.choice(len(X), NSRC, p=m / m.sum())]  # mass-weighted subsample, each of mass M / NSRC
    Rs, ph = np.hypot(src[:, 0], src[:, 1]), rng.uniform(0, 2 * np.pi, NSRC)
    src = np.c_[Rs * np.cos(ph), Rs * np.sin(ph), src[:, 2]]
    R, Z = np.meshgrid(GRID_R, GRID_R, indexing='ij')
    pts = np.c_[R.ravel(), np.zeros(R.size), Z.ravel()]
    acc = np.empty((len(pts), 3))
    for k in range(0, len(pts), 50):  # 50 grid nodes at a time (memory); acc: gradient of Phi [(km/s)^2 / kpc] at each node
        d = pts[k:k + 50, None, :] - src[None]
        acc[k:k + 50] = G * m.sum() / NSRC * np.sum(d / ((d ** 2).sum(-1, keepdims=True) + SOFT ** 2) ** 1.5, axis=1)
    dR, dz = (RegularGridInterpolator((GRID_R, GRID_R), acc[:, c].reshape(R.shape), bounds_error=False, fill_value=None) for c in (0, 2))

    def grad(x):
        Rx = np.hypot(x[:, 0], x[:, 1])
        pt = np.c_[np.minimum(Rx, GRID_R[-1]), np.minimum(np.abs(x[:, 2]), GRID_R[-1])]
        gR, gz = dR(pt), dz(pt) * np.sign(x[:, 2])
        c, s = (np.divide(x[:, k], Rx, out=np.zeros(len(x)), where=Rx > 0) for k in (0, 1))
        return np.c_[gR * c, gR * s, gz]
    return grad


def balance(X, V, grad):
    """2K / |W| = Sum v^2 / Sum x . grad Phi of positions X and velocities V (N, 3): 1 for a tracer in equilibrium in the potential
    (the trace of the tensor virial theorem; the ratios Omega_pot, Omega_kin do not test the potential's depth).

    X [kpc], V [km/s] (N, 3); grad: gradient function as POTENTIALS. Returns a float."""
    return float(np.sum(V ** 2) / np.sum(X * grad(X)))


def omega_pot(X, grad):
    """(W_xx + W_yy) / (2 W_zz) of positions X (N, 3) [kpc] in the potential with gradient function grad (as POTENTIALS); a float.
    W_ij = -Sum x_i dPhi/dx_j (module docstring); the minus signs cancel in the ratio."""
    g = grad(X)
    return 0.5 * np.sum(X[:, 0] * g[:, 0] + X[:, 1] * g[:, 1]) / np.sum(X[:, 2] * g[:, 2])


def omega_kin(V):
    """(Sum v_x^2 + Sum v_y^2) / (2 Sum v_z^2) of disc-frame velocities V (N, 3) [km/s], z along the symmetry axis; a float."""
    return 0.5 * np.sum(V[:, 0] ** 2 + V[:, 1] ** 2) / np.sum(V[:, 2] ** 2)


def solve_q(target, h, grad, rng):
    """Axis ratio q on QGRID at which a K0 spheroid of scale length h [kpc] (NREAL particles) has Omega_pot = target in the potential
    grad; NaN if target lies outside the range of Omega_pot on QGRID.

    target : float, the Omega_kin to match; h [kpc]; grad : gradient function (POTENTIALS or grid_potential());
    rng : numpy.random.Generator. Returns a float. The spheroid of axis ratio q is the q = 1 realisation squashed in z by q."""
    X = spheroid.spheroid(h, 1.0, NREAL, rng)  # the whole tracer, not truncated
    om = np.array([omega_pot(X * [1, 1, q], grad) for q in QGRID])  # decreasing in q
    return float(np.interp(target, om[::-1], QGRID[::-1], left=np.nan, right=np.nan))


def flattening(p, h, grad, rng, r_floor=0.0):
    """Axis ratio q implied by the kinematic model p (kinfit vector) with scale length h [kpc] in the potential grad, and Omega_kin
    (r_floor: spheroid.velocities). q is NaN if Omega_kin lies outside the range of Omega_pot on QGRID (no tracer with QGRID[0] <= q
    <= QGRID[-1] has these kinematics); q > 1 (up to QGRID[-1]) means a tracer elongated along the symmetry axis, not oblate.

    Parameters
    ----------
    p : array_like, shape (33,)
        lmcold.kinfit parameter vector (rotation curve, ellipsoid sigma_R, sigma_phi, sigma_z [km/s], dlnsig_dR [1/kpc]).
    h : float
        Exponential scale length [kpc] of the population.
    grad : callable
        Gradient function of the potential (POTENTIALS or grid_potential()).
    rng : numpy.random.Generator
    r_floor : float
        Radius [kpc] inside which the dispersions are held constant (spheroid.velocities).

    Returns
    -------
    q : float
        Implied axis ratio (NaN outside QGRID).
    omega_kin : float
        Omega_kin of the model realisation.
    """
    X = spheroid.spheroid(h, 1.0, NREAL, rng)
    target = omega_kin(spheroid.velocities(p, X, rng, r_floor))
    return solve_q(target, h, grad, rng), float(target)
