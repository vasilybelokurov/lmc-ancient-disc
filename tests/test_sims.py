"""lmcold.sims: placement on the sky, spin frame and the metallicity mapping."""
import h5py
import numpy as np
import pytest

from lmcold import kinfit, sims


def test_place_reproduces_the_kinfit_mean_field():
    """Particles on the disc plane moving with kinfit's mean rotation (no streaming) are given back kinfit.model's proper motions and
    v_los at their sky positions: same orientation, sense of rotation, systemic motion and perspective."""
    rng = np.random.default_rng(1)
    p = kinfit.start()
    p[[kinfit.PARAMS.index(n) for n in ('dx', 'dy', 'inc', 'theta', 'v0', 'R0')]] = [0.4, -0.2, 30.0, 140.0, 70.0, 2.0]
    R, phi = rng.uniform(1, 8, 500), rng.uniform(0, 2 * np.pi, 500)
    X = np.c_[R * np.cos(phi), R * np.sin(phi), np.zeros(500)]
    vphi = np.asarray(kinfit.rotation(p, R))
    V = np.c_[-vphi * np.sin(phi), vphi * np.cos(phi), np.zeros(500)]
    s = sims.place(X, V, p)
    d = kinfit.data(s['ra'], s['dec'], s['pmra'], s['pmdec'], np.ones(500), np.ones(500), np.zeros(500))
    mra, mde, mlos, D = (np.asarray(v) for v in kinfit.model(p, d['rhat'], d['ea'], d['ed']))
    assert np.allclose(D, s['dist'], rtol=1e-6)
    assert np.allclose(mra, s['pmra'], atol=1e-6) and np.allclose(mde, s['pmdec'], atol=1e-6)
    assert np.allclose(mlos, s['vlos'], atol=1e-4)


@pytest.mark.parametrize('tilt', [60, 5])
def test_disc_frame_puts_the_spin_on_plus_z(tilt):
    """A rotating disc with its spin tilted from the simulation z axis (60 deg, and 5 deg, the near-z branch) and a bulk velocity: in
    the disc frame the spin is +z, v_phi > 0, bulk motion 0, and the transform is a rotation."""
    rng = np.random.default_rng(2)
    R, phi = rng.uniform(1, 8, 2000), rng.uniform(0, 2 * np.pi, 2000)
    X = np.c_[R * np.cos(phi), R * np.sin(phi), rng.normal(0, 0.1, 2000)]
    V = np.c_[-50 * np.sin(phi), 50 * np.cos(phi), np.zeros(2000)]
    a = np.deg2rad(tilt)
    rot = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Xd, Vd, _ = sims.disc_frame(X @ rot.T, V @ rot.T + [10, -20, 30], np.ones(2000))
    vphi = (Xd[:, 0] * Vd[:, 1] - Xd[:, 1] * Vd[:, 0]) / np.hypot(Xd[:, 0], Xd[:, 1])
    # the centre of velocity of a finite sample differs from the bulk motion by ~50 / sqrt(N) km/s
    assert abs(np.median(vphi) - 50) < 3 and np.abs(Xd[:, 2]).max() < 1
    assert np.allclose(np.linalg.norm(Xd, axis=1), np.linalg.norm(X, axis=1))


def test_lmc_scale_matches_cumulative_mass():
    """Mapped [Fe/H] reproduces the LMC cumulative formed-mass fraction at the knots, whatever the analogue's own scale."""
    rng = np.random.default_rng(3)
    feh, w = rng.normal(-1.0, 0.4, 100000), rng.uniform(0.5, 1.5, 100000)
    knots = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0]), np.array([0.0, 0.01, 0.05, 0.2, 0.8, 1.0])
    x = sims.lmc_scale(feh, w, *knots)
    for k, f in zip(*knots):
        assert abs(w[x < k].sum() / w.sum() - f) < 2e-3


def test_birth_phase_space_recovers_a_disc_born_in_a_moving_progenitor():
    """Stars born over a = 0.25-0.35 in a tilted disc (v_phi 40, sigma_z 10 km/s) whose centre moves through the box (comoving
    position linear in a, far out in the box) with a peculiar velocity of 300 km/s: progenitor_track + birth_phase_space give back
    v_phi and sigma_z about the disc's spin axis."""
    rng = np.random.default_rng(4)
    n = 6000
    a = np.sort(rng.uniform(0.25, 0.35, n))
    R, phi = rng.exponential(1.0, n), rng.uniform(0, 2 * np.pi, n)
    X = np.c_[R * np.cos(phi), R * np.sin(phi), rng.normal(0, 0.1, n)]
    V = np.c_[-40 * np.sin(phi), 40 * np.cos(phi), rng.normal(0, 10, n)]
    t = np.deg2rad(50)
    rot = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    centre = np.array([2.1e4, 3.3e4, 1.7e4]) + np.outer((a - 0.25) / 0.1, [400.0, -200.0, 100.0])  # comoving kpc
    vbulk = np.array([300.0, -150.0, 80.0])
    bx, bv = X @ rot.T / a[:, None] + centre, V @ rot.T + vbulk
    track = sims.progenitor_track(bx, a, bv)
    P, B, k = sims.birth_phase_space(bx, a, bv, np.ones(n), track)
    vphi = (P[k, 0] * B[k, 1] - P[k, 1] * B[k, 0]) / np.hypot(P[k, 0], P[k, 1])
    assert abs(np.mean(vphi) - 40) < 2 and abs(np.std(B[k, 2]) - 10) < 1


SUITES = [('Auriga', 'Au8', (sims.AURIGA['Au8'] / 'snapdir_251').exists()),
          ('TNG50', '708636', (sims.TNG_BIRTH_DIR / 'birth_708636.hdf5').exists())]


@pytest.mark.parametrize('suite, name, present', SUITES)
def test_birth_velocity_is_peculiar_velocity_over_a(suite, name, present):
    """The progenitor moves through the box: between a = 0.3 and 0.6 its centre's peculiar velocity a dX/dt (progenitor_track, from
    the comoving birth positions of the stars within 10 kpc today) equals the track's median birth_v (BirthVel / a) to 25 per cent in
    size and is parallel to it. The snapshot convention BirthVel x sqrt(a) would be 4-6 times too small at these a."""
    if not present:
        pytest.skip(f'{suite} birth data not on this machine')
    d = sims.birth(suite, name)
    s = ~d['exsitu'] & (np.linalg.norm(d['x'], axis=1) < 10)
    track = sims.progenitor_track(d['birth_x'][s], d['a_form'][s], d['birth_v'][s])
    h, om = (0.6777, 0.307) if suite == 'Auriga' else (sims.TNG_H, 0.3089)
    for lo, hi in ((0.3, 0.45), (0.45, 0.6)):
        (c0, c1), (v0, v1) = track(np.array([lo, hi]))
        vc = 0.5 * (lo + hi) * (c1 - c0) / (sims.lookback(lo, h, om) - sims.lookback(hi, h, om)) * 0.9778  # kpc/Gyr -> km/s
        vb = 0.5 * (v0 + v1)
        assert 0.8 < np.linalg.norm(vc) / np.linalg.norm(vb) < 1.25 and vc @ vb / np.linalg.norm(vc) / np.linalg.norm(vb) > 0.95


def test_coherence_limits():
    """C = 1 for one disc rotating in one sense, ~0 for two equal counter-rotating discs, unchanged by a rotation of the frame, and
    ~ N^-1/2 (within a factor 1.5) for N randomly oriented orbits."""
    rng = np.random.default_rng(5)
    n = 4000
    R, phi = rng.uniform(1, 5, n), rng.uniform(0, 2 * np.pi, n)
    X = np.c_[R * np.cos(phi), R * np.sin(phi), np.zeros(n)]
    V = np.c_[-np.sin(phi), np.cos(phi), np.zeros(n)] * 50
    m = np.ones(n)
    assert abs(sims.coherence(X, V, m) - 1) < 1e-12
    s = np.where(np.arange(n) % 2, 1, -1)[:, None]
    assert sims.coherence(X, V * s, m) < 0.05
    t = np.deg2rad(37)
    rot = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    assert abs(sims.coherence(X @ rot.T, V * s @ rot.T, m) - sims.coherence(X, V * s, m)) < 1e-12
    x, v = rng.normal(size=(200, 3)), rng.normal(size=(200, 3))
    assert 1 / 1.5 < sims.coherence(x, v, np.ones(200)) * np.sqrt(200) < 1.5


@pytest.mark.skipif(not (sims.TNG_TRACK_DIR / '708636' / 'snap_099.hdf5').exists(), reason='TNG50 progenitor cutouts not on this machine')
def test_progenitor_at_z0_is_the_analogue():
    """At snapshot 99 the main progenitor is the analogue itself: progenitor('track') gives the particles of the z = 0 birth cutout,
    in the same order, at the positions of sims._read up to one constant offset (same units and periodic wrapping; the centres differ
    only by the rounding of the catalogue position in sample.json to 0.1 ckpc/h, < 0.15 kpc)."""
    p = sims.progenitor('708636', 99, 'track')['stars']
    with h5py.File(sims.TNG_BIRTH_DIR / 'birth_708636.hdf5') as f:
        g = f['PartType4']
        ids = g['ParticleIDs'][:][g['GFM_StellarFormationTime'][:] > 0]
    assert np.array_equal(p['ids'], ids)
    d = p['x'] - sims._read('TNG50', '708636')['x']
    assert d.std(axis=0).max() < 1e-3 and np.linalg.norm(d.mean(axis=0)) < 0.15


def test_progenitor_track_does_not_depend_on_how_ties_are_ordered(monkeypatch):
    """Star particles share formation times (TNG50, Auriga). The track must not depend on how a sort orders tied times: a platform
    whose default sort reverses ties (valid behaviour) must give the same track."""
    rng = np.random.default_rng(5)
    n = 3000
    a = np.round(rng.uniform(0.2, 0.5, n), 2)                    # heavy ties, as the stored formation times
    x = rng.normal(size=(n, 3)) + 20 * a[:, None]               # a moving progenitor
    v = rng.normal(size=(n, 3)) * 30
    ref = sims.progenitor_track(x, a, v)(np.linspace(0.25, 0.45, 9))
    argsort = np.argsort

    def reversed_ties(arr, *args, kind=None, **kw):             # default kind: ties in reverse order; explicit kinds unchanged
        if kind is not None or np.ndim(arr) != 1:
            return argsort(arr, *args, kind=kind, **kw)
        return len(arr) - 1 - argsort(arr[::-1], kind='stable')   # ascending values, tied elements in reverse input order

    monkeypatch.setattr(np, 'argsort', reversed_ties)
    got = sims.progenitor_track(x, a, v)(np.linspace(0.25, 0.45, 9))
    for r, g in zip(ref, got):
        np.testing.assert_allclose(g, r, atol=1e-9)
