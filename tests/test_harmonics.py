import numpy as np

from lmcold import harmonics


def test_recovers_dipole_and_quadrupole():
    rng = np.random.default_rng(6)
    pa = rng.uniform(0, 360, 20000)
    v = -50 + 15 * np.cos(np.deg2rad(pa - 240)) + 10 * np.cos(2 * np.deg2rad(pa - 140)) + rng.normal(0, 5, pa.size)
    c, e = harmonics.fit(pa, v)
    assert abs(c[0] + 50) < 0.2
    amp1, ph1 = harmonics.amplitude_phase(c, 1)
    amp2, ph2 = harmonics.amplitude_phase(c, 2)
    assert abs(amp1 - 15) < 0.3 and abs(ph1 - 240) < 1.5
    assert abs(amp2 - 10) < 0.3 and abs(ph2 - 140) < 1.5
    assert 0 < e[1] < 0.2
