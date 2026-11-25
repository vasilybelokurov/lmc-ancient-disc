import numpy as np

from lmcold import giants


def test_huber_fit_ignores_outliers():
    rng = np.random.default_rng(8)
    x = rng.uniform(-2, 0, 5000)
    y = 0.3 + 0.7 * x + rng.normal(0, 0.05, x.size)
    y[:250] += rng.choice([-2, 2], 250)  # 5% gross outliers
    beta, sd = giants.huber_fit(np.column_stack([np.ones_like(x), x]), y)
    np.testing.assert_allclose(beta, [0.3, 0.7], atol=0.01)
    assert abs(sd - 0.05) < 0.01


def test_calibration_round_trip():
    """BOSS values generated from APOGEE with a known relation: the fit recovers it and calibrating returns the APOGEE scale."""
    rng = np.random.default_rng(9)
    n = 3000
    feh_a, teff, snr = rng.uniform(-2.2, 0.0, n), rng.uniform(3900, 5100, n), rng.uniform(10, 80, n)
    v_a = rng.normal(260, 40, n)
    t, s = giants.regressors(teff, snr)
    truth = dict(d0=3.9, d1=-4.9, c0=-0.06, c1=0.73, c2=-0.03, c3=0.08)
    feh_b = truth["c0"] + truth["c1"] * feh_a + truth["c2"] * t + truth["c3"] * s + rng.normal(0, 0.03, n)
    v_b = v_a + truth["d0"] + truth["d1"] * t + rng.normal(0, 1.0, n)
    cal = giants.fit_calibration(teff, snr, v_b, v_a, feh_b, feh_a)
    for k, v in truth.items():
        assert abs(cal[k] - v) < 0.02 + 0.02 * abs(v), (k, cal[k], v)
    v_c, feh_c = giants.calibrate(cal, teff, snr, v_b, feh_b)
    assert abs(np.median(feh_c - feh_a)) < 0.005 and abs(np.median(v_c - v_a)) < 0.05


def test_quality_bits():
    assert giants.bit(np.array([16, 0, 1 << 23]), giants.EXTRATARG_DUPLICATE).tolist() == [True, False, False]
    assert giants.bit(np.array([1 << 23]), giants.ASPCAP_STAR_BAD).tolist() == [True]


def test_quality_masks_on_the_common_columns():
    """Masks read the common column names of data/spec_giants.parquet (v, ev, feh, teff, logg, snr + survey flags)."""
    import pandas as pd
    a = pd.DataFrame(dict(extratarg=[0, 16, 0], aspcapflag=[0, 0, 1 << 23], feh=[-1.0, -1.0, -1.0], snr=[80, 80, 80],
                          teff=[4500, 4500, 4500], logg=[2.0, 2.0, 2.0], ev=[0.1, 0.1, 0.1], vscatter=[0.2, 0.2, 0.2]))
    assert giants.apogee_ok(a).tolist() == [True, False, False]
    b = pd.DataFrame(dict(flag_bad=[0, 1, 0], feh=[-1.0, -1.0, np.nan], v=[260.0, 260.0, 260.0], ev=[2.0, 2.0, 2.0],
                          snr=[20, 20, 20], teff=[4500, 4500, 4500], logg=[2.0, 2.0, 2.0]))
    assert giants.boss_ok(b).tolist() == [True, False, False]


def test_boss_spectrum_quality_is_separate_from_the_giant_cuts():
    """A dwarf with a good spectrum passes the spectroscopic quality mask but not the giant cuts; boss_ok needs both."""
    import pandas as pd
    b = pd.DataFrame(dict(flag_bad=[0, 0], feh=[-1.0, -1.0], v=[260.0, 260.0], ev=[2.0, 2.0], snr=[20, 20], teff=[4500, 6000],
                          logg=[2.0, 4.5]))
    assert giants.boss_spectrum_ok(b).tolist() == [True, True]
    assert giants.is_giant(b).tolist() == [True, False]
    assert giants.boss_ok(b).tolist() == [True, False]


def test_apogee_spectrum_quality_is_separate_from_the_giant_and_velocity_cuts():
    """A low-S/N or velocity-variable giant can pass the spectroscopic mask at a lower S/N threshold; apogee_ok needs all cuts."""
    import pandas as pd
    a = pd.DataFrame(dict(extratarg=[0, 0, 0, 0], aspcapflag=[0, 0, 0, 0], feh=[-1.0, -2.2, -1.0, -1.0], snr=[80, 40, 80, 80],
                          teff=[4500, 4500, 4500, 6000], logg=[2.0, 1.0, 2.0, 4.5], ev=[0.1, 0.1, 0.1, 0.1],
                          vscatter=[0.2, 0.2, 3.0, 0.2]))
    assert giants.apogee_spectrum_ok(a, snr_min=30).tolist() == [True, True, True, True]
    assert giants.apogee_spectrum_ok(a).tolist() == [True, False, True, True]
    assert giants.rv_ok(a).tolist() == [True, True, False, True]
    assert giants.apogee_ok(a).tolist() == [True, False, False, False]


def test_boss_spectrum_needs_snr_16():
    """Below S/N 16 BOSSNet [Fe/H] barely tracks APOGEE (slope 0.16-0.51 against 0.79 at S/N > 40;
    scripts/fig_boss_apogee_offset.py): such spectra are failures, restored by the pass factor."""
    import pandas as pd
    b = pd.DataFrame(dict(flag_bad=[0, 0], feh=[-1.0, -1.0], v=[260.0, 260.0], ev=[2.0, 2.0], snr=[15.9, 16.0]))
    assert giants.boss_spectrum_ok(b).tolist() == [False, True]
