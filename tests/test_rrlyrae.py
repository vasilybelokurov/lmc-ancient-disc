import numpy as np

from lmcold import rrlyrae


def test_feh_invariant_under_2pi_shifts():
    P, phi = np.array([0.5, 0.6, 0.7]), np.array([2.0, 2.5, 3.1])
    for f in (rrlyrae.feh_ogle_I, rrlyrae.feh_gaia_G_m25, rrlyrae.feh_gaia_G_p26):
        np.testing.assert_allclose(f(P, phi + 2 * np.pi), f(P, phi))
        np.testing.assert_allclose(f(P, phi - 2 * np.pi), f(P, phi))


def test_feh_ogle_I_pivot():
    """At P = 0.58 d and phi31_V = 5.25 the Mullen et al. relation gives its zero point, -1.22."""
    phi31_I = (5.25 - np.pi + 0.039) / 0.788
    np.testing.assert_allclose(rrlyrae.feh_ogle_I(0.58, phi31_I), -1.22)


def test_feh_gaia_G_values():
    """Both G-band relations by hand at P = 0.6 d."""
    np.testing.assert_allclose(rrlyrae.feh_gaia_G_p26(0.6, 5.25 - np.pi), -6.93 - 6.04 * 0.6 + 1.65 * 5.25)
    np.testing.assert_allclose(rrlyrae.feh_gaia_G_m25(0.6, 2.0), -5.55 * 0.6 + 0.94 * 2.0 - 0.37)


def test_zw_to_c21_matches_muraveva25():
    """Muraveva et al. (2025) convert the Skowron et al. (2016) LMC median, -1.59 on the ZW scale, to -1.52 on the C21 scale."""
    np.testing.assert_allclose(rrlyrae.zw_to_c21(-1.59), -1.52, atol=0.005)


def test_relative_modulus_recovers_relation():
    rng = np.random.default_rng(3)
    n = 4000
    P, feh = rng.uniform(0.45, 0.75, n), rng.normal(-1.4, 0.3, n)
    W = 17.5 - 2.9 * np.log10(P) + 0.05 * feh + rng.normal(0, 0.1, n)
    W[:40] -= 3.0  # gross outliers (blends) must be clipped
    dmu, coef, fit = rrlyrae.relative_modulus(W, P, feh, np.ones(n, bool))
    for got, true, tol in zip(coef, (17.5, -2.9, 0.05), (0.05, 0.08, 0.02)):
        assert abs(got - true) < tol, (got, true)
    assert not fit[:40].any() and abs(np.median(dmu[40:])) < 0.01


def test_read_ogle_rrab(tmp_path):
    (tmp_path / "ident.dat").write_text(
        "OGLE-LMC-RRLYR-00001  RRab  04:27:45.45 -70:43:12.0  LMC545.10.329    LMC158.8.216\n"
        "OGLE-LMC-RRLYR-00009  RRd   04:30:00.00 -70:00:00.0  LMC545.10.330\n")
    (tmp_path / "RRab.dat").write_text(
        "OGLE-LMC-RRLYR-00001  18.772    -    0.6347521 0.0000010  6000.13200  0.533  0.545 4.395  0.351 2.747\n")
    t = rrlyrae.read_ogle_rrab(tmp_path)
    assert len(t) == 1 and np.isnan(t.V[0])
    np.testing.assert_allclose([t.ra[0], t.dec[0], t.P[0], t.phi31[0]],
                               [(4 + 27 / 60 + 45.45 / 3600) * 15, -(70 + 43 / 60 + 12 / 3600), 0.6347521, 2.747])


def test_ogle_own_discovery():
    assert rrlyrae.ogle_own_discovery(["OGLE-LMC-RRLYR-00001", "OGLE-LMC-RRLYR-39959", "OGLE-LMC-RRLYR-39960"]).tolist() == [True, True, False]
