import numpy as np

from lmcold import magcloud


def test_ms_coordinates_lmc_and_its_motion():
    """The carton's LMC centre (L, B) = (-0.6, 3.6) is 1.04 deg from the vdMK14 centre in the Nidever et al. frame; the LMC systemic
    PM is 0.073 mas/yr from the carton's reference (1.8, 0.40), well inside its 1.2 mas/yr window."""
    L, B, pmL, pmB = magcloud.ms_coordinates(np.array([79.88]), np.array([-69.59]), np.array([1.895]), np.array([0.287]))
    assert abs(np.hypot(L[0] + 0.6, B[0] - 3.6) - 1.04) < 0.01
    assert abs(np.hypot(pmL[0] - 1.8, pmB[0] - 0.40) - 0.073) < 0.002


def test_reddening_rescaled_only_in_the_inner_clouds_at_high_ebv():
    g0, c0 = magcloud.dereddened(np.full(3, 17.0), np.full(3, 17.6), np.full(3, 16.4), np.array([0.2, 0.2, 0.05]),
                                 np.array([1.0, 6.0, 1.0]), np.full(3, 30.0))
    av = 0.86 * 3.1 * np.array([0.2 * 0.07, 0.2, 0.05])
    np.testing.assert_allclose(g0, 17.0 - magcloud.A_OVER_AV["G"] * av)


def test_eligible_star_and_each_cut():
    """An LMC red giant passes; changing one property at a time fails it."""
    base = dict(ra=79.88, dec=-69.59, pmra=1.895, pmdec=0.287, parallax=0.02, parallax_error=0.05, g=16.5, bp=17.3, rp=15.6, ebv=0.05)
    a = lambda **kw: {k: np.array([kw.get(k, v)]) for k, v in base.items()}  # noqa: E731
    assert magcloud.eligible(**a())[0]
    for bad in (dict(g=17.6), dict(pmra=0.0), dict(parallax=1.0, parallax_error=0.05), dict(bp=16.5, rp=15.6), dict(g=12.0)):
        assert not magcloud.eligible(**a(**bad))[0], bad


def test_carton_ebv_rescales_at_the_lmc_centre_only():
    """At the LMC centre E(B-V) = 0.2 becomes 0.014; 20 deg away, and below 0.1 at the centre, SFD is kept."""
    e = magcloud.carton_ebv(np.array([79.88, 79.88, 30.0]), np.array([-69.59, -69.59, -60.0]), np.array([0.2, 0.05, 0.2]))
    np.testing.assert_allclose(e, [0.014, 0.05, 0.2])
