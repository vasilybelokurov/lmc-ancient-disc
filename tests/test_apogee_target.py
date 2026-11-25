import numpy as np

from lmcold import apogee_target as at
from lmcold import magcloud


def test_polygon_vertices():
    """The BrtRGB / FntRGB polygon of select_mc_targets.pro: (0.5632, 12.35), (1.2834, 12.35), (0.55455, 12.8), (1.2488, 12.8),
    and the FntRGB lower vertices (0.5219, maglim), (1.1182, maglim)."""
    blue, red = at.sides(np.array([12.35, 12.8, 14.5]), 14.5)
    np.testing.assert_allclose(blue, [0.5632, 0.55455, 0.5219])
    np.testing.assert_allclose(red, [1.2834, 1.2488, 1.1182])


def test_rgb_class():
    jk, h = np.array([0.9, 0.9, 0.9, 0.3, 0.9]), np.array([12.5, 13.5, 15.0, 13.5, 12.0])
    assert at.rgb_class(jk, h, 14.5).tolist() == [1, 2, 0, 0, 0]


def test_bright_neighbour_flags_only_the_fainter_star():
    ra, dec = np.array([80.0, 80.0, 80.0]), np.array([-69.0, -69.0 + 3 / 3600, -69.0 + 60 / 3600])
    assert at.bright_neighbour(ra, dec, np.array([10.0, 13.0, 13.0])).tolist() == [False, True, False]


def test_eligible_stars_pass_the_astrometric_cuts():
    """astrometry_ok is the astrometric part of the carton; a star failing only in colour still passes it."""
    base = dict(ra=79.88, dec=-69.59, pmra=1.895, pmdec=0.287, parallax=0.02, parallax_error=0.05)
    a = lambda **kw: {k: np.array([kw.get(k, v)]) for k, v in base.items()}  # noqa: E731
    assert magcloud.astrometry_ok(**a())[0]
    assert magcloud.eligible(**a(), g=np.array([16.5]), bp=np.array([17.3]), rp=np.array([15.6]), ebv=np.array([0.05]))[0]
    for bad in (dict(pmra=0.0), dict(pmdec=3.5), dict(parallax=1.0), dict(pmra=np.nan)):
        assert not magcloud.astrometry_ok(**a(**bad))[0], bad


def test_gaia_join_keeps_exact_ids_when_an_object_is_unmatched():
    import pandas as pd
    from lmcold.apogee_target import join_gaia
    source = 4621247719436568960  # not exactly representable in float64
    objects = pd.DataFrame({'apogee_id': ['2Mmatched', '2Mmissing']})
    gaia = pd.DataFrame({'tmass': ['matched'], 'gsep': [.1], 'source_id': [source]})
    joined = join_gaia(objects, gaia)
    assert int(joined.source_id.iloc[0]) == source
    assert pd.isna(joined.source_id.iloc[1])


def test_weights_count_dwarfs_as_good_spectra_not_failures():
    """One BrtRGB group of 100 candidates, all observed with good ASPCAP spectra: 80 giants and 20 dwarfs. The dwarfs are
    non-members removed after weighting, not failed spectra, so the giants must carry a total weight of 80 (counting them
    as failures gave 100)."""
    import pandas as pd
    c = pd.DataFrame(dict(group="g", cls=1, observed=True, giant=np.arange(100) < 80, snr=100.0, extratarg=0, aspcapflag=0,
                          feh=-0.5))
    q, covered, w = at.weights(c, 30, np.zeros(100, int))
    assert covered.all()
    np.testing.assert_allclose(w[c.giant.to_numpy()].sum(), 80)
