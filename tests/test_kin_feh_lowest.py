"""The mock generator of scripts/kin_feh_lowest.py draws from the ellipsoid model that lmcold.kinfit fits: a fit of OWN to a large
mock recovers the generating v_rot(4 kpc) / sigma_z and sigma_z."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import kin_feh_lowest as kf
from lmcold import kinfit
from lmcold.constants import LMC


def test_mock_is_recovered_by_the_fit():
    rng = np.random.default_rng(3)
    n = 20000
    r = np.deg2rad(rng.uniform(2, 9, n))
    a = rng.uniform(0, 2 * np.pi, n)
    dec = LMC['dec0'] + np.rad2deg(r * np.cos(a))
    ra = LMC['ra0'] + np.rad2deg(r * np.sin(a)) / np.cos(np.deg2rad(dec))
    cat = dict(ra=ra, dec=dec, pmra=np.zeros(n), pmdec=np.zeros(n), pmra_error=np.full(n, 0.03), pmdec_error=np.full(n, 0.03),
               pmra_pmdec_corr=np.zeros(n), vlos=np.where(rng.random(n) < 0.8, 0.0, np.nan), evlos=np.full(n, 1.0))
    q = kinfit.start()
    for name, v in dict(v0=60.0, R0=3.0, sigma_R=38.0, sigma_phi=32.0, sigma_z=30.0, dlnsig_dR=-0.05, k=1.0, logit_fout=-10.0).items():
        q[kinfit.PARAMS.index(name)] = v
    m = kf.mock(cat, kf.with_ratio(q, 1.0), rng)
    p0 = q.copy()
    for name, v in dict(v0=40.0, sigma_R=30.0, sigma_phi=30.0, sigma_z=40.0, logit_fout=-3.0).items():
        p0[kinfit.PARAMS.index(name)] = v
    p = kf.ml(m, p0)
    assert abs(kf.ratio_of(p) - 1.0) < 0.04
    assert abs(p[kinfit.PARAMS.index('sigma_z')] - 30.0) < 1.0
