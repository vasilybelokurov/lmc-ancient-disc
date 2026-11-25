import numpy as np

from lmcold.foreground import radial_fraction


def test_uniform_foreground_fraction_recovered_from_reference_mean():
    """LMC exponential in radius plus a uniform foreground: knowing only the mean fraction over 2-10 deg recovers it at every
    radius."""
    rng = np.random.default_rng(1)
    n_lmc, sigma_mw = 200000, 40.0  # stars per deg^2 of foreground
    rho_lmc = rng.gamma(2, 1.6, n_lmc)
    rho_mw = 12 * np.sqrt(rng.random(int(sigma_mw * np.pi * 144)))
    rho = np.concatenate([rho_lmc, rho_mw])
    rho, is_mw = rho[rho < 12], np.concatenate([np.zeros(n_lmc, bool), np.ones(len(rho_mw), bool)])[rho < 12]
    norm = (rho >= 2) & (rho < 10)
    f_ref = is_mw[norm].mean()
    f = radial_fraction(rho, np.ones(len(rho)), np.full(len(rho), f_ref))
    for lo, hi in ((2, 3), (4, 5), (8, 9), (10, 11)):  # >= 600 foreground stars per ring
        s = (rho >= lo) & (rho < hi)
        assert abs(f[s].mean() / is_mw[s].mean() - 1) < 0.06
