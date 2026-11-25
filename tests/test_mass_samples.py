import numpy as np
import pandas as pd

from lmcold.mass_samples import MEASURED_EDGES, boss_response


def test_response_follows_the_target_snr():
    """Overlap stars at one G: true -0.5 is measured -0.5 at S/N 50 and -0.9 at S/N 18 (true -1.8: -1.8 and -1.6). A target
    observed at S/N 18 must get the low-S/N measurements only; weighting in G alone would mix the two."""
    o = pd.DataFrame(dict(feh_a=[-.5, -.5, -1.8, -1.8], feh_b=[-.5, -.9, -1.8, -1.6], g=[17.2] * 4, snr=[50., 18., 50., 18.]))
    o = pd.concat([o] * 20, ignore_index=True)  # enough stars per true bin that empirical_response does not borrow
    r = boss_response(o, np.array([-.7, -1.7]), np.array([17.2, 17.2]), np.array([18., 18.]), np.ones(2))
    measured = lambda x: np.digitize(x, MEASURED_EDGES) - 1  # noqa: E731
    assert r[measured(-.9), 3] == 1 and r[measured(-1.6), 1] == 1


def test_metal_rich_split_follows_apogee_in_the_same_cell_and_falls_back_when_sparse():
    """BOSS giants above [Fe/H] = -1 are shared among the reporting bins above -1 as the APOGEE giants above -1 of their cell;
    a cell with too few APOGEE giants uses the next coarser level."""
    from lmcold.mass_samples import NMIN_SPLIT, metal_rich_split
    n = NMIN_SPLIT
    a = pd.DataFrame(dict(feh=[-.6] * n + [-.4] * n + [-.8] * n, w=1., rho=1., g=[16.] * 2 * n + [17.3] * n,
                          c0=[1.3] * 2 * n + [1.5] * n))
    b = pd.DataFrame(dict(feh=[-.3, -.3, -1.5], rho=1., g=[16., 17.3, 17.3], c0=[1.3, 1.9, 1.5]))
    p = metal_rich_split(b, a)
    np.testing.assert_allclose(p[0], [0, .5, .5, 0, 0])   # its own cell: half -0.75..-0.5, half -0.5..-0.25
    np.testing.assert_allclose(p[1], [1, 0, 0, 0, 0])     # empty colour cell: falls back to ring x G (all -0.8)
    np.testing.assert_allclose(p[2], 0)                   # not above -1
