"""scripts/kin_vc_drift.scale_length: exponential scale length of a weighted surface density."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from kin_vc_drift import scale_length  # noqa: E402


def test_compact_population_with_empty_outer_annuli():
    """A compact exponential (h = 0.7 kpc) leaves the outer annuli of the fitted range empty; the fit uses the filled ones (a
    simulated metal-rich bin, TNG50 614626, gave NaN)."""
    rng = np.random.default_rng(1)
    R = rng.gamma(2, 0.7, 3000)  # face-on exponential: R ~ R exp(-R / h)
    R = R[R < 6.5]
    h = scale_length(R, np.ones(len(R)), np.zeros(len(R)))[0]
    assert abs(h - 0.7) < 0.1


def test_no_stars_in_the_fitted_range_gives_nan():
    """A population with no stars at 2-8 kpc has no measurable scale length (a simulated bin with none)."""
    R = np.full(100, 0.5)
    assert np.isnan(scale_length(R, np.ones(100), np.zeros(100))[0])
