import numpy as np
import pandas as pd

from lmcold.groups import assign


def test_weights_sum_to_the_parent_count():
    """Coarse groups hold only the stars left after the finer levels, so sum(N / N_sel) over selected stars = parent count."""
    rng = np.random.default_rng(1)
    n = 5000
    coarse = rng.integers(0, 10, n)
    fine = np.where(rng.random(n) < 0.5, 0, rng.integers(1, 400, n))  # one dense fine group per coarse one, the rest sparse
    sel = rng.random(n) < np.where(fine == 0, 0.1, 0.5)
    s = pd.Series(assign([fine * 10 + coarse, coarse, np.zeros(n)], nmin=15))
    n_p, n_s = s.map(s.value_counts()), s.map(pd.Series(sel).groupby(s).sum())
    ok = (n_s > 0).to_numpy()
    np.testing.assert_allclose(np.sum((n_p / n_s)[sel & ok]), np.sum(ok))


def test_finest_level_with_enough_stars():
    s = assign([np.array([0, 0, 0, 1]), np.zeros(4)], nmin=3)
    assert s[0] == s[1] == s[2] != s[3]
