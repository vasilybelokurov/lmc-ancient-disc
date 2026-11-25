import numpy as np
import pandas as pd

from lmcold import magcloud, selection
from lmcold.groups import bin_index


def survey(n=400_000, seed=5):
    """Selection groups with very different observed fractions; within a group observation is random; the pass probability
    halves for metal-poor stars at faint G (as in BOSS)."""
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 50, n)
    obs = rng.random(n) < (0.05 + 0.9 * (s / 49))
    mh, g = rng.uniform(-2.5, 0, n), rng.uniform(15, 17.5, n)
    good = obs & (rng.random(n) < 0.9 * np.where((mh < -1.5) & (g > 16.75), 0.5, 1.0))
    pg = selection.pass_groups(g, mh)
    return s, obs, good, mh, g, pg


def test_stratum_weights_sum_to_the_parent():
    s, obs, good, mh, g, pg = survey()
    w = selection.group_weights(s, good, np.ones(len(s)))
    np.testing.assert_allclose(np.bincount(s, w), np.bincount(s), rtol=1e-12)


def test_final_weights_recover_the_metal_poor_count():
    """Group weights alone under-count the faint metal-poor stars; with the pass factor the weighted count below -1.5 matches the
    parent to within its Poisson-like noise, and the total is kept."""
    s, obs, good, mh, g, pg = survey()
    ones = np.ones(len(s))
    w0 = selection.group_weights(s, good, ones)
    w = w0 * selection.pass_factor(s, obs, good, pg, ones)
    true = np.sum(mh < -1.5)
    assert np.sum(w0[mh < -1.5]) / true < 0.92
    assert abs(np.sum(w[mh < -1.5]) / true - 1) < 0.02
    assert abs(w.sum() / len(s) - 1) < 0.01


def test_pass_factor_is_one_when_failures_ignore_metallicity():
    rng = np.random.default_rng(2)
    s, obs, _, mh, g, pg = survey()
    good = obs & (rng.random(len(s)) < 0.8)
    f = selection.pass_factor(s, obs, good, pg, np.ones(len(s)))
    assert abs(np.median(f[good]) - 1) < 0.01 and np.ptp(np.percentile(f[good], [1, 99])) < 0.1


def test_cells_put_missing_xp_in_its_own_bin():
    c = selection.pass_groups(np.array([15.0, 15.0, 17.4]), np.array([-1.7, np.nan, np.nan]))
    assert c[0] != c[1] and c[1] != c[2]


def test_bin_index_keeps_both_edges():
    assert bin_index(np.array([12.0, 15.0, 17.5, 18.0]), [12.0, 16.0, 17.5]).tolist() == [0, 0, 1, 1]


def test_carton_class_any_other_boss_carton():
    """Standards and A-star cartons count as 'other'; APOGEE-only cartons do not change the class."""
    c = pd.Series(["mwm_magcloud_rgb_boss", "mwm_magcloud_rgb_boss;ops_std_boss_gdr2", "mwm_magcloud_agb_apogee;mwm_magcloud_rgb_boss",
                   "manual_mwm_halo_distant_kgiant_far_boss;mwm_magcloud_rgb_boss", "mwm_halo_nmp_xp_boss;mwm_magcloud_rgb_boss",
                   "manual_mwm_halo_vmp_wise_boss;mwm_magcloud_rgb_boss", "mwm_magcloud_rgb_boss;openfibertargets_nov2020_6b"])
    assert magcloud.carton_class(c).tolist() == ["magcloud", "other", "magcloud", "kgiant", "xp", "xp", "other"]


def test_pass_groups_take_survey_magnitude_edges():
    """APOGEE pass groups bin H with its own edges; the same XP bin in different H bins is a different pass group."""
    h = np.array([12.5, 13.5, 14.5])
    c = selection.pass_groups(h, np.full(3, -1.2), edges=[12.35, 13.3, 14.3, 15.3])
    assert len(set(c.tolist())) == 3


def test_boss_pass_groups_split_by_ring():
    """Same G and XP [M/H]: the LMC centre, 4 deg and 8 deg south fall in three different BOSS pass groups."""
    pg = selection.boss_pass_groups(np.full(3, 17.2), np.full(3, -1.6), np.full(3, 79.88), np.array([-69.59, -73.59, -77.59]))
    assert len(set(pg.tolist())) == 3
    assert len(set((pg // 3).tolist())) == 1


def test_coverage_factor_fills_each_xp_bin_from_its_own_covered_stars():
    """One region: 60 metal-rich stars (40 covered) and 40 metal-poor (10 covered). Filled by XP bin, the covered metal-rich
    stars stand for 60 and the metal-poor for 40; a cell with too few covered stars falls back to the next, coarser key."""
    rich = np.arange(100) < 60
    covered = np.r_[np.arange(60) < 40, np.arange(40) < 10]
    f = selection.coverage_factor([np.where(rich, "r", "p"), np.zeros(100)], covered, 5, np.ones(100))
    np.testing.assert_allclose(f[covered & rich], 60 / 40)
    np.testing.assert_allclose(f[covered & ~rich], 40 / 10)
    f = selection.coverage_factor([np.where(rich, "r", "p"), np.zeros(100)], covered, 20, np.ones(100))
    np.testing.assert_allclose(f[covered & rich], 60 / 40)
    np.testing.assert_allclose(f[covered & ~rich], 40 / 10)  # the metal-poor cell (10 covered) pools at the last level alone


def test_coverage_factor_counts_multiplicities():
    """Doubling every star leaves the factor unchanged; dropping a gap star (m = 0) lowers it."""
    covered = np.arange(10) < 4
    keys = [np.zeros(10)]
    np.testing.assert_allclose(selection.coverage_factor(keys, covered, 1, np.full(10, 2.0))[covered], 10 / 4)
    m = np.ones(10)
    m[9] = 0
    np.testing.assert_allclose(selection.coverage_factor(keys, covered, 1, m)[covered], 9 / 4)
