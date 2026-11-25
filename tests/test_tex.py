from lmcold.tex import count, fout_percent, pm, pm_sym


def test_pm_symmetric_when_rounded_errors_agree():
    assert pm(57.1, 55.9, 58.3, 1) == r'$57.1\pm1.2$'


def test_pm_asymmetric_when_rounded_errors_differ():
    assert pm(23.9, 19.1, 27.9, 1) == r'$23.9^{+4.0}_{-4.8}$'


def test_pm_sym_and_count_and_fout():
    assert pm_sym(63.8, 0.84, 1) == r'$63.8\pm0.8$'
    assert count(2168) == r'2\,168' and count(846) == '846' and count(28909) == r'28\,909'
    assert abs(fout_percent(0.0) - 50) < 1e-12
