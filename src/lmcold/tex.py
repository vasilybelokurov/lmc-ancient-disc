"""LaTeX formatting of results for the paper tables (scripts/paper_table_*.py). Every number in a paper table is written by one of
those scripts through these functions; no table value is typed by hand.

Rule for a value with a 16th-84th percentile range: both half-ranges are rounded to the value's decimals; if they are equal the
cell is '$v\\pm e$', otherwise '$v^{+u}_{-l}$'.
"""
import numpy as np

# Paper row labels of the population names used in results/*.csv
GROUP_LABEL = {'giants [Fe/H] < -1.5': r'$\feh<-1.5$', 'giants -1.5 < [Fe/H] < -1.1': '$-1.5$ to $-1.1$',
               'giants -1.1 < [Fe/H] < -0.8': '$-1.1$ to $-0.8$', 'giants -0.8 < [Fe/H] < -0.5': '$-0.8$ to $-0.5$',
               'giants [Fe/H] > -0.5': r'$\feh>-0.5$', 'RR Lyrae': 'RR Lyrae'}


def pm(v, lo, hi, d):
    """'$v\\pm e$' or '$v^{+u}_{-l}$' with d decimals, from the value v and its 16th (lo) and 84th (hi) percentiles.

    Parameters
    ----------
    v, lo, hi : float
        Value (median) and the percentiles bracketing it.
    d : int
        Decimals of the value and of the errors.

    Returns
    -------
    str
        LaTeX math cell.
    """
    u, l = f'{hi - v:.{d}f}', f'{v - lo:.{d}f}'
    return f'${v:.{d}f}\\pm{u}$' if u == l else f'${v:.{d}f}^{{+{u}}}_{{-{l}}}$'


def pm_sym(v, e, d):
    """'$v\\pm e$' with d decimals, for a value with a symmetric error e (e.g. from an inverse Hessian)."""
    return f'${v:.{d}f}\\pm{e:.{d}f}$'


def count(n):
    """Integer with a thin space between groups of three digits for numbers >= 1000 (2\\,168), as in the paper."""
    s = f'{int(n):,}'
    return s.replace(',', r'\,')


def fout_percent(logit):
    """Outlier fraction in per cent from the fitted logit_fout."""
    return 100 / (1 + np.exp(-logit))
