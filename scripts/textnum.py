"""Helpers of the paper-text generators (scripts/paper_text_*.py): read settings from the code without running it, and write numbers
as they appear in the text."""
# %% [markdown]
# ### Reading settings from the code
# Module constants of a script and default arguments of a function, read without running anything, so that a number in the
# text follows the code that uses it.

# %%
import ast
import inspect

import numpy as np

from lmcold.sources import ROOT


def const(script):
    """Module-level constants of scripts/<script>.py whose values are literals (also tuple assignments), without running it."""
    # walk the top-level statements of the script's syntax tree; keep single-target assignments only
    out = {}
    for n in ast.parse((ROOT / 'scripts' / f'{script}.py').read_text()).body:
        # only plain 'NAME = value' or 'A, B = value' statements
        if not isinstance(n, ast.Assign) or len(n.targets) != 1:
            continue
        t = n.targets[0]
        # literal_eval accepts numbers, strings, tuples, lists, dicts; anything computed (calls, names) is skipped
        try:
            v = ast.literal_eval(n.value)
        except ValueError:
            continue
        # NAME = literal
        if isinstance(t, ast.Name):
            out[t.id] = v
        # A, B = (literal, literal): one entry per name
        elif isinstance(t, ast.Tuple) and isinstance(v, tuple) and len(v) == len(t.elts):
            out.update({e.id: x for e, x in zip(t.elts, v)})
    return out


def default(f, arg):
    """Default value of argument `arg` of function `f`."""
    # the default as written in the function's signature (inspect.Parameter.empty if it has none)
    return inspect.signature(f).parameters[arg].default


# %% [markdown]
# ### Writing numbers as in the text
# Integers without a decimal point, approximate values to one significant figure, lists joined with 'and', and powers of ten.

# %%
def num(x):
    """A number as it is written in the text: integers without a point, others as given (results are rounded where computed)."""
    # whole numbers (below 1e6) without a decimal point; anything else in the shortest general format
    x = float(x)
    return f'{x:.0f}' if x == round(x) and abs(x) < 1e6 else f'{x:g}'


def sigfig(x, n=1):
    """x rounded to n significant figures, as written in the text (approximate values, scatters, correlations)."""
    # zero has no logarithm; written as '0'
    x = float(x)
    if x == 0:
        # d: number of decimals that keeps n significant figures (negative d rounds to tens, hundreds, ...)
        return '0'
    d = n - 1 - int(np.floor(np.log10(abs(x))))
    # never print a negative number of decimals
    return f'{round(x, d):.{max(d, 0)}f}'


def listing(xs):
    """Strings xs joined as 'a, b and c'."""
    # all but the last joined with commas, the last with 'and'
    return ', '.join(xs[:-1]) + ' and ' + xs[-1]


def tex_sci(x):
    # decimal exponent e and mantissa x/10^e rounded to one decimal
    """x as m\\times10^e (two significant figures)."""
    e = int(np.floor(np.log10(x)))
    return rf'{num(round(x / 10 ** e, 1))}\times10^{{{e}}}'
