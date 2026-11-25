"""Paper table (Section 5, tab:shape): the axis ratio q = c/a implied by the kinematics of each giant [Fe/H] bin in the potentials
P1-P3 (results/shape_virial.csv, no central floor): q of the posterior-median model with the 16th-84th percentiles of the joint draws
as a range, and 2K/|W| of the median model at that q.

Writes results/table_shape.tex (the tabular; copied to the paper's tables/table_shape.tex).

Run: .venv/bin/python scripts/paper_table_shape.py   (after scripts/shape_virial.py)
"""
import pandas as pd

from lmcold.sources import ROOT

POTS = ('P1', 'P2', 'P3')   # the three potentials of scripts/shape_virial.py, one column each

if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data
    # The axis ratios of scripts/shape_virial.py; only the giant bins without a central floor (r_floor = 0) enter the table.

    # %%
    # One row per (population, potential, r_floor) with q, its percentiles q_p16, q_p84 and balance = 2K/|W|
    v = pd.read_csv(ROOT / 'results/shape_virial.csv')
    # Keep the giant [Fe/H] bins with no central floor
    v = v[(v.r_floor == 0) & v.population.str.startswith('giants')]

    # %% [markdown]
    # ### Write the table
    # One row per [Fe/H] bin; each cell is q [p16--p84]; 2K/|W|.

    # %%
    # Table header: [Fe/H] bin and one column per potential
    rows = [r'\begin{tabular}{lccc}', r'\hline', r'\feh & P1 & P2 & P3\\', r'\hline']
    for pop, g in v.groupby('population', sort=False):
        # Rows of one bin ordered P1, P2, P3; label: the bin's [Fe/H] range in LaTeX
        g = g.set_index('potential').loc[list(POTS)]
        label = '$' + pop.removeprefix('giants ').replace('[Fe/H]', r'\feh') + '$'
        # Cell: q [p16--p84]; 2K/|W|, all to two decimals
        cells = [f'{r.q:.2f} [{r.q_p16:.2f}--{r.q_p84:.2f}]; {r.balance:.2f}' for _, r in g.iterrows()]
        rows.append(label + ' & ' + ' & '.join(cells) + r'\\')
    # Close the tabular, write results/table_shape.tex and print it
    rows += [r'\hline', r'\end{tabular}']
    (ROOT / 'results/table_shape.tex').write_text('\n'.join(rows) + '\n')
    print('\n'.join(rows))
