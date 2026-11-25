"""Paper figure (Section 5): the flattening each giant [Fe/H] bin's kinematics imply, q = c/a of an oblate K0 tracer from the tensor
virial theorem (scripts/shape_virial.py, results/shape_virial.csv, no central floor), against the bin's median [Fe/H]
(lmcold.samples.giants), in the three potentials of lmcold.virial (posterior-median vector: markers; 16th-84th percentiles of the joint
posterior draws: bars). Points of P1 and P3 are offset in [Fe/H] by -DX and +DX for legibility. The grey band is the RR Lyrae (same old
population; median [Fe/H] from results/kin_support_ellipsoid.csv): q from their near-to-far distance gradient at the metal-poor giants' inclination (results/
shape_rrl_depth.csv: bootstrap 16-84 per cent, darker; and the range from a +-0.02 mag coherent modulus gradient, lighter).

Writes figures/paper_shape_virial.png (KB chapter 'The shape of the old population', Figure paper_shape_virial).

Run: .venv/bin/python scripts/paper_shape_virial.py   (after scripts/shape_virial.py and scripts/shape_rrl_cells.py)
"""
# %% [markdown]
# ### Settings
# Plotting choices: the offset of the P1 and P3 points in [Fe/H], and the marker, colour and legend label of each potential.

# %%
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmcold import samples
from lmcold.layout import panel_grid
from lmcold.sources import ROOT

DX = 0.025  # dex, horizontal offset of the points: (k - 1) DX for the k-th potential, so P1 at -DX, P2 at 0, P3 at +DX
STYLE = {'P1': ('o', 'k'), 'P2': ('s', '#2166ac'), 'P3': ('D', '#b2182b')}  # potential: (marker, colour)
# Legend label of each potential of lmcold.virial.
LABELS = {'P1': 'P1: cored logarithmic halo', 'P2': r'P2: P1 flattened, $q_\Phi=0.85$', 'P3': 'P3: disc + halo'}

if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data
    # The axis ratios implied by the motions from `results/shape_virial.csv` (`scripts/shape_virial.py`), the RR Lyrae flattening
    # from their depth from `results/shape_rrl_depth.csv` (`scripts/shape_rrl_cells.py`), the median [Fe/H] of the RR Lyrae from
    # `results/kin_support_ellipsoid.csv`, and the giant catalogues of Zenodo bundle 1 through `lmcold.samples.giants()`, split into the
    # five [Fe/H] bins (for the median [Fe/H] of each bin).

    # %%
    v = pd.read_csv(ROOT / 'results/shape_virial.csv')  # q per population, potential and central floor r_floor
    d = pd.read_csv(ROOT / 'results/shape_rrl_depth.csv')  # RR Lyrae q from the distance gradient, per case (bootstrap, dipole)
    # Median [Fe/H] of the RR Lyrae members.
    feh_rrl = pd.read_csv(ROOT / 'results/kin_support_ellipsoid.csv').set_index('population').feh_med['RR Lyrae']
    giants = samples.giants()  # (label, catalogue dict) per [Fe/H] bin, from data/giants_apogee.parquet and giants_boss.parquet

    # %% [markdown]
    # ### The axis ratios implied by the motions
    # `results/shape_virial.csv` holds, for each giant [Fe/H] bin and each of the three potentials, the axis ratio $q$ of the tracer
    # whose tensor-virial ratio matches the bin's kinematics: posterior-median model and the 16th-84th percentiles of the posterior
    # draws. We use the runs without a central dispersion floor, and the median [Fe/H] of each bin.

    # %%
    v = v[(v.r_floor == 0) & v.population.str.startswith('giants')]
    # Median [Fe/H] [dex] of each giant bin, keyed by the population name of the csv.
    feh = {f'giants {lab}': np.median(c['feh']) for lab, c in giants}

    # %% [markdown]
    # ### The RR Lyrae: flattening from their depth
    # `results/shape_rrl_depth.csv` gives $q$ from the near-to-far distance gradient of the RR Lyrae at the inclination of the
    # metal-poor giants: bootstrap resamples, and the two refits with a coherent distance error of $\pm0.02$ mag across the line of
    # nodes.

    # %%
    # Bootstrap values of the RR Lyrae q (dimensionless).
    b = d[d.case == 'bootstrap'].q

    # %% [markdown]
    # ### The figure
    # Grey bands: the RR Lyrae (light: the distance-error range; dark: bootstrap 16th-84th percentiles). Markers: the giants in the
    # three potentials, offset slightly in [Fe/H] for legibility.

    # %%
    plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
    # One panel; legend above it.
    fig, ax = panel_grid(1, 1, width=3.5, left=0.55, right=0.1, top=0.6)
    a = ax[0, 0]
    # RR Lyrae bands, 0.1 dex wide about their median [Fe/H]: light = range of the two dipole refits, dark = bootstrap 16th-84th
    a.add_patch(plt.Rectangle((feh_rrl - 0.05, d[d.case.str.startswith('dipole')].q.min()), 0.1,
                              np.ptp(d[d.case.str.startswith('dipole')].q), color='0.85', lw=0))
    a.add_patch(plt.Rectangle((feh_rrl - 0.05, b.quantile(0.16)), 0.1, b.quantile(0.84) - b.quantile(0.16), color='0.6', lw=0,
                              label='RR Lyrae: distance gradient'))
    for k, (pot, (mk, c)) in enumerate(STYLE.items()):
        # The bins in this potential: median-vector q with 16th-84th percentile bars, at the bin's median [Fe/H] offset by (k - 1)
        # DX.
        g = v[v.potential == pot]
        x = np.array([feh[p] for p in g.population]) + (k - 1) * DX
        a.errorbar(x, g.q, yerr=[g.q - g.q_p16, g.q_p84 - g.q], fmt=mk, color=c, ms=4, capsize=0, lw=1, label=LABELS[pot])
    # Axis ranges; the legend above the panel; the plotted values printed.
    a.set(xlabel='[Fe/H]', ylabel='axis ratio $q=c/a$ from rotation', ylim=(0, 0.8), xlim=(-2.0, -0.2))
    a.legend(loc='lower left', bbox_to_anchor=(0, 1.02), fontsize=6, frameon=False, ncol=2)
    fig.savefig(ROOT / 'figures/paper_shape_virial.png')
    print(v[['population', 'potential', 'q', 'q_p16', 'q_p84', 'balance']].round(3).to_string(index=False))
