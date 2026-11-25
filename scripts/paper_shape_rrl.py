"""Paper figure (Section 5): the inclination-flattening plane of the old population, from its RR Lyrae density and from its motions.

Curves (lmcold.shape.projected, at the values of the free sky-cell fit, results/shape_rrl_cells.csv): constant outline axis ratio
on the sky q_sky (counts; blue) and constant near-to-far depth gradient g (median moduli; red) of an inclined oblate body. They
cross at the free fit (black dot: counts + distances alone). Vertical bands: the inclination from the motions (scripts/
shape_rrl_cells.motion_incs: the metal-poor giants, darker; the RR Lyrae proper motions, lighter); open circle: q of the ridge
fit (inclination held) at the giants' inclination. Dashed lines: the rotation-implied q of the metal-poor giant bins (lowest 16th and
highest 84th percentile over P1-P3, results/shape_virial.csv). Arrow: an in-plane elongated mock at the giants' geometry
(results/shape_rrl_triaxial.csv, ARROW: c/a, b/a, angle of the major axis from the nodes), from its true (inclination, c/a) to
the oblate fit of its cells. The model fits the cells poorly (chi^2 per cell ~3-6, scripts/paper_shape_rrl_residuals.py), so no
likelihood contours are drawn.

Writes figures/paper_shape_rrl.png (KB chapter 'The shape of the old population', Figure paper_shape_rrl).

Run: .venv/bin/python scripts/paper_shape_rrl.py   (after scripts/shape_rrl_cells.py, shape_rrl_triaxial.py, shape_virial.py)
"""
# %% [markdown]
# ### Settings
# Plot limits, the colours of the two curves, and which elongated mock the arrow shows.

# %%
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmcold import shape
from lmcold.layout import panel_grid
from lmcold.sources import ROOT
from shape_rrl_cells import motion_incs, virial_q

XLIM, YLIM = (10.0, 70.0), (0.0, 1.0)  # deg, q
SKY, DEPTH = '#2166ac', '#b2182b'  # colours of the two curves
ARROW = (0.41, 0.85, 150.0)  # mock: c/a, b/a, major-axis angle [deg]

if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data
    # The fits of the RR Lyrae sky cells from `results/shape_rrl_cells.csv` (`scripts/shape_rrl_cells.py`); the tilts from the motions
    # through `motion_incs()` (posterior summaries in `results/kin_mcmc_data_ellipsoid.csv` and
    # `results/kin_mcmc_data_ellipsoid_outer.csv`) and the axis ratios the motions require through `virial_q()`
    # (`results/shape_virial.csv`), both of `scripts/shape_rrl_cells.py`; the elongated mocks from `results/shape_rrl_triaxial.csv`
    # (`scripts/shape_rrl_triaxial.py`).

    # %%
    R = pd.read_csv(ROOT / 'results/shape_rrl_cells.csv')  # free oblate fit and the ridge of fits at fixed inclination
    ki = motion_incs()  # {label: (median, p16, p84)} inclination [deg] from the motions
    v = virial_q()  # q required by the rotation of the metal-poor giant bins, all potentials, no central floor
    T = pd.read_csv(ROOT / 'results/shape_rrl_triaxial.csv')  # oblate fits of elongated mocks (c, ba, phi)

    # %% [markdown]
    # ### The fits of the RR Lyrae sky cells
    # The free oblate fit of the counts and median distances in the sky cells, the 'ridge' of fits with the inclination held at a grid
    # of values, and the elongated mock with the axis ratios and angle `ARROW`.

    # %%
    free = R[R.fit == 'free'].iloc[0]            # free oblate fit: inc [deg], q
    r = R[R.fit == 'ridge'].sort_values('inc')   # best q at each held inclination, sorted by inclination
    # The mock with c/a, b/a and angle equal to ARROW, and its oblate fit (inc, q).
    t = T[np.isclose(T.c, ARROW[0]) & np.isclose(T.ba, ARROW[1]) & np.isclose(T.phi, ARROW[2])].iloc[0]

    # %% [markdown]
    # ### The figure
    # The two curves are the $(i,q)$ of an inclined oblate body with the observed sky outline $q_{\rm sky}$ and depth gradient $g$
    # (`lmcold.shape.projected`); grey bands are the tilts from the motions; the open circle is $q$ from the depth gradient at the
    # giants' tilt; the arrow goes from the elongated mock's true $(i, c/a)$ to its oblate fit.

    # %%
    plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
    fig, ax = panel_grid(1, 1, width=3.6, left=0.55, right=0.15, top=0.25)
    a = ax[0, 0]   # the single panel
    I, Q = np.meshgrid(np.linspace(*XLIM, 601), np.linspace(*YLIM, 401))  # inclination [deg] and q grid for the two contours
    for k, colour, text, xy in ((0, SKY, 'outline on the sky\n$q_{\\rm sky}=%.2f$', (62, 0.55)),
                                (1, DEPTH, 'near-to-far distance\ngradient $g=%.2f$', (52, 0.85))):
        # Observed q_sky (k = 0) or depth gradient g (k = 1) of the free fit; its contour over (i, q); label at xy.
        z0 = shape.projected(free.inc, free.q)[k]
        a.contour(I, Q, shape.projected(I, Q)[k], levels=[z0], colors=[colour], linewidths=1.5)
        a.text(*xy, text % z0, color=colour, fontsize=7, ha='center', va='center')
    # Grey bands: 16th-84th percentiles of the inclination from the motions (first entry darker, label at its median).
    for k, (name, (m, lo, hi)) in enumerate(ki.items()):
        a.axvspan(lo, hi, color='k', alpha=0.15 if k == 0 else 0.06, lw=0)
        a.text(m if k == 0 else lo - 0.5, 0.97 - 0.09 * k, 'tilt from motions:\n' + name, fontsize=6, va='top',
               ha='left' if k == 0 else 'right')
    # Open circle: q of the ridge interpolated at the median tilt [deg] of the metal-poor giants.
    m = ki['metal-poor giants'][0]
    qk = np.interp(m, r.inc, r.q)
    a.plot(m, qk, 'o', mfc='w', mec='k', ms=5, zorder=5)
    a.annotate(f'$q={qk:.2f}$', (m, qk), xytext=(5, -10), textcoords='offset points', fontsize=7)
    # Dashed lines: lowest 16th and highest 84th percentile over results/shape_virial.csv of the q the rotation requires.
    for y in (v.q_p16.min(), v.q_p84.max()):
        a.axhline(y, color='k', ls='--', lw=0.8)
    a.text(XLIM[1] - 1, v.q_p84.max() + 0.01, 'q needed by the rotation (motions)', fontsize=6, ha='right', va='bottom')
    # Arrow from the mock's true (giants' tilt, c/a) to its oblate fit (inc, q); the text is placed at (44, 0.3).
    a.annotate('', (t.inc, t.q), (m, ARROW[0]), arrowprops=dict(arrowstyle='->', color='0.4', lw=0.8, shrinkA=3, shrinkB=3))
    a.text(44, 0.3, f'elongated mock at the\nmotions tilt\n($b/a={ARROW[1]}$, $c/a={ARROW[0]}$)\nfitted as oblate', fontsize=6, color='0.4', ha='center')
    # Filled circle: the free fit of counts and distances.
    a.plot(free.inc, free.q, 'ko', ms=5, zorder=5)
    a.annotate(f'counts + distances alone\n$i={free.inc:.0f}^\\circ$, $q={free.q:.2f}$', (free.inc, free.q), xytext=(-95, 12),
               textcoords='offset points', fontsize=7)
    a.set(xlabel='inclination $i$ [deg]', ylabel='axis ratio $q$ (0: thin disc, 1: sphere)', xlim=XLIM, ylim=YLIM)
    fig.savefig(ROOT / 'figures/paper_shape_rrl.png')
