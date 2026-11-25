"""Paper figure (Appendix): what the oblate sky-cell model leaves in the RR Lyrae counts and median moduli (paper Section 5).

The cells of lmcold.shape.cell_stats (8 rings x 24 sectors, 2-7 deg), for the free fit (counts + medians alone) and for the fit at the
old population's kinematic inclination (the ridge point at INC, results/shape_rrl_cells.csv), from lmcold.shape.cell_model:
(a, c) counts, Pearson residual (n - lam) / sqrt(lam); (b, d) median modulus minus model [mag]. Sky: east left, north up, rho [deg]
from the LMC centre; lines: the fit's line of nodes (solid) and the kinematic one of the metal-poor giants (dashed).

Writes figures/paper_shape_rrl_residuals.png (KB chapter 'The shape of the old population', Figure paper_shape_rrl_residuals) and results/shape_rrl_harmonics.csv (per ring of the free fit: m = 1 and m = 2 amplitude and
position angle of the fractional count residual n / lam - 1, and the fraction of the ring inside the OGLE-IV fields,
lmcold.ogle.in_footprint, to tell structure from coverage) and results/shape_rrl_fit_stats.csv (per fit: stars, cells, chi^2 of
the counts and of the medians, range of the count Pearson residuals, rms modulus residual and median modulus error [mag]).

Run: .venv/bin/python scripts/paper_shape_rrl_residuals.py   (after scripts/shape_rrl_cells.py)
"""
# %% [markdown]
# ### Settings
# The inclination of the second fit and the colour limits of the residual maps.

# %%
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Wedge

from lmcold import ogle, shape
from lmcold.layout import panel_grid
from lmcold.sources import ROOT
from shape_rrl_cells import OLD

INC = 25.0  # deg, ridge point at the metal-poor giants' inclination (24.9; the ridge grid has 25.0)
LIM = (4.0, 0.06)  # colour limits: Pearson residual, modulus residual [mag]


# %% [markdown]
# ### Drawing helpers
# `cells` colours the sky cells as wedges on the tangent plane; `axis` draws a line through the centre at a given position angle.


# %%
def cells(a, values, norm, cmap):
    """Cell wedges coloured by values (NaN: blank); position angle from north through east, east to the left.

    a: matplotlib axes (tangent-plane coordinates xi, eta [deg]); values: one number per cell of lmcold.shape.cell_stats, ring by ring
    (len(CELL_RHO) - 1 rings x CELL_NPA sectors); norm, cmap: colour normalisation and map."""
    # angular width of one position-angle sector [deg] (360 / 24 = 15)
    w = 360 / shape.CELL_NPA
    # cells numbered ring by ring: index c -> (ring, sector)
    for c, v in enumerate(values):
        if not np.isfinite(v):
            continue
        ring, sec = divmod(c, shape.CELL_NPA)
        # inner and outer radius of the ring [deg]
        r1, r2 = shape.CELL_RHO[ring], shape.CELL_RHO[ring + 1]
        # Wedge angles run from the +xi axis; with xi = rho sin PA, eta = rho cos PA, sector PA1..PA2 spans 90 - PA2..90 - PA1
        a.add_patch(Wedge((0, 0), r2, 90 - (sec + 1) * w, 90 - sec * w, width=r2 - r1, color=cmap(norm(v)), lw=0))  # xi = rho sin PA


def axis(a, theta, **kw):
    """Line through the centre at position angle theta [deg] (east of north) on axes a, out to 7.5 deg; kw passed to plot."""
    # PA east of north -> angle from the +xi axis [rad]
    t = np.deg2rad(90 - theta)
    a.plot([-7.5 * np.cos(t), 7.5 * np.cos(t)], [-7.5 * np.sin(t), 7.5 * np.sin(t)], color='k', lw=0.8, **kw)


if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data
    # The RR Lyrae table `data/rrab.parquet`, the fits of the sky cells from `results/shape_rrl_cells.csv`
    # (`scripts/shape_rrl_cells.py`) and the kinematic posterior summaries of `results/kin_mcmc_data_ellipsoid.csv`. The OGLE-IV field
    # centres (`data/raw/ogle4/fields-OGLE-IV-LMC.txt`, `fields-OGLE-IV-MBR.txt`) are read where they are used, inside
    # `lmcold.ogle.in_footprint` (ring coverage below).

    # %%
    t = pd.read_parquet(ROOT / 'data/rrab.parquet')  # RR Lyrae with membership, distance modulus dmu and its source dmu_src
    R = pd.read_csv(ROOT / 'results/shape_rrl_cells.csv')  # free oblate fit and the ridge of fits at fixed inclination
    k = pd.read_csv(ROOT / 'results/kin_mcmc_data_ellipsoid.csv')  # ellipsoid-model posterior summaries (line of nodes theta)

    # %% [markdown]
    # ### The RR Lyrae in the sky cells
    # Members with an OGLE Wesenheit distance modulus, inside the 2-7 deg footprint, counted in the sky cells of `lmcold.shape.cell_stats`
    # (number, median modulus and its error per cell); the free fit and the fit at the kinematic inclination; the mean line of nodes of
    # the old population from its motions.

    # %%
    # members whose distance modulus comes from the OGLE I-band (Wesenheit) photometry
    t = t[t.member & (t.dmu_src == 'I')]
    # inside the 2-7 deg annulus and the distance window (lmcold.shape.in_footprint)
    t = t[shape.in_footprint(t.ra.to_numpy(), t.dec.to_numpy(), t.dmu.to_numpy())]
    # per cell: star count n, median modulus med [mag] and its standard error se [mag]
    n, med, se = shape.cell_stats(t.ra.to_numpy(), t.dec.to_numpy(), t.dmu.to_numpy())
    # (unit vectors, cell index) of the quadrature points inside the cells
    sky = shape.cell_sky()
    # (label, fitted row): the free fit, and the ridge fit at inclination INC
    fits = [('free fit', R[R.fit == 'free'].iloc[0]), (f'$i={INC:.0f}^\\circ$ (kinematic)',
                                                                       R[(R.fit == 'ridge') & np.isclose(R.inc, INC)].iloc[0])]
    # kinematic line of nodes theta [deg]: mean of the stage-1 posterior medians of the old-population groups (OLD)
    theta_kin = k[(k.stage == 1) & k.population.isin(OLD) & (k.param == 'theta')]['median'].mean()

    # %% [markdown]
    # ### Lopsidedness of the counts, ring by ring
    # The $m=1$ and $m=2$ Fourier amplitudes of the fractional count residual $n/\lambda-1$ of the free fit in each ring, with the
    # fraction of the ring inside the OGLE-IV fields (to tell structure from coverage).

    # %%
    # parameters (inc, theta, q, h, delta) of the free fit
    p = [fits[0][1][c] for c in shape.CELL_PARAMS]
    # expected counts lam per cell of the free fit
    lam = np.asarray(shape.cell_model(p, n, sky)[0])
    # quadrature points of the whole annulus, unit vectors (M, 3)
    rs = shape.footprint()[0]
    # whether each point lies in an OGLE-IV field ((RA, Dec) [deg] from the unit vector)
    inside = ogle.in_footprint(np.rad2deg(np.arctan2(rs[:, 1], rs[:, 0])) % 360, np.rad2deg(np.arcsin(rs[:, 2])))
    # cell index of each point (-1 outside the rings)
    k = shape.cell_index(rs)
    # centre position angle of each sector [rad]
    pa = (np.arange(shape.CELL_NPA) + 0.5) * 2 * np.pi / shape.CELL_NPA
    rows = []
    for ring in range(len(shape.CELL_RHO) - 1):
        # the cells of this ring, and their fractional count residual n / lam - 1
        c = slice(ring * shape.CELL_NPA, (ring + 1) * shape.CELL_NPA)
        d = n[c] / lam[c] - 1
        # coverage: fraction of the ring's quadrature points inside the OGLE-IV fields
        row = dict(rho_lo=shape.CELL_RHO[ring], rho_hi=shape.CELL_RHO[ring + 1], coverage=inside[k // shape.CELL_NPA == ring].mean())
        for m in (1, 2):
            # Fourier cosine and sine coefficients of order m; amplitude a_m and phase angle pa_m [deg] (period 360 / m)
            a, b = 2 * np.mean(d * np.cos(m * pa)), 2 * np.mean(d * np.sin(m * pa))
            row[f'a{m}'], row[f'pa{m}'] = np.hypot(a, b), np.rad2deg(np.arctan2(b, a)) / m % (360 / m)
        rows.append(row)
    H = pd.DataFrame(rows)
    H.to_csv(ROOT / 'results/shape_rrl_harmonics.csv', index=False, float_format='%.3f')
    print(H.round(3).to_string(index=False))

    # %% [markdown]
    # ### The residual maps
    # Counts as Pearson residuals $(n-\lambda)/\sqrt{\lambda}$ and median moduli minus the model, for both fits; the fit statistics go to
    # `results/shape_rrl_fit_stats.csv`.

    # %%
    plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
    # 2 x 2 equal panels: rows = fits, columns = counts and moduli
    fig, ax = panel_grid(2, 2, width=5.0, left=0.5, right=0.1, hgap=0.45, vgap=0.3, top=0.75)
    # symmetric colour ranges: +-4 for the Pearson residual, +-0.06 mag for the modulus residual
    norms = [Normalize(-LIM[0], LIM[0]), Normalize(-LIM[1], LIM[1])]
    cmap = plt.get_cmap('RdBu_r')
    stats = []
    for i, (label, f) in enumerate(fits):
        # parameters of this fit; lam: expected counts; Mc: model mean modulus [mag] per cell
        p = [f[c] for c in shape.CELL_PARAMS]
        lam, Mc = (np.asarray(v) for v in shape.cell_model(p, n, sky))
        # res[0]: Pearson residual of the counts; res[1]: median modulus minus model [mag]
        res = [(n - lam) / np.sqrt(lam), med - Mc]
        print(f'{label}: counts chi2 {np.sum(res[0] ** 2):.0f}, medians chi2 {np.nansum((res[1] / se) ** 2):.0f}, '
              f'rms modulus residual {np.nanstd(res[1]):.3f} mag ({len(n)} cells)')
        # fit statistics written to results/shape_rrl_fit_stats.csv
        stats.append(dict(fit=('free', 'ridge')[i], inc=f.inc, n_stars=len(t), n_cells=len(n), chi2_counts=np.sum(res[0] ** 2),
                          chi2_medians=np.nansum((res[1] / se) ** 2), pearson_min=res[0].min(), pearson_max=res[0].max(),
                          rms_modulus=np.nanstd(res[1]), median_se=np.nanmedian(se)))
        for j in (0, 1):
            # one panel per residual: the cells, the fit's line of nodes (solid) and the kinematic one (dashed)
            a = ax[i, j]
            cells(a, res[j], norms[j], cmap)
            axis(a, f.theta)
            axis(a, theta_kin, ls='--')
            # east to the left, +-7.5 deg
            a.set(xlim=(7.5, -7.5), ylim=(-7.5, 7.5), aspect='equal')
            a.text(0.03, 0.97, f'({"abcd"[2 * i + j]}) {label}', transform=a.transAxes, fontsize=6, va='top')
        ax[i, 0].set_ylabel(r'$\eta$ [deg]')
    # x labels on the bottom row; one horizontal colour bar per column, above the top panel
    for j, lab in enumerate(('counts: $(n-\\lambda)/\\sqrt{\\lambda}$', 'median modulus $-$ model [mag]')):
        ax[1, j].set_xlabel(r'$\xi$ [deg] (east left)')
        pos = ax[0, j].get_position()
        cax = fig.add_axes([pos.x0, pos.y1 + 0.03, pos.width, 0.015])
        fig.colorbar(ScalarMappable(norms[j], cmap), cax=cax, orientation='horizontal').set_label(lab, fontsize=7)
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
    pd.DataFrame(stats).to_csv(ROOT / 'results/shape_rrl_fit_stats.csv', index=False, float_format='%.4f')
    fig.savefig(ROOT / 'figures/paper_shape_rrl_residuals.png')
