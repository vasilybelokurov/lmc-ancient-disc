"""Diagnostic figures of the shared-v_c + drift model (scripts/kin_vc_drift.py, scripts/kin_vc_drift_extensions.py): one fitted
circular-speed curve for the five giant [Fe/H] bins, each bin's mean rotation lagging it by its asymmetric drift (radial Jeans
equation), against free rotation curves per bin. Reads the fitted vectors (data/kin_vc_drift_ext_vectors.npz) and result tables;
fits nothing.

Data against model. Every giant with a proper motion and a line-of-sight velocity has a measured 3D velocity
V = kappa D (pmra e_E + pmdec e_N) + vlos r_hat [km/s], D its disc distance in the fit's frame. Its internal velocity in a fit p is
V - V_frame(p), where V_frame is the same fit with the rotation switched off (v0 = 0, h = 0: centre, systemic motion and
perspective only), projected on the star's e_phi (azimuthal; positive along the rotation, so v_phi > 0) and e_R. The model's
internal velocity of the star is V_model(p) - V_frame(p) (its mean rotation with the m = 2 streaming). Both are averaged over the
stars of a bin (mean, error = standard deviation / sqrt(n)); the measurement errors average out, the projection is linear.
Stars whose 3D residual from the free fit exceeds CLIP km/s are dropped (Milky Way stars, gross errors) in every model.

Writes (all diagnostic):
- figures/fig_kin_vc_drift.png: (a) the shared v_c(R) and each bin's mean rotation, free (solid) and shared-v_c + drift (dashed);
  (b) v_rot(4 kpc) shared-v_c + drift against free per bin, and the RR Lyrae prediction (results/kin_vc_drift_rrl.csv).
- figures/fig_kin_vc_drift_deficit.png: summed ln L(shared v_c + drift) - ln L(free) per bin of R (a) and azimuth (b)
  (results/kin_vc_drift_deficit.csv).
- figures/fig_kin_vc_drift_data.png: 2 x 5 panels, one column per [Fe/H] bin. Top: mean v_phi of the stars in 1-kpc bins of R
  (points, free-fit frame) and of the models on the same stars: free (grey), shared v_c + drift (black dashed), shared v_c + drift
  with population-dependent streaming and thickness (black dotted). Bottom: data - model of v_phi for free (open) and shared
  v_c + drift (filled), each in its own frame.
- figures/fig_kin_vc_drift_data_phi.png: 2 x 5 panels, data - model of v_phi (top) and v_R (bottom) in 45-deg bins of azimuth
  (from the line of nodes, along the rotation) at 2 < R < 9 kpc, free (open) and shared v_c + drift (filled).
- results/kin_vc_drift_profiles.csv: population, model, coordinate, lo, hi, n, data and
  model mean v_phi and v_R [km/s] and the error of the data mean.

Run: .venv/bin/python scripts/fig_kin_vc_drift.py   (after scripts/kin_vc_drift_extensions.py)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fit_kinematics_mcmc import COLS, ring
from kin_vc_drift import P, vrot4
from kin_vc_drift_extensions import azimuth
from lmcold import kinfit, samples
from lmcold.constants import KAPPA
from lmcold.layout import panel_grid
from lmcold.sources import DATA, ROOT

CLIP = 150.0  # km/s, as scripts/paper_kin_residuals.py: stars with a larger 3D residual from the free fit are dropped
R_EDGES = np.arange(2., 10., 1.)  # kpc: 1-kpc bins of in-plane radius R from 2 to 9 kpc
PHI_EDGES = np.linspace(0., 360., 9)  # deg: eight 45-deg bins of azimuth from the line of nodes
COLOURS = ['#d7191c', '#fdae61', '#a6d96a', '#1a9641', '#2878b5']  # giant [Fe/H] bins, metal-poor first
# key of the fitted vectors in data/kin_vc_drift_ext_vectors.npz: legend label
MODELS = {'free': 'free', 'drift': 'shared $v_c$ + drift', 'drift_p_stream_p_beta': '+ streaming + thickness'}


def velocities(p, d):
    """Measured and model internal velocities (v_phi, v_R) [km/s] of the stars of d in the frame of p, the 3D residual
    |V - V_model| [km/s], and R [kpc], phi [deg]."""
    q = p.copy()
    q[[P('v0'), P('h')]] = 0.0  # frame only: no rotation, no drift
    # Model proper motions [mas/yr], vlos [km/s], disc distance D [kpc] and unit vectors eR, ephi (N, 3) for the full fit p.
    mra, mde, mlos, D, _, eR, ephi, _ = (np.asarray(v) for v in kinfit._model(p, d['rhat'], d['ea'], d['ed']))
    # The same for the frame-only vector q.
    fra, fde, flos = (np.asarray(v) for v in kinfit.model(q, d['rhat'], d['ea'], d['ed'])[:3])
    ea, ed, rh = (np.asarray(d[k]) for k in ('ea', 'ed', 'rhat'))
    # 3D velocity [km/s], shape (N, 3): kappa D (pmra e_E + pmdec e_N) + vlos r_hat (KAPPA = 4.74 km/s per kpc mas/yr).
    vec = lambda a, b, c: KAPPA * D[:, None] * (a[:, None] * ea + b[:, None] * ed) + c[:, None] * rh  # noqa: E731
    V, Vm, Vf = vec(np.asarray(d['pmra']), np.asarray(d['pmdec']), np.asarray(d['vlos'])), vec(mra, mde, mlos), vec(fra, fde, flos)
    R, phi = azimuth(p, d)  # in-plane radius [kpc] and azimuth [deg] in the frame of p
    # Projection on (e_phi, e_R): returns (v_phi, v_R) [km/s].
    proj = lambda X: ((X * ephi).sum(1), (X * eR).sum(1))  # noqa: E731
    return proj(V - Vf), proj(Vm - Vf), np.linalg.norm(V - Vm, axis=1), R, phi


def binned(x, edges, *ys):
    """Mean of each y in bins of x (edges: bin edges, same units as x), the standard error (sd / sqrt n) of the first, and the counts;
    bins with 4 stars or fewer give NaN."""
    k = np.digitize(x, edges) - 1  # bin index per star; -1 or len(edges) - 1 outside the edges
    # Per y: the mean in each bin with more than 4 stars.
    out = [np.array([y[k == i].mean() if (k == i).sum() > 4 else np.nan for i in range(len(edges) - 1)]) for y in ys]
    # Standard error of the mean of the first y, and the star counts per bin.
    err = np.array([ys[0][k == i].std() / np.sqrt((k == i).sum()) if (k == i).sum() > 4 else np.nan for i in range(len(edges) - 1)])
    return out, err, np.bincount(k[(k >= 0) & (k < len(edges) - 1)], minlength=len(edges) - 1)


if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data
    # The fitted parameter vectors of `scripts/kin_vc_drift_extensions.py` (`data/kin_vc_drift_ext_vectors.npz`); the RR Lyrae
    # prediction (`results/kin_vc_drift_rrl.csv`) and the likelihood deficit per bin (`results/kin_vc_drift_deficit.csv`) from the same
    # script; the giant catalogues of Zenodo bundle 1 through `lmcold.samples.giants()`, split into the five [Fe/H] bins and cut to
    # $2<\rho<9^\circ$ by `ring`.

    # %%
    z = np.load(DATA / 'kin_vc_drift_ext_vectors.npz')  # parameter vectors per bin and model ('free', 'drift', ...), and 'names'
    names = list(z['names'])  # the giant bins, in the order of the vectors
    rrl = pd.read_csv(ROOT / 'results/kin_vc_drift_rrl.csv', index_col=0)  # RR Lyrae v_rot(4 kpc), measured and predicted
    Dd = pd.read_csv(ROOT / 'results/kin_vc_drift_deficit.csv')  # summed ln L(shared v_c + drift) - ln L(free) per bin of R, phi
    # Giant catalogue dict per [Fe/H] bin (data/giants_apogee.parquet, giants_boss.parquet), 2 < rho < 9 deg, no distance moduli.
    giants = [ring(cat | dict(dmu=np.full(len(cat['ra']), np.nan))) for _, cat in samples.giants()]

    # %% [markdown]
    # ### The fitted models
    # For each giant [Fe/H] bin the parameter vector of the free fit (its own rotation curve) and of the shared-$v_c$ + drift fit (one
    # circular-speed curve; the bin's mean rotation lags it by its asymmetric drift, `lmcold.kinfit.rotation` with $h>0$).

    # %%
    plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
    pf, pj = z['free'], z['drift']  # (5 bins, 33 parameters, order of lmcold.kinfit.PARAMS): free fit, shared-v_c + drift fit
    labels = [n.replace('giants ', '') for n in names]

    # %% [markdown]
    # ### Rotation curves and $v_{\rm rot}(4\,{\rm kpc})$
    # (a) The shared $v_c(R)$ and each bin's mean rotation in both models; (b) $v_{\rm rot}(4\,{\rm kpc})$ of the shared-$v_c$ +
    # drift model against the free fit, with the RR Lyrae prediction (`results/kin_vc_drift_rrl.csv`).

    # %%
    fig, ax = panel_grid(1, 2, width=6.0, left=0.55, right=0.1, hgap=0.55, top=0.75)
    R = np.linspace(1.75, 9, 100)  # in-plane radius grid [kpc]
    # Shared circular speed: the first 32 parameters with h (index 32) set to 0 switch the drift off (kinfit.rotation).
    ax[0, 0].plot(R, np.asarray(kinfit.rotation(np.r_[pj[0][:32], 0.], R)), 'k-', lw=1.5, label=r'shared $v_c$')
    # (a) Each bin's mean rotation [km/s]: free fit (solid) and shared v_c minus its drift (dashed).
    for a, j, c, lab in zip(pf, pj, COLOURS, labels):
        ax[0, 0].plot(R, np.asarray(kinfit.rotation(a, R)), '-', color=c, lw=0.9)
        ax[0, 0].plot(R, np.asarray(kinfit.rotation(j, R)), '--', color=c, lw=0.9, label=lab)
    ax[0, 0].set(xlabel='R [kpc]', ylabel=r'mean rotation [km s$^{-1}$]', ylim=(0, 110))
    ax[0, 0].legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=3, fontsize=5, frameon=False,
                    title='solid: free; dashed: shared $v_c$ + drift', title_fontsize=5)
    # (b) v_rot(4 kpc) [km/s] of the shared-v_c + drift fit against the free fit, one point per giant bin.
    for a, j, c in zip(pf, pj, COLOURS):
        ax[0, 1].plot(vrot4(a), vrot4(j), 'o', color=c, ms=5)
    # RR Lyrae: measured (x) and predicted (y) v_rot(4 kpc), median with 16-84 per cent errors; grey line is y = x.
    m, pr = rrl['measured'], rrl['predicted']
    ax[0, 1].errorbar(m['median'], pr['median'], xerr=[[m['median'] - m['p16']], [m['p84'] - m['median']]],
                      yerr=[[pr['median'] - pr['p16']], [pr['p84'] - pr['median']]], fmt='*', color='#7b3294', ms=8,
                      label='RR Lyrae (predicted)')
    ax[0, 1].plot([50, 90], [50, 90], color='0.6', lw=0.6)
    ax[0, 1].set(xlabel=r'free $v_{\rm rot}(4\,{\rm kpc})$ [km s$^{-1}$]', ylabel=r'shared $v_c$ + drift [km s$^{-1}$]',
                 xlim=(50, 90), ylim=(50, 90))
    ax[0, 1].legend(loc='lower left', bbox_to_anchor=(0, 1.02), fontsize=6, frameon=False)
    # Panel letters, then save.
    for a, t in zip(ax[0], 'ab'):
        a.text(0.04, 0.92, f'({t})', transform=a.transAxes)
    fig.savefig(ROOT / 'figures/fig_kin_vc_drift.png')

    # %% [markdown]
    # ### Where the shared model fits worse
    # Summed $\ln L$ of the shared-$v_c$ + drift fit minus the free fit, star by star, in bins of radius (a) and azimuth (b)
    # (`results/kin_vc_drift_deficit.csv`).

    # %%
    fig, ax = panel_grid(1, 2, width=6.0, left=0.6, right=0.1, hgap=0.6, top=0.5)
    # One line per giant bin: summed ln L difference per R bin (a) and per azimuth bin (b), at the bin centres; the 'survey' rows
    # of the table are not plotted.
    for (name, g), c in zip(Dd[Dd.coordinate != 'survey'].groupby('population', sort=False), COLOURS):
        for a, coord in zip(ax[0], ('R', 'phi')):
            q = g[g.coordinate == coord]
            a.plot(.5 * (q.lo + q.hi), q.dlnL, 'o-', color=c, ms=3, lw=0.8, label=name.replace('giants ', ''))
    # Zero line (both fits equally good), axis labels and panel letters.
    for a, lab, t in zip(ax[0], ('R [kpc]', r'$\phi$ from the line of nodes [deg]'), 'ab'):
        a.axhline(0, color='0.6', lw=0.5)
        a.set(xlabel=lab, ylabel=r'$\sum\,\ln L_{\rm shared\,v_c+drift}-\ln L_{\rm free}$')
        a.text(0.04, 0.92, f'({t})', transform=a.transAxes)
    ax[0, 0].legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=3, fontsize=5, frameon=False)
    fig.savefig(ROOT / 'figures/fig_kin_vc_drift_deficit.png')

    # %% [markdown]
    # ### Data against the models
    # The mean $v_\phi$ and $v_R$ of the giants with 3D velocities, in bins of radius and of azimuth, against the models averaged over
    # the same stars (module docstring); the binned values go to `results/kin_vc_drift_profiles.csv`.

    # %%
    rows = []  # one dict per population x model x coordinate x bin; written to results/kin_vc_drift_profiles.csv
    # Two figures of 2 x 5 panels: against R (figR) and against azimuth (figP); one column per giant bin.
    figR, axR = panel_grid(2, 5, width=7.2, left=0.5, right=0.08, hgap=0.12, vgap=0.12, top=0.4)
    figP, axP = panel_grid(2, 5, width=7.2, left=0.5, right=0.08, hgap=0.12, vgap=0.12, top=0.4)
    for col, (cat, lab, c) in enumerate(zip(giants, labels, COLOURS)):
        # Stars with a 3D velocity (both a line-of-sight velocity and a proper motion), in the fit's data layout.
        has = np.isfinite(cat['vlos']) & np.isfinite(cat['pmra'])
        d = kinfit.data(*(cat[k][has] for k in COLS))
        # Measured and model internal velocities of these stars for each model in MODELS (velocities()).
        res = {key: velocities(z[key][col], d) for key in MODELS}
        keep = res['free'][2] < CLIP  # 3D residual from the free fit below CLIP [km/s], same stars for every model
        for key, ((ophi, oR), (mphi, mR), _, R, phi) in res.items():
            for coord, x, edges in (('R', R, R_EDGES), ('phi', phi, PHI_EDGES)):
                # Azimuth bins use only 2 < R < 9 kpc (the R_EDGES range).
                sel = keep & ((R > R_EDGES[0]) & (R < R_EDGES[-1]) if coord == 'phi' else keep)
                # Binned means of data and model v_phi, v_R [km/s]; err is the standard error of data v_phi, errR of data v_R.
                (dphi, dR, mp, mr), err, n = binned(x[sel], edges, ophi[sel], oR[sel], mphi[sel], mR[sel])
                errR = binned(x[sel], edges, oR[sel])[1]
                rows += [dict(population=names[col], model=key, coordinate=coord, lo=edges[i], hi=edges[i + 1], n=n[i],
                              data_vphi=dphi[i], model_vphi=mp[i], err_vphi=err[i], data_vR=dR[i], model_vR=mr[i], err_vR=errR[i])
                         for i in range(len(edges) - 1)]
        # The rows of this giant bin.
        T = pd.DataFrame(rows)
        T = T[T.population == names[col]]
        # v_phi against R.
        x = .5 * (R_EDGES[1:] + R_EDGES[:-1])  # R bin centres [kpc]
        # Data points in the free-fit frame (top row).
        t = T[(T.model == 'free') & (T.coordinate == 'R')]
        axR[0, col].errorbar(x, t.data_vphi, t.err_vphi, fmt='o', color=c, ms=3, lw=0.8, zorder=3, label='data')
        for key, style in (('free', dict(color='0.55', ls='-')), ('drift', dict(color='k', ls='--')),
                           ('drift_p_stream_p_beta', dict(color='k', ls=':'))):
            t = T[(T.model == key) & (T.coordinate == 'R')]
            axR[0, col].plot(x, t.model_vphi, lw=1, label=MODELS[key], **style)
            # Bottom row: data - model for free (open, shifted -0.1 kpc) and shared v_c + drift (filled, +0.1 kpc).
            if key != 'drift_p_stream_p_beta':
                axR[1, col].errorbar(x + (0.1 if key == 'drift' else -0.1), t.data_vphi - t.model_vphi, t.err_vphi,
                                     fmt='o', mfc=c if key == 'drift' else 'w', mec=c, ecolor=c, ms=3, lw=0.8, label=MODELS[key])
        # Residuals against azimuth.
        xp = .5 * (PHI_EDGES[1:] + PHI_EDGES[:-1])  # azimuth bin centres [deg]; points shifted by -4 (free) and +4 deg (drift)
        for key in ('free', 'drift'):
            t = T[(T.model == key) & (T.coordinate == 'phi')]
            for row, (dv, mv, ev) in enumerate((('data_vphi', 'model_vphi', 'err_vphi'), ('data_vR', 'model_vR', 'err_vR'))):
                axP[row, col].errorbar(xp + (4 if key == 'drift' else -4), t[dv] - t[mv], t[ev], fmt='o', mfc=c if key == 'drift' else 'w',
                                       mec=c, ecolor=c, ms=3, lw=0.8, label=MODELS[key])
        axR[0, col].set_title(lab, fontsize=6)
        axP[0, col].set_title(lab, fontsize=6)
    # Zero lines on every residual panel.
    for a in axR[1].tolist() + axP.ravel().tolist():
        a.axhline(0, color='0.6', lw=0.5)
    for axs, xlab, ylabs, xlim in ((axR, 'R [kpc]', (r'mean $v_\phi$ [km s$^{-1}$]', r'data $-$ model $v_\phi$ [km s$^{-1}$]'), (1.6, 9.4)),
                                   (axP, r'$\phi$ [deg]', (r'data $-$ model $v_\phi$ [km s$^{-1}$]', r'data $-$ model $v_R$ [km s$^{-1}$]'), (0, 360))):
        # Common y range per row (the widest of its panels); y labels on the first column, x labels on the bottom row.
        for row in range(2):
            lim = np.array([a.get_ylim() for a in axs[row]])
            for col, a in enumerate(axs[row]):
                a.set(xlim=xlim, ylim=(lim[:, 0].min(), lim[:, 1].max()))
                if col:
                    a.tick_params(labelleft=False)
                else:
                    a.set_ylabel(ylabs[row], fontsize=6)
                if row == 0:
                    a.tick_params(labelbottom=False)
                else:
                    a.set_xlabel(xlab)
    # Azimuth ticks every 90 deg; legends above the first column (lower-left inside the bottom-left panel of figR).
    for a in axP.ravel():
        a.set_xticks([0, 90, 180, 270])
    axR[0, 0].legend(loc='lower left', bbox_to_anchor=(0, 1.15), ncol=4, fontsize=5, frameon=False)
    axR[1, 0].legend(loc='lower left', fontsize=5, frameon=False)
    axP[0, 0].legend(loc='lower left', bbox_to_anchor=(0, 1.15), ncol=2, fontsize=5, frameon=False)
    # Save both figures and the binned profiles.
    figR.savefig(ROOT / 'figures/fig_kin_vc_drift_data.png')
    figP.savefig(ROOT / 'figures/fig_kin_vc_drift_data_phi.png')
    pd.DataFrame(rows).to_csv(ROOT / 'results/kin_vc_drift_profiles.csv', index=False, float_format='%.3f')
