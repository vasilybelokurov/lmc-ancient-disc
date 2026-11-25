"""Paper figure (Section 6, after Fig. sims_kinematics): kinematics of the simulated analogues' stars at birth and the same stars
today, against age, for the 681 TNG50 and the five Auriga level-3 analogues, with the LMC giants placed at the ages their
metallicity implies.

Simulations: results/sims_birth.csv (scripts/build_sims_birth.py: cohorts of COHORT Gyr; v_phi, sigma_z at birth and of the same
stars today).
LMC (results/kin_support_ellipsoid.csv, stage-1 ellipsoid fits at R = 4 kpc, Table kin): each giant [Fe/H] bin (lmcold.samples.
FEH_BINS) at the formed-mass-weighted mean age, with the 16th-84th percentile range, that the reference age-metallicity relation
(lmcold.constants.LMC_AMR_POVICK24, Gaussian scatter, constant star formation, ages to the PARSEC limit AGE_MAX as in Table ages)
gives the bin; v_rot(4 kpc) is plotted with the simulations' v_phi.

Panels (1 x 3, equal, in the order of Fig. sims_kinematics): (a) v_phi, (b) sigma_z [km/s], (c) v_phi / sigma_z (each cohort's own
ratio) against age today [Gyr]. Plotted in PLOT_BIN-Gyr age bins: each galaxy's median of its cohort values in the bin (the cohorts
stay COHORT wide; the bins only smooth the display). TNG50: median over the analogues with a birth value, and 16th-84th percentile
band; dashed blue at birth, solid red today. Auriga: one thin line per halo, same colours. Panel letters above the panels; legend
in the top left of (b).
Writes figures/paper_sims_birth.png; prints the 9-12 Gyr summary.

Run: .venv/bin/python scripts/manifest.py run paper_sims_birth.py   (after scripts/build_sims_birth.py; seconds,
2 s in the recorded run of docs/run_times.txt)
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from lmcold.constants import LMC_AMR_POVICK24
from lmcold.layout import panel_grid
from lmcold.samples import FEH_BINS
from lmcold.sources import ROOT

PLOT_BIN = 0.5  # Gyr, plotted age bins: median of the cohort values in each bin, per galaxy (display only)
YTOP = dict(vphi=90, sz=65, vs=6)  # upper limits of panels (a)-(c) (PI): room above the curves, and for the legend in (b)
AGE_MAX = 12.59  # Gyr, oldest PARSEC isochrone (lmcold.constants, LMC_AMR_POVICK24 comment)


def lmc_ages():
    """Mean age and 16th-84th percentile ages [Gyr] of each giant [Fe/H] bin under the reference AMR (formed-mass weights)."""
    # Fine age grid [Gyr] (uniform: constant star formation); mean [Fe/H] and Gaussian scatter [dex] of the AMR at each age
    a = np.linspace(0, AGE_MAX, 5000)
    mu, sd = (np.interp(a, LMC_AMR_POVICK24['age'], LMC_AMR_POVICK24[k]) for k in ('feh', 'sigma'))
    out = []
    for lo, hi in FEH_BINS:
        # w: probability that a star of each age has [Fe/H] in the bin (the age weights); c: their cumulative fraction
        w = norm.cdf(hi, mu, sd) - norm.cdf(lo, mu, sd)
        c = np.cumsum(w) / w.sum()
        # Weighted mean age and the ages at which the cumulative weight reaches 0.16 and 0.84
        out.append((np.average(a, weights=w), np.interp(0.16, c, a), np.interp(0.84, c, a)))
    return out


if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data
    # The figure reads two result tables and no simulation: `results/sims_birth.csv`, made from the TNG50 and Auriga analogues by
    # `scripts/build_sims_birth.py` (the simulations are needed only to remake it), and the LMC giants' stage-1 ellipsoid fits
    # `results/kin_support_ellipsoid.csv`. The ages of the LMC bins come from the age-metallicity relation `LMC_AMR_POVICK24` of
    # `lmcold.constants`, not a file.

    # %%
    t = pd.read_csv(ROOT / 'results/sims_birth.csv', dtype={'halo': str})   # per analogue and cohort: v_phi, sigma_z at birth and today
    k = pd.read_csv(ROOT / 'results/kin_support_ellipsoid.csv')   # LMC: v_rot(4 kpc), sigma_z, v_rot/sigma_z per giant bin and RR Lyrae

    # %% [markdown]
    # ### Cohorts at birth and today
    # `t` holds, for every analogue and cohort of `COHORT` Gyr of age, the mean $v_\phi$ and $\sigma_z$ of its stars at birth, relative
    # to the galaxy they formed in, and of the same stars today. Each cohort's $v_\phi/\sigma_z$ is its own ratio; the age bins of
    # `PLOT_BIN` Gyr only smooth the display. Of the LMC fits `k` only the giant bins are kept.

    # %%
    # Each cohort's own v_phi/sigma_z at birth and today; age_bin: centre [Gyr] of its PLOT_BIN-wide age bin
    t['vs_birth'], t['vs_same'] = t.vphi_birth / t.sz_birth, t.vphi_same / t.sz_same
    t['age_bin'] = (np.floor(t.age / PLOT_BIN) + 0.5) * PLOT_BIN
    # LMC: the five giant [Fe/H] bins only (RR Lyrae dropped)
    k = k[k.population.str.startswith('giants')].reset_index(drop=True)

    # %% [markdown]
    # ### The figure
    # Per galaxy the median of its cohorts in each age bin; TNG50 median and 16th-84th percentiles over the analogues (dashed: at
    # birth; solid: today), Auriga one thin line per halo; the LMC bins at the ages that `lmc_ages` gives them.

    # %%
    # Three equal panels (lmcold.layout.panel_grid; sizes in inches); blue for birth, red for today
    plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
    fig, ax = panel_grid(1, 3, width=7.0, left=0.5, right=0.1, hgap=0.55, top=0.3)
    colour = {'birth': '#2b6cb0', 'same': '#c0392b'}
    # Per panel: column stem of the simulations, LMC column of k, y-axis label
    panels = [('vphi', 'vrot4', r'$v_\phi$ [km s$^{-1}$]'), ('sz', 'sigma_z', r'$\sigma_z$ [km s$^{-1}$]'),
              ('vs', 'ratio_z', r'$v_\phi/\sigma_z$')]
    # TNG50 cohorts with a birth value; all Auriga cohorts
    tng, au = t[t.suite == 'TNG50'].dropna(subset=['vphi_birth']), t[t.suite == 'Auriga']
    titles = ('(a) rotation', '(b) vertical velocity dispersion', '(c) rotational support')
    for a, (q, lq, lab), title in zip(ax[0], panels, titles):
        for when, ls in (('birth', '--'), ('same', '-')):
            # Per-analogue median in each age bin, grouped by bin: band = 16th-84th percentiles over analogues, line = median
            g = tng.groupby(['halo', 'age_bin'])[f'{q}_{when}'].median().groupby(level='age_bin')
            a.fill_between(g.median().index, g.quantile(0.16), g.quantile(0.84), color=colour[when], alpha=0.18, lw=0)
            a.plot(g.median().index, g.median(), ls, color=colour[when], lw=1.4)
            # Auriga: one thin line per halo of its median in each age bin
            for _, h in au.groupby('halo'):
                h = h.groupby('age_bin')[f'{q}_{when}'].median()
                a.plot(h.index, h, ls, color=colour[when], lw=0.6, alpha=0.7)
        # LMC giant bins at their mean age, horizontal bar = 16th-84th percentile ages
        for (m, lo, hi), v in zip(lmc_ages(), k[lq]):
            a.errorbar(m, v, xerr=[[m - lo], [hi - m]], fmt='s', color='k', ms=3.5, capsize=2, lw=0.8, zorder=5)
        a.set(xlabel='age today [Gyr]', ylabel=lab, xlim=(0, 13.5))
        a.set_ylim(top=YTOP[q])
        a.set_title(title, loc='left', fontsize=8)  # above the panel, as Fig. sims_sidm: the curves fill its corners
    # Legend handles built by hand (one per line style), placed in panel (b)
    h = [plt.Line2D([], [], color=colour['birth'], ls='--', lw=1.4), plt.Line2D([], [], color=colour['same'], lw=1.4),
         plt.Line2D([], [], color='0.4', lw=0.5), plt.Line2D([], [], color='k', marker='s', ls='', ms=3.5)]
    ax[0, 1].legend(h, ['TNG50 at birth (median, 16-84%)', 'the same stars today', 'Auriga, one line per halo', 'LMC giants'],
                    loc='upper left', fontsize=5, frameon=False)   # empty: sigma_z < 43 at ages < 5 Gyr
    fig.savefig(ROOT / 'figures/paper_sims_birth.png')

    # %% [markdown]
    # ### Numbers of the text
    # The stars now 9-12 Gyr old, the young-star check and the ages of the LMC bins (the text's numbers are written by
    # `scripts/paper_text_sims_birth.py` with the same definitions).

    # %%
    # Columns summarised: v_phi, sigma_z [km/s] and their ratio, at birth and today
    q = ['vphi_birth', 'sz_birth', 'vs_birth', 'vphi_same', 'sz_same', 'vs_same']
    for s, g in t[(t.age > 9) & (t.age < 12)].dropna(subset=['vphi_birth']).groupby('suite'):
        # Per-analogue medians over the cohorts aged 9-12 Gyr; printed as 16/50/84 percentiles over the analogues
        per = g.groupby('halo')[q].median()
        print(s, len(per), 'analogues, 9-12 Gyr: 16/50/84 per cent over analogues\n', per.quantile([0.16, 0.5, 0.84]).round(2).to_string())
        # Analogues at or above 1.76, the LMC's v_rot/sigma_z of the most metal-poor giant bin (results/kin_support_ellipsoid.csv)
        print(' v/sigma >= 1.76 at birth:', (per.vs_birth >= 1.76).sum(), ' today:', (per.vs_same >= 1.76).sum())
    # Young TNG50 cohorts (< 1.5 Gyr): birth and today values should be close; then the LMC bin ages [Gyr]
    y = tng[tng.age < 1.5]
    print('TNG50 young-star check (< 1.5 Gyr), medians:', y[['vphi_birth', 'vphi_same', 'sz_birth', 'sz_same']].median().round(1).to_dict())
    print('LMC ages (mean, p16, p84):', [tuple(round(x, 1) for x in a) for a in lmc_ages()])
