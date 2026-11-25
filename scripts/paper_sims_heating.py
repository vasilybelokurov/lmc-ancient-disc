"""Paper figure (Section 6, after Fig. sims_birth): the conditions in which the simulated old stars are born and how they lose their
rotation, for the main progenitors of 60 TNG50 analogues (scripts/fetch_tng50_progenitors.py: SAMPLE, and FULL for panel c).

(a) Birth conditions against the stellar mass of the main progenitor at the time (SubLink tree, lmcold.sims.main_branch, at the
    tracked snapshot nearest the cohort's birth): mass-weighted v_phi (solid) and sigma_z (dashed) of
    - stars at birth: the COHORT-Gyr cohorts of results/sims_birth.csv (Fig. sims_birth's method: 0.5 < R < 3 kpc relative to the
      progenitor track);
    - star-forming gas (StarFormationRate > 0) at 0.5 < R < 3 kpc of the progenitor at GAS_SNAPS, about the spin axis of the
      star-forming gas within 3 kpc, velocities relative to its mean there;
    galaxy-weighted medians (median per galaxy in 0.25-dex bins, then over galaxies; bins with >= MIN_GAL galaxies), band 16-84 per
    cent over galaxies for the stars. LMC giant bins (results/kin_support_ellipsoid.csv) at the present-day stellar mass of the LMC
    in stars more metal-poor than each bin's median [Fe/H] (results/giant_present_mass.csv, a different estimator): v_rot(4 kpc)
    squares, sigma_z open triangles (today's sigma_z bounds the birth value from above).
(b) The same stars through time: v_phi/sigma_z of each COHORT-Gyr cohort's stars (those used at birth in a) at every later tracked
    snapshot (TRACK_SNAPS), wherever they are in the main progenitor (R > 0.1 kpc), in the progenitor's disc frame
    (lmcold.sims.disc_frame of all its stars), against the time since the cohort's birth; cohorts born > 10, 8-10 and < 4 Gyr ago;
    the first point after birth is the first tracked snapshot after the whole cohort formed (0.25-0.75 Gyr later); galaxy-weighted
    medians and 16-84 per cent over galaxies. Dotted: the LMC's [Fe/H] < -1.5 giants today.
(c) Orbital-plane coherence C (lmcold.sims.coherence) of 100-Myr cohorts: the stars formed in the WINDOW before each snapshot within
    3 kpc of the progenitor, followed (ParticleIDs, within 10 kpc, disc frame) at every later snapshot up to 0.5 Gyr (FULL sample,
    FULL_SNAPS, z = 4-1.5); median and 16-84 per cent over galaxies in 0.1-Gyr bins. Late control: the same for cohorts formed at
    LATE_SNAPS (z ~ 0.05-0.5) from the every-third-snapshot cutouts (two points). Dotted: C of randomly oriented orbits for the median
    cohort size.
Inputs: results/sims_heating_gas.csv, sims_heating_history.csv, sims_heating_coherence.csv (scripts/build_sims_heating.py).
Writes figures/paper_sims_heating.png.

Run: .venv/bin/python scripts/paper_sims_heating.py   (after scripts/build_sims_heating.py; seconds,
1 s in the recorded run of docs/run_times.txt)
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmcold.layout import panel_grid
from lmcold.sources import ROOT

MIN_GAL = 5  # galaxies per plotted bin


def gal_median(d, by, q):
    """Galaxy-weighted median and 16-84 per cent of q per bin `by` (median per galaxy first); bins with >= MIN_GAL galaxies."""
    # Median of q per (bin, galaxy), then grouped by bin.
    g = d.groupby([by, 'halo'], observed=True)[q].median().groupby(level=0, observed=True)
    # Per bin: median and 16th, 84th percentiles over galaxies, and the number of galaxies.
    out = pd.DataFrame({'m': g.median(), 'lo': g.quantile(.16), 'hi': g.quantile(.84), 'n': g.size()})
    return out[out.n >= MIN_GAL]


if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data
    # The figure reads five result tables and no simulation. Three come from `scripts/build_sims_heating.py`, which needs the TNG50
    # progenitors only to remake them: the star-forming gas of each progenitor at the snapshots `GAS_SNAPS`
    # (`results/sims_heating_gas.csv`), the cohorts at birth and at later tracked snapshots (`results/sims_heating_history.csv`), and
    # the orbital-plane coherence of the 100-Myr cohorts (`results/sims_heating_coherence.csv`). The LMC side is the giants' stage-1
    # ellipsoid fits (`results/kin_support_ellipsoid.csv`) and the present-day stellar mass per [Fe/H] bin
    # (`results/giant_present_mass.csv`).

    # %%
    gas = pd.read_csv(ROOT / 'results/sims_heating_gas.csv', dtype={'halo': str})   # star-forming gas: v_phi, sigma_z per progenitor and snapshot
    hist = pd.read_csv(ROOT / 'results/sims_heating_history.csv', dtype={'halo': str})   # cohorts at birth (dt = 0) and later
    coh = pd.read_csv(ROOT / 'results/sims_heating_coherence.csv', dtype={'halo': str})   # coherence C of 100-Myr cohorts against time
    lmc = pd.read_csv(ROOT / 'results/kin_support_ellipsoid.csv')   # LMC: v_rot(4 kpc), sigma_z, v_rot/sigma_z per giant bin and RR Lyrae
    cm = pd.read_csv(ROOT / 'results/giant_present_mass.csv').iloc[:-1]  # cumulative present mass below each feh_hi

    # %% [markdown]
    # ### The progenitor tables and the LMC bins
    # The ratio $v_\phi/\sigma_z$ of the cohorts and the gas; the LMC giant bins are placed at the present-day stellar mass in stars more
    # metal-poor than each bin's median [Fe/H].

    # %%
    hist['vs'], gas['vs'] = hist.vphi / hist.sz, gas.vphi / gas.sz   # v_phi / sigma_z
    lmc = lmc[lmc.population.str.startswith('giants')].reset_index(drop=True)   # the five giant bins (RR Lyrae dropped)
    # log10 present-day mass [Msun] below each bin's median [Fe/H], interpolated in the cumulative mass.
    lmc_logM = np.interp(lmc.feh_med, cm.feh_hi, np.log10(cm.present_cumulative))


    # %% [markdown]
    # ### (a) Conditions at birth
    # Galaxy-weighted medians (`gal_median`) in 0.25-dex bins of the progenitor's stellar mass at the time: the cohorts at birth and
    # the star-forming gas, $v_\phi$ and $\sigma_z$; the LMC bins as squares ($v_{\rm rot}$) and triangles (today's $\sigma_z$).

    # %%
    plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
    fig, ax = panel_grid(1, 3, width=7.0, left=0.5, right=0.1, hgap=0.6, top=0.55)
    mbins = np.arange(6.5, 10.01, 0.25)   # log10 Msun: 0.25-dex bins of the progenitor's stellar mass
    a = ax[0, 0]
    # Cohorts at birth (dt = 0) and the gas, each row assigned its mass bin from logMs [log10 Msun].
    birth = hist[hist.dt == 0].assign(mb=lambda t: pd.cut(t.logMs, mbins))
    gas['mb'] = pd.cut(gas.logMs, mbins)
    # Stars (blue: band and line) and gas (green line with dots), v_phi solid and sigma_z dashed.
    for q, ls in (('vphi', '-'), ('sz', '--')):
        s_ = gal_median(birth, 'mb', q); x = np.array([i.mid for i in s_.index])
        a.fill_between(x, s_.lo, s_.hi, color='C0', alpha=0.15, lw=0)
        a.plot(x, s_.m, ls, color='C0', lw=1.4)
        g_ = gal_median(gas, 'mb', q)
        a.plot([i.mid for i in g_.index], g_.m, ls, color='C2', lw=1.4, marker='o', ms=2.5)
    # LMC giant bins: v_rot(4 kpc) squares, sigma_z open triangles [km/s].
    a.plot(lmc_logM, lmc.vrot4, 's', color='k', ms=4, zorder=5)
    a.plot(lmc_logM, lmc.sigma_z, 'v', color='k', ms=4, mfc='w', zorder=5)
    a.set(xlabel=r'log stellar mass at the time [M$_\odot$]', ylabel=r'$v_\phi$, $\sigma_z$ [km s$^{-1}$]', ylim=(0, 90), xlim=(6.8, 10))
    a.set_title('(a) conditions at birth', loc='left', fontsize=8)

    # %% [markdown]
    # ### (b) The same stars through time
    # $v_\phi/\sigma_z$ of the cohorts against the time since their birth, for three ranges of birth time.

    # %%
    a = ax[0, 1]
    # time since birth rounded to 0.5 Gyr (birth rows stay at 0, the first later point at least 0.5)
    hist['dtb'] = np.where(hist.dt == 0, 0, np.clip((hist.dt * 2).round() / 2, 0.5, None))
    # Cohorts by birth time (age [Gyr] = lookback time of birth): galaxy-weighted v_phi / sigma_z per 0.5-Gyr step.
    for lab, sel, c in (('> 10 Gyr ago', hist.age > 10, 'C3'), ('8-10 Gyr ago', (hist.age > 8) & (hist.age <= 10), 'C1'),
                        ('< 4 Gyr ago', hist.age < 4, 'C0')):
        s_ = gal_median(hist[sel], 'dtb', 'vs')
        a.fill_between(s_.index, s_.lo, s_.hi, color=c, alpha=0.15, lw=0)
        a.plot(s_.index, s_.m, '-o', color=c, ms=2.5, lw=1.2, label=lab)
    a.axhline(lmc.ratio_z[0], color='k', ls=':', lw=0.8)   # v_rot / sigma_z of the first giant bin, [Fe/H] < -1.5
    a.set(xlabel='time since birth [Gyr]', ylabel=r'$v_\phi/\sigma_z$', xlim=(-0.1, 3.0), ylim=(0, 5.2))  # room above the data (bands reach 3.7) for the legend
    a.legend(fontsize=6, frameon=False, loc='upper right', title='born', title_fontsize=6); a.set_title('(b) the same stars through time', loc='left', fontsize=8)

    # %% [markdown]
    # ### (c) Orbital-plane coherence
    # $C$ of the 100-Myr cohorts born at $z$ = 1.5-4, followed at every snapshot, and of the late control cohorts; dotted: $C$ of
    # randomly oriented orbits for the median cohort size.

    # %%
    a = ax[0, 2]
    # Early cohorts (every-snapshot cutouts), time since birth dt [Gyr] rounded to 0.1 Gyr.
    early = coh[coh.set == 'full'].assign(dtb=lambda t: (t.dt / 0.1).round() * 0.1)
    s_ = gal_median(early, 'dtb', 'C')
    a.fill_between(s_.index, s_.lo, s_.hi, color='C3', alpha=0.15, lw=0)
    a.plot(s_.index, s_.m, '-o', color='C3', ms=2.5, lw=1.2, label='born at z = 1.5-4')
    late = coh[(coh.set == 'track') & (coh.dt < 0.75)].assign(dtb=lambda t: np.where(t.dt < 0.2, 0.0, 0.45))  # birth and the next snapshot
    s_ = gal_median(late, 'dtb', 'C')
    # Late control: two points (birth at 0, next snapshot plotted at 0.45 Gyr) with 16-84 per cent bars.
    a.errorbar(s_.index, s_.m, yerr=[s_.m - s_.lo, s_.hi - s_.m], fmt='s-', color='C0', ms=3.5, capsize=2, label='born at z = 0.05-0.5')
    n_med = early.groupby(['halo', 's_birth']).n0.first().median()   # median number of stars per early cohort
    a.axhline(1 / np.sqrt(n_med), color='0.5', ls=':', lw=0.8)  # C of n randomly oriented orbits ~ n^-1/2 (tests/test_sims.py)
    a.set(xlabel='time since birth [Gyr]', ylabel=r'orbital-plane coherence $C$', xlim=(-0.03, 0.55), ylim=(0, 1.05))
    a.legend(fontsize=6, frameon=False, loc='center right', bbox_to_anchor=(1.0, 0.7))
    a.set_title('(c) orbits scrambled', loc='left', fontsize=8)

    # %% [markdown]
    # ### Legend, file and the numbers of the text
    # The text's numbers are written by `scripts/paper_text_sims_birth.py` from the same tables.

    # %%
    # Shared legend above the panels: proxy handles for the line styles and markers of (a) and (b).
    hs = [plt.Line2D([], [], color='C0', lw=1.4), plt.Line2D([], [], color='C2', lw=1.4, marker='o', ms=2.5), plt.Line2D([], [], color='0.3', ls='-'),
          plt.Line2D([], [], color='0.3', ls='--'), plt.Line2D([], [], color='k', marker='s', ls='', ms=4), plt.Line2D([], [], color='k', marker='v', ls='', mfc='w', ms=4), plt.Line2D([], [], color='k', ls=':', lw=0.8)]
    fig.legend(hs, ['(a) TNG50 stars at birth', '(a) star-forming gas', r'$v_\phi$', r'$\sigma_z$', r'LMC $v_{\rm rot}$', r'LMC $\sigma_z$ (today)',
                    r'(b) LMC $v_{\rm rot}/\sigma_z$, [Fe/H] $<-1.5$'], loc='upper center', ncol=7, fontsize=6, frameon=False)
    fig.savefig(ROOT / 'figures/paper_sims_heating.png')
    # Printed numbers: gas medians per snapshot, first four points of each cohort curve of (b), and the coherence curves of (c).
    print('gas by snapshot (medians):'); print(gas.groupby('snap')[['z', 'logMs', 'vphi', 'sz', 'cs', 'vs']].median().round(2).to_string())
    for lab, sel in (('> 10 Gyr', hist.age > 10), ('8-10 Gyr', (hist.age > 8) & (hist.age <= 10)), ('< 4 Gyr', hist.age < 4)):
        print(lab, gal_median(hist[sel], 'dtb', 'vs').m.head(4).round(2).to_dict())
    print('coherence early', gal_median(early, 'dtb', 'C').m.round(2).to_dict(), ' late', gal_median(late, 'dtb', 'C').m.round(2).to_dict(),
          ' noise', round(1 / np.sqrt(n_med), 3))
