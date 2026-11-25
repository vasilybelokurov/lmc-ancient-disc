"""Sizes of the LMC's populations: the selection-weighted surface density of the giants in each [Fe/H] bin and of the RR Lyrae
against in-plane radius, with the exponential scale lengths h that scripts/kin_vc_drift.py fits to them (results/kin_vc_drift_h.csv).

The surface densities are made exactly as in kin_vc_drift.py (its functions and settings are imported): BOSS giants of the mass
sample weighted by selection weight x coverage fill x (1 - Milky Way fraction), RR Lyrae members with weight 1, in 1-kpc annuli of
the in-plane radius R of the joint ellipsoid fit; h is the exponential fitted over H_FIT (2-8 kpc; inside 2 kpc the bar). The
half-mass radius of a projected exponential disc is R_e = RE_PER_H h; quoting it extrapolates the exponential inside 2 kpc.
These are present-day sizes: heating and radial migration since the stars formed may have changed them.

Reads the BOSS mass sample (lmcold.mass_samples), data/giant_foreground.parquet, data/rrab.parquet, results/kin_mcmc_data_ellipsoid.csv
and results/kin_vc_drift_h.csv. Writes figures/fig_sizes.png: (a) surface density (each normalised to its fitted exponential at
R = 4 kpc) against R, the fits as lines over H_FIT, the region inside H_FIT[0] shaded; (b) h (filled) and R_e (open) against the
median [Fe/H] of each giant bin, the RR Lyrae at their median photometric [Fe/H]. Diagnostic figure (KB, Shape chapter).

Run: .venv/bin/python scripts/fig_sizes.py   (about a minute)
"""
# %% [markdown]
# ### Settings
# The annuli, fit range and radius function are imported from `scripts/kin_vc_drift.py`, so that the profiles are the ones its scale
# lengths were fitted to.

# %%
import matplotlib

matplotlib.use('Agg')
import numpy as np
import pandas as pd

from kin_vc_drift import H_FIT, R_EDGES, joint_params, radius
from lmcold import samples
from lmcold.kinfit import R_SIG
from lmcold.layout import panel_grid
from lmcold.mass_samples import samples as mass_samples
from lmcold.sources import DATA, ROOT

RE_PER_H = 1.678   # half-mass radius / scale length of a projected exponential disc
R_NORM = R_SIG     # kpc, where each profile is normalised (the radius of the kinematic quantities)

# %% [markdown]
# ### Load the data
# The joint-fit disc frame, the BOSS mass sample with its weights, the RR Lyrae members and the fitted scale lengths.

# %%
# Disc frame: stage-2 posterior medians of the joint ellipsoid fit (shared parameters plus those of the first
# giant bin); only the centre, inclination and position angle enter radius()
frame = joint_params()['giants [Fe/H] < -1.5']
# BOSS mass sample (first of the three returned tables), with the coverage fill and Milky Way fraction per star
b, _, _ = mass_samples()
fg = pd.read_parquet(DATA / 'giant_foreground.parquet')
b = b.merge(fg[['source_id', 'fill', 'f_mw']], on='source_id', validate='one_to_one')
# W: total weight = selection weight x coverage fill x (1 - Milky Way fraction); R: in-plane radius [kpc]
b['W'] = b.w * b['fill'] * (1 - b.f_mw)
b['R'] = radius(frame, b.ra.to_numpy(), b.dec.to_numpy())
# RR Lyrae members; H: fitted scale lengths h [kpc] with percentiles, indexed by population
rr = pd.read_parquet(DATA / 'rrab.parquet')
rr = rr[rr.member]
H = pd.read_csv(ROOT / 'results/kin_vc_drift_h.csv').set_index('population')

# %% [markdown]
# ### Surface densities
# For each population the weighted counts per annulus over the annulus area, normalised to its fitted exponential at R_NORM.

# %%
# Annulus centres [kpc] and areas [kpc^2]; fit_range: annuli whose centre lies inside H_FIT
mid, area = .5 * (R_EDGES[1:] + R_EDGES[:-1]), np.pi * np.diff(R_EDGES ** 2)
fit_range = (mid > H_FIT[0]) & (mid < H_FIT[1])
# (name, R [kpc], weights, median [Fe/H]) of each giant [Fe/H] bin of lmcold.samples, then the RR Lyrae
pops = [(f'giants {lab}', b[(b.feh >= lo) & (b.feh < hi)].R.to_numpy(), b[(b.feh >= lo) & (b.feh < hi)].W.to_numpy(),
         np.median(b[(b.feh >= lo) & (b.feh < hi)].feh)) for lab, (lo, hi) in zip([lab for lab, _ in samples.giants()], samples.FEH_BINS)]
pops.append(('RR Lyrae', radius(frame, rr.ra.to_numpy(), rr.dec.to_numpy()), np.ones(len(rr)), np.nanmedian(rr.feh)))   # 4 per cent have no [Fe/H]
prof = {}
for name, R, w, feh in pops:
    sigma = np.histogram(R, R_EDGES, weights=w)[0] / area      # weighted stars per kpc^2
    # Fitted scale length [kpc]; k: annuli in the fit range with stars
    h = H.loc[name, 'h']
    k = fit_range & (sigma > 0)
    a0 = np.mean(np.log(sigma[k]) + mid[k] / h)                # intercept of ln Sigma = a0 - R / h at the fitted slope
    prof[name] = (sigma / np.exp(a0 - R_NORM / h), h, feh)

# %% [markdown]
# ### Figure
# (a) The normalised profiles and their exponentials; (b) the scale lengths and the extrapolated half-mass radii against [Fe/H].

# %%
# Two equal panels: a = profiles, c = sizes; one colour per population (blues metal-poor, oranges metal-rich)
fig, ax = panel_grid(1, 2, width=7.0, left=0.65, top=0.6, hgap=0.7)
a, c = ax[0, 0], ax[0, 1]
colours = dict(zip(prof, ['#08306b', '#2171b5', '#6baed6', '#fd8d3c', '#d94801', 'k']))
# Radii [kpc] at which the fitted exponentials are drawn
x = np.linspace(*H_FIT, 50)
for name, (s, h, feh) in prof.items():
    # Points: measured profile (non-empty annuli); line: exp(-(R - R_NORM)/h), equal to 1 at R_NORM
    ls = '--' if name == 'RR Lyrae' else '-'
    a.semilogy(mid[s > 0], s[s > 0], 'o' if ls == '-' else 's', color=colours[name], ms=3)
    a.semilogy(x, np.exp(-(x - R_NORM) / h), ls, color=colours[name], lw=1, label=f'{name.replace("giants ", "")}: h = {h:.1f} kpc')
# Shade the region inside H_FIT[0] (not fitted); legend above the panel
a.axvspan(0, H_FIT[0], color='0.9', zorder=0)
a.set(xlabel='in-plane radius R [kpc]', ylabel=rf'$\Sigma(R)/\Sigma_{{\rm exp}}({R_NORM:g}\,{{\rm kpc}})$', xlim=(0, R_EDGES[-1]))
a.legend(loc='lower left', bbox_to_anchor=(0, 1.02), fontsize=6, frameon=False, ncol=2)
for name, (s, h, feh) in prof.items():
    # Filled: h with its 16th-84th percentiles; open: R_e = RE_PER_H h
    lo, hi = H.loc[name, 'h_p16'], H.loc[name, 'h_p84']
    mk = 's' if name == 'RR Lyrae' else 'o'
    c.errorbar(feh, h, [[h - lo], [hi - h]], fmt=mk, color=colours[name])
    c.plot(feh, RE_PER_H * h, mk, mfc='none', color=colours[name])
# Legend entries only (empty data), placed above the panel
c.plot([], [], 'ko', label='scale length h (fitted, 2-8 kpc)')
c.plot([], [], 'ko', mfc='none', label=r'$R_e=1.68\,h$ (extrapolated inside 2 kpc)')
c.set(xlabel='median [Fe/H]', ylabel='size [kpc]', ylim=(0, None))
c.legend(loc='lower left', bbox_to_anchor=(0, 1.02), fontsize=6, frameon=False)
fig.savefig(ROOT / 'figures/fig_sizes.png', dpi=200)
