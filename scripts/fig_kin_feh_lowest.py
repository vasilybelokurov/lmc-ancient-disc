"""Figure of the check of the most metal-poor giants (scripts/kin_feh_lowest.py, which makes the tables): (a) v_rot(4 kpc) / sigma_z
against the median measured [Fe/H] of the narrow bins inside the paper's lowest bin (geometry held at that bin's fit; errors
include the spread over the held parameters), with the paper's five bins fitted on their own (results/kin_support_ellipsoid.csv);
(b) the null test for measured [Fe/H] < -2.0: v_rot / sigma_z fitted to mocks in which a fraction (purity) of the stars are truly
metal-poor with a given rotation and the rest are interlopers from -1.75..-1.5, without (filled) and with (open) a 15 per cent hot,
non-rotating component, against the observed range (grey band, 16th-84th percentiles).

Reads results/kin_feh_lowest.csv, kin_feh_lowest_null.csv and kin_support_ellipsoid.csv. Writes figures/fig_kin_feh_lowest.png.
Diagnostic figure (KB chapter 'The most metal-poor stars'), not in the paper.

Run: .venv/bin/python scripts/fig_kin_feh_lowest.py   (a few seconds)
"""
# %% [markdown]
# ### Settings
# The bins and the null-test grid are those of scripts/kin_feh_lowest.py.

# %%
import matplotlib

matplotlib.use('Agg')
import numpy as np
import pandas as pd

from kin_feh_lowest import BINS, HOT, PURITY, RATIO
from lmcold.layout import panel_grid
from lmcold.sources import ROOT

# %% [markdown]
# ### Load the data and draw the figure
# The three tables; then (a) the ratio against the median measured [Fe/H] and (b) the observed ratio of measured [Fe/H] < -2.0
# against the null mocks.

# %%
t = pd.read_csv(ROOT / 'results/kin_feh_lowest.csv').set_index('sample')   # one row per fitted sample (bins and subsets)
nt = pd.read_csv(ROOT / 'results/kin_feh_lowest_null.csv')                  # one row per fitted null mock
# The paper's five giant bins, each fitted on its own (stage 1 of the ellipsoid run), for reference.
ref = pd.read_csv(ROOT / 'results/kin_support_ellipsoid.csv').query("population.str.startswith('giants')")
fig, ax = panel_grid(1, 2, width=7.0, top=1.0, hgap=0.75)   # two equal panels, room above for the keys

# (a) The four narrow bins: median and 16th-84th percentiles of v_rot(4 kpc) / sigma_z, the posterior spread combined in
# quadrature with the spread over the held parameters (held_spread).
a = ax[0, 0]
narrow = t.loc[[f'{lo} < [Fe/H] < {hi}'.replace('-9 < ', '') for lo, hi in BINS[:4]]]
err = np.hypot([narrow.ratio_z - narrow.ratio_z_p16, narrow.ratio_z_p84 - narrow.ratio_z], narrow.held_spread.values)
a.errorbar(narrow.feh_med, narrow.ratio_z, err, fmt='o', color='C3', label='narrow bins of measured [Fe/H], geometry held')
a.errorbar(ref.feh_med, ref.ratio_z, [ref.ratio_z - ref.ratio_z_p16, ref.ratio_z_p84 - ref.ratio_z], fmt='s', mfc='none',
           color='k', label='paper bins (each fitted on its own)')
a.set(xlabel='median measured [Fe/H]', ylabel=r'$v_{\rm rot}(4\,{\rm kpc})/\sigma_z$', ylim=(0, None))
a.legend(loc='lower left', bbox_to_anchor=(0, 1.02), fontsize=7, frameon=False)

# (b) The null test for all stars below -2: the measured value as a grey band (16th-84th percentiles), and every fitted mock
# as a point at its purity, shifted sideways by its true ratio (colour) and by the hot fraction (filled: none; open: HOT[-1]).
b = ax[0, 1]
name = '[Fe/H] < -2.0'
o = t.loc[name]
b.axhspan(o.ratio_z_p16, o.ratio_z_p84, color='0.85', label=f'observed, measured {name}')
for jr, ratio in enumerate(RATIO):
    for hot, mk in zip(HOT, ('o', 's')):
        s = nt[(nt.bin == name) & (nt.ratio == ratio) & (nt.hot == hot)]   # the mocks of this true ratio and hot fraction
        # x position: index of the purity on the axis, offset by 0.12 per true ratio and 0.05 for the hot mocks.
        x = s.purity.map({p: i for i, p in enumerate(PURITY)}) + 0.12 * jr - 0.2 + 0.05 * (hot > 0)
        b.plot(x, s.fit_ratio, mk, color=f'C{jr}', ms=3, mfc='none' if hot else f'C{jr}',
               label=f'true ratio {ratio}' + (f', {hot:.0%} hot' if hot else ''))
b.set(xticks=range(len(PURITY)), xticklabels=[str(p) for p in PURITY], xlabel='purity (fraction of stars truly below -2)',
      ylabel=r'fitted $v_{\rm rot}/\sigma_z$ of the mocks', ylim=(0, None))
b.legend(loc='lower left', bbox_to_anchor=(0, 1.02), fontsize=6, frameon=False, ncol=2)
fig.savefig(ROOT / 'figures/fig_kin_feh_lowest.png', dpi=200)


