"""Corrected giant counts and an equal-parent APOGEE/BOSS comparison (§2.2/2.3).

Same common parent = Gaia IDs in covered BOSS and APOGEE eligible samples;
NOT just stars with both spectra. Pixel bootstrap uses the same draws for both
surveys, keeping fitted selection weights fixed (conditional counting errors).
Counts are per bin; first and last bins include all [Fe/H]<-2 and >=0.
No isochrone conversion or foreground subtraction is applied to this figure.

Inputs: lmcold.mass_samples.samples(), i.e. the BOSS and APOGEE mass samples (velocity-selected giants with selection weights w;
BOSS [Fe/H] calibrated to the APOGEE scale) from data/boss_weights.parquet (sources.yaml release_boss_weights),
data/apogee_weights.parquet (release_apogee_weights), data/spec_giants.parquet and results/boss_calibration.json, and the
common covered parent (Gaia source_id in a covered BOSS selection group and a covered APOGEE parent).
Method: weighted counts in the ten reporting [Fe/H] bins (lmcold.mass_samples.FEH_EDGES, open ends) per nside-16 HEALPix pixel;
500 multinomial redraws of the pixels (same draws for all four count sets) give the errors.
Outputs:
- results/giant_counts.csv, one row per [Fe/H] bin: feh_lo, feh_hi; boss_raw (unweighted BOSS members), boss_weighted (BOSS
  with w), boss_common and apogee_common (weighted, common parent only), boss_common_raw and apogee_common_raw (unweighted),
  ratio = boss_common / apogee_common with bootstrap ratio_p16, ratio_p84; *_sd: bootstrap standard deviation of each count.
- results/giant_counts.json: sample sizes and the total common-parent BOSS/APOGEE ratio with its 16th-84th percentiles.
- figures/paper_giant_counts.png (not placed in the paper; its content is panel (b) of paper_mass_method, docs/figures.md).
Panels (1 x 3; x: [Fe/H] reporting bin centre, ticks '<-2' ... '>=0'):
(a) y: giants per [Fe/H] bin (log); BOSS covered sample, observed (unweighted, grey) and selection-corrected (blue); vertical
    bars: bootstrap 16th-84th percentiles.
(b) y: giants per bin (log); common parent, corrected BOSS (blue) and corrected APOGEE (orange).
(c) y: corrected BOSS / APOGEE in the common parent with bootstrap 16th-84th percentiles; bins with >= 5 unweighted stars in
    both surveys only.
Run: .venv/bin/python scripts/paper_giant_counts.py   (about 2 s on a laptop, docs/run_times.txt)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import pandas as pd
import json

from lmcold.mass_samples import samples, FEH_EDGES, FEH_CENTRES
from lmcold.layout import panel_grid
from lmcold.sources import ROOT

# %% [markdown]
# ### Load the data
# The BOSS and APOGEE mass samples with their selection weights, and the common covered parent
# (`lmcold.mass_samples.samples`, from `data/boss_weights.parquet`, `data/apogee_weights.parquet`,
# `data/spec_giants.parquet` and `results/boss_calibration.json`).

# %%
b,a,common = samples()   # BOSS and APOGEE mass samples (DataFrames with w, feh [dex], ra, dec [deg]) and common-parent Gaia ids

# %% [markdown]
# ### Four count sets
# BOSS unweighted and weighted, and both surveys weighted within the common parent.

# %%
bc,ac = b[b.source_id.isin(common)],a[a.source_id.isin(common)]   # BOSS and APOGEE members inside the common parent
# Four count sets (sample, weight per star): BOSS unweighted, BOSS weighted, common-parent BOSS and APOGEE weighted.
sets=[(b,np.ones(len(b))), (b,b.w.to_numpy()), (bc,bc.w.to_numpy()), (ac,ac.w.to_numpy())]

# %% [markdown]
# ### Counts and pixel bootstrap
# Weighted counts per HEALPix pixel and [Fe/H] bin, and their sums over 500
# bootstrap draws of the pixels.

# %%
# nside-16 HEALPix pixels (~3.7 deg), nested ordering; upix: every pixel occupied by any set, shared by all sets.
pixels=[hp.ang2pix(16,t.ra.to_numpy(),t.dec.to_numpy(),lonlat=True,nest=True) for t,_ in sets]
upix=np.unique(np.concatenate(pixels))
rng=np.random.default_rng(20261026)
# Pixel bootstrap: each of 500 draws gives the number of times every pixel is picked (len(upix) picks); shape (500, n_pix).
draws=rng.multinomial(len(upix),np.ones(len(upix))/len(upix),size=500)
counts,boots=[],[]
for (t,w),pix in zip(sets,pixels):
    j=np.digitize(t.feh,FEH_EDGES)-1   # reporting [Fe/H] bin index 0..9 of each star (open end bins)
    mat=np.zeros((len(upix),len(FEH_CENTRES)))  # weighted counts per pixel x [Fe/H] bin
    np.add.at(mat,(np.searchsorted(upix,pix),j),w)
    counts.append(mat.sum(axis=0))   # total count per [Fe/H] bin
    boots.append(draws@mat)  # (500, n_bins): counts of each bootstrap draw
counts=np.array(counts);boots=np.array(boots)  # (4, n_bins) and (4, 500, n_bins), in the order of `sets`

# %% [markdown]
# ### Ratios and tables
# The BOSS/APOGEE ratio in the common parent with bootstrap percentiles; written to
# `results/giant_counts.csv` and `results/giant_counts.json`.

# %%
# Unweighted common-parent counts per bin, BOSS and APOGEE.
rawb=np.histogram(bc.feh,FEH_EDGES)[0];rawa=np.histogram(ac.feh,FEH_EDGES)[0]
# Weighted BOSS / APOGEE ratio per bin, and per bootstrap draw (NaN where the APOGEE draw is empty); shape (500, n_bins).
with np.errstate(divide='ignore',invalid='ignore'):
    ratio=counts[2]/counts[3]
    bratio=np.where(boots[3]>0,boots[2]/boots[3],np.nan)
# 16th and 84th percentiles of the bootstrap ratio, only for bins with at least one finite draw (NaN otherwise).
low,high=np.full(len(FEH_CENTRES),np.nan),np.full(len(FEH_CENTRES),np.nan)
has_ratio=np.any(np.isfinite(bratio),axis=0)
low[has_ratio],high[has_ratio]=np.nanpercentile(bratio[:,has_ratio],[16,84],axis=0)
# Table of counts, ratios and bootstrap SDs per [Fe/H] bin (columns in the module docstring).
t=pd.DataFrame({'feh_lo':FEH_EDGES[:-1],'feh_hi':FEH_EDGES[1:],
                'boss_raw':counts[0],'boss_weighted':counts[1],'boss_common':counts[2],'apogee_common':counts[3],
                'boss_common_raw':rawb,'apogee_common_raw':rawa,'ratio':ratio,'ratio_p16':low,'ratio_p84':high})
for i,name in enumerate(['boss_raw','boss_weighted','boss_common','apogee_common']):
    t[name+'_sd']=boots[i].std(axis=0)
t.to_csv(ROOT/'results/giant_counts.csv',index=False)
# Sample sizes and the total common-parent ratio with its bootstrap 16th-84th percentiles.
summary=dict(common_parent=len(common),boss_members=len(b),apogee_members=len(a),boss_common=len(bc),apogee_common=len(ac),
             common_total_ratio=float(counts[2].sum()/counts[3].sum()),
             common_total_ratio_p16_p84=np.percentile(boots[2].sum(axis=1)/boots[3].sum(axis=1),[16,84]).tolist())
(ROOT/'results/giant_counts.json').write_text(json.dumps(summary,indent=2))

# %% [markdown]
# ### The figure
# Counts per [Fe/H] bin before and after the correction, in the common parent, and their ratio.

# %%
plt.style.use(ROOT/'src/lmcold/paper.mplstyle')
fig,axes=panel_grid(1,3,width=7.,left=.5,right=.08,hgap=.48,bottom=.45,top=.65)
blue,orange='#2878b5','#d75a20'
# Panels (a) and (b): indices into `sets`, legend labels and colours.
for ax,inds,labels,cols in [(axes[0,0],[0,1],['Observed','Selection corrected'],['0.6',blue]),
                           (axes[0,1],[2,3],['BOSS','APOGEE'],[blue,orange])]:
    for i,label,color in zip(inds,labels,cols):
        # Bootstrap 16th-84th percentile bars and the counts, in bins with a positive count (log axis).
        lo,hi=np.percentile(boots[i],[16,84],axis=0)
        positive=counts[i]>0
        ax.vlines(FEH_CENTRES[positive],lo[positive],hi[positive],color=color,lw=.8)
        ax.plot(FEH_CENTRES[positive],counts[i][positive],'o-',ms=2.5,color=color,label=label,lw=.8)
    ax.set(yscale='log',ylabel='Giants per metallicity bin')
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.1),borderaxespad=0)
# Ratios are shown only where both surveys have >= 5 unweighted common-parent stars and the bootstrap interval is defined.
valid=(rawb>=5)&(rawa>=5)&np.isfinite(low)&np.isfinite(high)
ax=axes[0,2]
# Quantile intervals plotted directly; their midpoint need not equal the data ratio.
ax.vlines(FEH_CENTRES[valid],low[valid],high[valid],color=blue,lw=.8)
ax.plot(FEH_CENTRES[valid],ratio[valid],'o',color=blue,ms=3)
ax.axhline(1,color='0.5',ls='--',lw=.7)
ax.set(ylabel='Corrected BOSS / APOGEE',ylim=(0,max(2.0,float(np.max(high[valid]))*1.05)))
for ax,title in zip(axes[0],['(a) BOSS covered sample','(b) Common parent sample','(c) Common parent sample']):
    ax.set(title=title,xlabel=r'[Fe/H]',xlim=(-2.45,.45),xticks=[-2.25,-1.5,-.5,.25],xticklabels=['<−2','−1.5','−0.5','≥0'])
fig.savefig(ROOT/'figures/paper_giant_counts.png')
print(json.dumps(summary,indent=2))
