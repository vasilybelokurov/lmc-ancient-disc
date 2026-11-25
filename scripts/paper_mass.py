"""Paper figures for Sect. 2.3: stellar mass per [Fe/H] (main text) and the survey check in shared field cores (appendix).

Run build_giant_mass.py and build_present_mass.py first. Bands are model alternatives (age priors, alpha, foreground, extinction,
inclination), not probabilities; bars are 16th-84th percentiles of the pixel and response bootstrap with weights and
conversions fixed. The cumulative axis is metallicity, not age or lookback time.
paper_mass.png: (a) mass per bin, (b) mass below [Fe/H]; formed and present (living stars plus remnants) on the same axes.
paper_mass_cores.png: (a) BOSS and APOGEE formed mass per bin in the ten shared field cores, (b) their ratio.

Inputs (results/, from scripts/build_giant_mass.py and scripts/build_present_mass.py):
- giant_present_mass.csv: per reporting [Fe/H] bin (lmcold.mass_samples.FEH_EDGES, open ends), formed_mass, cumulative,
  present_mass, present_cumulative [Msun], each with _p16/_p84 (bootstrap; for formed mass the columns mass_p16/mass_p84) and
  _model_min/_model_max (range over the model alternatives).
- giant_mass_bootstrap.csv: formed mass per bin [Msun] of each bootstrap draw (columns = bin centres).
- giant_mass_common.csv: summed over the ten shared APOGEE/BOSS field cores, boss_mass and apogee_mass [Msun] per bin, their
  ratio with 16th-84th percentiles of a resampling of the cores, and the unweighted star counts boss_raw, apogee_raw.
- giant_mass_fields.csv: mass per field core, survey, age prior, alpha model and bin (for the model ranges of the cores).
- mass_conversion_factors.csv: giants per Msun formed (eta) per bin; the 'full' BOSS conversion with the reference prior and the
  measured alpha trend is used here. mass_alpha_trend.csv: median APOGEE [alpha/M] per [Fe/H] bin.
- data/boss_parent.parquet (release_boss_parent) and data/xp_andrae.parquet (scripts/build_xp_andrae.py, from sources.yaml
  andrae23_table1): Gaia XP [M/H] of the BOSS eligible parent; data/clusters_grocholski06.csv (clusters_grocholski06).
paper_mass.png panels (1 x 3; x: [Fe/H]):
(a) y: mass per reporting bin [Msun] (log), x at the bin centres (ticks '<-2' ... '>=0'); blue: formed, orange: present (living
    stars and remnants); bars: bootstrap 16th-84th percentiles; bands: model range.
(b) y: mass below [Fe/H] [Msun] (log), x at each bin's upper edge, the last point ('All', placed at 0.3) the total; same coding.
(c) y: fraction per dex (log, 5e-3 to 5), x -2.5 to 0.5; steps: our formed mass per bin divided by the total and the bin width
    (outer bins closed at -2.5 and 0.5), band from the bootstrap; purple dashed: Gaia XP [M/H] of the BOSS parent converted to
    [Fe/H] and weighted by 1/eta (i.e. as formed mass); black: Grocholski+06 clusters by number; both Gaussian KDEs with bootstrap
    16th-84th percentile bands.
paper_mass_cores.png panels (1 x 2; x: [Fe/H] reporting bin):
(a) y: formed mass per bin [Msun] (log) in the shared cores, BOSS (blue) and APOGEE (orange), each with its own CMD conversion;
    bands: range over age priors and alpha models.
(b) y: BOSS / APOGEE (log, 0.3-3), core-resampling 16th-84th percentiles, grey band: model range; bins with >= 5 unweighted stars
    in both surveys.
Placement: paper_mass.png Sect. 2.3 main text; paper_mass_cores.png appendix (docs/figures.md).
Run: .venv/bin/python scripts/paper_mass.py
Run time: 10 s on a MacBook (M-series), docs/run_times.txt.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from lmcold.layout import panel_grid
from lmcold.mass_conversion import REFERENCE_PRIOR
from lmcold.mass_samples import FEH_CENTRES, FEH_EDGES
from lmcold.sources import DATA, ROOT

# %% [markdown]
# ### Load the data
# The mass tables written by `scripts/build_giant_mass.py` and `scripts/build_present_mass.py` (`results/giant_present_mass.csv`,
# `giant_mass_common.csv`, `giant_mass_fields.csv`, `giant_mass_bootstrap.csv`, `mass_conversion_factors.csv`,
# `mass_alpha_trend.csv`); the BOSS parent with its Gaia XP [M/H] (`data/boss_parent.parquet`, `data/xp_andrae.parquet`, Zenodo
# bundle 1); and the cluster [Fe/H] of Grocholski et al. (2006), `data/clusters_grocholski06.csv`.

# %%
# results/giant_present_mass.csv: formed and present mass per [Fe/H] bin and below it, bootstrap percentiles and model ranges
t=pd.read_csv(ROOT/'results/giant_present_mass.csv')
# results/giant_mass_common.csv: BOSS and APOGEE formed mass in the ten shared field cores, and their ratio
c=pd.read_csv(ROOT/'results/giant_mass_common.csv')
# results/giant_mass_fields.csv: mass per field core, survey and [Fe/H] bin for each model (prior x alpha)
fields=pd.read_csv(ROOT/'results/giant_mass_fields.csv')
# results/giant_mass_bootstrap.csv: formed mass per [Fe/H] bin in each bootstrap draw (one row per draw)
boot=pd.read_csv(ROOT/'results/giant_mass_bootstrap.csv').to_numpy()
# results/mass_conversion_factors.csv: Msun formed per giant (eta) per bin; the full parent, reference prior, measured alpha
conv=pd.read_csv(ROOT/'results/mass_conversion_factors.csv').query("field=='full' and prior==@REFERENCE_PRIOR and alpha_model=='measured'")
# results/mass_alpha_trend.csv: the APOGEE [alpha/Fe] trend against [Fe/H]
trend=pd.read_csv(ROOT/'results/mass_alpha_trend.csv')
# data/boss_parent.parquet (eligible stars) joined with data/xp_andrae.parquet: XP [M/H] of the BOSS eligible parent (finite
# values only: build_xp_andrae sets NaN outside its cool-giant cuts)
xp=pd.read_parquet(DATA/'boss_parent.parquet',columns=['source_id','eligible']).query('eligible').merge(
    pd.read_parquet(DATA/'xp_andrae.parquet',columns=['source_id','mh_xp']),on='source_id').mh_xp.dropna().to_numpy()
# data/clusters_grocholski06.csv: Ca II triplet [Fe/H] of LMC clusters
cl=pd.read_csv(DATA/'clusters_grocholski06.csv').feh.dropna().to_numpy()

# %% [markdown]
# ### The mass tables
# Formed and present mass per bin with bootstrap percentiles and model ranges, the masses of the shared field cores, and the
# model range of the cores over the age and alpha alternatives.

# %%
plt.style.use(ROOT/'src/lmcold/paper.mplstyle')
# Mass summed over the cores for each model (prior x alpha) and survey; model_range: min/max over models of BOSS, APOGEE, ratio.
models=fields.groupby(['prior','alpha_model','survey','feh']).mass.sum().unstack('survey')
models['ratio']=models.BOSS/models.APOGEE
model_range=models.groupby('feh').agg(['min','max'])
# %% [markdown]
# ### Panels (a) and (b): mass per bin and mass below [Fe/H]

# %%
blue,orange='#2878b5','#d75a20'
fig,axes=panel_grid(1,3,width=7,left=.55,right=.1,hgap=.6,bottom=.5,top=.6)
# (a) mass per bin at the bin centres; (b) cumulative mass at the upper bin edges, the last (open) bin plotted at 0.3 as 'All'.
for ax,col,x in zip(axes[0,:2],['mass','cumulative'],[FEH_CENTRES,np.r_[t.feh_hi.to_numpy()[:-1],.3]]):
    for name,value,color,lab in [('formed','formed_mass' if col=='mass' else 'cumulative',blue,'formed'),
                                 ('present','present_'+col,orange,'present (living stars and remnants)')]:
        pre='' if name=='formed' else 'present_'
        ax.fill_between(x,t[pre+col+'_model_min'],t[pre+col+'_model_max'],color=color,alpha=.18,lw=0)
        ax.vlines(x,t[pre+col+'_p16'],t[pre+col+'_p84'],color=color,lw=.8)
        ax.plot(x,t[value],'-o',ms=3,color=color,label=lab)
    ax.set(yscale='log',xlabel=r'[Fe/H]',xlim=(-2.45,.45))
    ax.set_xticks([-2.25,-1.5,-.5,.25] if col=='mass' else [-2,-1.5,-.5,.3])
    ax.set_xticklabels(['<−2','−1.5','−0.5','≥0'] if col=='mass' else ['−2','−1.5','−0.5','All'])
axes[0,0].set(title='(a) Mass per metallicity bin',ylabel=r'Mass per bin ($M_\odot$)')
axes[0,1].set(title='(b) Mass below [Fe/H]',ylabel=r'Mass below [Fe/H] ($M_\odot$)')
# %% [markdown]
# ### Panel (c): comparison with Gaia XP and the clusters
# Our formed mass per dex, the Gaia XP metallicities of the BOSS parent weighted as mass, and the clusters by number.

# %%
# (c) Distributions per dex of [Fe/H]. Ours: formed mass per bin over the bin width (binned: the response and the split above -1
# work in bins), band from the bootstrap (results/giant_mass_bootstrap.csv). Gaia XP [M/H] of the BOSS parent (Andrae+23; to [Fe/H]
# with the alpha trend), each star weighted by 1/eta of its bin (results/mass_conversion_factors.csv, full parent, reference), and LMC
# clusters with Ca II triplet [Fe/H] (Grocholski+06) by number: Gaussian KDEs (bandwidths KDE_BW), bands from NBOOT_KDE bootstraps.
KDE_BW,NBOOT_KDE=dict(xp=.05,clusters=.1),300  # KDE bandwidths [dex] and number of bootstrap draws of the KDEs
rng=np.random.default_rng(5)
grid=np.linspace(-2.6,.5,311)  # [Fe/H] grid of the KDEs, 0.01-dex step
ax=axes[0,2]
edges=np.r_[-2.5,FEH_EDGES[1:-1],.5]  # reporting bins with the open ends closed at -2.5 and 0.5 for plotting
dens=lambda m: m/m.sum()/np.diff(edges)  # noqa: E731  fraction of the total mass per dex in each bin
lo,hi=np.percentile([dens(b) for b in boot],[16,84],axis=0)
ax.fill_between(edges,np.r_[lo,lo[-1]],np.r_[hi,hi[-1]],step='post',color=blue,alpha=.25,lw=0)
ax.stairs(dens(t.formed_mass.to_numpy()),edges,color=blue,lw=1.5,label='this work, formed mass')


def kde(x,w,bw):
    """Weighted Gaussian KDE on grid, through a 0.01-dex histogram; normalised to unit integral.

    Parameters
    ----------
    x : numpy.ndarray
        [Fe/H] values [dex].
    w : numpy.ndarray
        Weights per value.
    bw : float
        Gaussian standard deviation [dex].

    Returns
    -------
    numpy.ndarray
        Density per dex on `grid`; values outside the grid are dropped before smoothing.
    """
    h=np.histogram(x,np.r_[grid-.005,grid[-1]+.005],weights=w)[0]
    d=gaussian_filter1d(h.astype(float),bw/.01,mode='constant')
    return d/(d.sum()*.01)


# [Fe/H] = [M/H] - log10(0.638 x 10^[alpha/Fe] + 0.362) (Salaris, Chieffi & Straniero 1993; refs.bib Salaris1993); the APOGEE
# alpha trend, tabulated against [Fe/H], is interpolated at [M/H].
feh_xp=xp-np.log10(.638*10**np.interp(xp,trend.feh,trend.alpha)+.362)
w_xp=1/conv.sort_values('feh').eta.to_numpy()[np.digitize(feh_xp,FEH_EDGES)-1]  # Msun formed per giant of the star's bin
for x,w,bw,colour,ls,lab in [(feh_xp,w_xp,KDE_BW['xp'],'#7b3294','--',f'Gaia XP, BOSS parent ({len(xp):,} stars), as mass'),
                         (cl,np.ones(len(cl)),KDE_BW['clusters'],'k','-',f'clusters, Ca II triplet ({len(cl)}), by number')]:
    bs=[kde(x[i],w[i],bw) for i in (rng.integers(0,len(x),len(x)) for _ in range(NBOOT_KDE))]
    ax.fill_between(grid,*np.percentile(bs,[16,84],axis=0),color=colour,alpha=.15,lw=0)
    ax.plot(grid,kde(x,w,bw),color=colour,lw=1.2,ls=ls,label=lab)
ax.set(yscale='log',ylim=(5e-3,5),xlim=(-2.5,.5),xlabel='[Fe/H]',ylabel='fraction per dex',title='(c) Comparison')
ax.legend(loc='lower center',bbox_to_anchor=(.5,1.1),frameon=False,fontsize=6.5)
h,l=axes[0,0].get_legend_handles_labels()
fig.legend(h,l,loc='upper center',ncol=2,bbox_to_anchor=(.35,1))
fig.savefig(ROOT/'figures/paper_mass.png')

# %% [markdown]
# ### BOSS against APOGEE in the field cores

# %%
# --- paper_mass_cores.png: BOSS against APOGEE in the ten shared field cores ---
fig,axes=panel_grid(1,2,width=7,left=.6,right=.1,hgap=.65,bottom=.5,top=.7)
for survey,color in [('boss',blue),('apogee',orange)]:
    m=c[survey+'_mass'];valid=m>0  # bins with mass (log axis)
    axes[0,0].fill_between(FEH_CENTRES[valid],model_range[survey.upper()]['min'].to_numpy()[valid],model_range[survey.upper()]['max'].to_numpy()[valid],color=color,alpha=.15)
    axes[0,0].plot(FEH_CENTRES[valid],m[valid],'-o',ms=3,color=color,label=survey.upper())
axes[0,0].set(yscale='log',title='(a) Same field cores, separate CMD conversions',ylabel=r'Formed mass per bin ($M_\odot$)')
valid=(c.boss_raw>=5)&(c.apogee_raw>=5)&np.isfinite(c.ratio_p16)&np.isfinite(c.ratio_p84)
axes[0,1].fill_between(FEH_CENTRES[valid],model_range['ratio']['min'].to_numpy()[valid],model_range['ratio']['max'].to_numpy()[valid],color='0.7',alpha=.3)
axes[0,1].vlines(FEH_CENTRES[valid],c.ratio_p16[valid],c.ratio_p84[valid],color=blue,lw=.8)
axes[0,1].plot(FEH_CENTRES[valid],c.ratio[valid],'o',ms=3,color=blue)
axes[0,1].axhline(1,color='0.5',ls='--',lw=.7)
axes[0,1].set(title='(b) Common-footprint mass ratio',ylabel='BOSS / APOGEE',yscale='log',ylim=(.3,3),yticks=[.5,1,2],yticklabels=['0.5','1','2'])
axes[0,1].yaxis.set_minor_formatter(NullFormatter())
for ax in axes[0]:
    ax.set(xlabel=r'[Fe/H]',xlim=(-2.45,.45),xticks=[-2.25,-1.5,-.5,.25],xticklabels=['<−2','−1.5','−0.5','≥0'])
h,l=axes[0,0].get_legend_handles_labels()
fig.legend(h,l,loc='upper center',ncol=2,bbox_to_anchor=(.5,.99))
fig.savefig(ROOT/'figures/paper_mass_cores.png')
print('Saved paper_mass.png and paper_mass_cores.png')
