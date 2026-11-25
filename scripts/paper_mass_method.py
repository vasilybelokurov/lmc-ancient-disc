"""Paper figure (Sect. 2.3): the two steps from giant counts to mass.

(a) Observable BOSS giants per formed mass against age at four model metallicities, one per prior range of [Fe/H]; thick bars:
16th-84th percentiles of the reference age weighting (age-metallicity relation of Povick+24, lmcold.mass_conversion.age_weights)
for the reporting [Fe/H] bin of each curve; thin bars: the window of prior A. The table also holds a representative
APOGEE selection (H<13.3, maglim=14.5; not an average over the field depths), for the appendix comparison.
results/age_weighting.json: mean and 16th-84th percentile ages of the reference weighting per prior [Fe/H] range (Table 2).
(b) BOSS metallicity response: corrected BOSS counts in the common parent divided by the corrected APOGEE counts, and by the
APOGEE counts passed through the response measured on the stars with both spectra (results/boss_response.json from
scripts/boss_response.py); bars are 16th-84th percentiles of the joint pixel and overlap-star bootstrap. Uses the treatment of isolated age-grid spikes of lmcold.isochrones, retaining both raw and adjusted
values in the output table. Photometric distance and depth from constants; the median true reddening of the BOSS parent
(lmcold.extinction) and the median reddening the carton removed; no disc distance offset.

Inputs:
- data/boss_parent.parquet (sources.yaml release_boss_parent): the BOSS eligible parent, for the median reddenings.
- PARSEC isochrones (sources.yaml parsec_gaiaEDR3 for BOSS, parsec_2mass_spitzer for APOGEE; read through lmcold.sources.path,
  which checks the sha256).
- results/mass_alpha_trend.csv (scripts/build_giant_mass.py): median APOGEE [alpha/M] (ASPCAP ALPHA_M) per [Fe/H] bin.
- results/boss_response.json (scripts/boss_response.py): key closure.all with the observed and predicted BOSS counts, the
  direct APOGEE counts and the bootstrap ratio percentiles per reporting [Fe/H] bin (lmcold.mass_samples.FEH_CENTRES).
Method: lmcold.isochrones.count_grid integrates the Kroupa IMF along each isochrone inside the survey's colour-magnitude
selection (BOSS: the mwm_magcloud_rgb_boss polygon and G <= 17.5; APOGEE: its targeting polygon to H = 13.3), at the
photometric LMC distance modulus with a 0.1-mag Gaussian depth (lmcold.constants), giving giants per solar mass formed (n_box);
smooth_age_spikes replaces isolated spikes of the age grid (raw values kept in n_box_raw).
Outputs:
- results/giant_conversion.csv: per survey, isochrone [M/H] (mh) and age: logage, age [Gyr], n_box_raw and n_box [giants per Msun
  formed], spike_adjusted, and the reddenings used (ebv, ebv_carton [mag, SFD scale]).
- results/age_weighting.json: per prior [Fe/H] range, mean age and 16th-84th percentile ages [Gyr] of the reference weighting.
- figures/paper_mass_method.png (appendix D; docs/figures.md).
Panels (1 x 2):
(a) x: age [Gyr], log, 0.4-13.5; y: BOSS giants per 10^6 Msun formed (log, 3-3000); one curve per isochrone [M/H] (-2.0, -1.0,
    -0.6, -0.2 nearest grid values), labelled with the corresponding [Fe/H]; thick horizontal bars: 16th-84th percentile ages of
    the reference (AMR) weighting, thin bars: the age window of prior A (bar heights arbitrary, stacked by curve).
(b) x: [Fe/H] reporting bin centre (-2.45 to -0.1); y: observed BOSS / expected, 0-2.6; grey: corrected BOSS / corrected APOGEE
    counts (no response); blue: corrected BOSS / APOGEE counts passed through the BOSS response, with bootstrap 16th-84th
    percentiles; bins with >= 3 weighted APOGEE giants.
Run: .venv/bin/python scripts/paper_mass_method.py
Run time: 3 s on a MacBook (M-series), docs/run_times.txt.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import json

from lmcold.isochrones import count_grid, smooth_age_spikes
from lmcold.mass_conversion import AGE_RANGES, REFERENCE_PRIOR, age_weights
from lmcold.mass_samples import FEH_CENTRES
from lmcold.layout import panel_grid
from lmcold.sources import ROOT, DATA, path
from lmcold import magcloud
from lmcold.constants import LMC_TARGET_MS, SMC_TARGET_MS
from lmcold.extinction import rc_map, true_ebv

# %% [markdown]
# ### Load the data
# The BOSS parent `data/boss_parent.parquet` (Zenodo bundle 1), the APOGEE [alpha/Fe] trend `results/mass_alpha_trend.csv`
# (written by `scripts/build_giant_mass.py`) and the closure test of `results/boss_response.json` (`scripts/boss_response.py`).
# Two further inputs are read where they are used: the Skowron et al. (2021) red-clump reddening map by
# `lmcold.extinction.rc_map()` in the reddening step, and the two PARSEC isochrone grids (`lmcold.sources.path`) one per survey
# inside the loop that counts giants per solar mass.

# %%
# data/boss_parent.parquet: the BOSS parent (eligible stars kept) with ra, dec [deg] and SFD E(B-V) (column ebv)
p = pd.read_parquet(DATA/'boss_parent.parquet').query('eligible')
# results/mass_alpha_trend.csv: the APOGEE [alpha/Fe] trend against [Fe/H]
trend=pd.read_csv(ROOT/'results/mass_alpha_trend.csv')
# results/boss_response.json: closure test on the common parent, all stars (observed, predicted, APOGEE counts per bin)
c=json.loads((ROOT/'results/boss_response.json').read_text())['closure']['all']

# %% [markdown]
# ### Reddening of the parent
# The median reddening that the carton removed and the median true reddening of the BOSS parent.

# %%
# Median carton (effective) and true reddening of the BOSS eligible parent. The carton's E(B-V) is SFD98 times 0.07 where
# E(B-V) >= 0.1 within 4.5 deg of the LMC or 2.5 deg of the SMC target centres in Magellanic Stream coordinates (the rule of
# lmcold.magcloud.rescaled_ebv, written out here); the true E(B-V) (lmcold.extinction.true_ebv) is the Skowron et al. (2021)
# red-clump E(V-I) / 1.237 inside their map and SFD outside it.
L,B=magcloud.ms_coordinates(p.ra.to_numpy(),p.dec.to_numpy(),np.zeros(len(p)),np.zeros(len(p)))[:2]
effective=np.where(((np.hypot(L-LMC_TARGET_MS[0],B-LMC_TARGET_MS[1])<=4.5)|(np.hypot(L-SMC_TARGET_MS[0],B-SMC_TARGET_MS[1])<=2.5))&(p.ebv.to_numpy()>=.1),.07*p.ebv,p.ebv)
ebv_carton=float(np.median(effective))
ebv=float(np.median(true_ebv(p.ra.to_numpy(),p.dec.to_numpy(),p.ebv.to_numpy(),rc_map())))
# %% [markdown]
# ### Giants per solar mass formed
# For the four isochrone metallicities of panel (a), the number of selected giants per solar mass formed at each age
# (`lmcold.isochrones.count_grid`), for BOSS and APOGEE; written to `results/giant_conversion.csv`.

# %%
parts=[]
for survey,key in [('BOSS','parsec_gaiaEDR3'),('APOGEE','parsec_2mass_spitzer')]:
    g=pd.read_parquet(path(key))
    mh=np.array(sorted(g.MH.unique()))
    # The isochrone metallicities nearest to [M/H] = -2, -1, -0.6, -0.2: one per prior [Fe/H] range.
    chosen=[mh[np.argmin(abs(mh-v))] for v in [-2.,-1.,-.6,-.2]]  # [Fe/H] bins <-1.5, -1.5..-1, -1..-0.5, >-0.5
    # Giants per Msun formed for each (mh, age); dmu = 0: no inclined-disc distance offset.
    t=smooth_age_spikes(count_grid(g[g.MH.isin(chosen)],survey,ebv,ebv_carton,0.))
    t['survey']=survey;t['ebv']=ebv;t['ebv_carton']=ebv_carton
    parts.append(t)
t=pd.concat(parts,ignore_index=True)
t.to_csv(ROOT/'results/giant_conversion.csv',index=False)
# %% [markdown]
# ### Ages of the reference weighting
# The 16th-84th percentile and mean ages of the formed mass per [Fe/H] range (the table of ages), written to
# `results/age_weighting.json`.

# %%
# Ages of the reference weighting per prior [Fe/H] range (Table 2): the sum of the reporting-bin weights over the range.
ages=np.sort(t.age.unique())
ranges={}
for name,(lo,hi) in {'<-1.5':(-9,-1.5),'-1.5..-1':(-1.5,-1.),'-1..-0.5':(-1.,-.5),'>-0.5':(-.5,9)}.items():
    w=sum(age_weights(ages,f,REFERENCE_PRIOR) for f in FEH_CENTRES if lo<f<hi)  # formed-mass weight per grid age
    # Percentile ages: linear interpolation of the cumulative weight over the sorted grid ages.
    ranges[name]=dict(mean=float(np.average(ages,weights=w)),p16_p84=np.interp([.16,.84],np.cumsum(w)/w.sum(),ages).tolist())
(ROOT/'results/age_weighting.json').write_text(json.dumps(ranges,indent=1))
print(json.dumps(ranges,indent=1))
# %% [markdown]
# ### Panel (a): efficiency against age

# %%
plt.style.use(ROOT/'src/lmcold/paper.mplstyle')
fig,axes=panel_grid(1,2,width=7,left=.6,right=.1,hgap=.65,bottom=.5,top=.75)
colors=['#d75a20','#33976b','#2878b5','#7b3294']
ax=axes[0,0]
# [M/H] -> [Fe/H] by inverting [M/H] = [Fe/H] + log10(0.638 x 10^[alpha/Fe] + 0.362) (Salaris, Chieffi & Straniero 1993; refs.bib
# Salaris1993), with [alpha/Fe] from the measured APOGEE trend; the trend is interpolated at [M/H] rather than at [Fe/H].
mh_to_feh=lambda mh: mh-np.log10(.638*10**np.interp(mh,trend.feh,trend.alpha)+.362)  # noqa: E731, inverse of the alpha correction
for k,((mh,q),color) in enumerate(zip(t[t.survey=='BOSS'].groupby('mh'),colors)):
    feh=mh_to_feh(mh)
    w=age_weights(q.age.to_numpy(),feh,REFERENCE_PRIOR)  # reference (AMR) weighting of the reporting bin containing feh
    lo,hi=np.interp([.16,.84],np.cumsum(w)/w.sum(),q.age)
    alo,ahi=AGE_RANGES['A'][np.digitize(feh,[-1.5,-1.,-.5])]  # prior-A age window [Gyr] of the curve's [Fe/H] range
    ax.plot(q.age,q.n_box*1e6,color=color,label=rf'[M/H]$={mh:g}$ ([Fe/H]$\approx{feh:.1f}$)')
    # Bars at arbitrary heights, stepped by a factor 1.45 per curve; the prior-A bar just below the AMR bar.
    ax.plot([lo,hi],[600*1.45**k]*2,color=color,lw=2.5,solid_capstyle='butt')
    ax.plot([alo,min(ahi,q.age.max())],[600*1.45**k/1.12]*2,color=color,lw=.8,solid_capstyle='butt')
ax.set(xscale='log',yscale='log',xlim=(.4,13.5),ylim=(3,3000),xticks=[.5,1,2,5,10],xticklabels=['0.5','1','2','5','10'],
       xlabel='Age (Gyr)',ylabel=r'BOSS giants per $10^6\,M_\odot$ formed',title='(a) Conversion')
ax.legend(loc='lower left',bbox_to_anchor=(0,1.1),borderaxespad=0)
# %% [markdown]
# ### Panel (b): closure of the response on the common parent
# From `results/boss_response.json` (`scripts/boss_response.py`).

# %%
ax=axes[0,1]
obs,pred,apo=(np.array(c[k]) for k in ('observed','predicted','apogee'))
good=apo>=3  # bins with at least 3 (weighted) APOGEE giants in the common parent
ax.plot(FEH_CENTRES[good]-.03,obs[good]/apo[good],'o',ms=3,color='0.6',label='no response')
ax.vlines(FEH_CENTRES[good]+.03,np.array(c['ratio_p16'])[good],np.array(c['ratio_p84'])[good],color='#2878b5',lw=.8)
ax.plot(FEH_CENTRES[good]+.03,obs[good]/pred[good],'o',ms=3,color='#2878b5',label='with response')
ax.axhline(1,color='.3',lw=.6,ls='--')
ax.set(xlim=(-2.45,-.1),ylim=(0,2.6),xticks=[-2.25,-1.5,-1,-.5],xticklabels=['<−2','−1.5','−1','−0.5'],xlabel='[Fe/H]',
       ylabel='BOSS observed / expected',title='(b) BOSS response, common parent')
ax.legend(loc='lower left',bbox_to_anchor=(0,1.1),borderaxespad=0)
fig.savefig(ROOT/'figures/paper_mass_method.png')
print(t.groupby(['survey','mh']).n_box_raw.agg(['min','max']))
