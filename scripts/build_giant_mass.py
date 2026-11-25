"""Stellar mass formed per [Fe/H] bin from the BOSS giants (paper Sect. 2.3; Letter Fig. 1b through build_present_mass.py),
and the BOSS / APOGEE comparison in shared field cores (appendix figure paper_mass_cores).

The weighted BOSS giant counts per [Fe/H] bin are turned into formed mass by dividing by eta, the number of selected giants
per Msun formed computed from PARSEC isochrones (lmcold.isochrones, lmcold.mass_conversion), region by region, after removing
the Milky Way foreground and correcting for the BOSS [Fe/H] response. The KB chapter "Stellar mass per metallicity" describes
each step and its equations.

Four parts of the method go beyond a weighted count divided by one conversion:
- Above [Fe/H] = -1 the regional BOSS counts are shared among the reporting bins as the APOGEE giants of the same ring x G x
  dereddened-colour cell (lmcold.mass_samples.metal_rich_split): BOSSNet over-estimates [Fe/H] of hot, low-S/N giants, which the
  stars with both spectra barely contain (scripts/paper_boss_hot_bias.py). BOSS still sets the total above -1. The field cores
  use the same split, so above -1 only their total is an independent test.
- BOSS metallicity response measured on the stars with both spectra (lmcold.mass_samples.boss_response, closure tested in
  scripts/boss_response.py): masses per true [Fe/H] bin from a non-negative Poisson-mean fit (lmcold.mass_response.recover_mass)
  of the BOSS counts in fine measured bins into 0.5-dex true bins, with an effective conversion per bin (count-weighted harmonic
  mean over regions and reporting bins); regional masses are the direct ones (measured bins) times the ratio of fitted to
  direct mass of their true bin.
- Conversion per region with the true reddening (lmcold.extinction) and the inclined-disc distance (lmcold.geometry.disc_dmu),
  besides the reddening the carton removed: the full BOSS sample in 16 regions (rings 0-3-6-9-12 deg x quadrants), each at
  the median values of its parent stars; APOGEE and BOSS field cores at the medians over the core.
- Milky Way foreground varying with radius (lmcold.foreground), per reporting bin, normalised to the transferred reference mean
  fraction over 2-10 deg; bracketed by zero and twice its amplitude.
Model alternatives: age weightings AMR (reference; lmcold.mass_conversion) and windows A/B/C x alpha (measured, 0, 0.15, 0.3) x foreground (0, 1, 2); and, for the reference, the
Skowron et al. E(V-I) systematic (+-0.025 mag) and the vdMK14 inclination error (34 +- 7 deg). Non-overlapping APOGEE cores
r<0.8 deg, contiguous >=90%-covered magnitude chunks, H_hi>=13.3. Full BOSS coverage fill (lmcold.selection.coverage_factor) is
an explicit ring (0-3-6-12 deg) x G extrapolation. Bootstrap: NBOOT nside-16 pixel draws of the parent, selection weights and coverage recomputed per draw, overlap stars of the
response resampled.

Inputs. sources.yaml entries: apogee_dr17_allstarlite (ALPHA_M for the alpha trend), apogee2_field and apogee2_object_<field>
(core centres; SFD_EBV of the core stars), skowron21_table1 (via lmcold.extinction.rc_map), mass_foreground_reference,
parsec_gaiaEDR3 (BOSS) and parsec_2mass_spitzer (APOGEE). Data files: boss_weights, boss_parent, apogee_parent,
apogee_weights (release_* entries) and, through lmcold.mass_samples.samples, spec_giants.parquet and
results/boss_calibration.json.
Units: masses in Msun formed (Kroupa IMF, 0.08-100 Msun); eta in giants per Msun formed; E(B-V) [mag, SFD scale]; dmu [mag];
[Fe/H] in dex on the APOGEE scale; reporting bins lmcold.mass_samples.FEH_EDGES (centres FEH_CENTRES, -2.25 ... +0.25).
Outputs (results/ unless stated):
  mass_alpha_trend.csv            median [Fe/H], [alpha/M] and N per reporting bin with >= 10 APOGEE giants;
  mass_fields_apogee_candidates.csv, mass_fields.csv, mass_fields_rejected.csv
                                  APOGEE field cores (h_limit, maglim, centre, ebv_carton, ebv, dmu, boss_coverage);
  mass_regions.csv                the 16 BOSS regions: n_parent, median true/carton E(B-V), dmu (and at inc -+ 7 deg), rho;
  data/giant_foreground.parquet   per BOSS giant: source_id, feh, rho, fill, f_mw (foreground fraction);
  giant_mass_variants.csv         mass per reporting bin for every model alternative;
  giant_mass_response.csv         reference: direct, fitted and fitted/direct per bin;
  mass_tail_conversion.csv        eta at -2.5, -2.25, -2.0 and whether [M/H] is clipped at the grid edge;
  giant_mass_fields.csv, mass_apogee_fill.csv
                                  per field core and survey: counts, raw counts, eta, mass; APOGEE fill per H chunk;
  mass_conversion_factors.csv     factors() output per survey x field/region x prior x alpha;
  mass_model_grid.csv             giants per Msun formed on the isochrone grid for every conversion;
  giant_mass_bootstrap.csv        NBOOT draws of the reference mass per bin;
  giant_mass.csv                  reference mass, cumulative mass, bootstrap 16-84%, model min/max, covered and direct mass;
  giant_mass_common.csv           BOSS and APOGEE masses in the common cores, ratio and its 16-84% over 500 paired draws;
  giant_mass_assignment.csv       change of the direct mass if faint giants below -2 take their XP [M/H] instead;
  giant_mass.json                 summary numbers and definitions.
These feed build_present_mass.py, figures/paper_mass.png, paper_mass_cores.png, paper_mass_summary.png, and (through
giant_foreground.parquet) paper_feh_profile.py, fig_giant_foreground.py and other figure scripts.

Run: .venv/bin/python scripts/build_giant_mass.py   (after boss_selection.py and apogee_selection.py)
Run time: 162 s on a MacBook (M-series), docs/run_times.txt.
"""

# %% [markdown]
# ### Settings
# Modules, the output folder, the bin and ring edges of the conversion, the model alternatives and the size of the bootstrap.

# %%
import json
import healpy as hp
import numpy as np
import pandas as pd
from astropy.io import fits
from lmcold import magcloud, selection
from lmcold.constants import LMC
from lmcold.extinction import EVI_PER_EBV, rc_map, true_ebv
from lmcold.footprint import angular_distance, lmc_xy
from lmcold.foreground import radial_fraction
from lmcold.geometry import disc_dmu
from lmcold.groups import bin_index
from lmcold.isochrones import count_grid, smooth_age_spikes
from lmcold.mass_conversion import REFERENCE_PRIOR, factors
from lmcold.mass_response import recover_mass
from lmcold.mass_samples import (samples, boss_response, overlap, metal_rich_split, with_colour, FEH_EDGES, FEH_CENTRES,
                                 MEASURED_EDGES, TRUE_EDGES)
from lmcold.sources import DATA, ROOT, path, sha256

OUT=ROOT/'results'  # output folder of the tables
N=len(FEH_CENTRES)  # number of reporting [Fe/H] bins (10)
G_EDGES=[12,15.5,16.25,16.75,17.1,17.5]  # G bins [mag] of the BOSS coverage fill (as boss_selection.GEDGES)
H_EDGES=np.array([12.35,12.8,13.3,13.8,14.3,14.8,15.3])  # H chunks [mag] of the APOGEE cores, BrtRGB bright limit to 15.3
RINGS=[0,3,6,9,12]  # deg: ring edges of the 16 BOSS conversion regions (4 rings x 4 quadrants)
PRIORS,ALPHAS=[REFERENCE_PRIOR,'A','B','C'],['measured','0','0.15','0.3']  # age weightings; alpha models ('0.15': [alpha/Fe])
DEBV=.025/EVI_PER_EBV  # mag: Skowron et al. 2021 systematic (0.025 mag in E(V-I)) of the intrinsic RC colour, on the E(B-V) scale
DINC=7.  # deg: error of the disc inclination, van der Marel & Kallivayalil 2014
NBOOT=500  # bootstrap draws of the full BOSS mass


# %% [markdown]
# ### Helpers
# `dmu` gives the distance-modulus offset of the inclined disc at a sky position, `hist` the weighted counts per reporting
# [Fe/H] bin, and `fit` the response-corrected mass per bin from the counts and the conversions (lmcold.mass_response).

# %%
def dmu(ra,dec,inc=LMC['inc']):
    """Distance-modulus offset [mag] of the LMC disc plane at (ra, dec) [deg] relative to the centre (lmcold.geometry.disc_dmu).

    Parameters
    ----------
    ra, dec : array_like
        Positions [deg].
    inc : float
        Disc inclination [deg]; the line of nodes is constants.LMC['theta'].

    Returns
    -------
    numpy.ndarray
        dmu = 5 log10(d / D0) [mag], positive on the far side of the disc.
    """
    # All angles to radians: position, inclination, line of nodes and the adopted centre (constants.LMC)
    return np.asarray(disc_dmu(np.radians(np.asarray(ra,float)),np.radians(np.asarray(dec,float)),np.radians(inc),np.radians(LMC['theta']),
                               np.radians(LMC['ra0']),np.radians(LMC['dec0'])))


def hist(s,w):
    """Weighted counts of sample s (column feh [dex]) in the reporting bins FEH_EDGES; w: weight per row."""
    return np.histogram(s.feh,FEH_EDGES,weights=w)[0]


def fit(regional,fine,eta,r):
    """Response-corrected mass per reporting bin, the ratio fitted/direct per reporting bin, and the direct mass, from counts
    per region (regions x reporting bins), the BOSS counts in the fine measured bins, conversions per region (regions x
    reporting bins) and the response r (0.5-dex true bins TRUE_EDGES). Within a true bin the mass is shared out as the direct
    masses of its reporting bins."""
    direct=(regional/eta).sum(0)  # mass [Msun] = counts / (giants per Msun), summed over regions
    # tb: true bin of each reporting bin; per true bin, count-weighted harmonic-mean eta = sum(counts) / sum(direct mass).
    tb=np.searchsorted(TRUE_EDGES,FEH_CENTRES)-1
    c,d=np.bincount(tb,regional.sum(0),len(TRUE_EDGES)-1),np.bincount(tb,direct,len(TRUE_EDGES)-1)
    # Poisson-mean fit of the fine counts through the response into true-bin masses m [Msun] (eta.mean() where a
    # true bin has no direct mass); ratio: fitted / direct mass of each reporting bin's true bin
    m=recover_mass(fine,r,np.where(d>0,c/np.where(d>0,d,1),eta.mean()))
    ratio=np.where(d>0,m/np.where(d>0,d,1),0.)[tb]
    return direct*ratio,ratio,direct


# %% [markdown]
# ### Load the data
# The BOSS and APOGEE mass samples and the stars observed by both (lmcold.mass_samples), the BOSS selection weights and parent,
# the APOGEE parent and weights (Zenodo bundle 1), and the red-clump reddening map. Further inputs are read where they are used:
# APOGEE allStarLite and the field list below, the foreground reference and the PARSEC grids further down.

# %%
# BOSS (b) and APOGEE (a) mass samples: good giant spectra, positive weight, 150 < v_los < 420 km/s (lmcold.mass_samples).
b,a,_=samples()
o=overlap(b,a)  # stars with both spectra: feh_b (calibrated BOSS), feh_a (APOGEE) [dex], g [mag], snr
bw=pd.read_parquet(DATA/'boss_weights.parquet')  # BOSS eligible parent: selection group, good/observed flags, g, mh_xp, ra, dec
bp=pd.read_parquet(DATA/'boss_parent.parquet').query('eligible')  # SFD E(B-V) of the eligible parent stars
ap=pd.read_parquet(DATA/'apogee_parent.parquet')  # APOGEE candidates: field, h [mag], rfield [deg from field centre], maglim
# APOGEE weights joined with field, distance from the field centre and maglim; the same columns added to the sample
aw=pd.read_parquet(DATA/'apogee_weights.parquet').merge(ap[['field','apogee_id','rfield','maglim']],on=['field','apogee_id'],validate='one_to_one')
a=a.merge(ap[['field','apogee_id','rfield']],on=['field','apogee_id'],validate='one_to_one')
rc=rc_map()  # Skowron et al. 2021 red-clump E(V-I) boxes

# %% [markdown]
# ### The [alpha/Fe] trend
# Median ASPCAP [alpha/M] of the APOGEE mass sample per reporting [Fe/H] bin (bins with at least 10 giants), used to convert
# [Fe/H] to the model [M/H] (results/mass_alpha_trend.csv).

# %%
# Abundance trend: exactly selected APOGEE stars, best nonduplicate spectrum.
with fits.open(path('apogee_dr17_allstarlite'),memmap=True) as h:
    d=h[1].data
    # Rows of the mass sample observed with the du Pont telescope (lco25m), EXTRATARG bit 4 (duplicate) not set
    ids=np.char.strip(d['APOGEE_ID'].astype(str))
    mask=pd.Series(ids).isin(a.apogee_id).to_numpy()&(np.char.strip(d['TELESCOPE'].astype(str))=='lco25m')&((d['EXTRATARG'] & (1<<4))==0)
    al=pd.DataFrame(dict(apogee_id=ids[mask],alpha=d['ALPHA_M'][mask].astype(float),snr=d['SNR'][mask].astype(float)))
# One row per star: the highest-S/N spectrum; joined with the sample's [Fe/H]
al=al.sort_values('snr',ascending=False,kind='stable').drop_duplicates('apogee_id')
al=a[['apogee_id','feh']].merge(al,on='apogee_id',validate='one_to_one')
al=al[np.isfinite(al.alpha)&(abs(al.alpha)<2)]  # drop missing and out-of-range [alpha/M]
# Median [Fe/H], [alpha/M] and count per reporting bin; bins with >= 10 giants kept
trend=al.groupby(pd.cut(al.feh,FEH_EDGES,right=False),observed=True).agg(feh=('feh','median'),alpha=('alpha','median'),n=('alpha','size')).query('n>=10').reset_index(drop=True)
trend.to_csv(OUT/'mass_alpha_trend.csv',index=False)
print('Alpha trend:',trend.to_string(index=False),flush=True)

# %% [markdown]
# ### APOGEE field cores
# Cores of radius 0.8 deg about the APOGEE field centres where at least 90 per cent of the candidates have spectra down to
# H = 13.3 or fainter; for each, the median carton and true E(B-V) and disc offset of its targeting-file stars.

# %%
# Choose contiguous well-covered APOGEE cores.
# A core is r < 0.8 deg of an APOGEE field; walk up in H chunks from 12.35 while >= 90% of the candidates are covered (and below
# the field's maglim); keep fields whose covered range reaches H = 13.3 (h_limit = the last chunk's upper edge).
fields=[]
for field,q in aw[aw.rfield<.8].groupby('field'):
    # hi: upper H edge [mag] of the covered range so far, starting at the bright limit
    hi=12.35
    for low,high in zip(H_EDGES[:-1],H_EDGES[1:]):
        high=min(high,float(q.maglim.iloc[0]))
        part=q[(q.h>=low)&(q.h<high)]
        if high<=low or not len(part) or part.covered.mean()<.9:
            break
        hi=high
    # Keep the field if the covered range reaches H = 13.3
    if hi>=13.3:
        fields.append(dict(field=field,h_limit=hi,maglim=float(q.maglim.iloc[0])))
f=pd.DataFrame(fields)
# Field centres (ra, dec [deg]) from the apogee2_field table, one row per field name
with fits.open(path('apogee2_field')) as h:
    d=h[1].data
    centres=pd.DataFrame(dict(field=np.char.strip(d['FIELD_NAME'].astype(str)),ra=d['RA'].astype(float),dec=d['DEC'].astype(float))).drop_duplicates('field')
f=f.merge(centres,on='field',validate='one_to_one')
for i,r in f.iterrows():
    assert np.all(angular_distance(f.ra.drop(i),f.dec.drop(i),r.ra,r.dec)>1.6)  # 0.8-deg cores do not overlap
    # Median carton E(B-V), true E(B-V) and disc dmu over all targeting-file stars in the core.
    with fits.open(path('apogee2_object_'+r.field.lower())) as h:
        d=h[1].data
        # Targeting-file stars within 0.8 deg of the field centre: positions [deg] and SFD E(B-V) [mag]
        core=angular_distance(d['RA'],d['DEC'],r.ra,r.dec)<.8
        ra,dec,sfd=d['RA'][core].astype(float),d['DEC'][core].astype(float),d['SFD_EBV'][core].astype(float)
        f.loc[i,'ebv_carton']=np.nanmedian(magcloud.carton_ebv(ra,dec,sfd))
        f.loc[i,'ebv']=np.nanmedian(true_ebv(ra,dec,sfd,rc))
        f.loc[i,'dmu']=np.median(dmu(ra,dec))

# %% [markdown]
# ### BOSS coverage of the cores
# Cores are dropped until in every remaining one at least 90 per cent of the BOSS eligible stars lie in selection groups with
# enough good spectra inside the cores (results/mass_fields.csv, mass_fields_rejected.csv).

# %%
# Require BOSS coverage too. Recompute after pruning because groups can cross
# field boundaries. This avoids estimating a field from a handful of stars.
f.to_csv(OUT/'mass_fields_apogee_candidates.csv',index=False)
s=pd.factorize(bw.group)[0]  # integer selection group of each eligible BOSS star
good=bw.good.to_numpy();obs=bw.observed.to_numpy()  # good giant spectrum; observed by BOSS
pg=selection.boss_pass_groups(bw.g,bw.mh_xp,bw.ra,bw.dec)  # groups for the pass factor (lmcold.selection)
# Drop cores where < 90% of the BOSS eligible stars lie in selection groups with >= NMIN_GOOD good spectra inside the cores.
rejected=[]
while True:
    # region: index of the core holding each eligible BOSS star (-1 outside every core)
    region=np.full(len(bw),-1)
    for i,r in f.iterrows():
        region[angular_distance(bw.ra,bw.dec,r.ra,r.dec)<.8]=i
    inside=region>=0
    # Good spectra inside the cores in each star's group; cov: group has >= NMIN_GOOD; coverage: covered fraction per core
    ng=np.bincount(s,inside*good)[s]
    cov=ng>=selection.NMIN_GOOD
    coverage=np.array([cov[region==i].mean() for i in range(len(f))])
    f['boss_coverage']=coverage
    # Stop when every core is >= 90 per cent covered; else record and drop the failing cores and recompute
    if np.all(coverage>=.9):break
    rejected.extend(f.loc[coverage<.9,['field','boss_coverage']].to_dict('records'))
    f=f[coverage>=.9].reset_index(drop=True)
    assert len(f)>0
f.to_csv(OUT/'mass_fields.csv',index=False)
pd.DataFrame(rejected).to_csv(OUT/'mass_fields_rejected.csv',index=False)
print('Common fields:',f.to_string(index=False),flush=True)

# %% [markdown]
# ### BOSS weights in the cores
# Selection weights recomputed from the core stars only, a coverage fill per core and G bin, and the share of each BOSS giant
# above [Fe/H] = -1 among the reporting bins, taken from the APOGEE giants of the same cell.

# %%
# Local BOSS weights: group counts and pass factors from the core stars only (multiplicity = inside); fill per core x G bin.
wlocal=selection.group_weights(s,good,inside)*selection.pass_factor(s,obs,good,pg,inside)
gbin=bin_index(bw.g,G_EDGES)  # G bin of each eligible star
# Per eligible star in a core: core index, local weight and coverage fill per core x G cell; joined to the BOSS mass sample
local=bw[['source_id','g']].assign(region=region,local_w=wlocal,fill=selection.coverage_factor([region*10+gbin],cov,1,inside.astype(float)))[inside]
bc=b.drop(columns='g').merge(local,on='source_id',validate='one_to_one')
bc=bc[bc.local_w>0]
SPLIT_COLS=[f'split{k}' for k in range(5)]  # shares of the reporting bins above -1 (lmcold.mass_samples.metal_rich_split)
# Shares of each core giant among the reporting bins above -1, from the APOGEE giants of its cell
bc[SPLIT_COLS]=metal_rich_split(with_colour(bc[['source_id','ra','dec','feh']]),with_colour(a[['source_id','ra','dec','feh','w']]))


def split_hist(t,w):
    """Weighted counts per reporting bin, the giants above -1 shared as their split columns."""
    w=np.asarray(w,float)
    h=np.histogram(t.feh,FEH_EDGES,weights=w*(t.feh<=-1))[0]  # giants at or below -1 by their own [Fe/H]
    # Giants above -1 added to the last len(SPLIT_COLS) reporting bins as weight x share
    h[N-len(SPLIT_COLS):]+=(w[:,None]*t[SPLIT_COLS].to_numpy()).sum(0)
    return h

# %% [markdown]
# ### Conversion regions of the full BOSS sample
# Coverage fill per ring and G bin, and the 16 conversion regions (four rings by four quadrants) with the median true and carton
# E(B-V) and disc offset of their parent stars (results/mass_regions.csv).

# %%
# Full BOSS: coverage fill in rings x G, conversion per ring x quadrant region.
rho=angular_distance(bw.ra,bw.dec,LMC['ra0'],LMC['dec0'])
assert rho.max()<12.01
rings=np.digitize(rho,[3,6])  # 0, 1, 2 for 0-3, 3-6, 6-12 deg (coverage fill)
# Covered selection groups (>= NMIN_GOOD good spectra); fill = parent / covered parent per ring x G cell.
fullcov=bw.group.map(bw.groupby('group').good.sum()).to_numpy()>=selection.NMIN_GOOD
ones=np.ones(len(bw))
# Coverage fill per eligible star (cells ring x G, at least 1 covered star per cell)
bw['fill']=selection.coverage_factor([rings*10+gbin],fullcov,1,ones)
x,y=lmc_xy(bw.ra.to_numpy(),bw.dec.to_numpy())  # deg, tangent-plane coordinates about the LMC centre (quadrants)
# Per eligible star: SFD E(B-V) (boss_parent), carton and true E(B-V), disc dmu at inc and inc -+ DINC; region 0-15.
geo=bw[['source_id','ra','dec']].merge(bp[['source_id','ebv']],on='source_id',validate='one_to_one')
geo=geo.assign(rho=rho,region=4*np.clip(np.digitize(rho,RINGS)-1,0,3)+(x>=0)+2*(y>=0),ebv_carton=magcloud.carton_ebv(geo.ra,geo.dec,geo.ebv),
               ebv_true=true_ebv(geo.ra.to_numpy(),geo.dec.to_numpy(),geo.ebv.to_numpy(),rc),dmu=dmu(geo.ra,geo.dec),
               dmu_lo=dmu(geo.ra,geo.dec,LMC['inc']-DINC),dmu_hi=dmu(geo.ra,geo.dec,LMC['inc']+DINC))
# Per region: parent count and median true and carton E(B-V) [mag], dmu (and at inc -+ DINC) [mag], rho [deg]
regions=geo.groupby('region').agg(n_parent=('ra','size'),ebv=('ebv_true','median'),ebv_carton=('ebv_carton','median'),
                                  dmu=('dmu','median'),dmu_lo=('dmu_lo','median'),dmu_hi=('dmu_hi','median'),rho=('rho','median'))
regions.to_csv(OUT/'mass_regions.csv')
print('Regions:',regions.round(3).to_string(),flush=True)
# Attach fill, rho and region to each BOSS giant of the mass sample
b=b.merge(bw[['source_id','fill']],on='source_id',validate='one_to_one').merge(geo[['source_id','rho','region']],on='source_id',validate='one_to_one')

# %% [markdown]
# ### Milky Way foreground
# The foreground fraction f_mw of each BOSS giant: the reference mean fraction at its [Fe/H], spread in radius as a foreground of
# uniform surface density per reporting bin (lmcold.foreground); written per giant to data/giant_foreground.parquet.

# %%
# Foreground: reference mean fraction f_mw at [Fe/H] (interpolated in feh_med), spread in radius by lmcold.foreground.radial_fraction.
fg=pd.read_csv(path('mass_foreground_reference'))  # mean foreground fraction f_mw over 2-10 deg against feh_med
j=np.digitize(b.feh,FEH_EDGES)-1  # reporting bin of each BOSS giant
# Foreground fraction per giant, filled per reporting bin below
b['f_mw']=0.
for k in range(N):  # per reporting bin, so the surface-density profile is that of the same metallicities
    m=j==k
    if m.any():
        # Radial foreground fraction from the giants' radii [deg], weights w x fill and the reference mean fraction at their [Fe/H]
        b.loc[m,'f_mw']=radial_fraction(b.rho[m].to_numpy(),(b.w*b['fill'])[m].to_numpy(),np.interp(b.feh[m],fg.feh_med,fg.f_mw))
# Per-giant foreground table for the figure scripts
b[['source_id','feh','rho','fill','f_mw']].to_parquet(DATA/'giant_foreground.parquet',index=False)

# %% [markdown]
# ### Giants per solar mass formed
# The PARSEC grids are read and the selected giants per Msun formed are counted on them: once at the medians of the whole
# parent, and per region for the reference and the E(V-I) and inclination variants. `factors` averages them over age and
# converts [Fe/H] to [M/H] for each age weighting and [alpha/Fe] model, giving eta per region and reporting bin.

# %%
# PARSEC grids per survey (Gaia EDR3 photometry for BOSS, 2MASS for APOGEE); models: every computed grid
grids={name:pd.read_parquet(path(key)) for name,key in [('BOSS','parsec_gaiaEDR3'),('APOGEE','parsec_2mass_spitzer')]}
models=[]
def model(survey,field,ebv,ebv_carton,dmu,h_limit=13.3,maglim=14.5):
    """Giants per Msun formed on the isochrone grid for one survey and line of sight; also appended to `models`.

    Parameters
    ----------
    survey : {'BOSS', 'APOGEE'}
        Selects the PARSEC grid and the selection box (lmcold.isochrones.count_grid).
    field : str
        Label stored with the grid ('full', 'R<region><variant>' or an APOGEE field name).
    ebv, ebv_carton : float
        True and carton-removed E(B-V) [mag, SFD scale].
    dmu : float
        Disc distance-modulus offset [mag].
    h_limit, maglim : float
        APOGEE H limit of the core and the field's FntRGB limit [mag] (used for APOGEE).

    Returns
    -------
    pandas.DataFrame
        Spike-smoothed grid (lmcold.isochrones.smooth_age_spikes): mh, age, n_box per grid point.
    """
    print('Integrating',survey,field,f'E(B-V) {ebv:.4f} carton {ebv_carton:.4f} dmu {dmu:+.3f}',flush=True)
    # Selected giants per Msun on the grid for this line of sight, with the age spikes smoothed; stored in models
    g=smooth_age_spikes(count_grid(grids[survey],survey,ebv,ebv_carton,dmu,h_limit,maglim))
    models.append(g.assign(survey=survey,field=field,ebv=ebv,ebv_carton=ebv_carton,dmu=dmu,h_limit=h_limit,maglim=maglim))
    return g

# 'full': one conversion at the medians of the whole parent, for the age/metallicity bookkeeping of build_present_mass and
# the tail check; the masses use the regional conversions.
fullgrid=model('BOSS','full',float(geo.ebv_true.median()),float(geo.ebv_carton.median()),float(geo.dmu.median()))
# Regional grids: reference ('') and the systematic variants E(B-V) +-DEBV and inclination -+DINC.
regional={name:[model('BOSS',f'R{r}{name}',e.ebv+de,e.ebv_carton,e[dm]) for r,e in regions.iterrows()]
          for name,de,dm in [('',0,'dmu'),(' E+',DEBV,'dmu'),(' E-',-DEBV,'dmu'),(' i-',0,'dmu_lo'),(' i+',0,'dmu_hi')]}
# factors() of every grid for every age weighting and alpha model (saved to mass_conversion_factors.csv)
conversion=[]
for prior in PRIORS:
    for alpha in ALPHAS:
        conversion.append(factors(fullgrid,trend,prior,alpha).assign(survey='BOSS',field='full',prior=prior,alpha_model=alpha))
        for r,g in zip(regions.index,regional['']):
            conversion.append(factors(g,trend,prior,alpha).assign(survey='BOSS',field=f'R{r}',prior=prior,alpha_model=alpha))
# eta[(prior, alpha, variant)]: (regions x reporting bins); variants other than '' only for the reference prior and alpha.
eta={(prior,alpha,name):np.array([factors(g,trend,prior,alpha).eta.to_numpy() for g in gs])
     for name,gs in regional.items() for prior in PRIORS for alpha in ALPHAS if name=='' or (prior,alpha)==(REFERENCE_PRIOR,'measured')}

# %% [markdown]
# ### The BOSS response and the metal-rich split
# The response matrix measured on the stars observed by both surveys, weighted to the BOSS sample, and the share of each BOSS
# giant above [Fe/H] = -1 among the reporting bins; `counts` sums the weighted giants per region and reporting bin.

# %%
rid=np.searchsorted(regions.index.to_numpy(),b.region.to_numpy())  # row of each BOSS giant's region in `regions`
jf=np.digitize(b.feh,MEASURED_EDGES)-1  # fine measured bin of each BOSS giant
# BOSS response matrix from the overlap stars, weighted to the BOSS sample (weights w x fill)
response=boss_response(o,b.feh,b.g,b.snr,(b.w*b['fill']).to_numpy())
# Above [Fe/H] = -1 each BOSS giant is shared among the reporting bins as the APOGEE giants of its ring x G x colour cell
# (lmcold.mass_samples.metal_rich_split): BOSSNet over-estimates [Fe/H] of hot, low-S/N giants (scripts/paper_boss_hot_bias.py).
rich=(b.feh>-1).to_numpy()  # BOSS giants above [Fe/H] = -1, shared out by `split`
split=metal_rich_split(with_colour(b[['source_id','ra','dec','feh']]),with_colour(a[['source_id','ra','dec','feh','w']]))
assert len(split)==len(b)


def counts(weights):
    """BOSS counts per region x reporting bin (giants above -1 shared by `split`) and per fine measured bin.

    Parameters
    ----------
    weights : numpy.ndarray
        Weight per BOSS giant (selection weight x foreground and fill terms).

    Returns
    -------
    reg : numpy.ndarray
        (len(regions), N).
    fine : numpy.ndarray
        (len(MEASURED_EDGES) - 1,).
    """
    # Giants at or below -1 by their own bin; those above -1 shared over the last split.shape[1] bins; fine: all giants
    reg=np.zeros((len(regions),N))
    np.add.at(reg,(rid[~rich],j[~rich]),weights[~rich])
    np.add.at(reg[:,N-split.shape[1]:],rid[rich],weights[rich,None]*split[rich])
    return reg,np.bincount(jf,weights,len(MEASURED_EDGES)-1)


# %% [markdown]
# ### Mass for every model alternative
# The fit for every combination of conversion, age weighting, [alpha/Fe] model, foreground amplitude (0, 1, 2) and coverage fill
# (results/giant_mass_variants.csv), and the reference with its fitted-to-direct ratios (giant_mass_response.csv).

# %%
# Every model alternative; the weight per giant is w x (1 - foreground x f_mw) x fill (if filled).
variants=[];direct_rows=[]
for (prior,alpha,name),e in eta.items():
    for foreground in ([0,1,2] if name=='' else [1]):
        for filled in ([False,True] if name=='' else [True]):
            # Counts with this foreground amplitude and fill, response-corrected mass per bin [Msun]; one table row per bin
            reg,fine=counts((b.w*(1-foreground*b.f_mw)*(b['fill'] if filled else 1)).to_numpy())
            m,ratio,direct=fit(reg,fine,e,response)
            variants.append(pd.DataFrame(dict(conversion=name.strip() or 'reference',prior=prior,alpha_model=alpha,foreground=foreground,
                                              filled=filled,feh=FEH_CENTRES,mass=m,cumulative=np.cumsum(m),direct_mass=direct)))
# All alternatives to giant_mass_variants.csv; ref: rows of the reference model (reference conversion, prior,
# measured alpha, foreground 1)
v=pd.concat(variants,ignore_index=True)
v.to_csv(OUT/'giant_mass_variants.csv',index=False)
ref=(v.conversion=='reference')&(v.prior==REFERENCE_PRIOR)&(v.alpha_model=='measured')&(v.foreground==1)
# Reference mass recomputed directly and checked against the filled reference row; then giant_mass_response.csv
reg_ref,fine_ref=counts((b.w*(1-b.f_mw)*b['fill']).to_numpy())
mass_ref,ratio_ref,direct_ref=fit(reg_ref,fine_ref,eta[REFERENCE_PRIOR,'measured',''],response)
np.testing.assert_allclose(mass_ref,v[ref&v.filled].mass.to_numpy())
pd.DataFrame(dict(feh=FEH_CENTRES,fitted_over_direct=ratio_ref,direct=direct_ref,fitted=mass_ref)).to_csv(OUT/'giant_mass_response.csv',index=False)

# %% [markdown]
# ### Conversion of the metal-poor tail
# eta at [Fe/H] = -2.5, -2.25 and -2 against that of the first reporting bin, and whether [M/H] falls off the grid
# (results/mass_tail_conversion.csv).

# %%
# Conversion of the metal-poor tail: eta at single [Fe/H] values, relative to the reference first bin.
# reff: eta per reporting bin of the 'full' grid with the default prior and alpha
tail=[]
reff=factors(fullgrid,trend).eta.to_numpy()
for centre in [-2.5,-2.25,-2.]:
    # eta at this single [Fe/H], the grid [M/H] it maps to and whether it is clipped at the grid edge
    fac=factors(fullgrid,trend,centres=[centre])
    tail.append(dict(feh=centre,eta=fac.eta.iloc[0],mh_grid=fac.mh_grid.iloc[0],clipped=bool(fac.clipped_mh.iloc[0]),mass_ratio_to_reference=reff[0]/fac.eta.iloc[0]))
pd.DataFrame(tail).to_csv(OUT/'mass_tail_conversion.csv',index=False)

# %% [markdown]
# ### BOSS and APOGEE in the field cores
# Each survey with its own selection box and the extinction and disc offset of each core; APOGEE counts filled per H chunk,
# BOSS counts response-corrected with the response fitted to the summed core counts (results/giant_mass_fields.csv).

# %%
# Field-specific conversions and counts, with local BOSS weights; the BOSS response fitted to the summed core counts.
rows=[];afills=[]
# Summed BOSS counts of all cores in the fine measured bins (local weights x fill), and the core response
core_fine=np.zeros(len(MEASURED_EDGES)-1)
for i,r in f.iterrows():
    bs=bc[bc.region==i]
    core_fine+=np.histogram(bs.feh,MEASURED_EDGES,weights=bs.local_w*bs['fill'])[0]
core_response=boss_response(o,bc.feh,bc.g,bc.snr,(bc.local_w*bc['fill']).to_numpy())
core={}
for i,r in f.iterrows():
    # Giants per Msun for APOGEE (no carton reddening removal: ebv twice) and for BOSS at this core
    ag=model('APOGEE',r.field,r.ebv,r.ebv,r.dmu,r.h_limit,r.maglim)
    bg=model('BOSS',r.field,r.ebv,r.ebv_carton,r.dmu)
    # APOGEE mass-sample giants of the core within its covered H range; raw: unweighted counts per reporting bin
    ac=a[(a.field==r.field)&(a.rfield<.8)&(a.h>=12.35)&(a.h<r.h_limit)].copy()
    acount=np.zeros(N);raw=hist(ac,np.ones(len(ac)))
    for low,high in zip(H_EDGES[:-1],H_EDGES[1:]):
        high=min(high,r.h_limit)
        if high<=low:break
        # Parent rows (astro_ok) and covered candidates of the core in this H chunk
        p=ap[(ap.field==r.field)&(ap.rfield<.8)&ap.astro_ok&(ap.h>=low)&(ap.h<high)]
        q=aw[(aw.field==r.field)&(aw.rfield<.8)&aw.covered&(aw.h>=low)&(aw.h<high)]
        fill=len(p)/len(q)  # astro_ok parent rows (bright-neighbour stars included) / covered candidates in the chunk
        afills.append(dict(field=r.field,h_lo=low,h_hi=high,parent=len(p),covered=len(q),fill=fill))
        # Giants of the chunk, weighted by w x fill
        part=ac[(ac.h>=low)&(ac.h<high)]
        acount+=hist(part,part.w*fill)
    # Per core: (grid, weighted counts, raw counts) of each survey; BOSS counts with the metal-rich split
    bs=bc[bc.region==i]
    core[r.field]=dict(APOGEE=(ag,acount,raw),BOSS=(bg,split_hist(bs,bs.local_w*bs['fill']),hist(bs,np.ones(len(bs)))))
for prior in PRIORS:
    for alpha in ALPHAS:
        # eta per core and survey; BOSS fitted/direct ratios from the summed core counts and the core response
        fac={(fld,survey):factors(c[survey][0],trend,prior,alpha) for fld,c in core.items() for survey in ('APOGEE','BOSS')}
        e=np.array([fac[fld,'BOSS'].eta.to_numpy() for fld in core])
        bcounts=np.array([core[fld]['BOSS'][1] for fld in core])
        _,bratio,_=fit(bcounts,core_fine,e,core_response)
        for fld,c in core.items():
            for survey in ('APOGEE','BOSS'):
                g,count,nraw=c[survey]
                conversion.append(fac[fld,survey].assign(survey=survey,field=fld,prior=prior,alpha_model=alpha))
                direct=count/fac[fld,survey].eta.to_numpy()
                mass=np.nan_to_num(direct*bratio) if survey=='BOSS' else direct  # APOGEE: no response correction
                rows.append(pd.DataFrame(dict(field=fld,survey=survey,prior=prior,alpha_model=alpha,feh=FEH_CENTRES,count=count,raw=nraw,
                                              eta=fac[fld,survey].eta.to_numpy(),mass=mass,direct_mass=direct)))
# Write the core masses, the APOGEE fills, the conversion factors and every isochrone grid
fieldmass=pd.concat(rows,ignore_index=True)
fieldmass.to_csv(OUT/'giant_mass_fields.csv',index=False)
pd.DataFrame(afills).to_csv(OUT/'mass_apogee_fill.csv',index=False)
pd.concat(conversion,ignore_index=True).to_csv(OUT/'mass_conversion_factors.csv',index=False)
pd.concat(models,ignore_index=True).to_csv(OUT/'mass_model_grid.csv',index=False)

# %% [markdown]
# ### Bootstrap and the reference mass table
# NBOOT draws of nside-16 sky pixels of the parent, with the selection weights and coverage recomputed and the stars observed by
# both resampled; percentiles and model ranges go with the reference into results/giant_mass.csv.

# %%
# Bootstrap: nside-16 pixels of the parent, with the selection weights and coverage factors recomputed for each draw, and the
# overlap stars of the response; conversions fixed.
rng=np.random.default_rng(20261026)  # fixed seed
# nside-16 nested pixel of each eligible star; u: occupied pixels, ii: index of each star's pixel in u
pix=hp.ang2pix(16,bw.ra.to_numpy(),bw.dec.to_numpy(),lonlat=True,nest=True)
u,ii=np.unique(pix,return_inverse=True)
pos=pd.Index(bw.source_id).get_indexer(b.source_id)  # row of each BOSS giant in the parent
boot=[]
# Each draw: pixel multiplicities from a multinomial (resampling pixels with replacement); m: per-star multiplicity
for draw in rng.multinomial(len(u),np.full(len(u),1/len(u)),size=NBOOT):
    m=draw[ii].astype(float)
    # Selection weights and coverage fill recomputed with the multiplicities; wstar: weight x fill x m of each BOSS giant
    wb=selection.group_weights(s,good,m)*selection.pass_factor(s,obs,good,pg,m)
    fb=selection.coverage_factor([rings*10+gbin],np.bincount(s,m*good)[s]>=selection.NMIN_GOOD,1,m)
    wstar=(wb*fb*m)[pos]
    # Overlap stars resampled with replacement; mass per bin with the reference conversion and the redrawn response
    ob=o.iloc[rng.integers(0,len(o),len(o))]
    reg,fine=counts(wstar*(1-b.f_mw.to_numpy()))
    boot.append(fit(reg,fine,eta[REFERENCE_PRIOR,'measured',''],boss_response(ob,b.feh,b.g,b.snr,wstar))[0])
boot=np.array(boot)
pd.DataFrame(boot,columns=FEH_CENTRES).to_csv(OUT/'giant_mass_bootstrap.csv',index=False)
# Reference mass per bin [Msun] and cumulative mass, with bootstrap 16th-84th percentiles
summary=pd.DataFrame(dict(feh=FEH_CENTRES,feh_lo=FEH_EDGES[:-1],feh_hi=FEH_EDGES[1:],mass=mass_ref,cumulative=mass_ref.cumsum()))
summary[['mass_p16','mass_p84']]=np.percentile(boot,[16,84],axis=0).T
summary[['cumulative_p16','cumulative_p84']]=np.percentile(boot.cumsum(axis=1),[16,84],axis=0).T
# Model envelope: min and max over all filled alternatives; covered_mass: the reference without fill
for col in ['mass','cumulative']:
    summary[col+'_model_min']=v[v.filled].groupby('feh')[col].min().to_numpy()
    summary[col+'_model_max']=v[v.filled].groupby('feh')[col].max().to_numpy()
summary['covered_mass']=v[ref&~v.filled].mass.to_numpy()
summary['direct_mass']=direct_ref
summary.to_csv(OUT/'giant_mass.csv',index=False)

# %% [markdown]
# ### Survey ratio in the cores
# BOSS over APOGEE mass per bin summed over the cores, with 16th-84th percentiles from 500 resamplings of the cores
# (results/giant_mass_common.csv).

# %%
# Paired resampling of the same field cores for the survey ratio.
# Reference model; mb, ma: BOSS and APOGEE mass [Msun] per (core, reporting bin)
q=fieldmass[(fieldmass.prior==REFERENCE_PRIOR)&(fieldmass.alpha_model=='measured')]
mb=q[q.survey=='BOSS'].pivot(index='field',columns='feh',values='mass').to_numpy()
ma=q[q.survey=='APOGEE'].pivot(index='field',columns='feh',values='mass').to_numpy()
# 500 resamplings of the cores (multiplicities); ratio of the summed masses per bin for each, and for the data
draw=rng.multinomial(len(f),np.full(len(f),1/len(f)),size=500)
with np.errstate(divide='ignore',invalid='ignore'):
    ratios=(draw@mb)/(draw@ma)
    ratio=mb.sum(axis=0)/ma.sum(axis=0)
# Bins with no APOGEE mass give NaN ratios
ratio[~np.isfinite(ratio)]=np.nan
ratios[~np.isfinite(ratios)]=np.nan
# Per bin: summed masses, ratio, BOSS direct mass and raw counts of both surveys
common=pd.DataFrame(dict(feh=FEH_CENTRES,boss_mass=mb.sum(axis=0),apogee_mass=ma.sum(axis=0),ratio=ratio,
                         boss_direct_mass=q[q.survey=='BOSS'].groupby('feh').direct_mass.sum().to_numpy()))
for survey in ['BOSS','APOGEE']:
    common[survey.lower()+'_raw']=q[q.survey==survey].groupby('feh').raw.sum().to_numpy()
# 16th-84th percentiles of the ratio where any draw is finite
common[['ratio_p16','ratio_p84']]=np.nan
has=np.any(np.isfinite(ratios),axis=0)
common.loc[has,['ratio_p16','ratio_p84']]=np.nanpercentile(ratios[:,has],[16,84],axis=0).T
common.to_csv(OUT/'giant_mass_common.csv',index=False)

# %% [markdown]
# ### Faint giants below [Fe/H] = -2
# Change of the direct mass if the BOSS giants fainter than G = 17.1 below [Fe/H] = -2 take their Gaia XP [M/H] instead
# (results/giant_mass_assignment.csv).

# %%
# The identified faint-tail assignment comparison, on the direct masses (measured bins).
# Weight per star divided by eta of its region and (new) bin gives mass; foreground interpolated at the assigned [Fe/H].
sens=[]
e=eta[REFERENCE_PRIOR,'measured','']
# Two cases: giants in -2.25 <= [Fe/H] < -2, and all giants below -2 (G > 17.1, with XP [M/H])
for label,low in [('disputed_bin',-2.25),('all_faint_below_m2',-np.inf)]:
    changed=((b.g>17.1)&(b.feh>=low)&(b.feh<-2)&np.isfinite(b.mh_xp)).to_numpy()
    alt=b.feh.to_numpy().copy();alt[changed]=b.mh_xp.to_numpy()[changed]
    # Direct mass per bin with the alternative and the original [Fe/H]; relative changes below -2, -1.5 and in total
    jj=np.digitize(alt,FEH_EDGES)-1
    m=np.histogram(alt,FEH_EDGES,weights=(b.w*b['fill']).to_numpy()*(1-np.interp(alt,fg.feh_med,fg.f_mw))/e[rid,jj])[0]
    base=np.histogram(b.feh,FEH_EDGES,weights=(b.w*b['fill']).to_numpy()*(1-np.interp(b.feh,fg.feh_med,fg.f_mw))/e[rid,j])[0]
    sens.append(dict(case=label,changed=int(changed.sum()),below_m2=m[0]/base[0]-1,below_m15=m[:3].sum()/base[:3].sum()-1,total=m.sum()/base.sum()-1))
pd.DataFrame(sens).to_csv(OUT/'giant_mass_assignment.csv',index=False)

# %% [markdown]
# ### Summary numbers
# Totals, masses below -2 and -1.5, bootstrap ranges, core ratio, conversion alternatives and definitions, written to
# results/giant_mass.json.

# %%
# Area of the cores: len(f) spherical caps of radius 0.8 deg, 2 pi (1 - cos r) sr x (180/pi)^2 deg^2/sr.
cum=lambda m,n: float(np.sum(m[:n]))  # noqa: E731
# Summary numbers (masses in Msun formed; below_m2 = first bin, below_m15 = first three bins) and definitions
metadata=dict(fields=len(f),common_area_deg2=float(len(f)*2*np.pi*(1-np.cos(np.deg2rad(.8)))*(180/np.pi)**2),
    mass_type='formed, Kroupa 0.08–100 Msun',full_radius_deg=12,full_mass=float(mass_ref.sum()),covered_mass=float(summary.covered_mass.sum()),
    below_m2=cum(mass_ref,1),below_m15=cum(mass_ref,3),direct_full_mass=float(direct_ref.sum()),direct_below_m2=cum(direct_ref,1),
    direct_below_m15=cum(direct_ref,3),
    below_m2_p16_p84=np.percentile(boot[:,0],[16,84]).tolist(),below_m15_p16_p84=np.percentile(boot[:,:3].sum(1),[16,84]).tolist(),
    full_mass_p16_p84=np.percentile(boot.sum(1),[16,84]).tolist(),
    common_boss_apogee_ratio=float(mb.sum()/ma.sum()),
    common_ratio_p16_p84=np.percentile((draw@mb).sum(axis=1)/(draw@ma).sum(axis=1),[16,84]).tolist(),
    conversion_alternatives={name:dict(full=float(v[(v.conversion==name)].mass.sum()),below_m2=float(v[(v.conversion==name)].mass.iloc[0]),
                                       below_m15=float(v[(v.conversion==name)].mass.iloc[:3].sum()))
                             for name in ['E+','E-','i-','i+']},
    parent_median_ebv=dict(true=float(geo.ebv_true.median()),carton=float(geo.ebv_carton.median())),
    foreground='reference mean fraction over 2-10 deg, radial profile from a uniform foreground surface density per reporting bin; amplitude 0,1,2',
    response='empirical BOSS response from the stars with both spectra (scripts/boss_response.py); per-bin ratio fitted/direct applied to regions',
    envelope='A/B/C age priors x measured/0/0.15/0.3 alpha x 0/1/2 foreground, plus E(V-I) +-0.025 and inclination +-7 deg for the reference; not confidence bounds',
    bootstrap='500 nside16 pixel draws of the parent with selection weights and coverage recomputed, and resampled response overlap, for the full mass; 500 paired field cores for the ratio (weights fixed); conversions fixed',
    grid_age_max_gyr=float(fullgrid.age.max()),grid_mh_min=float(fullgrid.mh.min()),
    input_hashes={x:sha256(DATA/x) for x in ['boss_weights.parquet','apogee_weights.parquet','boss_parent.parquet','apogee_parent.parquet']})
# Write results/giant_mass.json and print it without the input hashes
(OUT/'giant_mass.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps({k:val for k,val in metadata.items() if k!='input_hashes'},indent=2),flush=True)
