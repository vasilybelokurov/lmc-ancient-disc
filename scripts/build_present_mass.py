"""Add living and present (living+remnant) masses to the formed-mass results: results/giant_present_mass.csv, the stellar
mass formed and present per [Fe/H] bin of Letter Fig. 1b (scripts/paper_letter_signal.py) and of paper Sect. 2.3.

Per isochrone (PARSEC gaiaEDR3 grid, sources.yaml entry parsec_gaiaEDR3), lmcold.surviving_mass.survival_grid gives the
fractions of the formed mass (Kroupa IMF, 0.08-100 Msun) now in living stars and in remnants (Conroy, Gunn & White 2009);
average_survival averages them over age with the same age weights as each formed-mass conversion. The formed masses of
scripts/build_giant_mass.py are multiplied by these fractions.
Inputs (results/): mass_conversion_factors.csv, giant_mass_variants.csv, giant_mass.csv, giant_mass_bootstrap.csv,
giant_mass_fields.csv (all from build_giant_mass.py).
Outputs (results/): mass_survival_grid.csv (fractions per [M/H] and age), mass_survival_factors.csv (conversion rows with the
fractions), giant_present_mass_variants.csv, giant_present_mass.csv (reference: formed, living, remnant and present mass
per reporting bin [Msun], cumulative, bootstrap 16-84% and model min/max), giant_present_mass_fields.csv,
giant_present_mass_common_totals.csv and giant_present_mass.json (totals, below -1.5 and -2, model range, definitions,
input sha256). The survival fractions come from the BOSS 'full' conversion (one per prior x alpha model x [Fe/H] bin) and
are applied to every variant. Used by figures/paper_mass.png, paper_mass_summary.png and Table 3.

Run: .venv/bin/python scripts/build_present_mass.py   (after build_giant_mass.py)
Run time: 2 s on a MacBook (M-series), docs/run_times.txt.
"""

# %% [markdown]
# ### Settings
# Modules and the output folder.

# %%
import json
import numpy as np
import pandas as pd
from lmcold.sources import ROOT,path,sha256
from lmcold.mass_conversion import REFERENCE_PRIOR
from lmcold.surviving_mass import survival_grid,average_survival

out=ROOT/'results'  # folder of the input and output tables

# %% [markdown]
# ### Load the data and the survival fractions
# The PARSEC grid (sources.yaml entry parsec_gaiaEDR3) gives, per isochrone, the fractions of the formed mass now in living
# stars and in remnants; they are averaged over age with the weights of each conversion of results/mass_conversion_factors.csv.
# The mass tables of build_giant_mass.py are read in the next steps, where they are used.

# %%
g=survival_grid(pd.read_parquet(path('parsec_gaiaEDR3')))  # living and remnant fractions of the formed mass per [M/H] and age
g.to_csv(out/'mass_survival_grid.csv',index=False)
f=pd.read_csv(out/'mass_conversion_factors.csv')  # one row per survey x field/region x prior x alpha model x [Fe/H] bin
f=average_survival(g,f)  # adds living_fraction, remnant_fraction, present_fraction (dimensionless)
f.to_csv(out/'mass_survival_factors.csv',index=False)
full=f[(f.survey=='BOSS')&(f.field=='full')]  # fractions of the whole-parent BOSS conversion

# %% [markdown]
# ### Present mass for every model alternative
# Each variant of results/giant_mass_variants.csv times the fractions of the whole-parent BOSS conversion with the same age
# weighting and [alpha/Fe] model (results/giant_present_mass_variants.csv).

# %%
v=pd.read_csv(out/'giant_mass_variants.csv').merge(full[['prior','alpha_model','feh','living_fraction','present_fraction','present_fraction_legacy_remnants']],on=['prior','alpha_model','feh'],validate='many_to_one')
# 'present_legacy_remnants': an alternative remnant prescription (lmcold.surviving_mass, legacy=True), kept as a sensitivity.
for name,fraction in [('living','living_fraction'),('present','present_fraction'),('present_legacy_remnants','present_fraction_legacy_remnants')]:
    v[name+'_mass']=v.mass*v[fraction]
    v[name+'_cumulative']=v.groupby(['conversion','prior','alpha_model','foreground','filled'])[name+'_mass'].cumsum()
v.to_csv(out/'giant_present_mass_variants.csv',index=False)

# %% [markdown]
# ### The reference table
# Formed, living, remnant and present mass per reporting bin and below each bin, with bootstrap percentiles (fractions fixed)
# and model ranges: results/giant_present_mass.csv, the table that Letter Fig. 1b and the mass figures read.

# %%
reference=full[(full.prior==REFERENCE_PRIOR)&(full.alpha_model=='measured')]  # AMR age weighting, measured alpha
t=pd.read_csv(out/'giant_mass.csv').merge(reference[['feh','living_fraction','remnant_fraction','present_fraction','present_fraction_legacy_remnants']],on='feh',validate='one_to_one')
t=t.rename(columns={'mass':'formed_mass'})  # Msun formed per reporting bin
boot=pd.read_csv(out/'giant_mass_bootstrap.csv').to_numpy()  # formed mass per bin, pixel and response bootstrap
for name in ['living','present']:
    t[name+'_mass']=t.formed_mass*t[name+'_fraction']
    t[name+'_cumulative']=t[name+'_mass'].cumsum()
    for p in ['p16','p84']:
        t[name+'_mass_'+p]=t['mass_'+p]*t[name+'_fraction']
    draws=boot*t[name+'_fraction'].to_numpy()  # (NBOOT, bins); fractions held fixed
    t[[name+'_cumulative_p16',name+'_cumulative_p84']]=np.percentile(draws.cumsum(axis=1),[16,84],axis=0).T
    # Model range over the filled variants (the rows of each feh group follow the sorted reporting bins).
    for suffix,method in [('min','min'),('max','max')]:
        t[name+'_mass_model_'+suffix]=v[v.filled].groupby('feh')[name+'_mass'].agg(method).to_numpy()
        t[name+'_cumulative_model_'+suffix]=v[v.filled].groupby('feh')[name+'_cumulative'].agg(method).to_numpy()
t['remnant_mass']=t.formed_mass*t.remnant_fraction  # Msun in white dwarfs, neutron stars and black holes
t.to_csv(out/'giant_present_mass.csv',index=False)

# %% [markdown]
# ### Summary numbers
# Totals below [Fe/H] = -1.5, below -2 and overall, the present-to-formed ratio and the model range of the present total.

# %%
# Sums [Msun] over the reporting bins: all bins, bins with [Fe/H] < -1.5 and bins with [Fe/H] < -2 (bin centres 'feh').
summary={}
for label,use in [('total',np.ones(len(t),bool)),('below_m15',t.feh<-1.5),('below_m2',t.feh<-2)]:
    q=t[use]
    summary[label]={c:float(q[c].sum()) for c in ['formed_mass','living_mass','remnant_mass','present_mass']}
    # Present-to-formed ratio, and the relative change of the present mass under the alternative remnant prescription.
    summary[label]['present_over_formed']=float(q.present_mass.sum()/q.formed_mass.sum())
    summary[label]['legacy_remnant_mass_change_fraction']=float((q.formed_mass*q.present_fraction_legacy_remnants).sum()/q.present_mass.sum()-1)
# Total present mass [Msun] of every filled model variant; its min and max give the model range.
totals=v[v.filled].groupby(['conversion','prior','alpha_model','foreground']).present_mass.sum()
summary['total_present_model_range']=[float(totals.min()),float(totals.max())]
summary['model_range_definition']='Age/alpha/foreground alternatives and the extinction/inclination conversion alternatives, not a confidence interval; fixed reference remnant prescription.'

# %% [markdown]
# ### Field cores
# Formed, living and present mass per core and survey, each with the fractions of its own conversion, and their sums; the
# summary is written to results/giant_present_mass.json.

# %%
# Field cores: each survey and core takes its own conversion's fractions.
fields=pd.read_csv(out/'giant_mass_fields.csv').merge(
    f[['survey','field','prior','alpha_model','feh','living_fraction','present_fraction']],
    on=['survey','field','prior','alpha_model','feh'],validate='one_to_one')
# Formed, living and present mass [Msun] per survey x core x prior x alpha model x [Fe/H] bin.
fields['formed_mass']=fields.mass
fields['living_mass']=fields.mass*fields.living_fraction
fields['present_mass']=fields.mass*fields.present_fraction
fields.to_csv(out/'giant_present_mass_fields.csv',index=False)
# Sum over the cores per survey, prior and alpha model; the reference model goes into the JSON summary.
common=fields.groupby(['survey','prior','alpha_model'])[['formed_mass','living_mass','present_mass']].sum().reset_index()
common.to_csv(out/'giant_present_mass_common_totals.csv',index=False)
reference=common.query("prior == @REFERENCE_PRIOR and alpha_model == 'measured'")
summary['common_field_totals']=reference.set_index('survey')[['formed_mass','living_mass','present_mass']].to_dict('index')
# Definitions and the sha256 of the input tables, stored with the numbers.
summary['common_field_definition']='Ten identical non-overlapping 0.8-degree field cores, 20.106 square degrees; velocity-selected, before foreground subtraction. Not full-LMC totals.'
summary['definition']='Present = living stars plus all retained remnants; returned gas excluded; no dynamical loss. Living stars separately tabulated.'
summary['remnants']='Conroy, Gunn & White 2009 section 2; alternative remnant prescription retained as sensitivity, not uncertainty interval.'
summary['input_hashes']={name:sha256(out/name) for name in ['giant_mass.csv','giant_mass_variants.csv','mass_conversion_factors.csv']}
(out/'giant_present_mass.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
