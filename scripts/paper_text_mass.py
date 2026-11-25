"""Knowledge-base text, chapter 'Stellar mass per metallicity' (extended draft Section 2.3 and its appendix on the mass): every number
of the text as a LaTeX macro (prefix \\ms), so that none is typed by hand.

As scripts/paper_text_shape.py: settings are read from the code (scripts/textnum.const, lmcold by import, or the source line where a
setting is written inline), results from the tables the figures and tables are drawn from. Literature values quoted with their
citation stay in the text. Three comparisons not written to any table are counted here from the catalogues: the Gaia XP metallicities
of the BOSS parent and the LMC clusters (as scripts/paper_mass.py, panel c), and the model foreground fraction of the outermost ring
(as scripts/fig_giant_foreground.py, panel a).
Inputs: results/{giant_present_mass, giant_mass_bootstrap, giant_mass_variants, giant_mass_response, giant_mass_common,
giant_mass_fields, mass_foreground_reference, giant_foreground_sideband, mass_alpha_trend, mass_regions, mass_conversion_factors,
giant_conversion, parent_extinction}.csv and {giant_mass, giant_present_mass, giant_counts, boss_response, boss_hot_bias}.json;
data/{boss_parent, xp_andrae, giant_foreground}.parquet, data/clusters_grocholski06.csv and the BOSS giants of
lmcold.mass_samples.samples(); settings of scripts/{build_giant_mass, boss_response, fig_giant_foreground, paper_parent_extinction}.py,
lmcold.{constants, extinction, isochrones, magcloud, mass_conversion, mass_samples, selection, foreground}.
Writes results/macros_mass.tex.
Run: .venv/bin/python scripts/manifest.py run paper_text_mass.py"""

# %% [markdown]
# ### Settings and helpers
# Imports, the dictionary M of macros (macro name without its prefix: the text it stands for) and the functions that format
# numbers as the text writes them. The inputs listed in the header are read in the step that uses them.

# %%
import json
import re

import healpy as hp
import numpy as np
import pandas as pd

from lmcold import foreground, isochrones, magcloud, mass_conversion, mass_samples, selection
from lmcold.constants import LMC, LMC_AMR_POVICK24, LMC_DEPTH_MAG, LMC_DISTANCE_MODULUS, LMC_MASS_LITERATURE
from lmcold.extinction import EVI_PER_EBV
from lmcold.mass_samples import FEH_CENTRES, FEH_EDGES
from lmcold.sources import DATA, ROOT
from textnum import const, default, listing, num, sigfig

R = ROOT / 'results'   # folder of the result tables
M = {}   # macro name (without the \ms prefix) -> text


def f1(x):
    """x with one decimal."""
    return f'{x:.1f}'


def f2(x):
    """x with two decimals."""
    return f'{x:.2f}'


def pct(x):
    """Fraction x as a whole-number percentage."""
    return num(round(100 * x))


def src(path, pattern):
    """The number captured by `pattern` in the source file `path` (a setting written inline)."""
    return float(re.search(pattern, (ROOT / path).read_text()).group(1))


def bounds(v, lo, hi, fmt=f2):
    """Value with its upper and lower distances to the 16th/84th percentiles, as three strings."""
    return fmt(v), fmt(hi - v), fmt(v - lo)


def sci2(v, lo, hi, e):
    """v, hi - v and v - lo in units of 10^e, v to two significant figures and the distances to the same decimals."""
    d = max(1 - int(np.floor(np.log10(v / 10 ** e))), 0)   # decimals that give v two significant figures
    return tuple(f'{x / 10 ** e:.{d}f}' for x in (v, hi - v, v - lo))


def binname(i):
    """Reporting [Fe/H] bin i as text, e.g. '−0.75 to −0.5'."""
    return f'{num(FEH_EDGES[i])} to {num(FEH_EDGES[i + 1])}'


# Result tables used in several steps: formed and present mass per [Fe/H] bin (t), mass summary (gm), present-day mass (gp),
# member counts (gc), BOSS response (br), hot-giant bias (hb).
t = pd.read_csv(R / 'giant_present_mass.csv')
gm = json.loads((R / 'giant_mass.json').read_text())
gp = json.loads((R / 'giant_present_mass.json').read_text())
gc = json.loads((R / 'giant_counts.json').read_text())
br = json.loads((R / 'boss_response.json').read_text())
hb = json.loads((R / 'boss_hot_bias.json').read_text())
# Module-level constants of four scripts, read from their source without running them.
bgm, bresp, ffg, ppe = const('build_giant_mass'), const('boss_response'), const('fig_giant_foreground'), const('paper_parent_extinction')

# %% [markdown]
# ### Samples, bins and foreground
# The mass samples and [Fe/H] bins, the Milky Way foreground, and the stars with both spectra.

# %%
# Mass samples and bins (lmcold.mass_samples)
# Velocity window [km/s] read from the line of lmcold.mass_samples that applies it.
vlos = re.search(r'\(b\.vlos > (\d+)\) & \(b\.vlos < (\d+)\)', (ROOT / 'src/lmcold/mass_samples.py').read_text()).groups()
# Window, member counts (thin-space thousands), radius [deg], reporting-bin width and the edges of the inner bins [dex].
M.update(VlosLo=vlos[0], VlosHi=vlos[1], NBoss=f"{gc['boss_members']:,}".replace(',', '\\,'),
         NApogee=f"{gc['apogee_members']:,}".replace(',', '\\,'), Rad=num(gm['full_radius_deg']), BinW=num(FEH_EDGES[2] - FEH_EDGES[1]),
         FehLo=num(FEH_EDGES[1]), FehHi=num(FEH_EDGES[-2]))

# Corrected counts: Milky Way foreground (results/mass_foreground_reference.csv, giant_foreground_sideband.csv)
fr = pd.read_csv(R / 'mass_foreground_reference.csv')   # foreground fraction f_mw per [Fe/H] bin (feh_med)
# Reference ring of lmcold.foreground.radial_fraction (default argument) [deg].
ref = default(foreground.radial_fraction, 'reference')
sb = pd.read_csv(R / 'giant_foreground_sideband.csv')   # sideband density and error per ring, n_raw stars
mean = np.sum(sb.density / sb.err ** 2) / np.sum(1 / sb.err ** 2)   # as scripts/fig_giant_foreground.py
top = sb.loc[sb.n_raw.idxmax()]   # the ring with the most sideband stars
# Foreground fractions [%] (metal-poor range, metal-rich mean), reference ring, sideband width, chi2 about the weighted mean.
M.update(FmwPoorLo=pct(fr[fr.feh_med < -1.5].f_mw.min()), FmwPoorHi=pct(fr[fr.feh_med < -1.5].f_mw.max()),
         FmwRich=sigfig(100 * fr[fr.feh_med > -1].f_mw.mean()), RefLo=num(ref[0]), RefHi=num(ref[1]), Side=num(ffg['SIDE']),
         SideChi=f1(np.sum(((sb.density - mean) / sb.err) ** 2)), SideDof=num(len(sb) - 1), SideN=num(top.n_raw),
         SideRhoLo=num(top.rho_lo), SideRhoHi=num(top.rho_hi))
# Model foreground fraction in the outermost 1-deg ring, per [Fe/H] range (scripts/fig_giant_foreground.py, panel a)
b, a, _ = mass_samples.samples()   # BOSS and APOGEE mass samples
# Stars with both spectra (lmcold.mass_samples.overlap): their median BOSS S/N against that of the BOSS giants in the faintest G bin of
# the response; and, at G >= XP_G, the median XP [M/H] - APOGEE [Fe/H] of metal-rich giants in the APOGEE [Fe/H] bins XP_RICH.
XP_G, XP_RICH = 16.5, (-0.9, -0.7, -0.5)   # mag; dex
o = mass_samples.overlap(b, a).merge(b[['source_id', 'mh_xp']], on='source_id')   # overlap stars with their XP [M/H]
faint = b.g >= mass_samples.RESPONSE_G_EDGES[-2]   # BOSS giants in the faintest G bin of the response
of = o[(o.g >= XP_G) & o.mh_xp.notna()]   # faint overlap stars with an XP [M/H]
# Median XP [M/H] - APOGEE [Fe/H] [dex] in each APOGEE [Fe/H] bin of XP_RICH.
dxp = [np.median((of.mh_xp - of.feh_a)[(of.feh_a >= lo) & (of.feh_a < hi)]) for lo, hi in zip(XP_RICH[:-1], XP_RICH[1:])]
M.update(SnrBoth=num(round(o.snr.median())), SnrFaint=num(round(b.snr[faint].median())), SnrGLo=num(mass_samples.RESPONSE_G_EDGES[-2]),
         SnrGHi=num(mass_samples.RESPONSE_G_EDGES[-1]), XpOffLo=f2(-max(dxp)), XpOffHi=f2(-min(dxp)), XpG=num(XP_G),
         XpRichLo=num(XP_RICH[0]), XpRichHi=num(XP_RICH[-1]))
b = b.merge(pd.read_parquet(DATA / 'giant_foreground.parquet')[['source_id', 'rho', 'fill', 'f_mw']], on='source_id')
outer = b[b.rho >= foreground.RINGS[-2]]   # BOSS giants in the outermost ring of lmcold.foreground.RINGS
w = outer.w * outer['fill']   # selection weight x coverage fill
# Weighted mean foreground fraction for [Fe/H] < -1.5, -1.5 to -1 and >= -1; quoted rounded to 10 per cent.
fo = [np.average(outer.f_mw[m], weights=w[m]) for m in (outer.feh < -1.5, (outer.feh >= -1.5) & (outer.feh < -1), outer.feh >= -1)]
M.update(FgOutLo=num(10 * round(10 * min(fo))), FgOutHi=num(10 * round(10 * max(fo))), FgRingLo=num(foreground.RINGS[-2]),
         FgRingHi=num(foreground.RINGS[-1]))

# %% [markdown]
# ### Giants per unit mass
# The PARSEC conversion from giants to formed mass, the age windows and the efficiency against age.

# %%
# Giants per unit mass (PARSEC conversion)
# IMF mass range [Msun] from the mass-type string of results/giant_mass.json.
imf = re.search(r'Kroupa ([0-9.]+)\D+([0-9.]+) Msun', gm['mass_type']).groups()
reg = pd.read_csv(R / 'mass_regions.csv')   # sky regions of the conversion, with their distance-modulus offset dmu [mag]
# Extinction coefficients written inline in lmcold.isochrones: the two factors of A_V per E(B-V), and A_H per E(B-V).
av = re.search(r'av, av_carton = ([.0-9]+) \* ([.0-9]+) \* ebv', (ROOT / 'src/lmcold/isochrones.py').read_text()).groups()
ah = src('src/lmcold/isochrones.py', r'Hmag\.to_numpy\(\) \+ LMC_DISTANCE_MODULUS \+ dmu \+ ([.0-9]+) \* ebv')
# IMF range, distance modulus, inclination and position angle [deg], depth [mag], E(V-I)/E(B-V), G limit, median E(B-V) of the
# parent (true and carton), subdivisions, regions, rings, dmu range and A_H / A_G.
M.update(ImfLo=imf[0], ImfHi=imf[1], DM=num(LMC_DISTANCE_MODULUS), Inc=num(LMC['inc']), Theta=num(LMC['theta']),
         Depth=num(LMC_DEPTH_MAG), EviEbv=f'{EVI_PER_EBV:.3f}', GLim=num(selection.PG_EDGES[-1]),
         EbvTrue=f"{gm['parent_median_ebv']['true']:.3f}", EbvCarton=f"{gm['parent_median_ebv']['carton']:.3f}",
         NSub=num(default(isochrones.selected_count, 'subdivisions')), NReg=num(len(reg)), Rings=listing([num(x) for x in bgm['RINGS']]),
         DmuLo=f2(reg.dmu.min()), DmuHi='+' + f2(reg.dmu.max()), AhAg=f2(ah / (magcloud.A_OVER_AV['G'] * float(av[0]) * float(av[1]))))
mc = (ROOT / 'src/lmcold/mass_conversion.py').read_text()
# The two coefficients of the [M/H]-[Fe/H]-[alpha/Fe] relation written in lmcold.mass_conversion (log10(A 10^al + B)).
sal = re.search(r'np\.log10\(([.0-9]+)\*10\*\*al\+([.0-9]+)\)', mc).groups()
at = pd.read_csv(R / 'mass_alpha_trend.csv')   # measured [alpha/Fe] against [Fe/H]
amr = LMC_AMR_POVICK24   # age-metallicity relation of lmcold.constants: feh, age [Gyr], sigma per point
# Relation coefficients, AMR end points and scatter range, and the oldest age and lowest [M/H] of the PARSEC grid.
M.update(SalA=num(float(sal[0])), SalB=num(float(sal[1])), AmrNow=f2(amr['feh'][0]), AmrOld=f2(amr['feh'][-1]), AmrAge=num(amr['age'][-1]), AmrSigLo=f2(min(amr['sigma'])),
         AmrSigHi=f2(max(amr['sigma'])), GridAge=f2(gm['grid_age_max_gyr']), GridMh=f2(gm['grid_mh_min']))
for k, i in zip(('One', 'Two', 'Three', 'Four'), (0, 2, 4, len(at) - 1)):   # four rows of the trend, from metal-poor to metal-rich
    M.update({f'Al{k}': f2(at.alpha[i]), f'AlFeh{k}': f1(at.feh[i])})
# Window A against the reference weighting: change of the masses below -2, -1.5, -1 and of the total (reference conversion,
# measured alpha, reference foreground, full coverage)
v = pd.read_csv(R / 'giant_mass_variants.csv')
vr = v[(v.conversion == 'reference') & (v.alpha_model == 'measured') & (v.foreground == 1) & v.filled].pivot(
    index='feh', columns='prior', values='mass').sort_index().cumsum()
# Fractional change, window A over AMR weighting, of the cumulative masses at rows 0, 2, 4 (upper edges -2, -1.5, -1) and the total.
ca = (vr.A / vr.AMR).iloc[[0, 2, 4, -1]] - 1
M.update(AWinLo=num(round(100 * ca.min())), AWinHi=num(round(100 * ca.max())))
# Efficiency eta against age: ratio between the grid ages nearest ETA_AGES at the lowest and highest [M/H] of figure (a)
ETA_AGES = (2., 12.)   # Gyr
gc_ = pd.read_csv(R / 'giant_conversion.csv').query("survey == 'BOSS'")   # n_box per (mh, age) of the BOSS grid
fac = {}   # [M/H] -> n_box at 2 Gyr / n_box at 12 Gyr
for mh, q in gc_.groupby('mh'):
    # n_box at the grid age nearest each of ETA_AGES.
    n = [q.n_box.iloc[np.argmin(abs(q.age - a))] for a in ETA_AGES]
    fac[mh] = n[0] / n[1]
M.update(EtaAgeLo=num(ETA_AGES[0]), EtaAgeHi=num(ETA_AGES[1]), EtaMhLo=num(min(fac)), EtaMhHi=num(max(fac)),
         EtaFacLo=f1(fac[min(fac)]), EtaFacHi=f1(fac[max(fac)]))

# %% [markdown]
# ### BOSS metallicity response and hot giants
# The response of the BOSS [Fe/H] and the offsets of hot giants.

# %%
# The BOSS metallicity response (results/boss_response.json)
bias = pd.DataFrame(br['bias'])   # per reporting bin: feh, n, median BOSS - APOGEE [dex]
mid = (bias.feh > FEH_EDGES[1]) & (bias.feh < FEH_EDGES[3])   # -2 < [Fe/H]_APOGEE < -1.5
c = br['closure']['all']   # closure on all stars
b2 = c['below_-2.0']   # its totals below [Fe/H] = -2
gr = pd.read_csv(R / 'giant_mass_response.csv')
# Distinct fitted / direct factors of the response correction, in table order.
fd = gr.drop_duplicates('fitted_over_direct').fitted_over_direct
# Overlap size, bias outside -2..-1.5 (rounded up to 0.01) and inside it (sign reversed), response bin settings, closure chi2,
# BOSS observed / APOGEE below -2, the measured-bin width, overlap stars below -1.5 and the four response factors.
M.update(NOverlap=f"{br['overlap']:,}".replace(',', '\\,'), BiasOk=f2(np.ceil(100 * bias['median'][~mid].abs().max()) / 100),
         BiasLo=f2(-bias['median'][mid].max()), BiasHi=f2(-bias['median'][mid].min()),
         GEdges=listing([num(x) for x in mass_samples.RESPONSE_G_EDGES[1:-1]]), SnrEdge=num(mass_samples.RESPONSE_SNR_EDGES[0]),
         RespSplit=num(src('src/lmcold/mass_samples.py', r'o\.feh_a\.to_numpy\(\) < (-[.0-9]+)')),
         RespChi=f1(c['chi2']), RespDof=num(c['dof']), DirectRatio=f2(b2['observed'] / b2['apogee']),
         DirectRatioOne=f1(b2['observed'] / b2['apogee']),
         MeasW=num(mass_samples.MEASURED_EDGES[2] - mass_samples.MEASURED_EDGES[1]),
         NPoor=num(sum(n for f, n in zip(bias.feh, bias.n) if f < FEH_EDGES[3])),
         FacOne=f2(fd.iloc[0]), FacTwo=f2(fd.iloc[1]), FacThree=f2(fd.iloc[2]), FacFour=f2(fd.iloc[3]))
# Observed / predicted below -2 with its distances to the 16th and 84th percentiles.
M.update(zip(('RespTwo', 'RespTwoUp', 'RespTwoDn'), bounds(b2['ratio'], *b2['ratio_p16_p84'])))
# Hot giants (results/boss_hot_bias.json): offsets of calibrated BOSS from APOGEE [Fe/H] per BOSS Teff bin, low and high S/N
te = hb['t_edges']   # K, Teff bin edges
k3 = te.index(4300)   # index of the bin starting at 4300 K
# Offsets [dex] (low and high S/N) in the 4300-4500 K bin (warm), the 4500-4700 K bin (hot), and the largest below 4300 K.
warm = (hb['offset_low'][k3], hb['offset_high'][k3])
hot = (hb['offset_low'][k3 + 1], hb['offset_high'][k3 + 1])
cool = max(abs(x) for x in hb['offset_low'][:k3] + hb['offset_high'][:k3])
# Teff edges, offset ranges, S/N split, hot low-S/N weight fractions [%], and the settings of the metal-rich split
# (rings [deg], G and colour edges [mag], minimum stars per cell).
M.update(TeffA=num(te[k3]), TeffB=num(te[k3 + 1]), TeffC=num(te[k3 + 2]), OffWarmLo=f2(min(warm)), OffWarmHi=f2(max(warm)),
         OffHotLo=f2(min(hot)), OffHotHi=f2(max(hot)), OffCool=f2(cool), SnrSplit=num(hb['snr_split']),
         HotWeight=pct(hb['target_weight_hot_lowsnr']), HotOverlap=pct(hb['overlap_fraction_hot_lowsnr']),
         HotAbove=pct(hb['boss_frac_above_m05']['hot_lowsnr']),
         SplitRings=', '.join(f'{a:g}–{b:g}' for a, b in zip([0] + mass_samples.SPLIT_RING_EDGES,
                                                              mass_samples.SPLIT_RING_EDGES + [gm['full_radius_deg']])),
         SplitG=listing([num(x) for x in mass_samples.SPLIT_G_EDGES]), SplitC=listing([num(x) for x in mass_samples.SPLIT_C_EDGES]),
         NMinSplit=num(mass_samples.NMIN_SPLIT))

# %% [markdown]
# ### Mass: present-day mass, uncertainties and result
# The present-day mass, the uncertainty budget, the masses per bin and the APOGEE check in the field cores.

# %%
# Present-day mass (results/giant_present_mass.json)
# Present over formed mass: all stars and [Fe/H] < -1.5.
M.update(PresAll=f2(gp['total']['present_over_formed']), PresPoor=f2(gp['below_m15']['present_over_formed']))

# Uncertainties (results/giant_mass.json; scripts/build_giant_mass.py)
alt = gm['conversion_alternatives']   # full mass under alternative conversions (E+-, i+-)
alphas = sorted(float(a) for a in v.alpha_model.unique() if a != 'measured')
# Bootstrap draws and nside, alpha models, alpha cap, E(B-V) change, the largest mass change from E(B-V) [%], the inclination
# change [deg] and the largest mass change from it [%, rounded up to 0.1].
M.update(NBoot=num(bgm['NBOOT']), Nside=re.search(r'nside(\d+)', gm['bootstrap']).group(1), Alphas=listing([num(a) for a in alphas]),
         AlphaCap=num(src('src/lmcold/mass_conversion.py', r'min\(float\(alpha\),([.0-9]+)\)')),
         DEvi=num(src('scripts/build_giant_mass.py', r'DEBV=([.0-9]+)/EVI_PER_EBV')),
         EChange=num(round(100 * max(abs(alt[k]['full'] / gm['full_mass'] - 1) for k in ('E+', 'E-')))),
         DInc=num(bgm['DINC']),
         IncChange=f1(np.ceil(1000 * max(abs(alt[k]['full'] / gm['full_mass'] - 1) for k in ('i+', 'i-'))) / 10))

# Result (results/giant_present_mass.csv, giant_mass_bootstrap.csv, giant_mass_variants.csv)
tot = t.iloc[-1]   # last row: cumulative over all bins
# Total formed mass [1e9 Msun] with its distances to the 16th and 84th percentiles.
M.update(zip(('Tot', 'TotUp', 'TotDn'), (f2(x / 1e9) for x in (tot.cumulative, tot.cumulative_p84 - tot.cumulative,
                                                                tot.cumulative - tot.cumulative_p16))))
# Range of the total over the model variants, present-day total and its bootstrap range [1e9 Msun].
M.update(TotMLo=f2(tot.cumulative_model_min / 1e9), TotMHi=f2(tot.cumulative_model_max / 1e9), Pres=f2(tot.present_cumulative / 1e9),
         PresBootLo=f2(tot.present_cumulative_p16 / 1e9), PresBootHi=f2(tot.present_cumulative_p84 / 1e9))
i2, i15 = 0, 2   # rows whose upper edge is -2 and -1.5
# Mass below -2 and -1.5 [1e7 Msun] and as a percentage of the total.
for k, i in (('Two', i2), ('OneFive', i15)):
    r = t.iloc[i]
    M.update(zip((f'Below{k}', f'Below{k}Up', f'Below{k}Dn'), sci2(r.cumulative, r.cumulative_p16, r.cumulative_p84, 7)))
    M[f'Pct{k}'] = f1(100 * r.cumulative / tot.cumulative)
rich = t.feh.to_numpy() > -.5   # bins centred above [Fe/H] = -0.5
boot = pd.read_csv(R / 'giant_mass_bootstrap.csv').to_numpy()   # (draws, bins) formed mass
fb = boot[:, rich].sum(1) / boot.sum(1)   # metal-rich fraction per draw
# Metal-rich fraction for each model variant (conversion, prior, alpha model, foreground).
fm = v[v.filled].groupby(['conversion', 'prior', 'alpha_model', 'foreground']).apply(
    lambda d: d.mass[d.feh > -.5].sum() / d.mass.sum(), include_groups=False)   # as scripts/paper_table_mass_summary.py
f0 = t.formed_mass[rich].sum() / t.formed_mass.sum()   # adopted metal-rich fraction
# Fractions [%] (adopted, bootstrap range, model range), and the masses below -2 and -1.5 without the response [1e7 Msun].
M.update(FracRich=pct(f0), FracRichLo=pct(np.percentile(fb, 16)), FracRichHi=pct(np.percentile(fb, 84)), FracRichMLo=pct(fm.min()),
         FracRichMHi=pct(fm.max()), NoRespTwo=f1(gm['direct_below_m2'] / 1e7), NoRespOneFive=f1(gm['direct_below_m15'] / 1e7))
peak = int(np.argmax(t.formed_mass))   # bin of the largest formed mass
# Mean factor of rise per bin from bin 1 to the peak, and the peak bin's range.
M.update(Rise=f1((t.formed_mass[peak] / t.formed_mass[1]) ** (1 / (peak - 1))), PeakBin=binname(peak))
# APOGEE check in the field cores: radius from the area of the caps, 2 pi (1 - cos r) per core
rcore = np.degrees(np.arccos(1 - gm['common_area_deg2'] * (np.pi / 180) ** 2 / (2 * np.pi * gm['fields'])))
# Number of fields, core radius [deg], coverage threshold [%], common area [deg^2].
M.update(NCore=num(gm['fields']), RCore=f1(rcore), CoreCov=num(100 * src('scripts/build_giant_mass.py', r'coverage>=([.0-9]+)')),
         CoreArea=f1(gm['common_area_deg2']))
# BOSS / APOGEE mass ratio in the cores with its distances to the percentiles.
M.update(zip(('CoreRatio', 'CoreRatioUp', 'CoreRatioDn'), bounds(gm['common_boss_apogee_ratio'], *gm['common_ratio_p16_p84'])))

# %% [markdown]
# ### Comparisons
# Published masses, the Gaia XP and cluster comparisons of panel (c), and the published field-star distributions.

# %%
# Published masses (lmcold.constants.LMC_MASS_LITERATURE)
lit = {r['key']: r['mass'] / 1e9 for r in LMC_MASS_LITERATURE}   # key -> mass [1e9 Msun]
# Present-day masses, van der Marel (2002) excluded from the range (quoted separately as LitVdM).
pres = [r['mass'] / 1e9 for r in LMC_MASS_LITERATURE if r['kind'] == 'present' and r['key'] != 'vanderMarel2002']
M.update(LitHZ=f1(lit['harris_zaritsky2009']), LitMazzi=f1(lit['mazzi2021']), LitPresLo=f1(min(pres)), LitPresHi=f1(max(pres)),
         LitVdM=f1(lit['vanderMarel2002']))
# Panel (c) comparisons: Gaia XP [M/H] of the BOSS parent as mass, and the clusters (as scripts/paper_mass.py)
# Conversion efficiency eta per [Fe/H] bin: whole field, reference prior, measured alpha.
conv = pd.read_csv(R / 'mass_conversion_factors.csv').query(
    "field == 'full' and prior == @mass_conversion.REFERENCE_PRIOR and alpha_model == 'measured'").sort_values('feh')
# XP [M/H] [dex] of the eligible BOSS parent stars (Andrae et al. 2023 catalogue).
xp = pd.read_parquet(DATA / 'boss_parent.parquet', columns=['source_id', 'eligible']).query('eligible').merge(
    pd.read_parquet(DATA / 'xp_andrae.parquet', columns=['source_id', 'mh_xp']), on='source_id').mh_xp.dropna().to_numpy()
# [M/H] to [Fe/H] with the measured alpha trend and the relation of lmcold.mass_conversion.
feh_xp = xp - np.log10(float(sal[0]) * 10 ** np.interp(xp, at.feh, at.alpha) + float(sal[1]))
k = np.digitize(feh_xp, FEH_EDGES) - 1   # reporting bin of each star
mx = np.bincount(k, 1 / conv.eta.to_numpy()[k], len(FEH_CENTRES))   # formed mass per reporting bin, up to a constant
cl = pd.read_csv(DATA / 'clusters_grocholski06.csv').feh.dropna()   # cluster [Fe/H] [dex]
# KDE bandwidths of panel (c), read from scripts/paper_mass.py.
kde = re.search(r'KDE_BW,NBOOT_KDE=dict\(xp=([.0-9]+),clusters=([.0-9]+)\)', (ROOT / 'scripts/paper_mass.py').read_text()).groups()
M.update(KdeXp=num(float(kde[0])), KdeCl=num(float(kde[1])), CoreMinN=num(src('scripts/paper_mass.py', r'\(c\.boss_raw>=(\d+)\)')),
         PeMinN=num(src('scripts/paper_parent_extinction.py', r'binned\.n_g >= (\d+)')),
         FgRingW=num(foreground.RINGS[1] - foreground.RINGS[0]))
# XP sample size, its mass fractions below -1.5 and -2 and above -0.5, its peak bin; clusters: number, median, fraction above -0.5.
M.update(NXp=f'{len(xp):,}'.replace(',', '\\,'), XpOneFive=f1(100 * mx[:i15 + 1].sum() / mx.sum()),
         XpTwo=f1(100 * mx[0] / mx.sum()), XpRich=pct(mx[rich].sum() / mx.sum()), XpPeakBin=binname(int(np.argmax(mx))),
         NCl=num(len(cl)), ClMed=f2(cl.median()), ClRich=pct((cl > -.5).mean()))
# Published field-star distributions (results/mdf_literature.json, from scripts/paper_mdf_literature.py)
md = json.loads((R / 'mdf_literature.json').read_text())
# Panel (a) (bar) and panel (b) of the literature figure.
pa, pb = md['(a) Bar'], next(v for k, v in md.items() if k.startswith('(b)'))
off = (pa['Cole+05, Ca II triplet']['median'] - pa['BOSS, as measured']['median'],
       pb['Carrera+08, Ca II triplet']['median'] - pb['BOSS, as measured']['median'])
sc = md['scale_check']   # BOSS minus literature [Fe/H] of matched stars
# Offset range, mean scale difference (sign reversed, 1 significant figure), and the matched-star counts.
M.update(MdfOffLo=f2(min(off)), MdfOffHi=f2(max(off)),
         MdfScale=sigfig(-np.mean([sc['BOSS - Carrera+08']['median'], sc['BOSS - Cole+05']['median']])),
         NCarMatch=num(sc['BOSS - Carrera+08']['n']), NColeMatch=num(sc['BOSS - Cole+05']['n']))

# %% [markdown]
# ### Appendix
# The response by APOGEE field type, the field cores bin by bin, and dust.

# %%
# Appendix: response by APOGEE field type (results/boss_response.json)
# Observed / predicted below -1.5 per APOGEE field type, with distances to the percentiles.
for k, key in (('Late', 'LMC18-35'), ('Early', 'LMC1-17 (W+D)')):
    x = br['closure'][key]['below_-1.5']
    M.update(zip((f'Resp{k}', f'Resp{k}Up', f'Resp{k}Dn'), bounds(x['ratio'], *x['ratio_p16_p84'])))
M.update(NBootResp=num(bresp['NBOOT']))
# Appendix: the field cores, bin by bin (results/giant_mass_common.csv, giant_mass_fields.csv)
cm = pd.read_csv(R / 'giant_mass_common.csv')   # BOSS / APOGEE in the field cores per bin
# Ratios of bin 0 (below -2) and bin 2.
for k, i in (('CoreTwo', 0), ('CoreMid', 2)):
    r = cm.iloc[i]
    M.update(zip((k, k + 'Up', k + 'Dn'), bounds(r.ratio, r.ratio_p16, r.ratio_p84)))
fields = pd.read_csv(R / 'giant_mass_fields.csv').field.unique()   # APOGEE field names 'LMC<n>'
up = cm.feh > FEH_EDGES[5]   # bins above [Fe/H] = -1
# APOGEE stars below -2 in the cores, bin-2 range, fields LMC1-17, BOSS / APOGEE mass above -1, H limit and lower mass edge.
M.update(CoreTwoN=num(cm.apogee_raw[0]), CoreMidBin=binname(2), NCoreWD=num(sum(int(f[3:]) <= 17 for f in fields)),
         CoreRich=f2(cm.boss_mass[up].sum() / cm.apogee_mass[up].sum()), HLim=num(default(isochrones.count_grid, 'h_limit')),
         HLo=num(src('src/lmcold/isochrones.py', r'\(m >= ([.0-9]+)\)')))
# Appendix: dust (results/parent_extinction.csv, scripts/paper_parent_extinction.py)
pe = pd.read_csv(R / 'parent_extinction.csv')
# Ratio q = observed / predicted parent counts per E(V-I) bin, for the region and the cell models.
rg, ce = (pe[pe.model == m].reset_index(drop=True) for m in ('region', 'cell'))
GROUPS = dict(A=[2], B=[3, 4, 5], C=[6], D=[7])   # rows (E(V-I) bins) of the region model quoted in the text, alone or pooled
dusty = rg.iloc[GROUPS['B'][0]:]   # cells above the lower edge of group B


def pooled(s):
    """q of several E(V-I) bins together: sum N_G over the sum of the predicted counts N_G / q."""
    return s.n_g.sum() / (s.n_g / s.q).sum()


# Pooled q and the E(V-I) range of each group.
for k, rows in GROUPS.items():
    s = rg.iloc[rows]
    M.update({f'Q{k}': f2(pooled(s)), f'Q{k}Lo': f2(s.evi_lo.iloc[0]), f'Q{k}Hi': f2(s.evi_hi.iloc[-1])})
    if k in 'CD':   # half the bootstrap 16th-84th range, quoted for the two dustiest bins
        M[f'Q{k}Err'] = f2((s.q_p84 - s.q_p16).iloc[0] / 2)
# Cell size [arcmin], radius, ages and [M/H] of the extinction check, low-E(V-I) edge, share of parent stars in dusty cells [%],
# stars missing there [%], and the largest excess of q above 1 [%].
M.update(Cell=f1(hp.nside2resol(ppe['NSIDE'], arcmin=True)), RMax=num(ppe['RMAX']), AgeLo=num(ppe['AGES'][0]),
         AgeHi=num(ppe['AGES'][1]), MhLo=f1(min(ppe['MH'])), MhHi=f1(max(ppe['MH'])), Low=f2(ce.evi_hi[1]),
         DustyShare=f1(100 * dusty.n_g.sum() / rg.n_g.sum()), Missing=f1(100 * (dusty.n_g / dusty.q - dusty.n_g).sum() / rg.n_g.sum()),
         Overshoot=sigfig(100 * (ce.q[:GROUPS['D'][0] + 1].max() - 1)))   # over the bins quoted, to the end of group D

# %% [markdown]
# ### Write the macros
# One \newcommand per entry of M, with the prefix \ms, into results/macros_mass.tex; the text shows them as macros, so no number
# is typed by hand.

# %%
# Header comment, then one \newcommand line per macro.
lines = ['% Numbers of the KB chapter Stellar mass per metallicity (scripts/paper_text_mass.py); do not edit.']
lines += [rf'\newcommand{{\ms{k}}}{{{x}}}' for k, x in M.items()]
(R / 'macros_mass.tex').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
