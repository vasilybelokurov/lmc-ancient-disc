"""APOGEE selection function: weights for the quality stars of the LMC1-35 RGB targeting (data/apogee_parent.parquet).

Candidates: astro_ok and no bright neighbour; a star in two fields keeps the row of the field that observed it (else the first).
Selection groups: field x zone (r < 0.8 deg | 0.8-0.95 deg; the 2017 designs stop at 0.8 deg) x class (BrtRGB one H bin; FntRGB 0.1-mag H
bins, filled brightest first) x colour bin (x = position across the polygon, edges XEDGES).
Quality: observed and lmcold.giants.apogee_spectrum_ok with S/N >= SNR_MIN, giant or not (lmcold.apogee_target.weights); the giant
cut and the velocity window are applied after weighting (no VSCATTER cut: it fails a quarter of the stars below [Fe/H] = -2).
  w = N_cand / N_quality x pass factor for quality stars in covered groups: BrtRGB groups with >= 3 quality stars; FntRGB groups
  with >= 2 and f_obs >= 0.2 (drops the thin bins at each field's fibre limit); any group with f_q >= 0.5 and >= 1 quality star.
  Pass factor (lmcold.selection, as for BOSS): pass groups H bin (PG_H_EDGES) x XP [M/H] of Andrae et al. 2023; ASPCAP fails
  (STAR_BAD, no FE_H) for 11% of the bright XP < -1.5 stars against 1.6% of XP > -1, at S/N ~ 150.
Checks (results/apogee_selection.json):
  T1 polygon reproduction: LMC1-17 targets with the W+D bit (APOGEE2_TARGET1 bit 7) inside the parent polygons (>= 99.5%);
     LMC18-35 Magellanic targets with H >= 12.35 and 0.5 < J - Ks < 1.3 inside them (>= 99%).
  T5 the weighted mean J - Ks of quality stars equals the mean of the covered candidates to < 0.005 mag (~0.03 dex in [Fe/H]),
     per group (W+D fields or not) x class.
  T7 XP closure: weighted quality stars / covered candidates, fractions of XP [M/H] below -1 and -1.5, with and without the pass
     factor, for the W+D fields and LMC18-35 separately. The W+D fields give 1.41 +- 0.33 below -1.5 without the pass factor
     (field bootstrap): the DDO51 giant flag, which is not reproduced, may favour metal-poor giants. Not corrected; a stated
     systematic of the APOGEE metal-poor counts.
  T6 S/N neutrality: weighted fractions of members (150 < v < 420 km/s) below [Fe/H] = -1.5 and -2 for S/N >= 30, 50, 70, in
     the groups covered at all three; they must agree within the group-bootstrap error of the first (0.0023). A lower S/N cut
     only adds coverage (faint groups).
Writes data/apogee_weights.parquet (one row per candidate star).

sources.yaml: reads release_apogee_parent and apogee_dr17_allvisit (for T1), plus data/xp_andrae.parquet
(scripts/build_xp_andrae.py, not registered); writes release_apogee_weights. Output columns: field, apogee_id, source_id,
ra, dec [deg], j, h, k [mag], cls, group, observed, quality, giant, covered, w, feh [dex, ASPCAP], v [km/s, heliocentric],
mh_xp [dex]. Note: lmcold.selection.pass_factor counts only groups with >= NMIN_GOOD = 3 quality stars, while the APOGEE
coverage rule above also admits FntRGB groups with 2 and any group with 1 (f_q >= 0.5). The weights serve
figures/paper_selection.png (c) and the APOGEE mass (build_giant_mass.py).

Run: .venv/bin/python scripts/apogee_selection.py   (after build_apogee_parent.py and build_xp_andrae.py)
"""
import json

import numpy as np
import pandas as pd
from astropy.io import fits

from lmcold import apogee_target as at
from lmcold import giants, selection
from lmcold.sources import DATA, ROOT, path

# %% [markdown]
# ### Settings
# Colour bins of the selection groups, the S/N cut of a quality spectrum and the $H$ bins of the pass groups.

# %%
XEDGES = [0.4, 0.6, 0.8, 0.9]  # edges of the relative colour x (0 = blue edge, 1 = red edge of the polygon): 5 colour bins
SNR_MIN = 30  # APOGEE S/N cut of a quality spectrum (check T6 compares 30, 50, 70)
PG_H_EDGES = [12.35, 13.3, 14.3, 15.3]  # H bins [mag] of the pass groups, from the BrtRGB bright limit to the faintest maglim

# %% [markdown]
# ### Load the data
# The APOGEE parent `data/apogee_parent.parquet`
# (`scripts/build_apogee_parent.py`). The XP metallicities `data/xp_andrae.parquet` are read in the
# pass-group step and the allVisit file in test T1.

# %%
# one row per (field, star) of the LMC1-35 RGB polygons; columns include j, h, k [mag], cls, maglim, rfield [deg], observed, snr
p = pd.read_parquet(DATA / "apogee_parent.parquet")

# %% [markdown]
# ### Candidates
# Polygon stars with the astrometric cuts and no bright neighbour, one row per star.

# %%
# Candidates; sorting puts a star's observed row first, so drop_duplicates keeps the field that observed it.
c = p[p.astro_ok & ~p.nbr].sort_values(["apogee_id", "observed"], ascending=[True, False]).drop_duplicates("apogee_id").copy()
# LMC1-17: fields whose targeting also used the Washington + DDO51 giant flag.
# field names are 'LMC<n>'; characters from index 3 give n
c["wd_field"] = c.field.str[3:].astype(int) <= 17

# %% [markdown]
# ### Selection groups
# Field, zone, class, $H$ bin and position across the colour range of the polygon.

# %%
# blue and red J - Ks [mag] edges of the targeting polygon at each star's H (lmcold.apogee_target.sides)
blue, red = at.sides(c.h.to_numpy(), c.maglim.to_numpy())
# x: relative colour across the polygon, 0 at the blue edge and 1 at the red edge
c["x"] = (c.j - c.k - blue) / (red - blue)
# (12.8 is the BrtRGB/FntRGB boundary of the polygons, lmcold.apogee_target)
# H bin: 0 for BrtRGB; FntRGB 1, 2, ... for 0.1-mag bins from H = 12.8.
hbin = np.where(c.cls == 1, 0, 1 + np.floor((c.h - 12.8) / 0.1).clip(0)).astype(int)
# Selection group label: field_zone_class_Hbin_colourbin, zone 0 for r < 0.8 deg and 1 for 0.8-0.95 deg.
c["group"] = (c.field + "_" + (c.rfield >= 0.8).astype(int).astype(str) + "_" + c.cls.astype(str) + "_" + hbin.astype(str) + "_"
             + np.digitize(c.x, XEDGES).astype(str))

# %% [markdown]
# ### Giant flag, XP metallicity and pass groups
# The giant flag of stars with spectra, the XP [M/H] of Andrae et al.
# (2023) and the pass groups ($H$ bin and XP [M/H]).

# %%
s = c[c.snr.notna()]  # stars with an allStarLite row
# is_giant evaluated on stars with spectra; stars without one get giant = False
c["giant"] = pd.Series(giants.is_giant(s), s.index).reindex(c.index, fill_value=False)
# XP [M/H] [dex] of Andrae et al. 2023 by Gaia source_id; NaN where missing
c["mh_xp"] = c.source_id.map(pd.read_parquet(DATA / "xp_andrae.parquet").set_index("source_id").mh_xp).astype(float)
# pass-group label of each candidate: H bin (PG_H_EDGES) x XP [M/H] bin (lmcold.selection.pass_groups)
pg = selection.pass_groups(c.h.to_numpy(), c.mh_xp.to_numpy(), PG_H_EDGES)


# %% [markdown]
# ### Weights
# Quality and coverage flags and the weight of each candidate (`lmcold.apogee_target.weights`).

# %%
# (quality, covered, w) for a given S/N cut, with or without the pass factor.
weights = lambda c, snr_min, with_pass=True: at.weights(c, snr_min, pg, with_pass)  # noqa: E731
# quality, covered: boolean (N,); w: weight (N,), 0 outside covered quality stars
c["quality"], c["covered"], c["w"] = weights(c, SNR_MIN)

# %% [markdown]
# ### Test T1: the targeting polygons
# Share of the stars APOGEE targeted that lie in the rebuilt parent.

# %%
# T1: Magellanic targets (LCO, bit 23 set, bit 22 unset) and whether each is in the parent.
with fits.open(path("apogee_dr17_allvisit"), memmap=True) as h:
    # allVisit table: one row per visit; keep field, star, telescope, targeting flag APOGEE2_TARGET1, H and J - K [mag]
    d = h[1].data
    v = pd.DataFrame({"field": np.char.strip(d["FIELD"].astype(str)), "apogee_id": np.char.strip(d["APOGEE_ID"].astype(str)),
                      "tel": np.char.strip(d["TELESCOPE"].astype(str)), "t1": d["APOGEE2_TARGET1"].astype(np.int64),
                      "h": d["H"].astype(float), "jk": (d["J"] - d["K"]).astype(float)})
# Magellanic targets observed at LCO (the du Pont 2.5-m) in the LMC1-35 fields, one row per (field, star)
v = v[(v.tel == "lco25m") & v.field.isin(at.FIELDS) & giants.bit(v.t1, 23) & ~giants.bit(v.t1, 22)].drop_duplicates(["field", "apogee_id"])
v["in_parent"] = pd.MultiIndex.from_frame(v[["field", "apogee_id"]]).isin(pd.MultiIndex.from_frame(p[["field", "apogee_id"]]))
# W+D targets: bit 7 of APOGEE2_TARGET1
wd = v[giants.bit(v.t1, 7)]
# LMC18-35 targets inside the plain colour-magnitude box of their targeting.
box = v[(v.field.str[3:].astype(int) >= 18) & (v.h >= 12.35) & (v.jk > 0.5) & (v.jk < 1.3)]
# share of each target set inside the parent; pass thresholds 99.5% (W+D) and 99% (LMC18-35), as in the module docstring
tests = {"T1": dict(wd_targets=len(wd), wd_in_parent=round(wd.in_parent.mean(), 4), lmc18_35_box=len(box),
                    lmc18_35_in_parent=round(box.in_parent.mean(), 4))}
tests["T1"]["pass"] = bool(tests["T1"]["wd_in_parent"] >= 0.995 and tests["T1"]["lmc18_35_in_parent"] >= 0.99)

# %% [markdown]
# ### Test T5: colour
# Weighted mean $J-K_s$ of the quality stars against that of the covered candidates.

# %%
# T5: weighted mean J - Ks of quality stars minus the unweighted mean of covered candidates [mag].
cov = c[c.covered]
t5 = {}
# two field groups (W+D or not) x two classes (1 = BrtRGB, 2 = FntRGB)
for lab, m in (("LMC1-17", cov.wd_field), ("LMC18-35", ~cov.wd_field)):
    for cl, nm in ((1, "BrtRGB"), (2, "FntRGB")):
        # covered candidates of the group, and its quality stars
        s = cov[m & (cov.cls == cl)]
        q = s[s.quality]
        t5[f"{lab} {nm}"] = round(float(np.average(q.j - q.k, weights=q.w) - (s.j - s.k).mean()), 4)
# pass if every difference is below 0.005 mag
tests["T5_dJK"] = t5 | {"pass": bool(all(abs(x) < 0.005 for x in t5.values()))}

# %% [markdown]
# ### Test T7: XP closure
# Weighted share of stars below two XP [M/H] thresholds over the share among covered
# candidates, with and without the pass factor.

# %%
# T7: weighted fraction below x over the fraction among covered candidates (1 = closure).
# stars with an XP [M/H]
xp = c.mh_xp.notna()
# per field group and weight set (final, or no pass factor): for x = -1 and -1.5,
# [sum of weights of stars below x / sum of weights] / [fraction below x among covered candidates]
tests["T7_xp_closure"] = {f"{fl} {lab}": {f"<{x}": round(float(np.sum(w[(w > 0) & xp & m & (c.mh_xp < x)]) / np.sum(w[(w > 0) & xp & m])
                                                           / np.mean(c.mh_xp[c.covered & xp & m] < x)), 3) for x in (-1.0, -1.5)}
                          for fl, m in (("LMC1-17", c.wd_field), ("LMC18-35", ~c.wd_field))
                          for lab, w in (("final", c.w.to_numpy()), ("no_pass", weights(c, SNR_MIN, False)[2]))}

# %% [markdown]
# ### Test T6: S/N cut
# Weighted fractions of metal-poor members for three S/N cuts.

# %%
# T6: members are giants with 150 < v < 420 km/s; compare S/N cuts on the groups covered under all three.
mem = (c.v > 150) & (c.v < 420)
# (quality, covered, w) for each S/N cut
runs = {snr: weights(c, snr) for snr in (30, 50, 70)}
# groups covered under all three cuts
common = np.logical_and.reduce([cov for _, cov, _ in runs.values()])
t6 = {}
for snr, (q, _, w) in runs.items():
    # quality giants that are members, in the common groups
    m = q & c.giant & mem & common
    # weighted fractions below [Fe/H] = -1.5 and -2 (ASPCAP feh)
    t6[f"snr>={snr}"] = dict(N=int(m.sum()), f_below_m15=round(float(w[m & (c.feh < -1.5)].sum() / w[m].sum()), 4),
                             f_below_m2=round(float(w[m & (c.feh < -2)].sum() / w[m].sum()), 4))
f15 = [r["f_below_m15"] for r in t6.values()]
t6["pass"] = bool(max(f15) - min(f15) < 0.0023)  # the group-bootstrap error of f_below_m15
tests["T6_snr_neutrality"] = t6
# counts of candidates, covered, observed and quality stars, and the sum of the weights
tests["summary"] = dict(candidates=len(c), covered=int(c.covered.sum()), observed=int(c.observed.sum()), quality=int(c.quality.sum()),
                        quality_covered=int((c.quality & c.covered).sum()), sum_w=round(float(c.w.sum()), 1))

# %% [markdown]
# ### Write the weights
# One row per candidate, `data/apogee_weights.parquet`, and the tests,
# `results/apogee_selection.json`.

# %%
# one row per candidate (columns listed in the module docstring)
c[["field", "apogee_id", "source_id", "ra", "dec", "j", "h", "k", "cls", "group", "observed", "quality", "giant", "covered", "w", "feh",
   "v", "mh_xp"]].to_parquet(DATA / "apogee_weights.parquet", index=False)
# the test results, also printed
(ROOT / "results" / "apogee_selection.json").write_text(json.dumps(tests, indent=1))
print(json.dumps(tests, indent=1))
