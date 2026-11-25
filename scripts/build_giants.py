"""The two giant samples: APOGEE DR17 and SDSS-V DR20 BOSS, each with Gaia DR3 astrometry, BOSS on the APOGEE scale.

Inputs: data/spec_giants.parquet (scripts/build_spec_giants.py, from the public SDSS files) and data/gaia_giants.parquet
(scripts/fetch_gaia_giants.py, public Gaia archive). Steps: quality masks (lmcold.giants); stars in both surveys (same Gaia
source) form the overlap; the BOSS -> APOGEE relations are fitted on a random half of the overlap and tested on the other half,
then refitted on all of it and applied to BOSS. Membership for the data section: within 12 deg and 100 < v_los < 450 km/s
(from build_spec_giants), parallax consistent with the LMC ((parallax + 0.017) / parallax_error < 3; 0.017 mas: Gaia DR3 zero
point, Lindegren et al. 2021). The Milky Way foreground is modelled later, in the kinematic fits.
Writes the release catalogues data/giants_apogee.parquet, data/giants_boss.parquet (columns ra, dec, pmra, pmdec, pmra_error,
pmdec_error, pmra_pmdec_corr, parallax, parallax_error, ruwe, G, vlos, evlos, feh, teff, logg, snr, source_id, in_both) and
results/boss_calibration.json (fit, held-out test, comparison with an earlier calibration) and
data/giants_overlap.parquet (the stars in both surveys, for the calibration figure).

sources.yaml entries written: release_giants_apogee (giants_apogee.parquet) and release_giants_boss (giants_boss.parquet).
Quality masks (lmcold.giants): Gaia: finite proper motions and covariance, RUWE <= 1.4. APOGEE: not a duplicate (EXTRATARG bit 4),
ASPCAPFLAG STAR_BAD (bit 23) unset, finite [Fe/H], S/N >= 50, v error < 1 km/s, visit scatter < 1.5 km/s. BOSS: flag_bad unset,
finite [Fe/H] and v, S/N >= 16 (BOSSNet [Fe/H] barely tracks APOGEE below; scripts/fig_boss_apogee_offset.py), v error < 10 km/s.
Both: giants, Teff < 5500 K and log g < 3.5.
Calibration (lmcold.giants.fit_calibration, Huber-weighted least squares), with t = (Teff_BOSS - 4200 K)/1000 K and
s = 20/(S/N)_BOSS - 1:  v_BOSS - v_APOGEE = d0 + d1 t;  [Fe/H]_BOSS = c0 + c1 [Fe/H]_APOGEE + c2 t + c3 s, inverted for BOSS.
Output units: ra, dec [deg, Gaia]; pmra, pmdec and errors [mas/yr]; parallax and error [mas]; G [mag]; vlos, evlos [km/s,
heliocentric; BOSS on the APOGEE zero point]; feh [dex, APOGEE scale]; teff [K]; logg [dex]; in_both: star also in the other
survey's quality sample. The calibration figure is figures/paper_boss_calibration.png (appendix B).

Run: .venv/bin/python scripts/build_giants.py   (after build_spec_giants.py and fetch_gaia_giants.py)
"""
import json

import numpy as np
import pandas as pd

from lmcold import giants
from lmcold.sources import DATA, ROOT

# %% [markdown]
# ### Settings
# The coefficients of an earlier calibration (kept only for comparison) and the Gaia columns of the release
# catalogues.

# %%
# Coefficients of an earlier calibration (S/N >= 10) for the comparison stored in boss_calibration.json.
RESEARCH_V2 = dict(d0=3.923, d1=-4.870, c0=-0.0569, c1=0.7310, c2=-0.0319, c3=0.0798)
# Gaia columns carried into the release catalogues (G = phot_g_mean_mag).
GAIA = ["source_id", "ra", "dec", "pmra", "pmdec", "pmra_error", "pmdec_error", "pmra_pmdec_corr", "parallax", "parallax_error",
        "ruwe", "G"]

# %% [markdown]
# ### Load the data
# The spectroscopic stars of both surveys, `data/spec_giants.parquet` (`scripts/build_spec_giants.py`), and
# their Gaia DR3 astrometry, `data/gaia_giants.parquet` (`scripts/fetch_gaia_giants.py`).

# %%
# One row per spectrum (APOGEE or BOSS): survey, source_id, v [km/s], ev, feh [dex], teff [K], logg, snr and quality flags.
spec = pd.read_parquet(DATA / "spec_giants.parquet")
# Gaia DR3 astrometry and photometry of the same sources; phot_g_mean_mag renamed G [mag].
gaia = pd.read_parquet(DATA / "gaia_giants.parquet").rename(columns={"phot_g_mean_mag": "G"})

# %% [markdown]
# ### Quality masks
# Gaia astrometry quality, then the spectroscopic quality and giant masks of each survey
# (`lmcold.giants`); one row per Gaia source in each survey.

# %%
gaia = gaia[giants.gaia_ok(gaia)]   # finite proper motions and covariance, RUWE <= 1.4 (module docstring)
spec = spec.drop(columns=["ra", "dec"]).merge(gaia, on="source_id")  # Gaia positions throughout
# One row per Gaia source in each survey (the first spectrum kept) after the quality and giant masks.
a = spec[spec.survey == "apogee"]
a = a[giants.apogee_ok(a)].drop_duplicates("source_id")
b = spec[spec.survey == "boss"]
b = b[giants.boss_ok(b)].drop_duplicates("source_id")

# %% [markdown]
# ### Held-out test of the calibration
# The stars in both surveys form the overlap. The BOSS-to-APOGEE relations are
# fitted on a random half and tested on the other half: median offsets and robust scatter, overall and in bins.

# %%
# Overlap: quality giants in both surveys (suffix _a = APOGEE, _b = BOSS). Fit on a random half (fixed seed), test on the rest.
m = a.merge(b, on="source_id", suffixes=("_a", "_b"))
rng = np.random.default_rng(12)   # fixed seed of the split
half = rng.random(len(m)) < 0.5   # (N,) bool, True: fitting half
# Arguments of lmcold.giants.fit_calibration: BOSS Teff, S/N and v, APOGEE v, BOSS and APOGEE [Fe/H].
args = lambda s: (s.teff_b, s.snr_b, s.v_b, s.v_a, s.feh_b, s.feh_a)  # noqa: E731
cal_half = giants.fit_calibration(*args(m[half]))   # dict of d0, d1, c0..c3 (module docstring)
# Calibrated BOSS v [km/s] and [Fe/H] [dex] of the held-out half.
v_t, feh_t = giants.calibrate(cal_half, m.teff_b[~half], m.snr_b[~half], m.v_b[~half], m.feh_b[~half])
d = feh_t - m.feh_a[~half]  # calibrated BOSS minus APOGEE [Fe/H] on the held-out half [dex]
# Held-out summary: median offsets, robust sd (1.4826 x MAD) and median residual in bins of APOGEE [Fe/H], BOSS S/N and Teff.
test = dict(N=int((~half).sum()), dv_median=float(np.median(v_t - m.v_a[~half])), dfeh_median=float(np.median(d)),
            dfeh_rsd=float(1.4826 * np.median(np.abs(d - np.median(d)))),
            dfeh_median_by_feh={f"{lo}..{hi}": float(np.median(d[(m.feh_a[~half] >= lo) & (m.feh_a[~half] < hi)]))
                                for lo, hi in [(-9, -1.5), (-1.5, -1.0), (-1.0, -0.5), (-0.5, 9)]},
            dfeh_median_by_snr={f"{lo}..{hi}": float(np.median(d[(m.snr_b[~half] >= lo) & (m.snr_b[~half] < hi)]))
                                for lo, hi in [(16, 20), (20, 25), (25, 30), (30, 40), (40, 1e4)]},
            dfeh_median_by_teff={f"{lo}..{hi}": float(np.median(d[(m.teff_b[~half] >= lo) & (m.teff_b[~half] < hi)]))
                                 for lo, hi in [(0, 3900), (3900, 4200), (4200, 4500), (4500, 1e4)]})

# %% [markdown]
# ### Final calibration, applied to BOSS
# The relations refitted on the whole overlap put the BOSS
# velocities and [Fe/H] on the APOGEE scale; the overlap is saved for the calibration figure.

# %%
cal = giants.fit_calibration(*args(m))  # final relations: refitted on the whole overlap
m[["source_id", "teff_b", "snr_b", "v_b", "v_a", "feh_b", "feh_a"]].to_parquet(DATA / "giants_overlap.parquet", index=False)
# BOSS on the APOGEE scale; the raw BOSSNet [Fe/H] is kept as feh_boss (not written to the release file).
b = b.assign(vlos=b.v, feh_boss=b.feh)
b["vlos"], b["feh"] = giants.calibrate(cal, b.teff, b.snr, b.v, b.feh_boss)
a = a.assign(vlos=a.v)   # APOGEE is the reference: vlos is its own velocity

# %% [markdown]
# ### The release catalogues
# Stars with a parallax consistent with the LMC distance, with the columns of the release
# files `data/giants_apogee.parquet` and `data/giants_boss.parquet`.

# %%
# Parallax cut: zero-point-corrected parallax (parallax + 0.017 mas) less than 3 sigma above zero, i.e. consistent with the
# ~50 kpc LMC distance (parallax ~0.02 mas); evlos is the survey's velocity error.
for name, s in (("apogee", a), ("boss", b)):
    s = s[(s.parallax + 0.017) / s.parallax_error < 3].assign(in_both=lambda t: t.source_id.isin(m.source_id), evlos=lambda t: t.ev)
    s = s[GAIA + ["vlos", "evlos", "feh", "teff", "logg", "snr", "in_both"]]   # release columns, in this order
    s.to_parquet(DATA / f"giants_{name}.parquet", index=False)
    print(f"{name}: {len(s)} giants; in both {int(s.in_both.sum())}")

# %% [markdown]
# ### Calibration summary
# The coefficients, the held-out test and the difference from the earlier calibration,
# in `results/boss_calibration.json`.

# %%
# Final coefficients, held-out test, the earlier coefficients and final minus earlier.
out = dict(calibration={k: float(v) for k, v in cal.items()}, held_out_test=test, research_v2=RESEARCH_V2,
           difference_from_research={k: float(cal[k] - RESEARCH_V2[k]) for k in RESEARCH_V2})
json.dump(out, open(ROOT / "results" / "boss_calibration.json", "w"), indent=1)
print(json.dumps(out, indent=1))
