"""RRab stars within 20 deg of the LMC: OGLE-IV RRab united with Gaia DR3 RRab (1 arcsec match). Writes data/rrab.parquet.

Type and period come from OGLE where a star is in OGLE, from Gaia otherwise. [Fe/H] columns: feh_I (OGLE I band),
feh_G_m25 and feh_G_p26 (Gaia G band, two relations), feh_sos (the Gaia catalogue value);
feh, the adopted value (feh_src: I or G). dmu: relative distance modulus (dmu_src: G or I Wesenheit);
ogle_own: found by OGLE's own search (not added from Gaia);
member: rho < RHO_MAX, outside the SMC mask and |dmu| < DMU_WINDOW.

sources.yaml entries read: ogle4_ident and ogle4_rrab (read_ogle_rrab takes their directory), gaia_rrlyrae_lmc20. Entry
written: release_rrab (data/rrab.parquet); also results/rrab_pw_fit.csv (coefficients a, b_logP, c_feh, N and robust scatter of
the two period-Wesenheit-[Fe/H] fits). Constants: RHO_MAX = 10 deg, DMU_WINDOW = 0.6 mag, SMC mask radius 13 deg
(lmcold.footprint).
Columns of rrab.parquet: the OGLE columns (ogle_id, ra, dec [deg], I, V [mag, intensity means], P, P_err [d], T0, amp_I [mag],
R21, phi21, R31, phi31 [rad]), the Gaia columns of fetch_gaia_rrlyrae.py (Gaia positions as ra_gaia, dec_gaia), source_id
(-1 for OGLE stars without a Gaia match), in_ogle, in_gaia, ogle_own, the [Fe/H] columns [dex], rho [deg], dmu_G, dmu_I, dmu
[mag], dmu_src, smc_masked, member. Gaia-only rows are kept only when Gaia classifies them RRab; matched stars keep the OGLE
type. Used by figures/paper_maps.png, paper_selection.png (a), paper_rrab_membership.png, paper_rrab_feh.png, the [Fe/H]
profile (paper_feh_profile.py), the proper-motion figures and the centre fits.

Run: .venv/bin/python scripts/build_rrab.py
"""
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from lmcold.constants import LMC
from lmcold.footprint import DMU_WINDOW, RHO_MAX, smc_masked
from lmcold.rrlyrae import feh_gaia_G_m25, feh_gaia_G_p26, feh_ogle_I, ogle_own_discovery, read_ogle_rrab, relative_modulus
from lmcold.sources import DATA, path

# %% [markdown]
# ### Load the data
# The OGLE-IV LMC RRab collection (identifications and light-curve parameters) and the Gaia DR3 RR Lyrae
# within 20 deg of the LMC (`scripts/fetch_ogle4_rrab.py`, `scripts/fetch_gaia_rrlyrae.py`).

# %%
ogle = read_ogle_rrab(path("ogle4_rrab").parent)
gaia = pd.read_parquet(path("gaia_rrlyrae_lmc20"))

# %% [markdown]
# ### Cross-match OGLE with Gaia
# Each OGLE RRab takes the source_id of the nearest Gaia star within 1 arcsec.

# %%
# Nearest Gaia star to each OGLE RRab; a match within 1 arcsec gives the OGLE star that Gaia source_id.
co, cg = SkyCoord(ogle.ra, ogle.dec, unit="deg"), SkyCoord(gaia.ra, gaia.dec, unit="deg")
j, sep, _ = co.match_to_catalog_sky(cg)
hit = sep.arcsec < 1.0
ogle["source_id"] = np.where(hit, gaia.source_id.to_numpy()[j], -1)
assert not ogle.source_id[hit].duplicated().any()  # no Gaia star matched to two OGLE stars

# %% [markdown]
# ### The union of the two catalogues
# Stars in both, in OGLE only, and in Gaia only when Gaia classifies them RRab;
# `ogle_own` marks OGLE's own discoveries.

# %%
g = gaia.rename(columns={"ra": "ra_gaia", "dec": "dec_gaia"})
# Union: both (matched), left_only (OGLE only), right_only (Gaia only; kept only if Gaia's class is RRab).
t = ogle.merge(g, on="source_id", how="outer", indicator=True)
t = t[(t._merge == "both") | (t._merge == "left_only") | (t.best_classification == "RRab")]
t = t.assign(in_ogle=t._merge != "right_only", in_gaia=t._merge != "left_only").drop(columns="_merge")
# Identifier placeholder for Gaia-only rows (>= FIRST_GAIA_ADDED, so not an own discovery; in_ogle is False anyway).
t["ogle_own"] = t.in_ogle & ogle_own_discovery(t.ogle_id.fillna("OGLE-LMC-RRLYR-99999"))
t["ra"] = t.ra.fillna(t.ra_gaia)
t["dec"] = t.dec.fillna(t.dec_gaia)
t["P"] = t.P.fillna(t.pf)  # period [d]: OGLE, else Gaia fundamental period

# %% [markdown]
# ### Photometric metallicities
# The $I$-band and $G$-band [Fe/H] from period and $\phi_{31}$, and the adopted value:
# $I$ band where available, otherwise $G$ band shifted by the median difference.

# %%
t["feh_I"] = feh_ogle_I(t.P, t.phi31)
t["feh_G_m25"] = feh_gaia_G_m25(t.pf, t.phi31_g)  # Gaia period: phi31_g was fitted with it
t["feh_G_p26"] = feh_gaia_G_p26(t.pf, t.phi31_g)
t["feh_sos"] = t.metallicity  # Gaia DR3 SOS photometric metallicity (Clementini et al. 2023)
# Adopted [Fe/H]: OGLE I where available (its zero point agrees with Gratton+04 spectroscopy); otherwise Gaia G (Muraveva+25)
# shifted onto the I scale by the median difference on the stars with both (the difference does not depend on [Fe/H]:
# Deming slope 0.98 +- 0.01, scripts/fig_rrab_feh_transform.py).
shift = (t.feh_I - t.feh_G_m25).median()
t["feh"] = t.feh_I.fillna(t.feh_G_m25 + shift)
t["feh_src"] = np.where(t.feh_I.notna(), "I", np.where(t.feh_G_m25.notna(), "G", ""))

# %% [markdown]
# ### Distance from the LMC centre
# Angular distance $\rho$ [deg] from the reference centre; the catalogue is cut at 20 deg.

# %%
c0 = SkyCoord(LMC["ra0"], LMC["dec0"], unit="deg")
t["rho"] = SkyCoord(t.ra, t.dec, unit="deg").separation(c0).deg
t = t[t.rho < 20].reset_index(drop=True)  # same radius as the Gaia query; drops OGLE stars beyond it

# %% [markdown]
# ### Relative distance modulus
# Period-Wesenheit-[Fe/H] relations fitted on the core ($\rho$ < 3 deg) give
# $\Delta\mu$ [mag] for each star, from the OGLE index where available and the Gaia index otherwise.

# %%
# Relative distance modulus, dmu = 0 at the mean LMC distance (only relative distances are needed: the SMC is ~0.45 mag behind).
# OGLE Wesenheit W_I = I - 1.55 (V - I) (Udalski et al. 2008) where available (scatter in the core 0.10 mag), otherwise the Gaia
# Wesenheit W_G = G - 1.90 (BP - RP) (intensity averages; Ripepi et al. 2019; 0.19 mag, crowded BP/RP); each period-Wesenheit-
# [Fe/H] relation is fitted on the LMC itself (rho < 3 deg).
# Stars without [Fe/H] take the median.
feh = t.feh.fillna(t.feh.median()).to_numpy()
core = ((t.rho < 3) & t.feh.notna()).to_numpy()  # fit only on stars with measured [Fe/H]
fits = []
for name, W in (("G", t.int_average_g - 1.90 * (t.int_average_bp - t.int_average_rp)), ("I", t.I - 1.55 * (t.V - t.I))):
    # Fit W = a + b log10 P + c [Fe/H] on the core with 3-sigma clipping; dmu = W - fit (lmcold.rrlyrae.relative_modulus).
    dmu, coef, fit = relative_modulus(W.to_numpy(), t.P.to_numpy(), feh, core)
    t["dmu_" + name] = dmu
    fits.append(dict(wesenheit=name, a=coef[0], b_logP=coef[1], c_feh=coef[2], N=fit.sum(),
                     rsd=1.4826 * np.median(np.abs(dmu[fit] - np.median(dmu[fit])))))
pd.DataFrame(fits).to_csv(DATA.parent / "results" / "rrab_pw_fit.csv", index=False, float_format="%.4f")
t["dmu"] = t.dmu_I.fillna(t.dmu_G)  # OGLE I Wesenheit preferred (smaller scatter)
t["dmu_src"] = np.where(t.dmu_I.notna(), "I", np.where(t.dmu_G.notna(), "G", ""))

# %% [markdown]
# ### Members
# Inside the science radius, outside the SMC mask and within the $\Delta\mu$ window; the catalogue is written
# to `data/rrab.parquet`.

# %%
# Members: inside the science radius, outside the SMC mask and within the distance window (no proper-motion cut, so the
# kinematics stay unbiased). The 20 deg sample is kept for the mask and the foreground estimate.
t["smc_masked"] = smc_masked(t.ra.to_numpy(), t.dec.to_numpy())
t["member"] = (t.rho < RHO_MAX) & ~t.smc_masked & (t.dmu.abs() < DMU_WINDOW)
t.to_parquet(DATA / "rrab.parquet", index=False)
print(f"{len(t)} RRab: OGLE and Gaia {(t.in_ogle & t.in_gaia).sum()}, OGLE only {(t.in_ogle & ~t.in_gaia).sum()}, "
      f"Gaia only {(~t.in_ogle).sum()}; G shift {shift:+.4f}; adopted [Fe/H] from {t.feh_src.value_counts().to_dict()}; "
      f"dmu from {t.dmu_src.value_counts().to_dict()}; SMC-masked {t.smc_masked.sum()}; members {t.member.sum()}")
