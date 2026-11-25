"""Which eligible stars of the BOSS parent (data/boss_parent.parquet) were observed, and which gave a good spectrum.

observed: at least one spAll-lite (SDSS-V DR20, v6_2_1) spectrum with ASSIGNED, ON_TARGET and VALID set, matched on GAIA_ID
          (Gaia DR3 source_id); firstcarton is that of the first such spectrum (lowest MJD), n_spec counts them.
in_bossnet: the star is in astraAllStarBossNet 0.8.1; spectrum_ok: lmcold.giants.boss_spectrum_ok on its highest-S/N row;
giant: lmcold.giants.is_giant on the same row. No velocity cut: membership is handled later.
Observed stars absent from BOSSNet (3,089 of 81,170) are failed spectra (median spAll S/N 6 against 14; ZWARNING set in 47%;
half classed GALAXY or QSO), so they count as spectrum_ok = False.
Writes data/boss_observed.parquet: source_id, the columns above and feh (BOSSNet, same row), one row per eligible parent star.

sources.yaml entries read: sdss5_spall_lite (through its decompressed copy, which must exist next to the registered .gz; the
checksum is verified on the .gz; no script in this repository writes the copy, so gunzip the file first) and
boss_dr20_bossnet; also data/boss_parent.parquet (release_boss_parent). Entry written: release_boss_observed.
Column types: n_spec int (0 if unobserved), firstcarton str ('' if unobserved), observed, in_bossnet, spectrum_ok,
giant bool, feh [dex, raw BOSSNet, NaN if not in BOSSNet]. Quality cuts of boss_spectrum_ok: flag_bad = 0, finite [Fe/H],
v and its error, S/N >= 16, v error < 10 km/s; is_giant: Teff < 5500 K and log g < 3.5.

Run: .venv/bin/python scripts/build_boss_observed.py   (after build_boss_parent.py)
"""
import numpy as np
import pandas as pd
from astropy.io import fits

from lmcold import giants
from lmcold.sources import DATA, path, sha256

# %% [markdown]
# ### Settings
# The BOSSNet columns kept and their common names.

# %%
# astraAllStarBossNet column -> common name used by lmcold.giants
BOSSNET = {"gaia_dr3_source_id": "source_id", "flag_bad": "flag_bad", "fe_h": "feh", "v_rad": "v", "e_v_rad": "ev", "snr": "snr",
           "teff": "teff", "logg": "logg"}


# %% [markdown]
# ### Helper
# `native` returns a FITS column in native byte order.

# %%
def native(a):
    """Return a FITS column as a numpy array in native byte order (FITS data are big-endian; pandas needs native order).

    Parameters
    ----------
    a : array_like
        Column of a FITS table.

    Returns
    -------
    numpy.ndarray
        Same values, native byte order.
    """
    # View as a numpy array.
    a = np.asarray(a)
    # Same values and dtype, with native ("=") byte order.
    return a.astype(a.dtype.newbyteorder("="))


# %% [markdown]
# ### Load the data
# The eligible stars of the BOSS parent. The spAll-lite and BOSSNet
# files are read, only for these stars, in the next two steps.

# %%
# Gaia DR3 source_id of every eligible parent star, shape (N,).
p = pd.read_parquet(DATA / "boss_parent.parquet", columns=["source_id", "eligible"])
p = p[p.eligible].drop(columns="eligible")

# %% [markdown]
# ### Observed stars
# Spectra in spAll-lite with ASSIGNED, ON_TARGET and VALID set, matched on the
# Gaia source_id: number of spectra and first carton per star.

# %%
with fits.open(path("sdss5_spall_lite").with_suffix(""), memmap=True) as f:  # the decompressed copy next to the registered .gz
    d = f[1].data
    # Rows of spAll-lite whose GAIA_ID is an eligible parent star; one row per spectrum (a star can have several).
    gid = native(d["GAIA_ID"])
    k = np.flatnonzero(np.isin(gid, p.source_id.to_numpy()))
    # Per spectrum: source_id, MJD of the observation, the three targeting/quality bits, and the first carton (string, blanks
    # stripped).
    s = pd.DataFrame({"source_id": gid[k], "mjd": native(d["MJD"][k]),
                      "good": (native(d["ASSIGNED"][k]) == 1) & (native(d["ON_TARGET"][k]) == 1) & (native(d["VALID"][k]) == 1),
                      "firstcarton": np.char.strip(np.asarray(d["FIRSTCARTON"][k]).astype(str))})
# Keep spectra with ASSIGNED, ON_TARGET and VALID all set; sorted by MJD so `first` below is the earliest good spectrum.
s = s[s.good].sort_values("mjd", kind="stable")
# Per observed star: number of good spectra and the carton of the earliest one.
obs = s.groupby("source_id").agg(n_spec=("mjd", "size"), firstcarton=("firstcarton", "first"))

# %% [markdown]
# ### Good spectra
# The highest-S/N BOSSNet row of each observed star, with the quality and giant
# flags of `lmcold.giants`.

# %%
with fits.open(path("boss_dr20_bossnet"), memmap=True) as f:
    d = f[1].data
    # BOSSNet rows of the observed stars only, renamed to the common names of BOSSNET (feh [dex], v and ev [km/s], teff [K]).
    gid = native(d["gaia_dr3_source_id"])
    k = np.flatnonzero(np.isin(gid, obs.index.to_numpy()))
    b = pd.DataFrame({new: native(d[old][k]) for old, new in BOSSNET.items()})
# One BOSSNet row per star: the highest-S/N one.
b = b.sort_values("snr", ascending=False, kind="stable").drop_duplicates("source_id").set_index("source_id")
# Quality and giant flags (cuts listed in the module docstring).
b["spectrum_ok"], b["giant"] = giants.boss_spectrum_ok(b), giants.is_giant(b)

# %% [markdown]
# ### One row per eligible star
# Flags with defaults for unobserved stars, written to
# `data/boss_observed.parquet`.

# %%
# Left joins on the eligible parent: unobserved stars get NaN, then the defaults below.
out = p.join(obs, on="source_id").join(b[["spectrum_ok", "giant", "feh"]], on="source_id")
# observed: has a good spAll spectrum; in_bossnet: has a BOSSNet row (absent ones count as failed spectra, module docstring).
out["observed"] = out.n_spec.notna()
out["in_bossnet"] = out.spectrum_ok.notna()
out["n_spec"] = out.n_spec.fillna(0).astype(int)
out["firstcarton"] = out.firstcarton.fillna("")
out[["spectrum_ok", "giant"]] = out[["spectrum_ok", "giant"]].fillna(False).astype(bool)
# Write the table and print the counts at each step, the file's sha256 and the most common first cartons.
out.to_parquet(DATA / "boss_observed.parquet", index=False)
print(f"{len(out)} eligible, {out.observed.sum()} observed, {out.in_bossnet.sum()} in BOSSNet, {out.spectrum_ok.sum()} good spectra,"
      f" {(out.spectrum_ok & out.giant).sum()} good giants;", sha256(DATA / "boss_observed.parquet"))
print(out[out.observed].firstcarton.value_counts().head(8).to_dict())
