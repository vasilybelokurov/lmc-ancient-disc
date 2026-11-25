"""Carton memberships and XP [M/H] of the eligible BOSS parent stars, from the public SDSS-V DR20 targetdb tables
(scripts/fetch_targetdb.py): Gaia DR3 source_id -> catalogid (catalog_to_gaia_dr3_source) -> target_pk (target) -> carton
(carton_to_target, carton). Every carton a star belongs to is kept, BOSS and APOGEE alike; no priority chain.
XP [M/H]: mos_xpfeh_gaia_dr3 (the XGBoost estimate the SDSS-V XP cartons used).
Writes data/boss_cartons.parquet: source_id, cartons (sorted, ';'-joined), mh_xp.

Inputs: data/boss_parent.parquet (release_boss_parent) and five targetdb parquet tables in ~/data/sdss/dr20/targetdb (sources.yaml
entries sdss5_targetdb_<table>). The output (sources.yaml: release_boss_cartons) is read by scripts/boss_selection.py, where the carton list
sets each star's fibre-priority class (lmcold.magcloud.carton_class). cartons is '' for stars in no carton; mh_xp [dex] is NaN
where mos_xpfeh_gaia_dr3 has no entry.

Run: .venv/bin/python scripts/build_boss_cartons.py   (after build_boss_parent.py and fetch_targetdb.py)
"""
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from lmcold.sources import DATA, sha256

# %% [markdown]
# ### Settings
# Where the SDSS-V targetdb tables are kept (`scripts/fetch_targetdb.py`).

# %%
# Local copy of the targetdb tables (scripts/fetch_targetdb.py).
TDB = Path("~/data/sdss/dr20/targetdb").expanduser()


# %% [markdown]
# ### Helper
# `read` returns the rows of a targetdb table whose key is in a given list, filtered while scanning.

# %%
def read(name, cols, key, values):
    """Rows of a targetdb table whose `key` is in `values`, filtered while scanning.

    Parameters
    ----------
    name : str
        Table name (file ~/data/sdss/dr20/targetdb/<name>.parquet).
    cols : list of str
        Columns to read.
    key : str
        Column the filter is applied to.
    values : array_like
        Values of `key` to keep.

    Returns
    -------
    pandas.DataFrame
        The matching rows, columns `cols`.
    """
    # open the parquet file as a dataset and push the 'key in values' filter into the scan, so only matching rows are loaded
    # (the targetdb tables are much larger than the parent sample)
    return ds.dataset(TDB / f"{name}.parquet").to_table(columns=cols, filter=pc.field(key).isin(values)).to_pandas()


# %% [markdown]
# ### Load the data
# The source_ids of the eligible BOSS parent stars. The targetdb
# tables are read, only for these stars, in the next steps.

# %%
# eligible: boolean column of boss_parent.parquet (set by build_boss_parent.py)
p = pd.read_parquet(DATA / "boss_parent.parquet", columns=["source_id", "eligible"])
sid = p.source_id[p.eligible].to_numpy()  # (N,) Gaia DR3 source_ids of the eligible stars

# %% [markdown]
# ### Carton memberships
# Gaia source_id to catalogid to target to carton; every carton of a star is kept.

# %%
# In catalog_to_gaia_dr3_source, target_id is the Gaia DR3 source_id.
g = read("mos_catalog_to_gaia_dr3_source", ["catalogid", "target_id"], "target_id", sid).rename(columns={"target_id": "source_id"})
# target rows of those catalogids, then their carton memberships (target_pk -> carton_pk)
t = read("mos_target", ["target_pk", "catalogid"], "catalogid", g.catalogid.unique())
c2t = read("mos_carton_to_target", ["target_pk", "carton_pk"], "target_pk", t.target_pk.unique())
# carton_pk -> carton name (small table, read whole)
carton = pd.read_parquet(TDB / "mos_carton.parquet", columns=["carton_pk", "carton"])
# One row per (star, carton) membership; a star may have several catalogids and targets.
m = c2t.merge(carton, on="carton_pk").merge(t, on="target_pk").merge(g, on="catalogid")
# per star: the set of its carton names, sorted and joined with ';'
cartons = m.groupby("source_id").carton.agg(lambda s: ";".join(sorted(set(s))))

# %% [markdown]
# ### XP [M/H] of SDSS-V
# The XP metallicity estimate that the SDSS-V XP cartons used.

# %%
# mh_xgboost [dex]: XP [M/H]; one row per source_id kept if the table repeats a star
xp = read("mos_xpfeh_gaia_dr3", ["source_id", "mh_xgboost"], "source_id", sid).drop_duplicates("source_id").set_index("source_id")

# %% [markdown]
# ### Write the table
# One row per eligible star: `data/boss_cartons.parquet`.

# %%
# one row per eligible star, in the order of boss_parent.parquet
out = pd.DataFrame({"source_id": sid})
out["cartons"] = out.source_id.map(cartons).fillna("")  # '' for stars in no carton
out["mh_xp"] = out.source_id.map(xp.mh_xgboost)  # NaN where the XP table has no entry
out.to_parquet(DATA / "boss_cartons.parquet", index=False)
# summary: counts in any carton, in the LMC/SMC red-giant BOSS carton, with XP [M/H]; and the output's sha256
print(f"{len(out)} eligible; {(out.cartons != '').sum()} in any carton, {out.cartons.str.contains('mwm_magcloud_rgb_boss').sum()} in"
      f" mwm_magcloud_rgb_boss; {out.mh_xp.notna().sum()} with XP [M/H];", sha256(DATA / "boss_cartons.parquet"))
