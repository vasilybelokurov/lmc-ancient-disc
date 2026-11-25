"""LMC star clusters with Ca II triplet [Fe/H] (Grocholski et al. 2006, VizieR J/AJ/132/1630, 2006AJ....132.1630G) and the
Gaia DR3 sources within RADIUS of each, G <= 18, from the public Gaia archive TAP service. For the colour calibration of the
isochrones on the giant branch (scripts/fig_cluster_colours.py). Writes data/clusters_grocholski06.csv and data/clusters_gaia.parquet.

sources.yaml entries written: clusters_grocholski06 (VizieR table, one row per cluster, with columns `feh`, `e_feh`, `ra`, `dec`
in deg added from the sexagesimal RAJ2000/DEJ2000) and clusters_gaia (Gaia DR3 sources with G <= 18 within RADIUS of each
cluster; column `cluster` gives the Grocholski name). clusters_grocholski06 is also read by scripts/paper_mass.py (cluster
[Fe/H] in panel c of figures/paper_mass.png).
Row order of clusters_gaia may differ between fetches (see sources.yaml), so its sha256 can change without a change of content.

Needs network access to VizieR and the Gaia archive (one TAP query per cluster); no run time is recorded.
Run: .venv/bin/python scripts/fetch_clusters.py
"""
# %% [markdown]
# ### Settings
# The cone radius around each cluster and the VizieR query of the cluster table.

# %%
import io
import urllib.request

import pandas as pd
import pyvo
from astropy.coordinates import SkyCoord
import astropy.units as u

from lmcold.sources import DATA, sha256

RADIUS = 6. / 60  # deg; clusters have diameters of 0.8-4.7 arcmin, the rest is the field annulus
# VizieR ASU query of table1 of J/AJ/132/1630: all columns, tab-separated, at most 100 rows (the table has 29).
URL = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/AJ/132/1630/table1&-out.all&-out.max=100"

# %% [markdown]
# ### The cluster table
# Table 1 of Grocholski et al. (2006) from VizieR: [Fe/H] of each cluster and its position, converted to degrees.

# %%
# VizieR TSV: comment lines start with '#'; then a header line, a units line and a dashes line before the data rows.
raw = urllib.request.urlopen(URL).read().decode()
lines = [x for x in raw.splitlines() if x and not x.startswith('#')]
c = pd.read_csv(io.StringIO('\n'.join([lines[0]] + lines[3:])), sep='\t')
c.columns = [x.strip() for x in c.columns]
# Each column becomes numeric if every non-empty value parses as a number; otherwise it stays a stripped string.
for col in c:
    v = c[col].astype(str).str.strip().replace('', None)
    num = pd.to_numeric(v, errors='coerce')
    c[col] = num if num.notna().sum() == v.notna().sum() else v
c = c.rename(columns={'[Fe/H]': 'feh', 'e_[Fe/H]': 'e_feh'})
pos = SkyCoord(c.RAJ2000, c.DEJ2000, unit=(u.hourangle, u.deg))
c['ra'], c['dec'] = pos.ra.deg, pos.dec.deg
c.to_csv(DATA / 'clusters_grocholski06.csv', index=False)

# %% [markdown]
# ### Gaia DR3 stars around each cluster
# One cone search per cluster on the Gaia archive; the stars are labelled with their cluster's name and saved.

# %%
# One cone search per cluster on the public Gaia archive TAP service (synchronous queries).
tap = pyvo.dal.TAPService("https://gea.esac.esa.int/tap-server/tap")
parts = []
for r in c.itertuples():
    # WHERE: G <= 18 (giant-branch stars of the clusters; the limit of the stored sample) and the source lies within RADIUS deg
    # of the cluster centre (CONTAINS(POINT, CIRCLE) = 1 is the ADQL cone test).
    q = ("SELECT source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, phot_bp_rp_excess_factor, pmra, pmdec, "
         "parallax, parallax_error, ruwe FROM gaiadr3.gaia_source WHERE phot_g_mean_mag <= 18 AND "
         f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {r.ra}, {r.dec}, {RADIUS})) = 1")
    t = tap.search(q).to_table().to_pandas()
    parts.append(t.assign(cluster=r.Cluster))
g = pd.concat(parts, ignore_index=True)
g.to_parquet(DATA / 'clusters_gaia.parquet', index=False)
print(len(c), 'clusters;', len(g), 'Gaia rows;', sha256(DATA / 'clusters_grocholski06.csv'), sha256(DATA / 'clusters_gaia.parquet'))
