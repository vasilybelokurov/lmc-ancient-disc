"""Published LMC field-star metallicities for the MDF comparison (scripts/paper_mdf_literature.py), from VizieR.

- Cole et al. (2005, 2005AJ....129.1465C, J/AJ/129/1465/stars): Ca II triplet [Fe/H] of 373 bar giants.
- Carrera et al. (2008, 2008AJ....135..836C, J/AJ/135/836/table1): Ca II triplet index (sum of the three equivalent widths), V, I and
  radial velocity of 702 stars in four fields 3-8 deg north of the bar (no [Fe/H] in the table; paper_mdf_literature.py applies
  their equation 1).
- Omkumar et al. (2025, 2025A&A...700A..74O, J/A+A/700/A74/lmc-rgb): Stroemgren photometric [Fe/H] from Gaia DR3 XP of LMC red giants.
Fetched from the VizieR mirror at IDIA (the Strasbourg server did not respond, 2026-10-28); same catalogues.
Writes data/mdf_cole05.csv, data/mdf_carrera08.csv and data/mdf_omkumar25.parquet.

sources.yaml entries written: mdf_cole05, mdf_carrera08 and mdf_omkumar25 (tier A). They serve figures/paper_mdf_literature.png
(appendix E). Columns are renamed to lower case: ra, dec [deg], feh, e_feh [dex], rv, e_rv [km/s], sum_ca, e_sum_ca [Angstrom]
(Carrera), rho [deg] (Omkumar: their angular distance from the LMC centre), source_id (Gaia DR3).
Note: sources.yaml records 635 rows for mdf_carrera08 (the VizieR table); the 702 above is the number of stars in the paper.

Needs network access to the VizieR mirror; no run time is recorded.
Run: .venv/bin/python scripts/fetch_mdf_literature.py
"""
import io
import urllib.request

import pandas as pd

from lmcold.sources import DATA, sha256

# %% [markdown]
# ### Reading a VizieR table
# One function downloads the chosen columns of a VizieR table and parses the tab-separated text.

# %%
# VizieR ASU TSV query at the IDIA mirror: all rows (-out.max=unlimited) of one table, the listed columns only.
VIZIER = "https://vizier.idia.ac.za/viz-bin/asu-tsv?-source={}&-out.max=unlimited&-out={}"


def vizier(table, cols):
    """Download columns of one VizieR table as a DataFrame.

    Parameters
    ----------
    table : str
        VizieR table name, e.g. 'J/AJ/129/1465/stars' ('+' must be URL-encoded as %2B).
    cols : list of str
        VizieR column names; computed columns such as _RA, _DE (J2000 position in deg) are allowed.

    Returns
    -------
    pandas.DataFrame
        One row per catalogue row. A column is numeric if every non-empty value parses as a number, otherwise a stripped
        string with empty values as None.
    """
    raw = urllib.request.urlopen(VIZIER.format(table, ','.join(cols)), timeout=600).read().decode()
    # TSV layout: '#' comment lines, then header, units and dashes lines, then the data rows.
    lines = [x for x in raw.splitlines() if x and not x.startswith('#')]
    t = pd.read_csv(io.StringIO('\n'.join([lines[0]] + lines[3:])), sep='\t', dtype=str)
    t.columns = [x.strip() for x in t.columns]
    # Per column: numeric if every non-empty value parses, else stripped strings (empty -> None).
    for col in t:
        v = t[col].str.strip().replace('', None)
        num = pd.to_numeric(v, errors='coerce')
        t[col] = num if num.notna().sum() == v.notna().sum() else v
    return t


if __name__ == '__main__':
    # %% [markdown]
    # ### The three catalogues
    # Cole et al. (2005), Carrera et al. (2008) and Omkumar et al. (2025), with their columns renamed, saved in data/.

    # %%
    # Cole et al. (2005): 2MASS id, field, V, I [mag], RV [km/s], CaT [Fe/H] and error [dex], J2000 position [deg].
    cole = vizier('J/AJ/129/1465/stars', ['2MASS', 'Field', 'Vmag', 'Imag', 'RV', '[Fe/H]', 'e_[Fe/H]', '_RA', '_DE'])
    cole = cole.rename(columns={'[Fe/H]': 'feh', 'e_[Fe/H]': 'e_feh', '_RA': 'ra', '_DE': 'dec', 'RV': 'rv'})
    # Carrera et al. (2008): position [deg], CaT index sum and error [Angstrom], V, I [mag], RV and error [km/s], comment.
    carrera = vizier('J/AJ/135/836/table1', ['_RAJ2000', '_DEJ2000', 'W.CaT', 'e_W.CaT', 'Vmag', 'Imag', 'RV', 'e_RV', 'Comm'])
    carrera = carrera.rename(columns={'_RAJ2000': 'ra', '_DEJ2000': 'dec', 'W.CaT': 'sum_ca', 'e_W.CaT': 'e_sum_ca', 'RV': 'rv', 'e_RV': 'e_rv'})
    # Omkumar et al. (2025): Gaia DR3 id, position [deg], distance from the LMC centre [deg], XP [Fe/H] and error [dex].
    omk = vizier('J/A%2BA/700/A74/lmc-rgb', ['GaiaDR3', 'RA_ICRS', 'DE_ICRS', 'Dist-deg', '[Fe/H]', 'e_[Fe/H]'])
    omk = omk.rename(columns={'GaiaDR3': 'source_id', 'RA_ICRS': 'ra', 'DE_ICRS': 'dec', 'Dist-deg': 'rho', '[Fe/H]': 'feh', 'e_[Fe/H]': 'e_feh'})
    # Save in data/ and print row count and sha256 of each file (the numbers recorded in sources.yaml).
    cole.to_csv(DATA / 'mdf_cole05.csv', index=False)
    carrera.to_csv(DATA / 'mdf_carrera08.csv', index=False)
    omk.to_parquet(DATA / 'mdf_omkumar25.parquet', index=False)
    for f, t in [('mdf_cole05.csv', cole), ('mdf_carrera08.csv', carrera), ('mdf_omkumar25.parquet', omk)]:
        print(f, len(t), sha256(DATA / f))
