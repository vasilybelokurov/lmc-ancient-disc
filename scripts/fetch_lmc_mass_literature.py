"""Archive primary references used by the LMC stellar-mass comparison table.

Numerical entries are manually transcribed with section/table references in
constants.py and docs/lmc_mass_literature.md; PDFs are retained for audit.

sources.yaml entries written: lmc_mass_reference_<name> for each key of `references` (data/raw/literature/<name>.pdf).
The numbers taken from these papers (constants.LMC_MASS_LITERATURE) are compared with this work's total mass in
figures/paper_mass_summary.png and Table 3 (scripts/paper_mass_summary.py; scripts/paper_table_mass_summary.py, which also reads
the lmc_mass_reference_* entries).

Needs network access to arxiv.org; no run time is recorded.
Run: .venv/bin/python scripts/fetch_lmc_mass_literature.py
"""
import requests
from lmcold.sources import DATA,sha256

# %% [markdown]
# ### The papers
# The arXiv identifier of each paper whose LMC stellar mass is quoted in the comparison.

# %%
# file stem -> arXiv identifier (bibcodes are in sources.yaml)
references={
    'keim2023':'2207.14277',
    'mazzi2021':'2108.07225',
    'harris_zaritsky2009':'0908.1422',
    'mcconnachie2012':'1204.1562',
    'vanderMarel2002':'astro-ph/0205161',
    'eskew2012':'1204.0524',
    'skibba2012':'1210.7812',
    'madau_dickinson2014':'1403.0007',
}
# %% [markdown]
# ### Download
# Each PDF goes to data/raw/literature/<name>.pdf; its sha256 is printed for data/sources.yaml.

# %%
for name,identifier in references.items():
    url='https://arxiv.org/pdf/'+identifier
    p=DATA/'raw/literature'/f'{name}.pdf'
    p.parent.mkdir(parents=True,exist_ok=True)
    r=requests.get(url,timeout=45);r.raise_for_status()
    assert r.content.startswith(b'%PDF'),url  # arXiv can return an HTML page instead of the PDF
    p.write_bytes(r.content)
    print(name,url,sha256(p),flush=True)
