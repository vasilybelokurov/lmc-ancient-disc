# Getting started

## Install

```bash
git clone https://github.com/vasilybelokurov/lmc-ancient-disc
cd lmc-ancient-disc
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```

`requirements-lock.txt` lists the exact package versions that made the results (Python 3.13). AGAMA is needed only to rerun the
halo and orbit experiments behind Figs 4 and 5; see `pyproject.toml`.

## Three routes

| Route | You need | You can remake | Time |
|---|---|---|---|
| Quick | the repository and Zenodo bundle 1 (`python scripts/fetch_release.py`) | every figure, table and number of the paper | minutes |
| Full | the public archives, through the `scripts/fetch_*.py` scripts | everything, checked against the sha256 in `data/sources.yaml` | days |

Bundle 1 holds the derived catalogues (giants, RR Lyrae, selection-function parents and weights), the posterior samples of the
kinematic fits, and the small public tables the scripts read. `results/` in the repository holds the tables behind every figure.

## Checking the results

`results/manifest.yaml` records every script run with the sha256 of each file it read and wrote.

```bash
python scripts/manifest.py check
```

lists any result that no longer matches the code and inputs that made it (none in the released version).

## Data

`data/sources.yaml` lists every input: its origin, the query or URL, the script that fetches or builds it, the number of rows, the
sha256 and the reference. One input is not public: the high-redshift snapshots of the SIDM Concerto halo used for Fig. 5, available
from the simulation authors on request.
