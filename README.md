# The puzzle of the ancient stellar disc of the Large Magellanic Cloud: code and data products

Code, result tables and figures for the MNRAS Letter *The puzzle of the ancient stellar disc of the Large Magellanic Cloud* (Belokurov,
in preparation). The knowledge base in `kb/` explains every figure and table, including the material of the extended version.

**Read online:** https://vasilybelokurov.github.io/lmc-ancient-disc/ (the knowledge base, each page a notebook that opens in Colab)

**What the paper finds.** The LMC's red giants (APOGEE and SDSS-V spectra, *Gaia* proper motions) rotate as a disc at every
metallicity, even the most metal-poor ones, which formed when the LMC had made only a few per cent of its stars. None of the LMC
analogues in the TNG50, Auriga and FIRE-2 simulations keeps such an early disc: their oldest stars are born rotating but lose the
rotation within one or two orbital periods, because the young dark-matter halo is aspherical and misaligned with the disc.

## Three ways to reproduce the results

| Route | You need | Then you can remake | Time |
|---|---|---|---|
| Quick | this repository and Zenodo bundle 1 (`python scripts/fetch_release.py`) | every figure, table and number of the paper | minutes |
| Full | the public archives, via the `scripts/fetch_*.py` scripts | everything, checked against the sha256 in `data/sources.yaml` | days |

One input is not public: the SIDM Concerto snapshots at z = 4-1.5 used for Fig. 5, available from the simulation authors on request.
Fig. 5 is remade from `results/sims_sidm_shapes.csv`.

## Install and test

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```

The exact package versions that made the results are in `requirements-lock.txt` (Python 3.13). AGAMA, needed only for the
halo and orbit experiments of Figs 4 and 5, is an optional dependency (see `pyproject.toml`).

## Layout

| Path | Contents |
|---|---|
| `src/lmcold/` | library: LMC geometry, sky projections, kinematic model, selection functions, stellar mass, simulations |
| `scripts/` | `fetch_*` download public data; `build_*` make the catalogues and simulation tables; `paper_*` make the paper's figures, tables and numbers |
| `results/` | the tables behind every figure and number; `results/manifest.yaml` records each script run with the sha256 of what it read and wrote |
| `figures/` | the figures (PNG) |
| `data/sources.yaml` | every input: origin, query or URL, fetch script, rows, sha256, reference |
| `paper/` | the Letter source |
| `kb/` | the knowledge base |
| `tests/` | unit and reproduction tests |

`python scripts/manifest.py check` confirms that every committed result matches the code and inputs that made it.

## Conventions

- Angles are in radians and distances in kpc inside the library.
- Θ is the position angle of the line of nodes (north through east), and i is the inclination (0 = face-on).
- The near side is at PA Θ − 90°, as in van der Marel & Kallivayalil (2014, ApJ 781, 121; [arXiv:1305.4641](https://arxiv.org/abs/1305.4641)).

## Licence

MIT (see `LICENSE`). Data products derived from public surveys and simulations follow the terms of their sources, listed in
`data/sources.yaml`.
