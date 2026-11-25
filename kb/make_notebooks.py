"""Assemble the knowledge-base notebooks: kb/notebooks/<page>.ipynb from each page source in kb/ (MyST Markdown), which the site shows
and the 'Open in Colab' buttons open.

Each notebook is built from three things, so that nothing is written twice:
- a first cell that sets the notebook up: in Google Colab it downloads and installs the repository; elsewhere it moves to the
  repository root;
- the page text, in which every number {glue:text}`name` is replaced by its value from the macro files that the scripts write
  (results/macros_*.tex), so that the numbers read the same on the site and in Colab;
- at each line <!-- script: paper_xxx.py --> of the text (or <!-- script: x.py show: <what it needs, how long> -->: its code
  shown as text, not run), the cells of that script: its '# %% [markdown]' comments become text
  cells and its code (the body of `if __name__ == '__main__':` moved to the top level) becomes code cells. The script still runs
  unchanged as `python scripts/paper_xxx.py`;
- numbered captions, as in the tutorials (../tutorials): a paragraph of the page that starts '**Figure (stem).**' becomes
  '**Figure 8.2.**' (chapter 8, its second figure) and is shown, with figures/<stem>.png, right after the script cell that saves
  that figure (or where it stands, if no script of the page saves it); '**Table (key).**' becomes '**Table 8.1.**'. In the text,
  {fig}`stem` and {tab}`key` become 'Figure 8.2' and 'Table 8.1', also across pages. The build stops if a figure a script of the
  page saves has no caption, or a caption is never cited in the text.

Run: .venv/bin/python kb/make_notebooks.py   (before jupyter-book build kb)
"""
import re
from pathlib import Path

import jupytext
import nbformat
import yaml

KB = Path(__file__).resolve().parent
ROOT = KB.parent
PAGES = {'letter/fig1_tracers.md': 'Letter 1', 'letter/fig2_rotational_support.md': 'Letter 2', 'letter/fig3_birth.md': 'Letter 3',
         'letter/fig4_scrambling.md': 'Letter 4', 'letter/fig5_sidm.md': 'Letter 5',
         'chapters/data_samples.md': 1, 'chapters/data_selection.md': 2, 'chapters/data_calibration.md': 3,
         'chapters/mass.md': 4, 'chapters/metallicity.md': 5, 'chapters/kinematics_model.md': 6,
         'chapters/kinematics_support.md': 7, 'chapters/kinematics_vc.md': 7, 'chapters/kinematics_lowfeh.md': 7, 'chapters/shape.md': 8,
         'chapters/sims_analogues.md': 9, 'chapters/sims_evolution.md': 10, 'chapters/getting_data.md': 11}   # page: chapter number (docs/kb_chapters.md) or 'Letter N'
REPO = 'https://github.com/vasilybelokurov/lmc-ancient-disc'
MARK = re.compile(r'^<!-- script: (\w+\.py)(?: show: (.+?))? -->$', re.M)   # 'show: <why>': shown, not run (needs, run time)
CELL = re.compile(r'^# %%( \[markdown\])?\s*$')
CAPTION = re.compile(r'^\*\*(Figure|Table) \((\w+)\)\.\*\*', re.M)
SAVES = re.compile(r"""savefig\(ROOT\s*/\s*['"]figures(?:['"]\s*/\s*['"]|/)(\w+)\.png['"]""")   # ROOT / 'figures/x.png' or "figures" / "x.png"
KIND = {'fig': 'Figure', 'tab': 'Table'}
REF = re.compile(r'\{(fig|tab)\}`(\w+)`')
MAIN = "if __name__ == '__main__':"

SETUP = f'''# Setup: in Google Colab, download and install the repository (a few minutes, once per session); elsewhere, work from the
# repository. Pages that read the derived catalogues need Zenodo bundle 1 (python scripts/fetch_release.py).
import os, subprocess, sys
os.environ['MPLBACKEND'] = 'Agg'   # as in a plain script run: Jupyter's inline settings would change the saved figures
if 'google.colab' in sys.modules:
    if not os.path.exists('/content/lmc-ancient-disc'):
        subprocess.run(['git', 'clone', '--depth', '1', '{REPO}', '/content/lmc-ancient-disc'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '/content/lmc-ancient-disc'], check=True)
    os.chdir('/content/lmc-ancient-disc')
else:
    while not os.path.exists('pyproject.toml'):
        os.chdir('..')
sys.path[:0] = ['src', 'scripts', 'kb']
if not os.path.exists('data/giants_apogee.parquet'):   # the catalogues of Zenodo bundle 1, checked against data/sources.yaml
    subprocess.run([sys.executable, 'scripts/fetch_release.py'], check=True)'''


def macros():
    """{name: value} of every \\newcommand in results/macros_*.tex."""
    out = {}
    for f in sorted((ROOT / 'results').glob('macros_*.tex')):
        out.update(re.findall(r'^\\newcommand\{\\(\w+)\}\{(.*)\}\s*$', f.read_text(), re.M))
    return {k: v.replace('\\,', '\u2009') for k, v in out.items()}


def numbers():
    """{('Figure' or 'Table', name): number} over all pages: 'C.k' in a chapter C, counted in order of the captions over its pages;
    on a page of a Letter figure ('Letter 3'), the Letter's number of that figure ('3') for its first caption and '3b', '3c' for
    further figures on that page."""
    out, k = {}, {}
    for page, chapter in PAGES.items():
        for kind, name in CAPTION.findall((KB / page).read_text()):
            assert (kind, name) not in out, f'{page}: {kind} ({name}) is captioned on two pages'
            if isinstance(chapter, str):   # the Letter's figure, then 'Nb', 'Nc' for more figures on its page
                k[page, kind] = k.get((page, kind), 0) + 1
                out[kind, name] = chapter.split()[1] + ('' if k[page, kind] == 1 else 'abcdefgh'[k[page, kind] - 1])
                continue
            k[chapter, kind] = k.get((chapter, kind), 0) + 1
            out[kind, name] = f'{chapter}.{k[chapter, kind]}'
    return out


def filler():
    """Function that puts the numbers into page text: {glue:text}`name` macros and {fig}/{tab} citations."""
    values, num = macros(), numbers()

    def fill(text):
        text = re.sub(r'\{glue:text\}`(\w+)`', lambda m: values[m.group(1)], text)
        return REF.sub(lambda m: KIND[m.group(1)] + ' ' + num[KIND[m.group(1)], m.group(2)], text)
    return fill, num


def captions(page, fill, num):
    """{figure stem: numbered caption} of a page."""
    out = {}
    for para in re.split(r'\n\s*\n', fill((KB / page).read_text())):
        m = CAPTION.match(para.strip())
        if m and m.group(1) == 'Figure':
            out[m.group(2)] = para.strip().replace(f'**Figure ({m.group(2)}).**', f'**Figure {num["Figure", m.group(2)]}.**', 1)
    return out


def figure_cells(stem, caption):
    """A cell showing figures/<stem>.png (code hidden on the site) and its numbered caption."""
    return [nbformat.v4.new_code_cell(f"from IPython.display import Image\nImage(filename='figures/{stem}.png')",
                                      metadata={'tags': ['remove-input']}),
            nbformat.v4.new_markdown_cell(caption)]


def data_used(name):
    """Markdown table of the files scripts/<name> reads, from its runs in the run log (results/manifest.yaml): each file of data/
    with its entry of data/sources.yaml (tier B: Zenodo bundle 1; tier A: public archive), each file of results/ with the logged
    script that writes it (or, for results made outside the run log, the command of its tier-C entry), and the external inputs (simulations, surveys) the run declared."""
    man = yaml.safe_load((ROOT / 'results/manifest.yaml').read_text())
    src = yaml.safe_load((ROOT / 'data/sources.yaml').read_text())
    by_file = {str(e['file']): (k, e) for k, e in src.items()}
    prod = {o: c for c, e in man.items() for o in e['outputs']}
    inputs = sorted({i for c, e in man.items() if c.split()[0] == name for i in e['inputs']})
    keys = [i.split('#')[1] for i in inputs if '#' in i]
    if len(keys) > 20:   # a record converted with every entry of sources.yaml (runs before per-entry tracing): not informative
        keys = []
    rows = []
    for f in inputs:
        if f.startswith('data/') and '#' not in f:
            k, e = by_file.get(f[5:], (None, {}))
            where = 'Zenodo bundle 1' if e.get('tier') == 'B' else ('public archive' if e.get('tier') == 'A' else 'repository')
            rows.append(f'| `{f}` | {where} | {e.get("origin", "")} |')
        elif f.startswith('results/') and f != 'results/manifest.yaml':
            if f in prod and prod[f].split()[0] == name:
                continue   # written by this script itself
            k, e = by_file.get('../' + f, (None, {}))   # tier C: results made outside the run log, with their command
            maker = (f'written by `python scripts/{prod[f]}`' if f in prod else
                     f'{e["origin"]}; written by `{e["build"]}`' if 'build' in e else e.get('origin', 'fixed input'))
            rows.append(f'| `{f}` | repository | {maker} |')
    for k in keys:
        e = src[k]
        if str(e['file']).startswith('~'):
            rows.append(f'| `{e["file"]}` | external ({k}) | {e.get("origin", "")}; fetched by `{e.get("fetch", "-")}` |')
    if not rows:
        return None
    return nbformat.v4.new_markdown_cell(f'**Data read by `scripts/{name}`** (from the run log)\n\n| File | Where it comes from | What it is |\n'
                                         '|---|---|---|\n' + '\n'.join(rows))


def script_cells(name, captions, level=2, show=None):
    """Notebook cells of scripts/<name>: the module docstring as a collapsed note, then its '# %%' cells (main block at top level),
    each code cell that saves a figure followed by that figure and its caption (from `captions`, {stem: text}). The script's step
    titles become headings of `level` (one below the page heading they stand under), so they are listed in the page contents.
    With `show` (what the script needs and how long it takes), its code is shown as code blocks in text cells (not run, on the
    site or in Colab), a note says why, and no figure is shown."""
    src = (ROOT / 'scripts' / name).read_text()
    doc = re.match(r'\s*"""(.*?)"""\n', src, re.S)
    body = src[doc.end():] if doc else src
    lines, main = [], False
    for line in body.splitlines():
        if line.strip() == MAIN:
            main = True
            continue
        lines.append(line[4:] if main and line.startswith('    ') else line)
    cells, kind, buf = [], 'code', []

    def flush():
        text = '\n'.join(buf).strip('\n')
        if not text.strip():
            return
        if kind == 'markdown':
            text = re.sub(r'^# ?', '', text, flags=re.M)
            cells.append(nbformat.v4.new_markdown_cell(re.sub(r'^#+ ', '#' * level + ' ', text, flags=re.M)))   # step titles at `level`
            return
        if show:   # a code block in a text cell, so that nothing runs it (Colab's 'Run all' ignores skip-execution tags)
            cells.append(nbformat.v4.new_markdown_cell(f'```python\n{text}\n```'))
            return
        cells.append(nbformat.v4.new_code_cell(text))
        for stem in dict.fromkeys(SAVES.findall(text)):
            assert stem in captions, f'scripts/{name} saves figures/{stem}.png, which has no caption on its page'
            cells.extend(figure_cells(stem, captions[stem]))

    for line in lines:
        m = CELL.match(line)
        if m:
            flush()
            kind, buf = ('markdown' if m.group(1) else 'code'), []
        else:
            buf.append(line)
    flush()
    cells.insert(0, nbformat.v4.new_code_cell(   # each script starts from matplotlib's defaults, as in a process of its own
        "import matplotlib\nmatplotlib.rcdefaults()\nimport matplotlib.pyplot as plt\nplt.close('all')", metadata={'tags': ['remove-cell']}))
    if show:
        cells.insert(0, nbformat.v4.new_markdown_cell(f'*Shown, not run here: {show}. The cells below are the script itself; run it from the '
                                                      f'repository with* `python scripts/{name}`*.*'))
    table = data_used(name)
    if table:
        cells.insert(0, table)
    if doc:
        cells.insert(0, nbformat.v4.new_markdown_cell(
            f'<details><summary>About <code>scripts/{name}</code></summary>\n\n```text\n{doc.group(1).strip()}\n```\n</details>'))
    return cells


def build(page):
    """Write kb/notebooks/<page>.ipynb. A figure saved by a script of the page is shown with its caption, from whichever page
    captions it."""
    nb = jupytext.read(KB / page, fmt='md:myst')
    fill, num = filler()
    source = (KB / page).read_text()
    cited = {(KIND[k], n) for k, n in REF.findall(source)}
    saved = {s for name, show in MARK.findall(source) if not show for s in SAVES.findall((ROOT / 'scripts' / name).read_text())}
    every = {k: v for other in PAGES for k, v in captions(other, fill, num).items()}

    paragraphs = {}   # markdown cell index: its paragraphs, a figure shown where its caption stands as (stem, caption)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown':
            paragraphs[i] = []
            for para in re.split(r'\n\s*\n', fill(cell.source)):
                m = CAPTION.match(para.strip())
                if m:
                    kind, name = m.groups()
                    assert (kind, name) in cited, f'{page}: {kind} ({name}) is never cited in the text'
                    para = para.strip().replace(f'**{kind} ({name}).**', f'**{kind} {num[kind, name]}.**', 1)
                    if kind == 'Figure':
                        if name in saved:
                            continue   # shown after the script cell that saves it
                        para = (name, para)
                paragraphs[i].append(para)

    out = [nbformat.v4.new_code_cell(SETUP, metadata={'tags': ['remove-input']})]
    level = 1   # level of the last page heading
    colab = REPO.replace('https://github.com/', 'https://colab.research.google.com/github/') + \
        f'/blob/main/kb/notebooks/{Path(page).with_suffix(".ipynb")}'
    badge = f'[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab})'   # under the title, as the tutorials
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'markdown':
            out.append(cell)
            continue
        if i == next(k for k, c in enumerate(nb.cells) if c.cell_type == 'markdown'):
            title = paragraphs[i][0].split('\n', 1)
            assert title[0].startswith('# '), f'{page}: the page must start with its title'
            paragraphs[i][0:1] = [title[0], badge] + title[1:]
        buf = []
        for para in paragraphs[i] + [None]:
            if isinstance(para, str) and MARK.match(para.strip()) is None:
                buf.append(para)
                level = ([len(h) for h in re.findall(r'^(#+) ', para, re.M)] or [level])[-1]
                continue
            if '\n\n'.join(buf).strip():
                out.append(nbformat.v4.new_markdown_cell('\n\n'.join(buf).strip('\n')))
            buf = []
            if isinstance(para, tuple):
                out.extend(figure_cells(*para))
            elif isinstance(para, str):
                name, show = MARK.match(para.strip()).groups()
                out.extend(script_cells(name, every, min(level + 1, 6), show))
    for i, cell in enumerate(out):   # fixed ids, so that a rebuilt notebook is byte-identical
        cell['id'] = f'cell-{i:03d}'
    nb.cells = out
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    target = KB / 'notebooks' / Path(page).with_suffix('.ipynb')
    target.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, target)
    return target


if __name__ == '__main__':
    for page in PAGES:
        print(build(page))
