"""kb/make_notebooks.py numbers captions per chapter, puts each figure and its caption right after the script cell that saves it,
replaces {fig}`stem` by 'Figure C.k', and stops on a figure without a caption or a caption never cited."""
import sys
from pathlib import Path

import pytest

pytest.importorskip('jupytext')   # the knowledge-base tools; not in requirements-lock.txt (the GitHub test machine)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kb'))
import make_notebooks as mn  # noqa: E402

SCRIPT = """\"\"\"Doc.\"\"\"
from lmcold.sources import ROOT

if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data

    # %%
    fig.savefig(ROOT / 'figures/one.png')

    # %%
    x = 1
"""


def page(tmp_path, monkeypatch, text):
    (tmp_path / 'kb/chapters').mkdir(parents=True)
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'results').mkdir()
    (tmp_path / 'scripts/s.py').write_text(SCRIPT)
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/sources.yaml').write_text("cat:\n  tier: B\n  file: cat.parquet\n  origin: the catalogue\n")
    (tmp_path / 'results/manifest.yaml').write_text(
        "s.py:\n  inputs: {data/cat.parquet: x, results/t.csv: x}\n  outputs: {figures/one.png: x}\n"
        "t.py:\n  inputs: {}\n  outputs: {results/t.csv: x}\n")
    (tmp_path / 'kb/chapters/p.md').write_text('# Title\n\n' + text)
    monkeypatch.setattr(mn, 'KB', tmp_path / 'kb')
    monkeypatch.setattr(mn, 'ROOT', tmp_path)
    monkeypatch.setattr(mn, 'PAGES', {'chapters/p.md': 4})
    return mn.build('chapters/p.md')


def test_caption_follows_the_saving_cell(tmp_path, monkeypatch):
    import nbformat
    nb = nbformat.read(page(tmp_path, monkeypatch, 'See {fig}`one`.\n\n<!-- script: s.py -->\n\n**Figure (one).** A plot.\n'), 4)
    src = [c.source for c in nb.cells]
    i = next(k for k, t in enumerate(src) if 'savefig' in t)
    assert "figures/one.png" in src[i + 1] and src[i + 2] == '**Figure 4.1.** A plot.' and 'x = 1' in src[i + 3]
    assert any('See Figure 4.1.' in t for t in src)
    colab = mn.REPO.replace('https://github.com/', 'https://colab.research.google.com/github/')
    assert f'({colab}/blob/main/kb/notebooks/chapters/p.ipynb)' in src[1]


def test_uncaptioned_or_uncited_figure_stops_the_build(tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match='no caption'):
        page(tmp_path, monkeypatch, '<!-- script: s.py -->\n')


def test_uncited_caption_stops_the_build(tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match='never cited'):
        page(tmp_path, monkeypatch, '<!-- script: s.py -->\n\n**Figure (one).** A plot.\n')


def test_figure_captioned_on_another_page(tmp_path, monkeypatch):
    """A script on a second page shows its figure with the caption (and number) of the page that captions it."""
    import nbformat
    page(tmp_path, monkeypatch, 'See {fig}`one`.\n\n**Figure (one).** A plot.\n\n<!-- script: s.py -->\n')
    (tmp_path / 'kb/letter').mkdir()
    (tmp_path / 'kb/letter/q.md').write_text('# Q\n\nAs {fig}`one`.\n\n<!-- script: s.py -->\n')
    monkeypatch.setattr(mn, 'PAGES', {'letter/q.md': 'Letter 2', 'chapters/p.md': 4})
    src = [c.source for c in nbformat.read(mn.build('letter/q.md'), 4).cells]
    assert '**Figure 4.1.** A plot.' in src and any('As Figure 4.1.' in x for x in src)


def test_savefig_forms():
    assert mn.SAVES.findall("""f.savefig(ROOT / 'figures/a.png', dpi=3); f.savefig(ROOT/'figures/b.png')
f.savefig(ROOT / "figures" / "c.png")""") == ['a', 'b', 'c']


def test_macro_values_with_braces(tmp_path, monkeypatch):
    (tmp_path / 'results').mkdir()
    (tmp_path / 'results/macros_x.tex').write_text('% c\n\\newcommand{\\xM}{4.1\\times10^{9}}\n\\newcommand{\\xN}{3}\n')
    monkeypatch.setattr(mn, 'ROOT', tmp_path)
    assert mn.macros() == {'xM': '4.1\\times10^{9}', 'xN': '3'}


def test_data_table_before_the_script(tmp_path, monkeypatch):
    import nbformat
    src = [c.source for c in nbformat.read(page(tmp_path, monkeypatch, 'See {fig}`one`.\n\n<!-- script: s.py -->\n\n**Figure (one).** A.\n'), 4).cells]
    table = next(t for t in src if t.startswith('**Data read by `scripts/s.py`**'))
    assert '| `data/cat.parquet` | Zenodo bundle 1 | the catalogue |' in table
    assert '| `results/t.csv` | repository | written by `python scripts/t.py` |' in table


def test_step_titles_are_headings_below_the_page_heading(tmp_path, monkeypatch):
    import nbformat
    src = [c.source for c in nbformat.read(page(tmp_path, monkeypatch, 'See {fig}`one`.\n\n## Making it\n\n<!-- script: s.py -->\n\n'
                                                '**Figure (one).** A.\n'), 4).cells]
    assert '### Load the data' in src


def test_show_only_script_is_code_text_not_a_cell_and_needs_no_caption(tmp_path, monkeypatch):
    # shown as a Python code block in a text cell: Colab's 'Run all' (which ignores skip-execution tags) must not run it
    import nbformat
    nb = nbformat.read(page(tmp_path, monkeypatch, '<!-- script: s.py show: the raw surveys; about 2 h -->\n'), 4)
    assert not any('savefig' in c.source for c in nb.cells if c.cell_type == 'code')
    assert any(c.source.startswith('```python\n') and 'savefig' in c.source for c in nb.cells if c.cell_type == 'markdown')
    assert any('Shown, not run here' in c.source and 'the raw surveys; about 2 h' in c.source for c in nb.cells if c.cell_type == 'markdown')
    assert not any('figures/one.png' in c.source for c in nb.cells if 'Image' in c.source)


def test_second_figure_on_a_letter_page_gets_a_letter(tmp_path, monkeypatch):
    (tmp_path / 'kb/letter').mkdir(parents=True)
    (tmp_path / 'kb/letter/f4.md').write_text('# F\n\n**Figure (main).** A.\n\n**Figure (demo).** B.\n')
    monkeypatch.setattr(mn, 'KB', tmp_path / 'kb')
    monkeypatch.setattr(mn, 'PAGES', {'letter/f4.md': 'Letter 4'})
    assert mn.numbers() == {('Figure', 'main'): '4', ('Figure', 'demo'): '4b'}


def test_bundle_table_rows_have_as_many_columns_as_its_header():
    """A header with a missing '|' turns the table into plain text on the site (2026-11-25)."""
    import re
    lines = [line for line in mn.bundle_table().splitlines() if line.startswith('|')]
    cells = {len(re.sub(r'`[^`]*`', 'x', line).split('|')) for line in lines}   # code spans may hold no '|', but be safe
    assert len(lines) > 2 and len(cells) == 1, cells
