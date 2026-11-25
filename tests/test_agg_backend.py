"""Every plotting script selects the Agg backend before pyplot, so a figure has the same bytes when made on a Mac (whose default
backend is macosx), on Linux, or in a notebook of the knowledge base (which runs with MPLBACKEND=Agg)."""
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def test_plotting_scripts_use_agg():
    missing = []
    for f in sorted(SCRIPTS.glob('*.py')):
        s = f.read_text()
        if 'matplotlib.pyplot' in s and not (0 <= s.find("matplotlib.use('Agg')") < s.find('matplotlib.pyplot')):
            missing.append(f.name)
    assert not missing, missing
