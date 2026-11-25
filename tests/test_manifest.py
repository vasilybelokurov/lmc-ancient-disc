"""scripts/manifest.py: a traced run records what a script reads and writes, and check() flags changed inputs and outputs."""
import pytest
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import manifest

SCRIPT = """import pandas as pd
a = pd.read_csv('../data/in.csv')
b = pd.read_parquet('../data/in.parquet')
(a + b).to_csv('../results/out.csv', index=False)
"""


def setup(tmp_path):
    for d in ('data', 'results', 'scripts'):
        (tmp_path / d).mkdir()
    pd.DataFrame(dict(x=[1, 2])).to_csv(tmp_path / 'data/in.csv', index=False)
    pd.DataFrame(dict(x=[10, 20])).to_parquet(tmp_path / 'data/in.parquet')
    (tmp_path / 'scripts/make.py').write_text(SCRIPT)
    return tmp_path / 'scripts/make.py', tmp_path / 'results/manifest.yaml'


def test_trace_records_inputs_and_outputs(tmp_path):
    script, path = setup(tmp_path)
    e = manifest.record(script, [], root=tmp_path.resolve(), path=path)
    assert set(e['inputs']) == {'data/in.csv', 'data/in.parquet', 'scripts/make.py'}
    assert set(e['outputs']) == {'results/out.csv'}
    assert manifest.check(root=tmp_path.resolve(), path=path) == []


def test_check_flags_stale_modified_missing(tmp_path):
    script, path = setup(tmp_path)
    manifest.record(script, [], root=tmp_path.resolve(), path=path)
    pd.DataFrame(dict(x=[1, 3])).to_csv(tmp_path / 'data/in.csv', index=False)
    (tmp_path / 'results/out.csv').write_text('x\n0\n')
    (tmp_path / 'data/in.parquet').unlink()
    got = {(s, f) for s, _, f in manifest.check(root=tmp_path.resolve(), path=path)}
    assert got == {('stale', 'data/in.csv'), ('modified', 'results/out.csv'), ('missing', 'data/in.parquet')}


def test_shared_output_checked_against_latest_writer(tmp_path):
    """Two commands append to one table (as the parallel sims_observe workers do): only the later one's hash counts."""
    script, path = setup(tmp_path)
    (tmp_path / 'scripts/append.py').write_text("open('../results/out.csv', 'a').write('9\\n')\n")
    manifest.record(script, [], root=tmp_path.resolve(), path=path)
    manifest.record(tmp_path / 'scripts/append.py', [], root=tmp_path.resolve(), path=path)
    assert manifest.check(root=tmp_path.resolve(), path=path) == []


def test_imported_module_is_an_input_even_when_already_imported(tmp_path):
    """A module of the repository the script imports is recorded, also when it was imported before the run (or loaded from its
    __pycache__ bytecode, which the open hook skips): its code is an input of the results."""
    script, path = setup(tmp_path)
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src/helper_mod_xyz.py').write_text('K = 2\n')
    sys.path.insert(0, str(tmp_path / 'src'))
    import helper_mod_xyz  # noqa: F401  imported before the traced run
    script.write_text('import helper_mod_xyz\n' + SCRIPT)
    e = manifest.record(script, [], root=tmp_path.resolve(), path=path)
    assert 'src/helper_mod_xyz.py' in e['inputs']


def test_file_renamed_into_place_is_an_output(tmp_path):
    """A script that writes a temporary file and renames it into place (os.replace, as sims_observe.py does) has the final file
    as its output."""
    script, path = setup(tmp_path)
    (tmp_path / 'scripts/atomic.py').write_text("import os\nopen('../results/x.tmp', 'w').write('1\\n')\n"
                                                "os.replace('../results/x.tmp', '../results/x.csv')\n")
    e = manifest.record(tmp_path / 'scripts/atomic.py', [], root=tmp_path.resolve(), path=path)
    assert set(e['outputs']) == {'results/x.csv'}


def test_sources_entries_are_inputs_not_the_whole_list(tmp_path, monkeypatch):
    """A script that reads an input through lmcold.sources.path depends on that entry of data/sources.yaml, not on the whole list:
    adding another entry leaves it up to date; changing its own entry makes it stale."""
    import yaml
    from lmcold import sources
    script, path = setup(tmp_path)
    src = tmp_path / 'data/sources.yaml'
    entry = dict(file='in.csv', sha256=sources.sha256(tmp_path / 'data/in.csv'))
    src.write_text(yaml.safe_dump({'a': entry}))
    monkeypatch.setattr(sources, 'DATA', tmp_path / 'data')
    script.write_text("import pandas as pd\nfrom lmcold.sources import path\npd.read_csv(path('a')).to_csv('../results/out.csv')\n")
    e = manifest.record(script, [], root=tmp_path.resolve(), path=path)
    assert 'data/sources.yaml#a' in e['inputs'] and 'data/sources.yaml' not in e['inputs']
    src.write_text(yaml.safe_dump({'a': entry, 'b': dict(file='x', sha256='0')}))
    assert manifest.check(root=tmp_path.resolve(), path=path) == []
    src.write_text(yaml.safe_dump({'a': dict(entry, origin='changed'), 'b': dict(file='x', sha256='0')}))
    assert ('stale', 'data/sources.yaml#a') in {(s, f) for s, _, f in manifest.check(root=tmp_path.resolve(), path=path)}


def test_script_ending_with_sys_exit_is_recorded(tmp_path):
    """A script that ends with sys.exit() (status 0) is a successful run and is recorded; a non-zero exit is not."""
    script, path = setup(tmp_path)
    script.write_text(script.read_text() + '\nimport sys\nsys.exit()\n')
    e = manifest.record(script, [], root=tmp_path.resolve(), path=path)
    assert 'results/out.csv' in e['outputs']
    script.write_text(script.read_text().replace('sys.exit()', "sys.exit('failed')"))
    with pytest.raises(SystemExit):
        manifest.record(script, [], root=tmp_path.resolve(), path=path)


def test_comment_and_docstring_edits_keep_a_record_valid_code_edits_do_not(tmp_path):
    script, path = setup(tmp_path)
    root = tmp_path.resolve()
    manifest.record(script, [], root=root, path=path)
    script.write_text('"""Add the two inputs."""\n# read both tables\n' + SCRIPT.replace("index=False)", "index=False)   # the sum"))
    assert manifest.check(root=root, path=path) == []
    script.write_text(SCRIPT.replace('a + b', 'a - b'))
    assert manifest.check(root=root, path=path) == [('stale', 'make.py', 'scripts/make.py')]


def test_docstring_counts_as_code_where_the_file_uses_it(tmp_path):
    script, path = setup(tmp_path)
    root = tmp_path.resolve()
    script.write_text('"""Usage: make."""\nprint(__doc__)\n' + SCRIPT)
    manifest.record(script, [], root=root, path=path)
    script.write_text('"""Usage: make it."""\nprint(__doc__)\n' + SCRIPT)
    assert manifest.check(root=root, path=path) == [('stale', 'make.py', 'scripts/make.py')]


def test_convert_replaces_byte_hashes_of_unchanged_code_only(tmp_path):
    script, path = setup(tmp_path)
    root = tmp_path.resolve()
    e = manifest.record(script, [], root=root, path=path)
    m = manifest.load(path)
    m['make.py']['inputs']['scripts/make.py'] = manifest.sha256(script)   # an old record: byte hash of the script
    m['other.py'] = dict(recorded='x', inputs={'scripts/make.py': '0' * 64}, outputs={})   # recorded with different code
    path.write_text(manifest.yaml.safe_dump(m, sort_keys=False))
    assert manifest.convert(root=root, path=path) == 1
    m = manifest.load(path)
    assert m['make.py']['inputs']['scripts/make.py'] == e['inputs']['scripts/make.py'] == manifest.fingerprint(script)
    assert e['inputs']['scripts/make.py'].startswith('code:')
    assert m['other.py']['inputs']['scripts/make.py'] == '0' * 64
