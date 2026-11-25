"""Record what each script reads and writes, and check that the committed results are up to date with their inputs.

results/manifest.yaml has one entry per script run, keyed by the command ('script.py arg ...'): the repository files the run
read (inputs) and wrote (outputs), each with its sha256 (Python inputs: a fingerprint of the code without comments and
docstrings, see fingerprint(), so comment edits leave records valid); an entry of data/sources.yaml read through lmcold.sources.path is
recorded on its own, as 'data/sources.yaml#<key>' with the sha256 of that entry, so that adding or changing another entry leaves the
run up to date. The tracer itself (this file) is not an input. It is written by running the script through this module, never by hand.

  run   python scripts/manifest.py run <script.py> [args ...]
        Runs the script as __main__ with an audit hook on file opens and wrappers on the readers that open files outside Python
        (pandas.read_parquet, pyarrow.parquet, h5py.File), then replaces the command's entry. Files outside the repository (large
        inputs in ~/data, listed with their sha256 in data/sources.yaml) and files that are both read and written count as outputs
        only if written.
  check python scripts/manifest.py check
        For every entry: 'modified' if an output no longer has its recorded sha256 (changed after the run), 'stale' if an input
        has changed since the run (rerun the command), 'missing' if a recorded file is gone. Exit status 1 if anything is listed.

Posterior samples (data/*.npz) and other large derived files stay out of git but are listed with their sha256, so a stale chain is
seen even where git is not. MCMC runs are deterministic on one machine (fixed seed: the 2026-11-09 rerun of the main ellipsoid fit
reproduced the 6 Nov samples byte for byte); across machines or library versions they need not be.
Usage: .venv/bin/python scripts/manifest.py run paper_table_shape.py ; .venv/bin/python scripts/manifest.py check"""
import ast, fcntl, hashlib, os, runpy, sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lmcold.sources import ROOT, sha256

MANIFEST = ROOT / 'results/manifest.yaml'   # run record: inputs, outputs and their hashes per script invocation
SOURCES = 'data/sources.yaml'   # read through lmcold.sources.path: recorded per entry, as 'data/sources.yaml#<key>'
SKIP = ('.git/', '.venv/', '__pycache__', '.pytest_cache', 'results/manifest.yaml', 'scripts/manifest.py')   # the tracer is not an input


def _rel(path, root):
    """Path relative to root as a string, or None if outside it or ignored."""
    try:
        p = Path(os.fspath(path)).expanduser().resolve().relative_to(root)
    except (TypeError, ValueError):
        return None
    s = p.as_posix()
    return None if any(k in s for k in SKIP) else s


def entry_hash(key, root=ROOT):
    """sha256 of entry `key` of data/sources.yaml (canonical YAML), or None if there is no such entry."""
    entries = yaml.safe_load((root / SOURCES).read_text()) or {}
    return hashlib.sha256(yaml.safe_dump(entries[key], sort_keys=True).encode()).hexdigest() if key in entries else None


def fingerprint(path):
    """Hash of an input file as recorded: for Python, 'code:' + sha256 of its syntax tree without comments and docstrings, so that
    editing comments leaves a record valid while any change to the code makes it stale (docstrings count as code in a file that
    reads __doc__); for any other file, its sha256. The tree depends on the Python version (3.13 here)."""
    p = Path(path)
    if p.suffix != '.py':
        return sha256(p)
    src = p.read_text()
    tree = ast.parse(src)
    if '__doc__' not in src:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body and \
                    isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return 'code:' + hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def trace(script, args, root=ROOT):
    """Run script (path) with args as __main__ in its own directory; return (sorted inputs, sorted outputs) as repo-relative paths,
    the entries of data/sources.yaml read through lmcold.sources.path as 'data/sources.yaml#<key>' instead of the whole file."""
    reads, writes, active, keys, inside = set(), set(), [True], set(), [False]

    def note(path, write):
        r = _rel(path, root)
        if r == SOURCES and inside[0] and not write:   # opened by lmcold.sources.path: the entry is the input, not the list
            return
        if r and (root / r).is_file() or (r and write):
            (writes if write else reads).add(r)

    def hook(event, a):
        if active[0] and event == 'open' and a and isinstance(a[0], (str, bytes, os.PathLike)):
            mode, flags = (a[1] or '') if len(a) > 1 else '', a[2] if len(a) > 2 else 0
            write = any(c in str(mode) for c in 'wax+') or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
            note(a[0], write)
        elif active[0] and event == 'os.rename':   # os.rename and os.replace: a file written elsewhere and moved into place
            note(a[1], True)

    wrapped = []

    def wrap(mod, name, write=False, arg=0):
        f = getattr(mod, name)
        def g(*a, **k):
            p = a[arg] if len(a) > arg else k.get('path', k.get('source', k.get('name')))
            if isinstance(p, (str, os.PathLike)):
                note(p, write or (name == 'File' and (a[1:2] or [k.get('mode', 'r')])[0] in ('w', 'a', 'x', 'r+', 'w-')))
            return f(*a, **k)
        setattr(mod, name, g); wrapped.append((mod, name, f))

    from lmcold import sources
    source_path = sources.path

    def entry(name):
        keys.add(name); inside[0] = True
        try:
            return source_path(name)
        finally:
            inside[0] = False
    sources.path = entry; wrapped.append((sources, 'path', source_path))
    import pandas, pyarrow.parquet, h5py
    wrap(pandas, 'read_parquet'); wrap(pyarrow.parquet, 'read_table'); wrap(pyarrow.parquet, 'ParquetFile')
    wrap(h5py, 'File'); wrap(pandas.DataFrame, 'to_parquet', write=True, arg=1)
    sys.addaudithook(hook)   # audit hooks cannot be removed: switched off by 'active' after the run
    argv, cwd = sys.argv, os.getcwd()
    try:
        sys.argv = [str(script)] + list(args)
        os.chdir(Path(script).parent)
        sys.path.insert(0, str(Path(script).resolve().parent))
        try:
            runpy.run_path(str(script), run_name='__main__')
        except SystemExit as e:   # sys.exit() or sys.exit(0) ends a successful run
            if e.code not in (None, 0):
                raise
    finally:
        active[0] = False; sys.argv = argv; os.chdir(cwd)
        for mod, name, f in wrapped:
            setattr(mod, name, f)
    # Repository modules in use: imported before the run (by this module) or loaded from __pycache__ bytecode, they raise no
    # 'open' event for their source, but their code is an input.
    for m in list(sys.modules.values()):
        if getattr(m, '__file__', None):
            note(m.__file__, False)
    return sorted(reads - writes) + [f'{SOURCES}#{k}' for k in sorted(keys)], sorted(writes)


# Read the manifest YAML into a dict {script key: entry}; an empty dict if the file is missing or empty. (A comment, not a
# docstring: this file prints its own __doc__, so docstrings here count as code in the run-log fingerprint.)
def load(path=MANIFEST):
    return yaml.safe_load(path.read_text()) or {} if path.exists() else {}


def record(script, args, root=ROOT, path=MANIFEST):
    """Run script through trace() and replace its entry in the manifest."""
    inputs, outputs = trace(script, args, root)
    key = ' '.join([Path(script).name] + list(args))
    e = dict(recorded=datetime.now(timezone.utc).isoformat(timespec='microseconds'),
             inputs={f: entry_hash(f.split('#', 1)[1], root) if '#' in f else fingerprint(root / f) for f in inputs
                     if '#' in f or (root / f).is_file()},
             outputs={f: sha256(root / f) for f in outputs if (root / f).is_file()})
    with open(path.with_suffix('.lock'), 'w') as lock:   # parallel workers (sims_observe.py i n) record at once
        fcntl.flock(lock, fcntl.LOCK_EX)
        m = load(path); m[key] = e
        path.write_text(yaml.safe_dump(dict(sorted(m.items())), sort_keys=False, width=200))
    return e


def check(root=ROOT, path=MANIFEST):
    """List (status, command, file) for every recorded file that is modified, stale or missing. A file written by several
    commands (parallel workers appending to one table, or every fit redrawing one figure) is checked against the latest of them."""
    m, out = load(path), []
    latest = {}
    for key, e in m.items():
        for f in e['outputs']:
            if e.get('recorded', '') >= m.get(latest.get(f), {}).get('recorded', ''):
                latest[f] = key
    for key, e in m.items():
        for kind, files in (('outputs', e['outputs']), ('inputs', e['inputs'])):
            for f, h in files.items():
                p = root / f.split('#', 1)[0]
                if kind == 'outputs' and latest[f] != key:
                    continue
                if '#' in f:   # one entry of data/sources.yaml
                    now = entry_hash(f.split('#', 1)[1], root) if p.is_file() else None
                    if now != h:
                        out.append(('missing' if now is None else 'stale', key, f))
                    continue
                if not p.is_file():
                    out.append(('missing', key, f))
                elif (fingerprint(p) if h.startswith('code:') else sha256(p)) != h:
                    out.append(('modified' if kind == 'outputs' else 'stale', key, f))
    return out


def convert(root=ROOT, path=MANIFEST):
    """Replace the byte sha256 of each recorded Python input by its fingerprint(), only where the file still has the recorded bytes
    (so the code fingerprinted is the code that ran); returns the number of replaced hashes. Used once, 2026-11-25."""
    m, n = load(path), 0
    for e in m.values():
        for f, h in e['inputs'].items():
            p = root / f
            if f.endswith('.py') and not h.startswith('code:') and p.is_file() and sha256(p) == h:
                e['inputs'][f] = fingerprint(p)
                n += 1
    path.write_text(yaml.safe_dump(m, sort_keys=False, width=200))
    return n


if __name__ == '__main__':
    if sys.argv[1:2] == ['run']:
        s = Path(sys.argv[2])
        s = s if s.exists() else ROOT / 'scripts' / s
        e = record(s, sys.argv[3:])
        print(f'{len(e["inputs"])} inputs, {len(e["outputs"])} outputs recorded')
    elif sys.argv[1:2] == ['convert']:
        print(f'{convert()} Python input hashes replaced by code fingerprints')
    elif sys.argv[1:2] == ['check']:
        bad = check()
        for status, key, f in bad:
            print(f'{status:8s} {key}: {f}')
        print(f'{len(load())} commands, {len(bad)} problems')
        sys.exit(1 if bad else 0)
    else:
        sys.exit(__doc__)
