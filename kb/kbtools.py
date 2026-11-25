"""Helper for the knowledge-base pages: list the steps upstream of a figure from the run log (results/manifest.yaml)."""
from pathlib import Path

import yaml
from IPython.display import Markdown, display

ROOT = Path(__file__).resolve().parents[1]


def upstream(png):
    """The logged runs upstream of figures/<png>, nearest first, with the data/ files they read."""
    man = yaml.safe_load((ROOT / 'results/manifest.yaml').read_text())
    prod = {o: c for c, e in man.items() for o in e['outputs']}
    order, todo = [], [prod['figures/' + png]]
    while todo:
        c = todo.pop(0)
        if c in order:
            continue
        order.append(c)
        todo += [prod[i] for i in man[c]['inputs'] if i in prod and prod[i] != c]
    rows = ['| Step | Reads (results/, data/) | Writes |', '|---|---|---|']
    for c in order:
        e = man[c]
        ins = ', '.join(f'`{i}`' for i in e['inputs'] if i.startswith(('results/', 'data/')) and not i.startswith('data/sources.yaml'))
        outs = ', '.join(f'`{o}`' for o in e['outputs'])
        rows.append(f'| `python scripts/{c}` | {ins or "–"} | {outs} |')
    display(Markdown('\n'.join(rows)))
