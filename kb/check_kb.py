"""Check the knowledge base: every code cell has comments, and every link of the built site works.

1. Cells: in every notebook of kb/notebooks (the run cells, and the code shown as text in ```python blocks), each cell with
   4 or more code lines (imports not counted) must have at least one comment line per 10 code lines.
2. Links (if the site is built, kb/_build/html): every href and src of every page. A link inside the site must point to an
   existing file and, with #anchor, to an existing id on that page. An 'Open in Colab' link must point to a notebook of
   kb/notebooks. Every other web link is fetched once: 404 or 410, or no answer at all, is an error; other refusals (403, 429,
   5xx: sites that block scripts) are listed as warnings.

Run: python kb/check_kb.py   (after jupyter-book build kb; about a minute for the web links)
"""
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urldefrag, urlparse

KB = Path(__file__).resolve().parent
SITE = KB / '_build/html'
MIN_LINES, PER_LINES = 4, 10   # a cell with MIN_LINES+ code lines needs a comment line per PER_LINES code lines
COLAB = re.compile(r'https://colab\.research\.google\.com/github/[^/]+/[^/]+/blob/main/(kb/notebooks/.+\.ipynb)$')


def bare_cells():
    """(notebook, cell number, code lines, comment lines) of every cell with too few comments."""
    bad = []
    for nb in sorted((KB / 'notebooks').rglob('*.ipynb')):
        for i, cell in enumerate(json.loads(nb.read_text())['cells']):
            src = ''.join(cell['source'])
            # Run cells, or the ```python blocks of text cells (scripts shown, not run)
            blocks = [src] if cell['cell_type'] == 'code' else re.findall(r'```python\n(.*?)```', src, re.S)
            for block in blocks:
                lines = [s.strip() for s in block.splitlines() if s.strip()]
                code = [s for s in lines if not s.startswith(('#', 'import ', 'from '))]
                comments = [s for s in lines if '#' in s]
                if len(code) >= MIN_LINES and len(comments) * PER_LINES < len(code):
                    bad.append((nb.relative_to(KB), i, len(code), len(comments)))
    return bad


class Page(HTMLParser):
    """Collects the links (href, src) and the ids of one HTML page."""

    def __init__(self):
        super().__init__()
        self.links, self.ids = [], set()

    def handle_starttag(self, tag, attrs):
        for k, v in attrs:
            if k in ('href', 'src') and v:
                self.links.append(v)
            if k in ('id', 'name') and v:
                self.ids.add(v)


def status(url):
    """HTTP status of a web link (0: no answer)."""
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (link check)'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def broken_links():
    """(errors, warnings): lists of (page, link, reason)."""
    pages = {p: Page() for p in SITE.rglob('*.html') if '_static' not in p.parts}   # _static: theme templates, not pages
    for p, parser in pages.items():
        parser.feed(p.read_text(errors='replace'))
    errors, web = [], {}
    for p, parser in pages.items():
        for link in parser.links:
            if link.startswith(('mailto:', 'javascript:', 'data:')):
                continue
            if COLAB.match(link):
                # The notebook Colab opens is the one in this commit of the repository
                if not (KB.parent / COLAB.match(link).group(1)).exists():
                    errors.append((p, link, 'no such notebook'))
            elif urlparse(link).scheme in ('http', 'https'):
                web.setdefault(urldefrag(link)[0], p)
            else:
                # A link inside the site: the file, then the anchor on it
                path, anchor = urlparse(link).path, urlparse(link).fragment
                target = (p.parent / unquote(path)).resolve() if path else p
                if target.is_dir():
                    target = target / 'index.html'
                if not target.exists():
                    errors.append((p, link, 'no such file'))
                elif anchor and target.suffix == '.html' and unquote(anchor) not in pages[target].ids:
                    errors.append((p, link, 'no such anchor'))
    with ThreadPoolExecutor(16) as pool:
        codes = dict(zip(web, pool.map(status, web)))
    warnings = []
    for url, code in codes.items():
        if code in (0, 404, 410):
            errors.append((web[url], url, f'HTTP {code}' if code else 'no answer'))
        elif code >= 400:
            warnings.append((web[url], url, f'HTTP {code}'))
    return errors, warnings


if __name__ == '__main__':
    bad = bare_cells()
    for b in bad:
        print('cell with too few comments: %s, cell %d (%d code lines, %d comments)' % b)
    errors = []
    if (SITE / 'index.html').exists():
        errors, warnings = broken_links()
        for p, link, why in warnings:
            print(f'warning: {p.relative_to(SITE)}: {link} ({why})')
        for p, link, why in errors:
            print(f'broken link: {p.relative_to(SITE)}: {link} ({why})')
    else:
        print(f'{SITE} not built: links not checked')
    print(f'{len(bad)} cells with too few comments, {len(errors)} broken links')
    sys.exit(1 if bad or errors else 0)
