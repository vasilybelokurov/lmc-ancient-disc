"""results/macros_letter.tex: ratios quoted to one decimal keep it when it is zero (1.0, not 1), as in the text."""
import re

from lmcold.sources import ROOT


def test_one_decimal_ratios_keep_their_decimal():
    m = dict(re.findall(r'\\newcommand\{\\lt(\w+)\}\{([^}]*)\}', (ROOT / 'results/macros_letter.tex').read_text()))
    for k in ('TngRatioMP', 'TngRatioMPLo', 'TngRatioMPHi', 'TngRatioMPMax', 'AuRatioMP', 'FireRatioMP', 'AuRatioMRMax', 'FireRatioMRMax'):
        assert re.fullmatch(r'\d+\.\d', m[k]), (k, m[k])
