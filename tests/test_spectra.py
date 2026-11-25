import pandas as pd

from lmcold import spectra


def test_gratton_one_row_per_star_with_the_average():
    """Gratton et al. (2004) table 7 has one row per spectrum; the star average [Fe/H]a is on the first row of each star only."""
    rows = pd.DataFrame({"Star": ["A1-07247", "A1-07247", "A1-06332"], "Vmag": [19.4, 19.4, 19.43],
                         "[Fe/H]": [-1.38, -1.42, -0.91], "[Fe/H]a": [-1.41, float("nan"), -0.91]})
    out = spectra.gratton_per_star(rows)
    assert sorted(out.Star) == ["A1-06332", "A1-07247"]
    assert out.set_index("Star").loc["A1-07247", "feh"] == -1.41
