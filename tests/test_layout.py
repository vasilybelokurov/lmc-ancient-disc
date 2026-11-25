import numpy as np

from lmcold.layout import panel_grid


def test_panels_equal_and_square():
    fig, ax = panel_grid(2, 3)
    w, h = fig.get_size_inches()
    sizes = [(a.get_position().width * w, a.get_position().height * h) for a in ax.ravel()]
    np.testing.assert_allclose(sizes, [sizes[0]] * 6)
    np.testing.assert_allclose(sizes[0][0], sizes[0][1])
    assert ax[0, 0].get_position().y0 > ax[1, 0].get_position().y0  # row 0 is the top row
