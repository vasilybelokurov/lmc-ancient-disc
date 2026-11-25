"""Figure layout for the paper: a grid of equal square panels, with room above the top row for legends or colour bars.

The repository rule (CLAUDE.md) is that all panels of a figure have the same size; the figure scripts (scripts/paper_*.py,
scripts/fig_*.py) build their axes with panel_grid.
"""
import matplotlib.pyplot as plt
import numpy as np


def panel_grid(nrows, ncols, width=7.0, left=0.5, right=0.08, hgap=0.55, vgap=0.15, bottom=0.42, top=0.5, aspect=1.0):
    """Figure of `width` inches with nrows x ncols equal axes of height/width `aspect` (square by default); margins and gaps in
    inches. Returns (fig, axes[nrows, ncols]).

    Parameters
    ----------
    nrows, ncols : int
        Panel grid.
    width : float
        Figure width [inch]; the height follows from the panel size.
    left, right, bottom, top : float
        Margins [inch]; `top` leaves room for legends or colour bars above the first row.
    hgap, vgap : float
        Horizontal and vertical gaps between panels [inch].
    aspect : float
        Panel height / width.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of Axes, shape (nrows, ncols)
        axes[0, 0] is top left.
    """
    # Panel width [inch] from the figure width minus margins and gaps; axes positions are then in figure fractions.
    size = (width - left - right - (ncols - 1) * hgap) / ncols
    fig = plt.figure(figsize=(width, bottom + nrows * size * aspect + (nrows - 1) * vgap + top), layout="none")
    width, height = fig.get_size_inches()  # matplotlib rounds the size to whole pixels
    axes = np.array([[fig.add_axes([(left + j * (size + hgap)) / width, (bottom + (nrows - 1 - i) * (size * aspect + vgap)) / height,
                                    size / width, size * aspect / height]) for j in range(ncols)] for i in range(nrows)])
    return fig, axes
