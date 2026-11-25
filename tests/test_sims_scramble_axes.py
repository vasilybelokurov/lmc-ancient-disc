"""scripts/sims_scramble.halo_axes returns axes with a fixed sign convention: the eigensolver may return either sign of each
eigenvector (platform-dependent), and the H2 variant rotates the dark matter with these axes."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('agama')   # scripts/sims_scramble.py imports AGAMA, which the GitHub test machine does not install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from sims_scramble import halo_axes  # noqa: E402


def test_axes_do_not_depend_on_eigenvector_signs(monkeypatch):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5000, 3)) * [3., 2., 1.] @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
    ref = halo_axes(x)
    eigh = np.linalg.eigh
    for flip in ([1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, -1, -1]):
        monkeypatch.setattr(np.linalg, 'eigh', lambda a, f=np.array(flip): (eigh(a)[0], eigh(a)[1] * f))
        np.testing.assert_allclose(halo_axes(x), ref, atol=1e-12)
    monkeypatch.setattr(np.linalg, 'eigh', eigh)
    assert np.isclose(np.linalg.det(ref), 1)
