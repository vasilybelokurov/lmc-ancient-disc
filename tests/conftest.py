"""Shared fixtures: the literature LMC geometry and helpers that place stars on the sky."""
import numpy as np
import pytest

import lmcold  # noqa: F401  (enables JAX float64)
from lmcold import geometry as geo
from lmcold.constants import LMC


def radec_from_offset(ra0, dec0, rho, pa):
    """(ra, dec) in radians of points at angular distance rho and PA pa (radians) from (ra0, dec0)."""
    r0, eE0, eN0 = (np.asarray(v) for v in geo.sky_basis(ra0, dec0))
    rho = np.atleast_1d(rho)[:, None]
    pa = np.atleast_1d(pa)[:, None]
    u = np.sin(pa) * eE0 + np.cos(pa) * eN0
    rhat = np.cos(rho) * r0 + np.sin(rho) * u
    ra = np.mod(np.arctan2(rhat[:, 1], rhat[:, 0]), 2 * np.pi)
    dec = np.arcsin(np.clip(rhat[:, 2], -1, 1))
    return ra, dec


@pytest.fixture
def lit():
    """vdMK14 geometry in radians/kpc."""
    p = dict(LMC)
    for k in ("ra0", "dec0", "inc", "theta"):
        p[k] = np.deg2rad(p[k])
    return p


@pytest.fixture
def rng():
    return np.random.default_rng(20260929)
