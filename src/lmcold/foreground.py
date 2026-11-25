"""Milky Way foreground fraction of the LMC giants as a function of radius.

The foreground has a uniform surface density over the 12-deg field (supported by the proper-motion sideband of the metal-poor
BOSS giants, scripts/build_giant_mass.py); its mean fraction over the reference radii is f_ref (the reference fit covered in-plane
radii 2-10 kpc, about 2-10 deg). Then f(rho) = f_ref x Sigma(reference) / Sigma(rho), with Sigma the weighted surface density of
the sample in 1-deg rings.

Input: f_ref from the reference fit (data/sources.yaml key mass_foreground_reference). Used by scripts/build_giant_mass.py
for the mass and by scripts/fig_giant_foreground.py (appendix figure fig_giant_foreground, §2.3).
"""
import numpy as np

RINGS = np.arange(0.0, 12.01, 1.0)  # deg, ring edges 0, 1, ..., 12 in angular distance from the LMC centre


def radial_fraction(rho, w, f_ref, reference=(2.0, 10.0)):
    """Foreground fraction per star (rho [deg], weights w, reference mean fraction f_ref per star); at most 1.

    Parameters
    ----------
    rho : ndarray
        Angular distance of each star from the LMC centre [deg].
    w : ndarray
        Selection weight of each star (the surface densities are weighted sums).
    f_ref : float or ndarray
        Foreground fraction over the reference radii (per star, so it may depend on e.g. [Fe/H]).
    reference : (float, float)
        Reference range of rho [deg], lower edge included.

    Returns
    -------
    ndarray
        f_ref x Sigma_ref / Sigma(ring of the star), capped at 1. Stars beyond the last ring take the outermost ring.
    """
    k = np.clip(np.digitize(rho, RINGS) - 1, 0, len(RINGS) - 2)
    # Solid angle of each ring [deg^2]: 2 pi (cos rho_in - cos rho_out) sr x (180/pi)^2.
    area = 2 * np.pi * -np.diff(np.cos(np.radians(RINGS))) * (180 / np.pi) ** 2
    sigma = np.bincount(k, w, len(RINGS) - 1) / area
    ref = (rho >= reference[0]) & (rho < reference[1])
    sigma_ref = np.sum(w[ref]) / (2 * np.pi * (np.cos(np.radians(reference[0])) - np.cos(np.radians(reference[1]))) * (180 / np.pi) ** 2)
    return np.minimum(f_ref * sigma_ref / sigma[k], 1.0)
