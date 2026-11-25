"""Observable giants per unit formed mass for the two survey CMD selections.

PARSEC cumulative IMF differences are integrated between adjacent isochrone
points. Forty midpoint samples resolve colour/magnitude selection boundaries.
The normalization 0.9499 converts PARSEC's cumulative IMF convention to a
Kroupa IMF normalized over initial masses 0.08--100 solar masses.

Inputs: PARSEC v1.2S + COLIBRI grids (data/sources.yaml keys parsec_gaiaEDR3 for BOSS, parsec_2mass_spitzer for APOGEE; scripts/
fetch_isochrones.py), with columns MH, logAge, Mini, int_IMF, label and the magnitudes. The grid is placed at
LMC_DISTANCE_MODULUS plus a disc offset dmu, smeared by a Gaussian depth LMC_DEPTH_MAG (lmcold.constants), and reddened. The
result n_box (giants per Msun formed, Kroupa 0.08-100 Msun) is the efficiency eta by which scripts/build_giant_mass.py divides
selection-corrected giant counts to give formed mass; it is shown in figures/paper_mass_method.png (a) (appendix D).
"""
import numpy as np
import pandas as pd
from matplotlib.path import Path

from . import apogee_target, magcloud
from .constants import LMC_DISTANCE_MODULUS, LMC_DEPTH_MAG

PARSEC_IMF_NORMALIZATION = .9499  # checked against normalized Kroupa counts on the main sequence


def selected_count(cumulative_imf, colour, magnitude, inside, subdivisions=40):
    """Selected stars per PARSEC mass unit, with linear segment interpolation.

    Between adjacent isochrone points the stars (number = difference of int_IMF) are spread uniformly along the straight
    segment in colour and magnitude; the fraction of `subdivisions` midpoint samples that fall inside the selection is the
    selected fraction of that segment.

    Parameters
    ----------
    cumulative_imf : ndarray, shape (N,)
        PARSEC int_IMF along the isochrone, sorted by initial mass.
    colour, magnitude : array_like, shape (N,)
        Observable colour and magnitude [mag] of the points.
    inside : callable
        inside(colour, magnitude) -> bool array, the selection.
    subdivisions : int
        Samples per segment.

    Returns
    -------
    float
        Selected stars per unit mass in the PARSEC IMF normalisation (divide by PARSEC_IMF_NORMALIZATION for Kroupa 0.08-100).
    """
    frac = (np.arange(subdivisions) + .5) / subdivisions
    colour = np.asarray(colour)
    magnitude = np.asarray(magnitude)
    cc = colour[:-1, None] + np.diff(colour)[:, None] * frac  # shape (N - 1, subdivisions)
    mm = magnitude[:-1, None] + np.diff(magnitude)[:, None] * frac
    selected = inside(cc.ravel(), mm.ravel()).reshape(cc.shape).mean(axis=1)
    return float(np.dot(np.diff(cumulative_imf), selected))


def count_grid(grid, survey, ebv, ebv_carton, dmu, h_limit=13.3, maglim=14.5):
    """Grid of giants per Msun formed; APOGEE limits are observed magnitudes.

    ebv: true reddening (SFD-scale E(B-V), lmcold.extinction.true_ebv); ebv_carton: the reddening the BOSS carton removed
    (its rescaled SFD, lmcold.magcloud.dereddened); dmu: distance-modulus offset of the region from the inclined disc
    (lmcold.geometry.disc_dmu). BOSS applies the carton polygon to the photometry the carton dereddened and the observed
    G<=17.5; APOGEE applies its targeting polygon (H<=h_limit) to observed 2MASS photometry.

    Parameters
    ----------
    grid : pandas.DataFrame
        PARSEC grid (columns MH, logAge, Mini, int_IMF, label and G_BPmag, G_RPmag, Gmag for BOSS or Jmag, Hmag, Ksmag for
        APOGEE; absolute magnitudes).
    survey : {'BOSS', 'APOGEE'}
    ebv, ebv_carton : float
        E(B-V) [mag] as above (ebv_carton is not used for APOGEE).
    dmu : float
        Distance-modulus offset [mag] added to LMC_DISTANCE_MODULUS.
    h_limit : float
        APOGEE faint limit in observed H [mag] of the counted stars.
    maglim : float
        APOGEE FntRGB faint limit [mag] that sets the polygon's sides (lmcold.apogee_target.MAGLIM of the field).

    Returns
    -------
    pandas.DataFrame
        One row per ([M/H], age): mh [dex], logage, age [Gyr], n_box_raw [selected giants per Msun formed, Kroupa 0.08-100].
    """
    # BOSS carton polygon in dereddened (BP - RP, G) (lmcold.magcloud).
    polygon = Path(np.column_stack([magcloud.BPRP_CUT, magcloud.G_CUT]))
    # Line-of-sight depth: 7 magnitude offsets from -2.5 to +2.5 sigma, with normalised Gaussian weights.
    shifts = np.linspace(-2.5, 2.5, 7) * LMC_DEPTH_MAG
    probability = np.exp(-.5 * (shifts / LMC_DEPTH_MAG)**2)
    probability /= probability.sum()
    rows = []
    for (mh, logage), iso in grid.groupby(['MH','logAge'], sort=True):
        iso = iso.sort_values('Mini', kind='stable')  # points of equal Mini keep their order in the PARSEC file (evolutionary order), on every machine
        iso = iso[iso.label <= 8]  # PARSEC phase labels 0-8 (pre-main sequence to thermally pulsing AGB); later phases dropped
        if survey == 'BOSS':
            # A_V = 0.86 x 3.1 E(B-V) as in the carton; the carton removed av_carton but the true extinction is av.
            av, av_carton = .86 * 3.1 * ebv, .86 * 3.1 * ebv_carton
            k = magcloud.A_OVER_AV
            colour = (iso.G_BPmag - iso.G_RPmag).to_numpy() + (k['BP'] - k['RP']) * (av - av_carton)
            magnitude = iso.Gmag.to_numpy() + LMC_DISTANCE_MODULUS + dmu + k['G'] * (av - av_carton)  # as the carton dereddened it
            ag = k['G'] * av_carton  # m + ag is the observed G, for the G <= 17.5 cut
            def inside(c, m):
                """True where (BP - RP, G) [mag] lies in the carton polygon and the observed G is <= 17.5."""
                return polygon.contains_points(np.column_stack([c,m])) & (m + ag <= 17.5)
        elif survey == 'APOGEE':
            # Extinction coefficients of the 2MASS conversion: A_J, A_H, A_Ks = 0.709, 0.449, 0.302 E(B-V)
            # (Schlafly & Finkbeiner 2011); observed J - Ks and H.
            colour = (iso.Jmag - iso.Ksmag).to_numpy() + (.709 - .302) * ebv
            magnitude = iso.Hmag.to_numpy() + LMC_DISTANCE_MODULUS + dmu + .449 * ebv
            def inside(c, m):
                """True where (J - Ks, H) [mag] lies in the APOGEE-2 LMC giant selection (lmcold.apogee_target.sides)."""
                blue, red = apogee_target.sides(m, maglim)
                return (m >= 12.35) & (m <= h_limit) & (c >= blue) & (c <= red)  # 12.35: bright edge of BrtRGB
        else:
            raise ValueError('survey must be BOSS or APOGEE')
        # Depth-averaged count, converted from the PARSEC to the Kroupa 0.08-100 Msun normalisation.
        n = sum(p * selected_count(iso.int_IMF.to_numpy(), colour, magnitude+d, inside)
                for p,d in zip(probability,shifts)) / PARSEC_IMF_NORMALIZATION
        rows.append(dict(mh=round(mh,2), logage=logage, age=10**logage/1e9, n_box_raw=n))
    return pd.DataFrame(rows)


def smooth_age_spikes(table):
    """Treatment of isolated PARSEC age-grid spikes; retain raw values.

    Replace a value exceeding twice BOTH neighbours with their mean. Repeat
    until no isolated spike remains; endpoints are never altered.

    Each pass replaces only the point with the largest ratio to its larger neighbour, and stops when that ratio is <= 2 or the
    point is below 5e-5 giants per Msun (the floor).

    Parameters
    ----------
    table : pandas.DataFrame
        count_grid output (columns mh, age, n_box_raw).

    Returns
    -------
    pandas.DataFrame
        A copy with n_box (adjusted) and spike_adjusted (bool) added; n_box_raw kept.
    """
    result=table.copy()
    result['n_box']=result.n_box_raw
    for _,part in result.groupby('mh',sort=False):
        order=part.sort_values('age').index
        values=result.loc[order,'n_box'].to_numpy().copy()
        while len(values)>2:
            neighbour=np.maximum(values[:-2],values[2:])
            ratio=values[1:-1]/np.maximum(neighbour,1e-30)
            k=int(np.argmax(ratio))+1
            if ratio[k-1]<=2 or values[k]<5e-5:
                break
            values[k]=(values[k-1]+values[k+1])/2
        result.loc[order,'n_box']=values
    result['spike_adjusted']=result.n_box!=result.n_box_raw
    return result
