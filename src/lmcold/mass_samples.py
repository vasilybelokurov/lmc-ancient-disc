"""Selection-weighted giant samples used for number and mass comparisons.

Use the quality cuts of the selection functions, without adding the stricter
kinematic release cuts. Both surveys use 150 < calibrated vlos < 420 km/s for
this velocity-selected candidate sample. Foreground subtraction is separate.

Method in plain words. The formed stellar mass per [Fe/H] bin (Sect. 2.3 of the paper; scripts/build_giant_mass.py) is built from
the giants observed by BOSS (SDSS-V DR20) and APOGEE (DR17), each star weighted by the inverse of its selection probability
(column w of the tier-B weight catalogues, lmcold.selection). This module supplies the pieces that are specific to the mass
samples:
- samples(): the two weighted samples and the Gaia IDs of the parent stars both surveys cover;
- overlap() and boss_response(): the BOSS metallicity response on the APOGEE scale, measured on the stars with both spectra and
  reweighted to the magnitude and S/N distribution of the BOSS target sample (lmcold.mass_response.empirical_response);
- with_colour() and metal_rich_split(): the split of the BOSS mass above [Fe/H] = -1 among the reporting bins, taken from APOGEE
  giants of similar radius, G and dereddened colour.
Assumptions: the selection weights remove the targeting and spectrum-failure selection (lmcold.selection); the stars with both
spectra, once reweighted, represent the BOSS response; APOGEE [Fe/H] is the reference ("true") scale.
Units: [Fe/H] in dex on the APOGEE scale; G and colours in mag; velocities in km/s; angles in deg.
Inputs: data/boss_weights.parquet, data/apogee_weights.parquet, data/spec_giants.parquet, data/boss_parent.parquet (sources.yaml
entries release_boss_weights, release_apogee_weights, release_spec_giants, release_boss_parent) and results/boss_calibration.json
(the BOSS-to-APOGEE calibration of lmcold.giants.fit_calibration). Tested in tests/test_mass_samples.py.
"""
import json
import numpy as np
import pandas as pd

from . import giants, selection
from .mass_response import empirical_response
from .sources import DATA, ROOT

# Reporting [Fe/H] bins [dex] of the masses (open ends below -2 and above 0) and the values at which each bin is plotted.
FEH_EDGES = np.array([-np.inf, -2., -1.75, -1.5, -1.25, -1., -.75, -.5, -.25, 0., np.inf])
FEH_CENTRES = np.array([-2.25, -1.875, -1.625, -1.375, -1.125, -.875, -.625, -.375, -.125, .25])
# Bins of the metallicity-response fit (lmcold.mass_response): true bins below -2, -2 to -1.5, -1.5 to -1 and above -1 (finite
# ends), and 0.125-dex measured bins with open ends. The response is measured on 2,001 stars: with 0.25-dex true bins the fit
# oscillated between neighbours (fitted/direct 0.57, 1.10, 0.47, ...; below -2 only 0.55 of APOGEE in LMC18-35), and splitting
# the metal-rich side at -0.5 (50 overlap stars above it) still did (1.28, 0.58, 1.78). These bins give 0.90, 0.90, 1.04 of
# APOGEE in LMC18-35 and smooth ratios for the full sample.
TRUE_EDGES = [-3., -2., -1.5, -1., .5]
MEASURED_EDGES = np.array([-np.inf, *np.arange(-2.75, .4, .125), np.inf])


def samples():
    """Return (BOSS members, APOGEE members, common covered-parent Gaia IDs).

    Returns
    -------
    b : pandas.DataFrame
        BOSS giants with a good spectrum (good, giant) and positive selection weight w, joined one-to-one with their highest-S/N
        BOSS spectrum (v, teff, snr), with vlos [km/s] and feh [dex] calibrated to the APOGEE scale (lmcold.giants.calibrate)
        and 150 < vlos < 420 km/s. Columns of data/boss_weights.parquet plus v, teff, snr, vlos.
    a : pandas.DataFrame
        APOGEE giants with quality, giant and covered flags, w > 0 and 150 < v < 420 km/s; columns of
        data/apogee_weights.parquet, source_id as int64.
    common : ndarray of int64
        Gaia DR3 source_id of parent stars in a covered selection group of both surveys (BOSS: the group holds at least
        lmcold.selection.NMIN_GOOD good spectra; APOGEE: covered).
    """
    bw = pd.read_parquet(DATA / 'boss_weights.parquet')
    aw = pd.read_parquet(DATA / 'apogee_weights.parquet')
    assert pd.api.types.is_integer_dtype(aw.source_id.dtype), 'Gaia IDs must not be floating point'
    # BOSS coverage per star: its selection group has at least NMIN_GOOD good spectra (lmcold.selection).
    bcov = bw.group.map(bw.groupby('group').good.sum()) >= selection.NMIN_GOOD
    common = np.intersect1d(bw.loc[bcov,'source_id'], aw.loc[aw.covered,'source_id'].dropna().astype('int64'))
    spec = pd.read_parquet(DATA / 'spec_giants.parquet')
    # One BOSS spectrum per star: the one with the highest S/N.
    spec = spec[spec.survey == 'boss'].sort_values('snr', ascending=False, kind='stable').drop_duplicates('source_id')
    b = bw[bw.good & bw.giant & (bw.w > 0)].merge(spec[['source_id','v','teff','snr']], on='source_id', validate='one_to_one')
    cal = json.loads((ROOT / 'results/boss_calibration.json').read_text())['calibration']
    b['vlos'], b['feh'] = giants.calibrate(cal, b.teff, b.snr, b.v, b.feh)
    b = b[(b.vlos > 150) & (b.vlos < 420)].copy()  # velocity window [km/s] of the LMC candidates (module docstring)
    a = aw[aw.quality & aw.giant & aw.covered & (aw.w > 0) & (aw.v > 150) & (aw.v < 420)].copy()
    a['source_id'] = a.source_id.astype('int64')
    assert b.source_id.is_unique and a.source_id.is_unique
    return b, a, common


# G [mag] edges of the magnitude cells used to reweight the overlap stars in boss_response (np.digitize also gives open end cells).
RESPONSE_G_EDGES = [12., 16.25, 16.75, 17.5]
# The scatter of BOSS [Fe/H] is set by S/N more than by G; the overlap has median S/N 30 at G > 16.5, the BOSS sample 18. Two
# S/N bins: the 78 overlap stars below [Fe/H] = -1.5 leave 6-34 per G x S/N cell, none at S/N 16-20 in the faintest G bin.
RESPONSE_SNR_EDGES = [30.]  # BOSS S/N edge of the two S/N cells


def overlap(b, a):
    """Stars in both mass samples: feh_b (calibrated BOSS), feh_a (APOGEE), g, snr (BOSS), ra, dec.

    Parameters
    ----------
    b, a : pandas.DataFrame
        The BOSS and APOGEE samples of samples().

    Returns
    -------
    pandas.DataFrame
        One row per star in both (inner join on source_id): source_id, feh_b, g [mag], snr, ra, dec [deg], feh_a [dex].
    """
    return b[['source_id', 'feh', 'g', 'snr', 'ra', 'dec']].merge(a[['source_id', 'feh']], on='source_id', suffixes=('_b', '_a'))


def boss_response(o, feh, g, snr, w):
    """BOSS metallicity response on the APOGEE scale (lmcold.mass_response.empirical_response, TRUE_EDGES x MEASURED_EDGES)
    for a target sample with measured [Fe/H] feh, G, BOSS S/N and weights w. The overlap stars o are weighted to the joint
    G x S/N distribution (RESPONSE_G_EDGES x RESPONSE_SNR_EDGES) of the target, separately for [Fe/H] < -1.5 (true for o,
    measured for the target) and above: the overlap is brighter and has higher S/N than the BOSS sample.

    Parameters
    ----------
    o : pandas.DataFrame
        Overlap stars from overlap() (feh_a, feh_b, g, snr).
    feh, g, snr, w : array_like, shape (N,)
        Measured (calibrated BOSS) [Fe/H] [dex], Gaia G [mag], BOSS S/N and weights of the N target stars.

    Returns
    -------
    ndarray, shape (len(MEASURED_EDGES) - 1, len(TRUE_EDGES) - 1)
        Response matrix R (rows: measured bins, columns: true bins), as returned by lmcold.mass_response.empirical_response.
    """
    feh, w = np.asarray(feh), np.asarray(w)
    # Cell index of the G x S/N grid for each star (np.digitize gives len(edges) + 1 cells per axis).
    cell = lambda g_, s_: (np.digitize(g_, RESPONSE_G_EDGES) * (len(RESPONSE_SNR_EDGES) + 1)  # noqa: E731
                           + np.digitize(s_, RESPONSE_SNR_EDGES))
    n = (len(RESPONSE_G_EDGES) + 1) * (len(RESPONSE_SNR_EDGES) + 1)
    out = np.ones(len(o))  # weight of each overlap star
    # Metal-poor and metal-rich halves separately: overlap stars by APOGEE [Fe/H], target stars by measured [Fe/H].
    for po, pt in ((o.feh_a.to_numpy() < -1.5, feh < -1.5), (o.feh_a.to_numpy() >= -1.5, feh >= -1.5)):
        ko, kt = cell(o.g.to_numpy()[po], o.snr.to_numpy()[po]), cell(np.asarray(g)[pt], np.asarray(snr)[pt])
        target = np.bincount(kt, w[pt], n) / w[pt].sum()
        have = np.bincount(ko, minlength=n) / po.sum()
        out[po] = (target / np.where(have > 0, have, 1))[ko]  # target cell fraction / overlap cell fraction
    return empirical_response(o.feh_a.to_numpy(), o.feh_b.to_numpy(), out, TRUE_EDGES, MEASURED_EDGES)


# Split of the BOSS mass above [Fe/H] = -1 among the reporting bins above -1 (Sect. 2.3): BOSSNet over-estimates [Fe/H] of hot,
# low-S/N giants (scripts/paper_boss_hot_bias.py), which the stars with both spectra barely sample, so the split comes from the
# APOGEE giants above -1 of the same cell of ring x G x dereddened BP-RP; BOSS gives the total above -1, which agrees with APOGEE.
SPLIT_RING_EDGES = [3., 6.]  # deg
SPLIT_G_EDGES = [16.25, 16.75, 17.1]  # G [mag] bin edges of the split cells (no further source in the code)
SPLIT_C_EDGES = [1.2, 1.4, 1.6]  # dereddened BP-RP c0 [mag] bin edges of the split cells (no further source in the code)
NMIN_SPLIT = 30  # APOGEE giants above -1 a cell needs to set the split (otherwise the next coarser cell is used)


def with_colour(t):
    """t (columns source_id, ra, dec) with Gaia G (g), dereddened BP-RP (c0; true reddening, lmcold.extinction.true_ebv; A_V =
    0.86 x 3.1 E(B-V)) and angular distance rho [deg], from the BOSS parent photometry; stars without it are dropped.

    Parameters
    ----------
    t : pandas.DataFrame
        Stars with source_id, ra, dec [deg] (any other columns are kept; a column g is replaced).

    Returns
    -------
    pandas.DataFrame
        t joined with data/boss_parent.parquet (g, bp, rp [mag], ebv [SFD E(B-V), mag]) plus c0 [mag] and rho [deg] from the
        adopted LMC centre (lmcold.constants.LMC).
    """
    from .constants import LMC
    from .extinction import rc_map, true_ebv
    from .footprint import angular_distance
    from .magcloud import A_OVER_AV as k
    from .sources import DATA
    phot = pd.read_parquet(DATA / 'boss_parent.parquet')[['source_id', 'g', 'bp', 'rp', 'ebv']].drop_duplicates('source_id')
    t = t.drop(columns=[c for c in ('g',) if c in t]).merge(phot, on='source_id', how='inner', validate='one_to_one')
    ra, dec = t.ra.to_numpy(), t.dec.to_numpy()
    # A_V [mag] from the true E(B-V) (red-clump map, lmcold.extinction); A_X / A_V of the BOSS carton (lmcold.magcloud).
    av = .86 * 3.1 * true_ebv(ra, dec, t.ebv.to_numpy(), rc_map())
    return t.assign(c0=t.bp - t.rp - (k['BP'] - k['RP']) * av, rho=angular_distance(ra, dec, LMC['ra0'], LMC['dec0']))


def metal_rich_split(b, a):
    """Fractions of each BOSS giant with [Fe/H] > -1 in the reporting bins above -1 (FEH_EDGES): the weighted [Fe/H] distribution of
    the APOGEE giants above -1 (weights w) in the same cell of ring x G x c0, else of ring x G, ring, or all, whichever first holds
    NMIN_SPLIT APOGEE giants. b and a need feh, rho, g, c0. Returns (len(b), 5); rows of stars at or below -1 are zero.

    Parameters
    ----------
    b : pandas.DataFrame, N rows
        BOSS giants with feh [dex], rho [deg], g, c0 [mag] (with_colour()).
    a : pandas.DataFrame
        APOGEE giants with the same columns and the selection weight w.

    Returns
    -------
    ndarray, shape (N, 5)
        For each BOSS star above -1, the weighted fractions of APOGEE giants in the five reporting bins above -1
        (FEH_EDGES[5:]); each such row sums to 1.
    """
    edges = FEH_EDGES[5:]  # reporting-bin edges from -1 up
    ar = a[a.feh > -1]
    j = np.digitize(ar.feh, edges) - 1

    # Cell keys at four levels, finest first: ring x G x c0, ring x G, ring, everything.
    def keys(t):
        r, g, c = (np.digitize(t[x], e) for x, e in (('rho', SPLIT_RING_EDGES), ('g', SPLIT_G_EDGES), ('c0', SPLIT_C_EDGES)))
        return [(r * 10 + g) * 10 + c, r * 10 + g, r, np.zeros(len(t), int)]
    ka, kb = keys(ar), keys(b)
    out = np.zeros((len(b), len(edges) - 1))
    todo = (b.feh > -1).to_numpy()  # BOSS stars still without a split
    for level in range(4):
        for key in np.unique(kb[level][todo]):
            m = ka[level] == key
            if m.sum() >= NMIN_SPLIT:
                h = np.bincount(j[m], ar.w.to_numpy()[m], len(edges) - 1)
                sel = todo & (kb[level] == key)
                out[sel], todo = h / h.sum(), todo & ~sel
    assert not todo.any()
    return out
