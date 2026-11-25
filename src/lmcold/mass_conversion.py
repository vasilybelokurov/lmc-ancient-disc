"""Age/abundance averaging, returning formed (not surviving) mass factors.

Age weighting of the grid ages for a reporting [Fe/H] bin (age_weights): 'AMR', the reference, is constant star formation times the
probability that a star of each age falls in the bin under the LMC age-metallicity relation of Povick et al. (2024;
lmcold.constants.LMC_AMR_POVICK24, Gaussian scatter); 'A', 'B', 'C' are uniform formed mass in fixed age windows per [Fe/H] range
(set by hand; with the AMR, A gives masses within -2 to +7 per cent), kept as alternatives.

factors() turns the grid of giants per formed mass (lmcold.isochrones, n_box per [M/H] and age) into one efficiency eta per
reporting [Fe/H] bin (lmcold.mass_samples.FEH_EDGES / FEH_CENTRES): [Fe/H] -> [M/H] with the alpha enhancement, nearest grid
[M/H], then a weighted mean over age. Used by scripts/build_giant_mass.py, build_present_mass.py, paper_mass.py and
paper_mass_method.py, and by lmcold.surviving_mass.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from .constants import LMC_AMR_POVICK24
from .mass_samples import FEH_CENTRES, FEH_EDGES

# Age windows [Gyr] (lo, hi) per prior, for [Fe/H] < -1.5, -1.5 to -1, -1 to -0.5 and >= -0.5 (in that order).
AGE_RANGES = {
    'A': [(9,13.5),(6,12),(2,10),(.7,4)],
    'B': [(11,13.5),(8,13),(4,11),(1,5)],
    'C': [(7,12),(4,10),(1.5,7),(.5,3)],
}
REFERENCE_PRIOR = 'AMR'  # age weighting of the reference mass


def age_weights(age, feh, prior):
    """Formed-mass weights of the grid ages `age` [Gyr, sorted] for the reporting bin (FEH_EDGES) that contains feh.

    Parameters
    ----------
    age : array_like
        Grid ages [Gyr], increasing.
    feh : float
        [Fe/H] that selects the reporting bin (AMR) or the age window (A, B, C).
    prior : {'AMR', 'A', 'B', 'C'}

    Returns
    -------
    ndarray
        Unnormalised weights per grid age: the age-step width (np.gradient) times, for 'AMR', the Gaussian probability that a
        star of that age has [Fe/H] in the bin; for A, B, C the step width inside the window and 0 outside.
    """
    age = np.asarray(age, float)
    if prior == 'AMR':
        i = np.digitize(feh, FEH_EDGES) - 1
        # Mean and scatter of [Fe/H] at each age (linear interpolation; held constant beyond the tabulated ages).
        mu, sig = (np.interp(age, LMC_AMR_POVICK24['age'], LMC_AMR_POVICK24[k]) for k in ('feh', 'sigma'))
        dt = np.gradient(age) if len(age) > 1 else np.ones(1)
        return dt * (norm.cdf(FEH_EDGES[i + 1], mu, sig) - norm.cdf(FEH_EDGES[i], mu, sig))
    lo, hi = AGE_RANGES[prior][np.digitize(feh, [-1.5, -1., -.5])]
    inside = (age >= lo) & (age <= hi)
    w = np.zeros(len(age))
    w[inside] = np.gradient(age[inside]) if inside.sum() > 1 else 1.
    return w


def factors(grid, trend, prior='A', alpha='measured', centres=FEH_CENTRES):
    """Average efficiency over the age weighting of `prior` (age_weights), for the windows A, B, C.

    Nearest metallicity; grid ages weighted by their age-step widths (and, for 'AMR', by the probability of the bin); age_lo,
    age_hi are the window or, for 'AMR', the 16th-84th percentiles of the weights. Report grid limits explicitly: no extrapolation in metallicity/age.
    ASPCAP ALPHA_M is the measured alpha proxy used in the conversion.

    Parameters
    ----------
    grid : pandas.DataFrame
        Spike-adjusted efficiencies (lmcold.isochrones.smooth_age_spikes): columns mh, age, n_box.
    trend : pandas.DataFrame
        Measured alpha trend: columns feh, alpha (median APOGEE ALPHA_M per [Fe/H] bin; scripts/build_giant_mass.py).
    prior : {'AMR', 'A', 'B', 'C'}
        Age weighting (age_weights).
    alpha : 'measured' or float
        'measured' interpolates trend; a number is [alpha/Fe] below [Fe/H] = -1, capped at 0.1 above.
    centres : array_like
        [Fe/H] of the reporting bins.

    Returns
    -------
    pandas.DataFrame
        One row per centre: feh, alpha, mh (model [M/H]), mh_grid (nearest grid [M/H]), clipped_mh (mh outside the grid),
        age_lo, age_hi [Gyr], grid_age_lo, grid_age_hi (extreme grid ages with weight > 0) [Gyr], eta [giants per Msun formed].
    """
    rows=[]
    metallicities=np.sort(grid.mh.unique())
    for feh in centres:
        al=(np.interp(feh,trend.feh,trend.alpha) if alpha=='measured'
            else float(alpha) if feh < -1 else min(float(alpha),.1))
        mh=feh+np.log10(.638*10**al+.362)  # alpha-enhanced [M/H] (Salaris et al. 1993; docs/outline.md)
        chosen=metallicities[np.argmin(abs(metallicities-mh))]
        q=grid[grid.mh==chosen].sort_values('age')
        w=age_weights(q.age.to_numpy(),feh,prior)
        assert w.sum()>0
        eta=np.average(q.n_box,weights=w)
        assert eta>0 and np.isfinite(eta)
        lo,hi=np.interp([.16,.84],np.cumsum(w)/w.sum(),q.age) if prior=='AMR' else AGE_RANGES[prior][np.digitize(feh,[-1.5,-1.,-.5])]
        rows.append(dict(feh=feh,alpha=al,mh=mh,mh_grid=chosen,
                         clipped_mh=bool(mh<metallicities[0] or mh>metallicities[-1]),
                         age_lo=lo,age_hi=hi,grid_age_lo=q.age[w>0].min(),grid_age_hi=q.age[w>0].max(),eta=eta))
    return pd.DataFrame(rows)
