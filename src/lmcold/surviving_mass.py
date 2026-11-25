"""Present mass per initial Kroupa (0.08--100 Msun) population mass.

Living stars: current PARSEC Mass integrated against the same int_IMF used
for giant efficiencies. Complete the missing 0.08--0.09 Msun interval with
unevolved stars. Remnants: Conroy, Gunn & White 2009, ApJ 699, 486, section 2
(doi:10.1088/0004-637X/699/1/486), citing Renzini & Ciotti 1993.
All remnants retained; no dynamical loss. Gas returned to the ISM is excluded.
An alternative WD/NS/BH prescription is saved for comparison, not averaged in.

The present fraction (living + remnant mass per formed mass) multiplies the formed masses of lmcold.mass_conversion to give
present-day stellar mass (scripts/build_present_mass.py; figures/paper_mass.png and paper_mass_summary.png). Input: the same
PARSEC grid as lmcold.isochrones (columns MH, logAge, Mini, Mass, int_IMF, label). Masses in Msun.
"""
import numpy as np
import pandas as pd
from .isochrones import PARSEC_IMF_NORMALIZATION
from .mass_conversion import age_weights


def power_integral(lo, hi, exponent):
    """Integral of m^exponent dm from lo to hi (exponent != -1)."""
    return (hi**(exponent+1)-lo**(exponent+1))/(exponent+1)


# dN/dm proportional to m^-1.3 below 0.5, 0.5*m^-2.3 above it.
# IMF_MASS: total initial mass over 0.08-100 Msun of that IMF, the integral of m dN/dm (exponents +1); dividing by it gives
# fractions per unit formed mass.
IMF_MASS = power_integral(.08,.5,-.3)+.5*power_integral(.5,100,-1.3)


def remnant_fraction(last_initial_mass, legacy=False):
    """Integrate piecewise linear initial-final relations analytically.

    Stars born above last_initial_mass (the most massive star still on the isochrone) are remnants of mass
    slope x M_ini + intercept. Default (Conroy et al. 2009, section 2): white dwarfs 0.077 M_ini + 0.48 below 8.5 Msun,
    neutron stars 1.4 Msun for 8.5-40 Msun, black holes 0.5 M_ini above 40 Msun. legacy=True: the alternative
    (0.109 M_ini + 0.394 below 8 Msun, 1.4 Msun for 8-25, 0.5 M_ini above 25).

    Parameters
    ----------
    last_initial_mass : float
        Largest initial mass still alive [Msun]; must be >= 0.5 for any remnants to form (asserted).
    legacy : bool
        Use the alternative prescription.

    Returns
    -------
    float
        Remnant mass per unit formed mass (Kroupa 0.08-100 Msun).
    """
    # (lo, hi, slope, intercept) per remnant type, initial masses in Msun; the 0.5 factor is the IMF above 0.5 Msun.
    segments=([(0,8,.109,.394),(8,25,0,1.4),(25,100,.5,0)] if legacy else
              [(0,8.5,.077,.48),(8.5,40,0,1.4),(40,100,.5,0)])
    total=0.
    for lo,hi,slope,intercept in segments:
        lo=max(lo,last_initial_mass)
        if lo<hi:
            assert lo>=.5
            total+=.5*(slope*power_integral(lo,hi,-1.3)+intercept*power_integral(lo,hi,-2.3))
    return total/IMF_MASS


def survival_grid(grid):
    """Living, remnant and present mass per formed mass for each ([M/H], age) of a PARSEC grid.

    Parameters
    ----------
    grid : pandas.DataFrame
        PARSEC grid: MH, logAge, Mini, Mass (current mass) [Msun], int_IMF, label.

    Returns
    -------
    pandas.DataFrame
        One row per isochrone: mh, logage, age [Gyr], living_fraction, remnant_fraction, present_fraction (their sum),
        low_mass_fraction (0.08 Msun to the first tabulated mass), present_fraction_legacy_remnants (alternative remnants),
        living_tabulated_fraction (tabulated points only); all per unit formed mass.
    """
    rows=[]
    for (mh,logage),iso in grid.groupby(['MH','logAge'],sort=True):
        iso=iso.sort_values('Mini', kind='stable')  # equal Mini keep their PARSEC file order, as lmcold.isochrones.count_grid
        iso=iso[iso.label<=8]  # same phase cut as lmcold.isochrones.count_grid
        initial=iso.Mini.to_numpy();current=iso.Mass.to_numpy()
        assert .08<=initial[0]<.5
        dn=np.diff(iso.int_IMF.to_numpy())/PARSEC_IMF_NORMALIZATION  # stars per Msun formed between adjacent points
        # Trapezoidal integration in cumulative number
        # midpoint-cell weights, with the missing low-mass interval added.
        tabulated=float(np.dot(dn,.5*(current[:-1]+current[1:])))
        low=power_integral(.08,initial[0],-.3)/IMF_MASS  # unevolved stars below the grid's first mass: current = initial
        live=tabulated+low
        rem=remnant_fraction(initial[-1]);legacy=remnant_fraction(initial[-1],legacy=True)
        rows.append(dict(mh=round(mh,2),logage=logage,age=10**logage/1e9,
            living_fraction=live,remnant_fraction=rem,present_fraction=live+rem,
            low_mass_fraction=low,present_fraction_legacy_remnants=live+legacy,
            living_tabulated_fraction=tabulated))
    t=pd.DataFrame(rows)
    assert t.present_fraction.between(0,1).all()
    assert (t.living_fraction<t.present_fraction).all()
    return t


def average_survival(grid, conversions):
    """Use exactly the metallicity and age weighting (lmcold.mass_conversion.age_weights) of each conversion.

    Parameters
    ----------
    grid : pandas.DataFrame
        survival_grid output.
    conversions : pandas.DataFrame
        Rows of lmcold.mass_conversion.factors output with a prior column: mh_grid, feh, prior are used.

    Returns
    -------
    pandas.DataFrame
        conversions with the age-weighted living_fraction, remnant_fraction, present_fraction and
        present_fraction_legacy_remnants appended.
    """
    rows=[]
    columns=['living_fraction','remnant_fraction','present_fraction','present_fraction_legacy_remnants']
    for r in conversions.itertuples():
        q=grid[grid.mh==r.mh_grid].sort_values('age')
        weights=age_weights(q.age.to_numpy(),r.feh,r.prior)
        assert weights.sum()>0
        rows.append({c:np.average(q[c],weights=weights) for c in columns})
    return pd.concat([conversions.reset_index(drop=True),pd.DataFrame(rows)],axis=1)
