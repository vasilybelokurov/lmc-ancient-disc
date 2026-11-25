import numpy as np
import pandas as pd
from scipy.integrate import quad
from lmcold.surviving_mass import IMF_MASS,remnant_fraction,average_survival


def test_remnant_integral_against_independent_quadrature():
    norm=quad(lambda m:m*m**-1.3,.08,.5)[0]+quad(lambda m:.5*m*m**-2.3,.5,100)[0]
    np.testing.assert_allclose(IMF_MASS,norm)
    expected=(quad(lambda m:.5*m**-2.3*(.077*m+.48),.9,8.5)[0]
              +quad(lambda m:.5*m**-2.3*1.4,8.5,40)[0]
              +quad(lambda m:.5*m**-2.3*.5*m,40,100)[0])/norm
    np.testing.assert_allclose(remnant_fraction(.9),expected,rtol=1e-10)
    assert remnant_fraction(100)==0


def test_survival_uses_formed_mass_age_weights():
    g=pd.DataFrame(dict(mh=[-1.]*3,age=[1.,2.,4.],living_fraction=[.6,.5,.4],
        remnant_fraction=[.1]*3,present_fraction=[.7,.6,.5],present_fraction_legacy_remnants=[.7,.6,.5]))
    c=pd.DataFrame(dict(mh_grid=[-1.],feh=[-.25],prior=['C']))  # prior C at [Fe/H] > -0.5: 0.5-3 Gyr, so ages 1 and 2
    a=average_survival(g,c)
    np.testing.assert_allclose(a.present_fraction,[(.7+.6)/2])
    np.testing.assert_allclose(a.present_fraction,a.living_fraction+a.remnant_fraction)


def test_living_mass_and_low_mass_completion():
    from lmcold.surviving_mass import survival_grid
    from lmcold.isochrones import PARSEC_IMF_NORMALIZATION
    # An independently specified number interval; Mass differs from Mini,
    # so confusing initial with current mass fails this check.
    iso=pd.DataFrame(dict(MH=[0.,0.],logAge=[10.,10.],Mini=[.09,1.],
        Mass=[.09,.7],int_IMF=[1.,1.+PARSEC_IMF_NORMALIZATION],label=[1,8]))
    result=survival_grid(iso).iloc[0]
    low=quad(lambda m:m*m**-1.3,.08,.09)[0]/IMF_MASS
    np.testing.assert_allclose(result.living_fraction,(.09+.7)/2+low)
    np.testing.assert_allclose(result.present_fraction,result.living_fraction+result.remnant_fraction)
