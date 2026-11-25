"""Numerical regression of the independent age averaging against frozen reference counts."""
from pathlib import Path
import numpy as np
import pandas as pd
from lmcold.mass_conversion import factors


def test_age_average_reproduces_e054():
    # Frozen reference model counts, not its inferred masses. Expected prior-A
    # average is independently recorded in results/mass_mock_conversion.csv.
    g=pd.read_csv(Path(__file__).parent/'data/age_average_reference.csv')
    trend=pd.DataFrame(dict(feh=[-2.,0.],alpha=[.15,.15]))
    result=factors(g,trend,alpha='.15',centres=[-1.625])
    np.testing.assert_allclose(result.eta,[.00017644551799660376],rtol=1e-11)
    assert result.mh_grid.iloc[0] == -1.5
    assert result.grid_age_hi.iloc[0] < result.age_hi.iloc[0]


def test_uniform_age_prior_weights_linear_age_not_log_age():
    g=pd.DataFrame(dict(mh=[-1.5]*3,age=[9.,10.,12.],n_box=[1.,2.,4.]))
    trend=pd.DataFrame(dict(feh=[-2.,0.],alpha=[0.,0.]))
    result=factors(g,trend,centres=[-1.625])
    np.testing.assert_allclose(result.eta,[(1+1.5*2+2*4)/4.5])


def test_amr_weights_follow_the_age_metallicity_relation():
    """With the Povick+24 relation, a metal-rich bin is weighted to young ages and a metal-poor bin to old ones."""
    from lmcold.mass_conversion import age_weights
    age = np.linspace(.5, 12.5, 25)
    young, old = age_weights(age, -.375, 'AMR'), age_weights(age, -1.625, 'AMR')
    assert np.average(age, weights=young) < 5 < 9 < np.average(age, weights=old)


def test_window_weights_unchanged():
    """The window priors keep their weighting: age steps of the ages inside the window, zero outside."""
    from lmcold.mass_conversion import age_weights
    np.testing.assert_allclose(age_weights([5., 9., 10., 12., 13., 14.], -1.625, 'A'), [0, 1, 1.5, 1.5, 1, 0])
