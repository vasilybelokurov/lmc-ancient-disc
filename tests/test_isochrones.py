import numpy as np
from lmcold.isochrones import selected_count


def test_segment_selection_integrates_across_boundary():
    # Exactly one quarter of a segment's two stars lies in the selected box.
    n=selected_count(np.array([0.,2.]),[0.,1.],[0.,1.],lambda c,m: (c>=.25)&(m<.5))
    assert n == .5


def test_full_isochrone_count_is_independent_of_sampling():
    imf=np.array([0.,.1,.1,.4,1.])
    for subdivisions in [1,7,40]:
        n=selected_count(imf,np.arange(5.),np.arange(5.),lambda c,m: np.ones(len(c),bool),subdivisions)
        np.testing.assert_allclose(n,1.)


def test_boss_conversion_keeps_file_order_of_equal_masses():
    # Public PARSEC fixture: [M/H]=-2, logAge=9.65; 2 of its points share an initial mass. They must keep their order in the PARSEC
    # file (evolutionary order), the same on every machine; the default sort ordered them differently on macOS (2.523e-4) and Linux.
    from pathlib import Path
    import pandas as pd
    from lmcold.isochrones import count_grid
    grid=pd.read_csv(Path(__file__).parent/'data/parsec_m2_age965.csv')
    e=.03791307285428047
    result=count_grid(grid,'BOSS',ebv=e,ebv_carton=e,dmu=0.)
    np.testing.assert_allclose(result.n_box_raw.iloc[0],.00023299035718091695,rtol=1e-10)


def test_boss_conversion_uses_true_extinction_and_disc_distance():
    # Neither effect has a fixed sign: the polygon's bright edge is a diagonal in (BP-RP, G), so dimming or reddening a blue
    # metal-poor RGB moves stars into it while the G <= 17.5 limit removes others. For this isochrone both raise the count.
    from pathlib import Path
    import pandas as pd
    from lmcold.isochrones import count_grid
    grid=pd.read_csv(Path(__file__).parent/'data/parsec_m2_age965.csv')
    n=lambda **k: count_grid(grid,'BOSS',**k).n_box_raw.iloc[0]
    ref=n(ebv=.04,ebv_carton=.04,dmu=0.)
    assert n(ebv=.08,ebv_carton=.04,dmu=0.)>1.1*ref
    assert n(ebv=.04,ebv_carton=.04,dmu=.2)>1.05*ref and abs(n(ebv=.04,ebv_carton=.04,dmu=-.2)/ref-1)>.05


def test_age_spike_treatment_preserves_raw_values_and_endpoints():
    import pandas as pd
    from lmcold.isochrones import smooth_age_spikes
    t=pd.DataFrame({'mh':[-1.]*5,'age':[1.,2.,3.,4.,5.],
                    'n_box_raw':[.001,.0001,.001,.0001,.001]})
    result=smooth_age_spikes(t)
    np.testing.assert_allclose(result.n_box,[.001,.0001,.0001,.0001,.001])
    np.testing.assert_allclose(result.n_box_raw,t.n_box_raw)

