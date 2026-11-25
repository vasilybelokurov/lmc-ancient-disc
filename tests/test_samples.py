import numpy as np
import pandas as pd
import pytest

from lmcold import samples, sources


def _table(ids, feh, survey_snr):
    """Minimal release-catalogue table: Gaia columns, vlos, evlos, feh, teff, snr."""
    n = len(ids)
    return pd.DataFrame(dict(source_id=ids, ra=np.full(n, 80.), dec=np.full(n, -69.), pmra=np.zeros(n), pmdec=np.zeros(n),
                             pmra_error=np.full(n, .05), pmdec_error=np.full(n, .05), pmra_pmdec_corr=np.zeros(n),
                             vlos=np.full(n, 260.), evlos=np.ones(n), feh=feh, teff=np.full(n, 4200.), snr=np.full(n, survey_snr)))


def test_merged_giants_keep_apogee_for_stars_in_both_surveys():
    """A star in both release catalogues appears once, with its APOGEE values and survey 'apogee'; the others keep their own."""
    a = _table([1, 2], [-0.6, -1.2], 100.)
    b = _table([2, 3], [-0.4, -1.6], 30.)
    m = samples.merge_giants(a, b)
    assert sorted(m.source_id) == [1, 2, 3]
    assert m.set_index("source_id").loc[2, "survey"] == "apogee"
    assert m.set_index("source_id").loc[2, "feh"] == -1.2
    assert m.set_index("source_id").loc[3, "survey"] == "boss"


@pytest.mark.skipif(not (sources.DATA / "giants_apogee.parquet").exists(), reason="release catalogues absent (Zenodo bundle 1)")
def test_giants_read_the_release_catalogues():
    """samples.giants() is built from the release catalogues: the bins hold every merged star once, no BOSS spectrum below
    S/N 16 (lmcold.giants.BOSS_SNR_MIN), and as many APOGEE stars as the APOGEE release catalogue."""
    from lmcold.giants import BOSS_SNR_MIN
    from lmcold.sources import path
    a, b = pd.read_parquet(path("release_giants_apogee")), pd.read_parquet(path("release_giants_boss"))
    bins = samples.giants()
    n = sum(len(c["ra"]) for _, c in bins)
    survey = np.concatenate([c["survey"] for _, c in bins])
    snr = np.concatenate([c["snr"] for _, c in bins])
    assert n == len(samples.merge_giants(a, b))
    assert (survey == "apogee").sum() == len(a)
    assert snr[survey == "boss"].min() >= BOSS_SNR_MIN


