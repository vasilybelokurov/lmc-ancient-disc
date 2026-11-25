from pathlib import Path

import pytest

from lmcold import sources


def test_resolve_home_and_relative():
    assert sources.resolve("~/data/x.fits") == Path("~/data/x.fits").expanduser()
    assert sources.resolve("raw/ogle4/ident.dat") == sources.DATA / "raw/ogle4/ident.dat"


@pytest.mark.skipif(not (sources.DATA / "boss_parent.parquet").exists(), reason="release catalogues absent (Zenodo bundle 1)")
def test_boss_catalogues_are_sorted_by_source_id():
    """The BOSS parent comes from a TAP query whose row order the server does not fix; the build sorts by source_id, and the catalogues
    derived from it keep that order, so a rebuild reproduces the recorded sha256."""
    import numpy as np
    import pandas as pd
    for name in ("boss_parent", "boss_observed", "boss_cartons"):
        sid = pd.read_parquet(sources.DATA / f"{name}.parquet", columns=["source_id"]).source_id.to_numpy()
        assert np.all(np.diff(sid) > 0), name
