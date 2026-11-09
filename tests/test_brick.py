import os
import numpy as np
import fitsio
import pytest
from legacygaap import brick, io, measure as ms

pytestmark = pytest.mark.nersc


def test_run_brick_end_to_end(tmp_path):
    """30 rows of S1 in g and z: schema, finite calibrated fluxes for valid apertures, the
    point-source response near the coadd flux scale, and a bitwise-identical rerun."""
    st = brick.Settings()
    out = []
    for k in range(2):
        fn = brick.run_brick('0332m045', bands=['g', 'z'], settings=st, limit=30, outdir=str(tmp_path / f'run{k}'),
                             log=lambda *a: None)
        out.append(fitsio.read(fn))
    a, b = out
    assert len(a) == 30 and a.dtype == b.dtype
    for name in a.dtype.names:
        x, y = a[name], b[name]
        assert np.array_equal(x, y) or (x.dtype.kind == 'f' and np.array_equal(np.isnan(x), np.isnan(y)) and np.array_equal(x[~np.isnan(x)], y[~np.isnan(y)])), name
    h = fitsio.read_header(fn, ext=1)
    assert h['NAP'] == st.nap and h['FLUXDEF'] == 'raw/(psfcorr*wcorr)'
    for band in 'gz':
        flags = a[f'gaap_flags_{band}']
        valid = (flags & (ms.FLAG_BIGPSF | ms.FLAG_NOPSF)) == 0
        assert np.all(np.isfinite(a[f'gaapflux_{band}'][valid])) and np.all(a[f'gaapflux_ivar_{band}'][valid] > 0)
        fs = a[f'gaap_fluxscale_{band}']
        assert np.all((fs > 0.93) & (fs < 1.03))
        pc = a[f'gaap_psfcorr_{band}'][valid] / np.repeat(fs[:, None], st.nap, axis=1)[valid]
        assert np.all(np.abs(pc - 1) < 0.01)
        wc = a[f'gaap_wcorr_{band}'][valid]
        assert np.all((wc > 0.8) & (wc <= 1.001))
