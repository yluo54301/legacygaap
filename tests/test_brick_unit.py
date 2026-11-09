"""Synthetic tests of the brick driver (run anywhere)."""
import numpy as np
import fitsio
from legacygaap import brick


def test_aperture_names_and_cutout_sizes():
    """Names keep a second decimal only when needed; per-source cutouts span 4 sigma of the
    largest aperture and never shrink below the configured size."""
    from legacygaap import brick, aperture as ap
    assert brick.Settings().aperture_names() == ['circ1.0', 'circ1.5', 'circ2.0', 'ell1.0', 'ell1.5']
    st = brick.Settings(circ=(0.8, 1.25, 3.0), min_apers=(1.0, 3.0), max_aper=4.0)
    assert st.aperture_names() == ['circ0.8', 'circ1.25', 'circ3.0', 'ell1.0', 'ell3.0']
    M = np.eye(2) / 0.168
    small = [M @ ap.circular(s) @ M.T for s in (0.8, 1.0)]
    big = [M @ ap.circular(s) @ M.T for s in (0.8, 4.0)]
    n = brick.cutout_sizes(np.array([small, big]), 97)
    assert n[0] == 97                                        # 4 x 1.0" / 0.168" = 24 px < 48
    assert n[1] == 2 * int(np.ceil(4 * 4.0 / 0.168)) + 1     # 191
    assert np.all(n % 2 == 1)


def test_read_blob_map_both_layouts(tmp_path):
    """DR10 writes the blob map in the primary HDU, DR11 and later in extension BLOB-MAP."""
    blobs = np.full((8, 8), -1, np.int32)
    blobs[2:4, 3:6] = 7
    old, new = str(tmp_path / 'blobs-dr10.fits'), str(tmp_path / 'blobs-dr11.fits')
    fitsio.write(old, blobs)
    with fitsio.FITS(new, 'rw') as F:
        F.write(None)
        F.write(blobs, extname='BLOB-MAP')
    assert np.array_equal(brick.read_blob_map(old), blobs)
    assert np.array_equal(brick.read_blob_map(new), blobs)
