import numpy as np
import pytest
from legacygaap import io, psf as P

pytestmark = pytest.mark.nersc


@pytest.mark.parametrize('brick,bands', [('0332m045', ['g', 'r', 'i', 'z']), ('1501p022', ['g']), ('1700p352', ['g', 'z'])])
def test_coadd_psf_against_pipeline_maps(brick, bands):
    p = io.BrickPaths(brick)
    model = P.CoaddPsfModel.for_brick(p, bands=bands)
    rng = np.random.default_rng(5)
    for band in bands:
        psz = io.read_coadd(p, 'psfsize', band)
        n_ok = 0
        for _ in range(20):
            x, y = rng.uniform(100, 3500, 2)
            ps = model.psf_at(x, y, band)
            assert ps.nccd > 0 and np.isfinite(ps.stamp).all() and abs(ps.stamp.sum() - 1.) < 1e-9
            iy, ix = np.unravel_index(np.argmax(ps.stamp), ps.stamp.shape)
            assert abs(ix - (x - ps.x0)) <= 1 and abs(iy - (y - ps.y0)) <= 1, 'peak not at the source position'
            # the pipeline psfsize at a pixel uses exactly the CCDs counted in nexp there (flag_nexp is
            # deliberately more tolerant, for deep stacks)
            if ps.nexp_at == ps.nccd:
                n_ok += 1
                ratio = ps.psfsize_pipeline / psz[int(round(y)), int(round(x))]
                assert abs(ratio - 1.) < 0.01, (band, x, y, ratio)    # nominal (legacypipe) pixel scales
                assert ps.neff_fwhm >= 0.95 * ps.psfsize_pipeline
        # positions inside our 10-px edge margin where the CP mask is narrower disagree with nexp (10-20%)
        assert n_ok >= 14, (band, n_ok)


def test_spline_mapping_matches_exact():
    """The spline brick->CCD map agrees with the exact TPV mapping to < 1e-4 px, and the PSF stamps
    of the fast (spline) and exact paths agree to 1e-6 of peak."""
    from legacygaap import resample as rs
    p = io.BrickPaths('0332m045')
    fast = P.CoaddPsfModel.for_brick(p, bands=['g'], load_nexp=False)
    exact = P.CoaddPsfModel(fast.bwcs, fast.ccds, fast.wcs, fast._loader, mapping='exact', sbscale_by_band=fast.sbscale,
                            nominal_scale_by_band=fast._ccd_scale)
    rng = np.random.default_rng(2)
    worst_map = 0.
    for i, (m, w) in enumerate(zip(fast.ccd_maps('g'), fast.wcs['g'])):
        if m.empty:
            continue
        xs = rng.uniform(max(m.bbox[0], -50), min(m.bbox[1], 3650), 200)
        ys = rng.uniform(max(m.bbox[2], -50), min(m.bbox[3], 3650), 200)
        fx, fy = rs.map_pixels(fast.bwcs, xs, ys, w)
        sx, sy = m.sx.ev(ys, xs), m.sy.ev(ys, xs)
        worst_map = max(worst_map, np.abs(sx - fx).max(), np.abs(sy - fy).max())
    assert worst_map < 1e-4, worst_map
    for _ in range(10):
        x, y = rng.uniform(100, 3500, 2)
        a, b = fast.psf_at(x, y, 'g'), exact.psf_at(x, y, 'g')
        assert a.ccd_idx.tolist() == b.ccd_idx.tolist()
        assert np.abs(a.stamp - b.stamp).max() < 1e-6 * b.stamp.max()
