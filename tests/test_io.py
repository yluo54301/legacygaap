import os
import numpy as np
import pytest
from legacygaap import io, resample as rs

MARGIN = io.EDGE_MARGIN
TEST_BRICKS = [('0332m045', 'south'), ('1501p022', 'south'), ('1500p032', 'south'), ('1700p352', 'north')]


# ---------------------------------------------------------------- synthetic (run anywhere)

def test_brick_dec_and_region_rule():
    assert io.brick_dec('1501p022') == 2.2 and io.brick_dec('0332m045') == -4.5 and io.brick_dec('1700p352') == 35.2


def test_psfex_path_rule():
    sd = '/survey'
    assert io.psfex_path(sd, 'decam/CP/V4.8.2a/CP20141223/c4d_141224_030948_ooi_g_ls9.fits.fz') == \
        '/survey/calib/psfex/decam/CP/V4.8.2a/CP20141223/c4d_141224_030948_ooi_g_ls9-psfex.fits'
    assert io.psfex_path(sd, 'mosaic/CP/V4.3/CP20171109/k4m_171110_072640_ooi_zd_ls9.fits') == \
        '/survey/calib/psfex/mosaic/CP/V4.3/CP20171109/k4m_171110_072640_ooi_zd_ls9-psfex.fits'
    assert io.psfex_path(sd, '90prime/CP/V2.3/CP20151112/ksb_151112_012825_ooi_g_ls9.fits.fz ') == \
        '/survey/calib/psfex/90prime/CP/V2.3/CP20151112/ksb_151112_012825_ooi_g_ls9-psfex.fits'


def _brick_like_wcs(ra=33.25, dec=-4.5, n=3600, scale=0.262):
    cd = scale / 3600.
    return rs.tan_wcs(ra, dec, n / 2 + 0.5, n / 2 + 0.5, -cd, 0., 0., cd, n, n)


def test_tan_wcs_roundtrip_and_scale():
    w = _brick_like_wcs()
    x = np.array([0., 100.5, 3599.])
    y = np.array([0., 2000.25, 3599.])
    ra, dec = w.wcs_pix2world(x, y, 0)
    x2, y2 = w.wcs_world2pix(ra, dec, 0)
    assert np.allclose(x, x2, atol=1e-6) and np.allclose(y, y2, atol=1e-6)
    assert abs(rs.pixel_scale(w) - 0.262) < 1e-9
    # CRPIX is 1-based: the 0-based pixel (1799.5, 1799.5) is the tangent point
    ra0, dec0 = w.wcs_pix2world(1799.5, 1799.5, 0)
    assert abs(ra0 - 33.25) < 1e-9 and abs(dec0 + 4.5) < 1e-9


def test_footprint_and_rasterize_rectangle():
    target = _brick_like_wcs(n=400)
    # a "CCD" on the same tangent plane and scale, offset by (100, 50) pixels, 80x60 pixels
    cd = 0.262 / 3600.
    ccd = rs.tan_wcs(33.25, -4.5, 200.5 - 100, 200.5 - 50, -cd, 0., 0., cd, 80, 60)
    poly = rs.footprint_pix(ccd, target)
    assert np.allclose(poly[:, 0].min(), 100 - 0.5, atol=1e-6) and np.allclose(poly[:, 0].max(), 180 - 0.5, atol=1e-6)
    assert np.allclose(poly[:, 1].min(), 50 - 0.5, atol=1e-6) and np.allclose(poly[:, 1].max(), 110 - 0.5, atol=1e-6)
    mask = rs.rasterize(poly, (400, 400), step=1)
    expected = np.zeros((400, 400), bool)
    expected[50:110, 100:180] = True
    assert mask.sum() == expected.sum() == 80 * 60
    assert np.array_equal(mask, expected)
    # subsampled grid is consistent with downsample()
    m4 = rs.rasterize(poly, (400, 400), step=4)
    assert np.array_equal(m4, rs.downsample(expected, 4))


def test_good_subregion_rules():
    row = np.zeros(1, dtype=io.CCD_DTYPE)[0]
    row['camera'], row['ccdname'], row['filter'], row['expnum'], row['height'] = 'decam', 'S7', 'g', 500000, 4094
    assert io.good_subregion(row) == (None, 1023, None, None)
    row['ccdname'], row['filter'], row['expnum'] = 'N4', 'r', 200000
    assert io.good_subregion(row) == (None, None, 100, None)
    row['ccdname'] = 'S4'
    assert io.good_subregion(row) == (None, None, None, 4094 - 100)
    row['camera'], row['ccdname'] = 'mosaic', 'S7'
    assert io.good_subregion(row) == (None, None, None, None)


def test_footprint_bounds_and_margin():
    target = _brick_like_wcs(n=400)
    cd = 0.262 / 3600.
    ccd = rs.tan_wcs(33.25, -4.5, 200.5 - 100, 200.5 - 50, -cd, 0., 0., cd, 80, 60)
    poly = rs.footprint_pix(ccd, target, bounds=(None, 40, 10, None), margin=2)
    assert np.allclose([poly[:, 0].min(), poly[:, 0].max()], [100 - 0.5 + 2, 100 + 40 - 0.5 - 2], atol=1e-6)
    assert np.allclose([poly[:, 1].min(), poly[:, 1].max()], [50 - 0.5 + 10 + 2, 110 - 0.5 - 2], atol=1e-6)


def _fake_ccds(n):
    T = np.zeros(n, dtype=io.CCD_DTYPE)
    T['camera'] = 'fake'
    return T


def test_select_ccds_by_nexp_synthetic():
    rng = np.random.default_rng(1)
    n = 400
    target = _brick_like_wcs(n=n)
    cd = 0.262 / 3600.

    def ccd(x0, y0, w, h):
        return rs.tan_wcs(33.25, -4.5, n / 2 + 0.5 - x0, n / 2 + 0.5 - y0, -cd, 0., 0., cd, w, h)

    used = [ccd(0, 0, 250, 300), ccd(150, 100, 250, 300), ccd(-100, 250, 300, 200)]
    unused = [ccd(50, 50, 200, 200), ccd(200, 250, 200, 200)]   # the second overlaps the first
    nexp = np.zeros((n, n), np.int16)
    for w in used:
        nexp += rs.rasterize(rs.footprint_pix(w, target), (n, n))
    nexp[rng.random((n, n)) < 0.03] = 0          # 3% "masked" pixels, like saturation/CR holes
    T = _fake_ccds(5)
    keep, count, nexp_ds, agreement, score = io.select_ccds_by_nexp(T, used + unused, target, nexp, step=2, margin=0.)
    assert keep.tolist() == [True, True, True, False, False]
    assert np.all(score[3:] > 0.9) and np.all(score[:3] < 0.1)
    assert agreement > 0.95


# ---------------------------------------------------------------- DR11 (CFS)

@pytest.mark.nersc
@pytest.mark.parametrize('brick,region', TEST_BRICKS)
def test_brick_products_resolve(brick, region):
    p = io.BrickPaths(brick)
    assert p.region == region and p.exists()
    bands = p.bands()
    assert 'g' in bands and 'r' in bands and 'z' in bands
    assert ('i' in bands) == (region == 'south')
    w = io.brick_wcs(p)
    assert abs(rs.pixel_scale(w) - 0.262) < 1e-3 and w.pixel_shape == (io.BRICK_SIZE, io.BRICK_SIZE)
    T = io.read_tractor(p, columns=['ra', 'dec', 'bx', 'by', 'type', 'flux_r', 'shape_r', 'shape_e1', 'shape_e2'])
    assert len(T) > 1000
    C = io.read_brick_ccds(p)
    assert len(C) > 10 and set(C['filter']) <= set(io.OPTICAL_BANDS)
    missing = [c['image_filename'] for c in C if not os.path.exists(io.psfex_path(p.survey_dir, c['image_filename']))]
    assert missing == []
    assert set(C['camera']) <= set(io.CAMERAS[region])


@pytest.mark.nersc
def test_iband_ccds_recovered_from_survey_table():
    p = io.BrickPaths('0332m045')
    bw = io.brick_wcs(p)
    T, info = io.ccds_for_brick(p, 'i', bwcs=bw, use_cache=False)
    assert info['source'] == 'survey-table'
    assert 35 <= info['n_candidates'] <= 70 and 35 <= info['n_selected'] <= info['n_candidates']
    # every dropped candidate was over-counted on (nearly) all of its footprint
    assert np.all(info['score'][~info['keep']] > 0.9)
    assert all(os.path.exists(io.psfex_path(p.survey_dir, c['image_filename'])) for c in T)
    # the cached list (written by the call above) is the same selection
    Tc, ic = io.ccds_for_brick(p, 'i', bwcs=bw)
    assert ic.get('cached') and ic['n_candidates'] == info['n_candidates']
    key = lambda X: sorted((str(c['image_filename']).strip(), str(c['ccdname']).strip()) for c in X)
    assert key(Tc) == key(T)


@pytest.mark.nersc
def test_survey_route_reproduces_brick_table():
    """On bands where the brick CCD table is the truth, the survey-table route must select the
    same CCDs, except for CCDs that contribute no unmasked pixel to the brick (score ~1)."""
    p = io.BrickPaths('0332m045')
    bw = io.brick_wcs(p)
    for band in ['g', 'r', 'z']:
        truth, info = io.ccds_for_brick(p, band, bwcs=bw)
        assert info['source'] == 'brick-table'
        C = io.candidate_ccds(p, band, bwcs=bw)
        nexp = io.read_coadd(p, 'nexp', band)
        keep, count, nexp_ds, agreement, score = io.select_ccds_by_nexp(C, io.ccd_wcses(C), bw, nexp, step=4)
        key = lambda t: set((int(e), c) for e, c in zip(t['expnum'], t['ccdname']))
        chosen = key(C[keep])
        assert not (chosen - key(truth)), (band, 'extra CCDs selected')
        missing = key(truth) - chosen
        for i in np.flatnonzero(~keep):
            if (int(C['expnum'][i]), C['ccdname'][i]) in missing:
                assert score[i] > 0.9, (band, C['ccdname'][i], score[i])
        # footprints never miss pixels the pipeline used, except in thin edge bands
        assert np.mean(nexp_ds > count) < 0.02, (band, np.mean(nexp_ds > count))
