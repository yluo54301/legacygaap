"""IBIS DR1 and HSC (ODIN hsc-co-4) support: flat run layout, band discovery, forced-flux columns,
WCS override (IBIS DR1 CCDs with V6.0LS headers), oversampled PsfEx models and SIP WCSs."""
import os
import numpy as np
import fitsio
import pytest

from legacygaap import io, psfex, resample as rs

IBIS = os.environ.get('GAAP_TEST_IBIS', '/global/cfs/cdirs/cosmo/work/legacysurvey/ibis')
IBIS_RUN = f'{IBIS}/reductions/ibis-dr1'
IBIS_V2 = f'{IBIS}/survey-ccds-ibis-v2.kd.fits'
IBIS_IMAGES = f'{IBIS}/images'
HSC = os.environ.get('GAAP_TEST_HSC', '/global/cfs/cdirs/cosmo/work/users/dstn/ODIN')
HSC_RUN = f'{HSC}/2024-a/hsc-co-4'
HSC_IMAGES = f'{HSC}/images'
# HSC-forced IBIS catalogue (same rows as the DR1 tractor catalogue), a pattern with {bbb} and {brick}
FORCED = os.environ.get('GAAP_TEST_FORCED')
BRICK = '1501p022'


# ----------------------------------------------------------------------------- unit

def _gauss(n, sigma, cx, cy):
    y, x = np.mgrid[:n, :n]
    return np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / sigma ** 2)


@pytest.mark.parametrize('dx,dy', [(0., 0.), (0.3, -0.2), (-0.45, 0.41)])
def test_oversampled_native_sampling(dx, dy):
    """An oversampled (sampling 0.5) Gaussian PsfEx basis sampled at the image pixel centres equals
    the native Gaussian at the shifted centre (what tractor's PixelizedPSF._sampleImage does)."""
    sig = 1.8                                    # native pixels
    W = 81
    basis = _gauss(W, sig / 0.5, W // 2, W // 2)[None]
    m = psfex.PsfExModel(basis, 0, (0., 0.), (1., 1.), sampling=0.5)
    img = m.native(500., 700., dx, dy)
    n = int(np.ceil(W * 0.5))
    assert img.shape == (n, n)
    ref = _gauss(n, sig, n // 2 + dx, n // 2 + dy)
    ref /= ref.sum()
    assert abs(img.sum() - 1) < 1e-12
    # exact at whole-pixel shifts (samples fall on model pixels); elsewhere Lanczos-3 interpolation
    # error, 0.1-0.4% of the peak for these widths (as in tractor, which samples the same way)
    tol = 1e-6 if (dx == 0 and dy == 0) else 5e-3
    assert np.max(np.abs(img - ref)) < tol * ref.max()
    y, x = np.mgrid[:n, :n]
    assert abs((img * x).sum() - (n // 2 + dx)) < 1e-2 and abs((img * y).sum() - (n // 2 + dy)) < 1e-2


def test_native_sampling_one_is_lanczos_shift():
    W = 63
    basis = _gauss(W, 2.2, W // 2, W // 2)[None]
    m = psfex.PsfExModel(basis, 0, (0., 0.), (1., 1.), sampling=1.)
    a = m.native(10., 20., 0.3, -0.1)
    b = rs.lanczos_shift(m.at(10., 20.), 0.3, -0.1)
    assert np.allclose(a, b / b.sum(), atol=1e-14)


def _touch(fn):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    open(fn, 'w').close()


def test_flat_layout_and_bands(tmp_path):
    d = str(tmp_path)
    cd = os.path.join(d, 'coadd', '150', BRICK)
    for b in ['M464', 'M411', 'M517', 'W1', 'g', 'r2']:
        _touch(os.path.join(cd, f'legacysurvey-{BRICK}-image-{b}.fits.fz'))
    assert io.is_flat(d)
    p = io.BrickPaths(BRICK, survey_dir=d)
    assert p.region == ''
    assert p.coadd_dir == cd
    assert p.bands() == ['M411', 'M464', 'g', 'M517', 'r2']      # by wavelength, WISE excluded
    assert p.tractor == os.path.join(d, 'tractor', '150', f'tractor-{BRICK}.fits')
    q = io.BrickPaths(BRICK, survey_dir=d, catalog='/x/{bbb}/forced-{brick}.fits')
    assert q.tractor == f'/x/150/forced-{BRICK}.fits'


def test_flux_column():
    names = ['flux_M411', 'forced_flux_g', 'forced_flux_ivar_g', 'flux_ivar_M411']
    assert io.flux_column(names, 'M411') == 'flux_M411'
    assert io.flux_column(names, 'g') == 'forced_flux_g'
    assert io.flux_column(names, 'z') is None
    assert io.flux_column(['flux_m464'], 'M464') == 'flux_m464'


def test_wcs_override_lookup(tmp_path):
    fn = str(tmp_path / 'ccds.fits')
    T = np.zeros(2, dtype=[('expnum', 'i8'), ('ccdname', 'S4'), ('image_filename', 'S40'), ('image_hdu', 'i2'),
                           ('ccdraoff', 'f4'), ('ccddecoff', 'f4')])
    T['expnum'] = [101, 102]
    T['ccdname'] = [b'N4', b'S7']
    T['image_filename'] = [b'a/x_v2.fits.fz', b'a/y_v2.fits.fz']
    T['image_hdu'] = [5, 7]
    T['ccdraoff'] = [0.01, -0.02]
    T['ccddecoff'] = [0.03, 0.04]
    fitsio.write(fn, T, clobber=True)
    try:
        assert io.set_wcs_override(fn, '/root', 'tst', expnums=[102]) == 1
        ccd = dict(expnum=102, ccdname='S7', camera='decam', image_filename='old.fits.fz', image_hdu=1,
                   ccdraoff=9., ccddecoff=9.)
        src = io.wcs_source(ccd)
        assert src[:2] == ('a/y_v2.fits.fz', 7) and src[4:] == ('/root', 'decam-tst')
        assert np.allclose(src[2:4], (-0.02, 0.04))
        with pytest.raises(KeyError):
            io.wcs_source(dict(ccd, expnum=101, ccdname='N4'))
    finally:
        io.clear_wcs_override()
    assert io.wcs_source(ccd)[0] == 'old.fits.fz'


# ----------------------------------------------------------------------------- data (CFS)

def _coverage_check(paths, band, npos=300, seed=3):
    """Fraction of random positions where the PSF model's CCD count passes the nexp test
    (PsfStamp.flag_nexp False): checks WCS, tim boxes, edge margins and mask maps together."""
    from legacygaap import psf as P
    m = P.CoaddPsfModel.for_brick(paths, bands=[band])
    W, H = m.bwcs.pixel_shape
    rng = np.random.default_rng(seed)
    xs, ys = rng.uniform(20, W - 20, npos), rng.uniform(20, H - 20, npos)
    lo, hi = m._nexp_range(band)
    ok = 0
    for x, y in zip(xs, ys):
        n = len(m.covering_fast(band, x, y)[0])
        cx, cy = int(round(x)), int(round(y))
        off = max(int(lo[cy, cx]) - n, n - int(hi[cy, cx]), 0)
        ok += off <= m.NEXP_TOL * max(n, 1)
    return ok / npos


@pytest.mark.nersc
def test_ibis_v2_wcs_reproduces_dr1_coverage():
    """IBIS DR1 CCDs with V6.0LS headers and V6.0LS offsets cover the brick as the DR1 nexp map says."""
    p = io.BrickPaths(BRICK, survey_dir=IBIS_RUN)
    assert p.region == '' and p.bands() == ['M411', 'M438', 'M464', 'M490', 'M517']
    try:
        n = io.set_wcs_override(IBIS_V2, IBIS_IMAGES, 'ibisv2', expnums=np.unique(io.read_brick_ccds(p)['expnum']))
        assert n >= len(io.read_brick_ccds(p))
        assert _coverage_check(p, 'M464') > 0.95
    finally:
        io.clear_wcs_override()


@pytest.mark.nersc
def test_hsc_sip_wcs_and_psf():
    old = io.IMAGES_ROOT
    io.set_images_root(HSC_IMAGES)
    try:
        p = io.BrickPaths(BRICK, survey_dir=HSC_RUN)
        assert p.region == '' and p.bands() == ['g', 'r2', 'i2', 'z', 'y']
        assert _coverage_check(p, 'i2') > 0.9
        T = io.read_brick_ccds(p)
        c = T[0]
        keys = io.ccd_wcs_keys(c)
        assert keys['CTYPE1'].endswith('-SIP') and 'A_ORDER' in keys
        m = psfex.hsc_psfex(os.path.join(HSC_IMAGES, c['image_filename']))
        assert m.sampling == 0.5 and m.bases.shape[1:] == (81, 81)
        img = m.native(1000., 2000.)
        assert img.shape == (41, 41) and abs(img.sum() - 1) < 1e-12
        assert np.unravel_index(np.argmax(img), img.shape) == (20, 20)
        # half-maximum width of the oversampled model vs the CCD table FWHM (pixels); the
        # noise-equivalent FWHM is ~1.4x larger (extended HSC wings)
        os_img = m.at(1000., 2000.)
        cx = os_img.shape[1] // 2
        prof = os_img[cx, cx:]
        k = np.argmax(prof < prof[0] / 2)
        hwhm = np.interp(prof[0] / 2, prof[k::-1], np.arange(k + 1)[::-1]) * m.sampling
        assert 0.85 < 2 * hwhm / float(c['fwhm']) < 1.25, (2 * hwhm, c['fwhm'])
    finally:
        io.set_images_root(old)


@pytest.mark.nersc
def test_forced_catalog_rows_match_tractor():
    if not FORCED or not os.path.exists(FORCED.format(bbb=BRICK[:3], brick=BRICK)):
        pytest.skip('set GAAP_TEST_FORCED to the forced-catalogue pattern')
    p = io.BrickPaths(BRICK, survey_dir=IBIS_RUN)
    q = io.BrickPaths(BRICK, survey_dir=IBIS_RUN, catalog=FORCED)
    a = io.read_tractor(p, columns=['objid'])
    b = io.read_tractor(q, columns=['objid', 'forced_flux_g'])
    assert len(a) == len(b) and np.all(a['objid'] == b['objid'])


@pytest.mark.nersc
def test_pixscale_convention_detection():
    """IBIS DR1 (legacypipe ibis-1.8) uses the CD-matrix pixel scale for sbscale and psfsize; DR11
    (DR11.0.6) folds in the linear TPV terms. Detected from the psfsize maps."""
    from legacygaap import psf as P
    p = io.BrickPaths(BRICK, survey_dir=IBIS_RUN)
    try:
        io.set_wcs_override(IBIS_V2, IBIS_IMAGES, 'ibisv2', expnums=np.unique(io.read_brick_ccds(p)['expnum']))
        m = P.CoaddPsfModel.for_brick(p, bands=['M464'])
        conv, r = m.detect_pixscale_convention(npos=25)
        assert conv == 'cd' and abs(r['cd'] - 1) < 1e-3 and r['nominal'] > 1.005, r
        assert 0.995 < np.median(m.sbscale['M464']) < 1.0   # (0.262 / 0.2623)^2
    finally:
        io.clear_wcs_override()
    q = io.BrickPaths(BRICK, survey_dir=io.DEFAULT_SURVEY_DIR)
    m = P.CoaddPsfModel.for_brick(q, bands=['r'])
    conv, r = m.detect_pixscale_convention(npos=25)
    assert conv == 'nominal' and abs(r['nominal'] - 1) < 1e-3, r
