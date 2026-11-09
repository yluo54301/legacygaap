"""Comparison tests against legacypipe/tractor; run inside the legacypipe Shifter image (docs/nersc.md):
    python -m pytest tests -m legacypipe
"""
import numpy as np
import pytest
from legacygaap import io, psfex, resample as rs, psf as P

pytestmark = [pytest.mark.legacypipe, pytest.mark.nersc]
compat = pytest.importorskip('legacygaap.compat_legacypipe')
pytest.importorskip('legacypipe')


@pytest.fixture(scope='module')
def s1():
    p = io.BrickPaths('0332m045')
    return p, compat.get_survey(p.survey_dir), compat.brick_ccds_fits_table(p)


def test_psfex_evaluation_matches_tractor(s1):
    p, survey, T = s1
    rng = np.random.default_rng(0)
    C = io.read_brick_ccds(p)
    for i in [0, 7, 40]:
        ccd = T[compat.match_rows(T, C[i:i + 1])[0]]
        im, psf = compat.ref_psf_object(survey, ccd)
        ours = psfex.psfex_for_ccd(io.psfex_path(p.survey_dir, C['image_filename'][i]), C['expnum'][i], C['ccdname'][i])
        for _ in range(50):
            x, y = rng.uniform(0, ccd.width - 1), rng.uniform(0, ccd.height - 1)
            ref = psf.getImage(x, y)
            assert np.abs(ours.at(x, y) - ref).max() < 1e-6


def test_lanczos_shift_matches_tractor():
    rng = np.random.default_rng(1)
    img = rng.random((63, 63))
    img /= img.sum()
    for dx, dy in [(0.3, -0.2), (-0.49, 0.49), (0.0, 0.1), (0.25, 0.25)]:
        assert np.abs(rs.lanczos_shift(img, dx, dy) - compat.ref_lanczos_shift(img, dx, dy)).max() < 1e-6


@pytest.mark.parametrize('brick,bands', [('0332m045', ['r', 'g']), ('1700p352', ['g', 'z'])])
def test_coadd_stamp_matches_reference(brick, bands):
    p = io.BrickPaths(brick)
    survey = compat.get_survey(p.survey_dir)
    T = compat.brick_ccds_fits_table(p)
    model = P.CoaddPsfModel.for_brick(p, bands=bands, load_nexp=False)
    rng = np.random.default_rng(3)
    worst = 0.
    for band in bands:
        for _ in range(5):
            x, y = rng.uniform(200, 3400, 2)
            ps = model.psf_at(x, y, band)
            assert ps.nccd > 0
            rows = T[compat.match_rows(T, model.ccds[band][ps.ccd_idx])]
            ref, n, fs = compat.ref_coadd_psf(survey, brick, rows, x, y, stamp=model.stamp, return_flux_scale=True)
            assert n == ps.nccd
            assert abs(ps.flux_scale / fs - 1) < 2e-4, (brick, band, ps.flux_scale, fs)
            diff = np.abs(ps.stamp - ref).max()
            worst = max(worst, diff)
            # astrometry.net's C resampler uses a Lanczos lookup table and float32 offsets: ~2e-4 of peak
            assert diff < 5e-4 * ps.stamp.max(), (brick, band, x, y, diff, ps.stamp.max())
    print(f'{brick} {bands}: worst |ours - reference| = {worst:.2e} (peak ~{ps.stamp.max():.3f})')
