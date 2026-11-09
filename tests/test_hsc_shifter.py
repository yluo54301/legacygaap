"""HSC images (Dustin's ODIN hsc-co-4 run): our SIP WCS and oversampled PsfEx sampling against
legacypipe's HscImage (astrometry.net Sip + tractor PixelizedPsfEx). Run via bin/sh.sh."""
import os
import numpy as np
import pytest

from legacygaap import io, psfex

pytestmark = [pytest.mark.legacypipe, pytest.mark.nersc]

HSC = os.environ.get('GAAP_TEST_HSC', '/global/cfs/cdirs/cosmo/work/users/dstn/ODIN')
HSC_RUN = f'{HSC}/2024-a/hsc-co-4'
HSC_IMAGES = f'{HSC}/images'
BRICK = '1501p022'


@pytest.fixture(scope='module')
def hsc():
    from legacypipe.survey import LegacySurveyData
    from astrometry.util.fits import fits_table
    survey = LegacySurveyData(survey_dir=HSC_RUN)
    p = io.BrickPaths(BRICK, survey_dir=HSC_RUN)
    old = io.IMAGES_ROOT
    io.set_images_root(HSC_IMAGES)
    yield survey, fits_table(p.ccds), io.read_brick_ccds(p)
    io.set_images_root(old)


def test_hsc_sip_wcs_matches_legacypipe(hsc):
    survey, T, C = hsc
    X, Y = np.meshgrid(np.linspace(0, 2047, 7), np.linspace(0, 4175, 9))
    for i in [0, 7, 40]:
        im = survey.get_image_object(T[i])
        lw = im.get_wcs()                         # astrometry.net Sip + ccdraoff/ccddecoff
        w = io.ccd_wcs_exact(C[i])
        r1, d1 = w.all_pix2world(X.ravel(), Y.ravel(), 0)
        r2, d2 = lw.pixelxy2radec(X.ravel() + 1., Y.ravel() + 1.)
        sep = 3600. * np.hypot((r1 - r2) * np.cos(np.deg2rad(d1)), d1 - d2) / 0.168
        assert sep.max() < 0.01, (i, sep.max())


def test_hsc_psf_matches_legacypipe(hsc):
    """Native-pixel PSF at sub-pixel positions: ours vs tractor's oversampled patch. legacypipe
    multiplies columns 0-5 of the oversampled bases by PsfEx's ``coeff`` scratch vector (see
    psfex.hsc_psfex), which zeroes the outer 3 native columns on one side; compare the inner
    25 x 25 pixels (radius 12, i.e. 2")."""
    survey, T, C = hsc
    for i in [0, 40]:
        im = survey.get_image_object(T[i])
        psf = im.read_psf_model(0, 0, pixPsf=True, normalizePsf=True)
        m = psfex.hsc_psfex(os.path.join(HSC_IMAGES, str(C[i]['image_filename']).strip()))
        for px, py in [(500.3, 1200.7), (1500.6, 3000.2), (1024., 2088.)]:
            patch = psf.getPointSourcePatch(px, py)
            ref = np.asarray(patch.patch, dtype=np.float64)
            ix, iy = round(px), round(py)
            ours = m.native(px, py, px - ix, py - iy)
            n = ours.shape[0]
            ox, oy = ix - n // 2, iy - n // 2              # our stamp origin in CCD pixels
            R = 12
            sl_o = (slice(iy - R - oy, iy + R + 1 - oy), slice(ix - R - ox, ix + R + 1 - ox))
            sl_r = (slice(iy - R - patch.y0, iy + R + 1 - patch.y0), slice(ix - R - patch.x0, ix + R + 1 - patch.x0))
            a, b = ours[sl_o], ref[sl_r]
            a, b = a / a.sum(), b / b.sum()
            err = np.max(np.abs(a - b)) / b.max()
            assert err < 3e-3, (i, px, py, err)
