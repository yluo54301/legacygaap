"""Reference implementations built on legacypipe, tractor and astrometry.net (Shifter only).

Used by the ``shifter`` tests to check the pure-python re-implementation against the code that
made the coadds. Nothing in ``legacygaap`` imports this module.
"""
import numpy as np


def get_survey(survey_dir):
    from legacypipe.survey import LegacySurveyData
    return LegacySurveyData(survey_dir=survey_dir)


def brick_ccds_fits_table(paths):
    from astrometry.util.fits import fits_table
    return fits_table(paths.ccds)


def match_rows(T, canonical):
    """Indices into fits_table ``T`` of the (expnum, ccdname) pairs of a canonical CCD table."""
    key = {(int(e), c.strip()): i for i, (e, c) in enumerate(zip(T.expnum, T.ccdname))}
    return np.array([key[(int(e), str(c).strip())] for e, c in zip(canonical['expnum'], canonical['ccdname'])])


def ref_psf_object(survey, ccd, hybrid=False):
    """legacypipe image object and its (normalised) PsfEx PSF model for a fits_table CCD row."""
    im = survey.get_image_object(ccd)
    psf = im.read_psf_model(0, 0, pixPsf=True, hybridPsf=hybrid, normalizePsf=True,
                            psf_sigma=ccd.fwhm / 2.35, w=ccd.width, h=ccd.height)
    return im, psf


def ref_lanczos_shift(img, dx, dy):
    from tractor.psf import lanczos_shift_image
    return lanczos_shift_image(np.asarray(img, dtype=np.float32), dx, dy)


def ref_coadd_psf(survey, brickname, ccd_rows, x, y, stamp=65, return_flux_scale=False):
    """Coadd PSF at brick pixel (x, y) from the given fits_table CCD rows, computed with legacypipe's
    WCS, tractor's PsfEx evaluation and sub-pixel shift, astrometry.net's Lanczos-3 resampling, and
    legacypipe's coadd scaling (image x sbscale, weight 1/sig1^2/sbscale^2). Unit sum; with
    ``return_flux_scale`` also the coadd flux of a unit-flux star."""
    from legacypipe.survey import wcs_for_brick
    from astrometry.util.resample import resample_with_wcs, OverlapError
    brick = survey.get_brick_by_name(brickname)
    targetwcs = wcs_for_brick(brick)
    R = stamp // 2
    cx, cy = int(round(float(x))), int(round(float(y)))
    ra, dec = targetwcs.pixelxy2radec(x + 1, y + 1)
    subwcs = targetwcs.get_subimage(cx - R, cy - R, stamp, stamp)
    acc = np.zeros((stamp, stamp), np.float64)
    n = 0
    wsum = 0.
    for ccd in ccd_rows:
        im, psf = ref_psf_object(survey, ccd)
        wcs = im.get_wcs()
        f = (targetwcs.pixel_scale() / wcs.pixel_scale()) ** 2
        ok, px, py = wcs.radec2pixelxy(ra, dec)
        px -= 1.
        py -= 1.
        patch = psf.getPointSourcePatch(px, py)
        p = patch.patch.astype(np.float64)
        ph, pw = p.shape
        pwcs = wcs.get_subimage(patch.x0, patch.y0, pw, ph)
        try:
            Yo, Xo, Yi, Xi, rims = resample_with_wcs(subwcs, pwcs, [p], 3)
        except OverlapError:
            continue
        w = 1. / float(ccd.sig1) ** 2 / f ** 2
        acc[Yo, Xo] += w * f * rims[0]
        wsum += w
        n += 1
    if return_flux_scale:
        return acc / acc.sum(), n, acc.sum() / wsum
    return acc / acc.sum(), n
