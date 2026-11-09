"""``gaap-brick``: GAaP photometry for one legacypipe brick.

Reads a brick's coadds, its tractor catalogue (or another catalogue with the same rows) and its CCD
list, rebuilds the coadd PSF at every source from the per-CCD PSF models, and writes one row per
catalogue row to ``<outdir>/<bbb>/gaap-<brick>.fits``; with ``--sky N`` also the same measurement
at N blank-sky positions, ``<outdir>/<bbb>/gaap-sky-<brick>.fits``. ``--tag T`` appends ``-T`` to
both names.

Examples::

    export LEGACY_SURVEY_DIR=/path/to/legacysurvey/dr10
    gaap-brick --brick 0001m002 --bands g,r,z --threads 32 --sky 1000 --outdir gaap
    gaap-brick --brick 0001m002 --circ 1.0,1.5 --min-aper 1.0 --limit 200 --outdir test

    # a single legacypipe run (no south/north level), its own catalogue rows from a forced catalogue
    gaap-brick --brick 1501p022 --survey-dir /path/to/run --catalog /path/to/forced/{bbb}/tractor-forced-{brick}.fits
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')          # one BLAS thread per process (the workers are processes)
import argparse  # noqa: E402
import numpy as np  # noqa: E402
from . import __version__, brick, io  # noqa: E402


def floats(s):
    return tuple(float(v) for v in s.split(',') if v.strip())


def parser():
    d = brick.Settings()
    p = argparse.ArgumentParser(prog='gaap-brick', description='GAaP photometry for one legacypipe brick',
                                epilog='Environment: LEGACY_SURVEY_DIR (default --survey-dir), GAAP_IMAGES_ROOT '
                                       '(default --images-root), GAAP_CACHE_DIR (WCS/CCD caches).')
    p.add_argument('--version', action='version', version=f'legacygaap {__version__}')
    g = p.add_argument_group('input')
    g.add_argument('--brick', required=True, help='brick name, e.g. 0001m002')
    g.add_argument('--survey-dir', default=None,
                   help='legacypipe survey directory: a data release (south/ and north/ trees) or a single run '
                        '(coadd/ directly under it); default $LEGACY_SURVEY_DIR')
    g.add_argument('--region', default=None, help='south or north (default: inferred; single runs have none)')
    g.add_argument('--bands', default=None, help='comma list (default: every optical band with a coadd)')
    g.add_argument('--catalog', default=None,
                   help="source catalogue instead of the run's tractor catalogue (same rows; may contain {brick}, "
                        '{bbb}); band fluxes from flux_<band> or forced_flux_<band>')
    g.add_argument('--image-kind', default='image', help='coadd to measure: image (default), model, blobmodel, ...')
    g.add_argument('--images-root', default=None,
                   help='directory that the CCD tables\' image_filename is relative to (default $GAAP_IMAGES_ROOT, '
                        'else <survey-dir>/images)')
    g.add_argument('--rows', default=None, help='file with the catalogue row indices to measure (one per line)')
    g.add_argument('--limit', type=int, default=None, help='only the first N catalogue rows (testing)')
    a = p.add_argument_group('apertures and PSF Gaussianization')
    a.add_argument('--circ', type=floats, default=d.circ, help='circular pre-seeing aperture dispersions (arcsec)')
    a.add_argument('--min-aper', type=floats, default=d.min_apers,
                   help='MIN_APER of the shape-matched elliptical apertures (arcsec)')
    a.add_argument('--max-aper', type=float, default=d.max_aper, help='cap on the elliptical aperture axes (arcsec)')
    a.add_argument('--kappa', type=float, default=d.kappa, help='elliptical aperture axes = kappa x tractor axes')
    a.add_argument('--scale', type=float, default=d.scale, help='target Gaussian PSF dispersion / PSF dispersion')
    a.add_argument('--ksize', type=int, default=d.ksize, help='Gaussianization kernel size (pixels)')
    a.add_argument('--lam', type=float, default=d.lam, help='kernel regularisation')
    a.add_argument('--stamp', type=int, default=d.stamp, help='PSF stamp size (pixels)')
    a.add_argument('--cutout', type=int, default=d.cutout, help='minimum measurement cutout (pixels)')
    s = p.add_argument_group('special inputs')
    s.add_argument('--wcs-ccds', default=None,
                   help="CCD table whose image of the same (expnum, ccdname) gives each CCD's WCS and offsets "
                        '(for runs whose original images were replaced by a re-reduction)')
    s.add_argument('--wcs-images-root', default=None, help='image root for --wcs-ccds')
    s.add_argument('--wcs-tag', default='alt', help='WCS cache subdirectory tag for --wcs-ccds')
    s.add_argument('--survey-ccds', default=None,
                   help='DIR:CAM[,CAM]: survey-ccds tables for bands missing from the brick CCD table (forced bands)')
    s.add_argument('--sky-ref-dir', default=None, help='run whose blob map and maskbits give the blank-sky positions')
    s.add_argument('--star-mags', type=floats, default=None, help='magnitude window of the shot-noise calibration stars')
    s.add_argument('--iso-dmag', type=float, default=None,
                   help='shot-noise calibration stars: ignore neighbours more than this many mag fainter (deep catalogues)')
    s.add_argument('--pixscale', default='auto', choices=['auto', 'nominal', 'cd'],
                   help="legacypipe's CCD pixel-scale convention for sbscale/psfsize (auto: match the psfsize map)")
    o = p.add_argument_group('output and execution')
    o.add_argument('--outdir', default='.', help='output directory (files go to <outdir>/<bbb>/)')
    o.add_argument('--tag', default=None, help='output name suffix: gaap-<brick>-<tag>.fits')
    o.add_argument('--sky', type=int, default=0, help='also measure N blank-sky positions (empirical errors, depth)')
    o.add_argument('--threads', type=int, default=1, help='worker processes')
    o.add_argument('--quiet', action='store_true', help='no progress messages')
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    st = brick.Settings(circ=a.circ, min_apers=a.min_aper, max_aper=a.max_aper, kappa=a.kappa, scale=a.scale,
                        ksize=a.ksize, lam=a.lam, stamp=a.stamp, cutout=a.cutout)
    rows = np.loadtxt(a.rows, dtype=int, ndmin=1) if a.rows else None
    survey_ccds = None
    if a.survey_ccds:
        d, cams = a.survey_ccds.split(':')
        survey_ccds = (d, cams.split(','))
    log = (lambda *s: None) if a.quiet else (lambda *s: print(*s, flush=True))
    brick.run_brick(a.brick, survey_dir=a.survey_dir, region=a.region, bands=a.bands.split(',') if a.bands else None,
                    settings=st, threads=a.threads, rows=rows, limit=a.limit, sky=a.sky, outdir=a.outdir,
                    image_kind=a.image_kind, log=log, catalog=a.catalog, images_root=a.images_root,
                    wcs_ccds=a.wcs_ccds, wcs_images_root=a.wcs_images_root, wcs_tag=a.wcs_tag, tag=a.tag,
                    star_mags=a.star_mags, sky_ref_dir=a.sky_ref_dir, pixscale=a.pixscale, survey_ccds=survey_ccds,
                    iso_dmag=a.iso_dmag)


if __name__ == '__main__':
    main()
