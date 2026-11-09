"""Paths and readers for legacypipe brick products, CCD tables and PsfEx calibration files.

Layout (DR10/DR11): ``<survey_dir>/<region>/coadd/<bbb>/<brick>/legacysurvey-<brick>-<kind>-<band>.fits.fz``,
``<survey_dir>/<region>/tractor/<bbb>/tractor-<brick>.fits``, CCD table in the coadd directory,
PsfEx models under ``<survey_dir>/calib/psfex/``, survey-wide CCD tables ``<survey_dir>/survey-ccds-<camera>-*.kd.fits``.
Single-run directories (one legacypipe run, e.g. a medium-band or an HSC reduction) have the same
layout without the region level ("flat"). HSC CALEXP/CORR images carry their own PsfEx model and SIP WCS.
No legacypipe or tractor imports.
"""
import os
import socket
import glob
import json
import numpy as np
import fitsio
from . import resample as rs

# Default survey directory: $LEGACY_SURVEY_DIR (as for legacypipe), else $GAAP_SURVEY_DIR. At NERSC the
# read-only DVS mount (/dvs_ro/cfs/...) is much faster on compute nodes than /global/cfs.
DEFAULT_SURVEY_DIR = os.environ.get('LEGACY_SURVEY_DIR') or os.environ.get('GAAP_SURVEY_DIR')
REGIONS = ('south', 'north')
CAMERAS = {'south': ('decam',), 'north': ('90prime', 'mosaic')}
OPTICAL_BANDS = ('g', 'r', 'i', 'z')
BRICK_SIZE = 3600
# Where the files named by the CCD tables' image_filename live (their headers give the exact CCD WCS;
# HSC and Rubin coadd files also hold the PSF): $GAAP_IMAGES_ROOT, else <survey_dir>/images (run_brick).
IMAGES_ROOT = os.environ.get('GAAP_IMAGES_ROOT')
# Caches of CCD WCS headers (JSON), per-brick CCD lists and HSC mask maps: $GAAP_CACHE_DIR, else
# $XDG_CACHE_HOME/legacygaap (default ~/.cache/legacygaap). Safe to share between concurrent jobs.
CACHE_DIR = os.environ.get('GAAP_CACHE_DIR') or os.path.join(
    os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache'), 'legacygaap')
WCS_CACHE = os.path.join(CACHE_DIR, 'wcs_cache')
CCD_CACHE = os.path.join(CACHE_DIR, 'ccd_cache')
# CCD edge band (pixels) that the CP masks (invvar = 0), measured on DR11 brick 0332m045: ~10-15 px
EDGE_MARGIN = 10.
# over-counted fraction above which a candidate CCD is deemed unused by the pipeline (unused: 1.0)
OVERCOUNT_THRESHOLD = 0.7
# half-diagonal of one CCD on the sky (deg), used to pre-select survey CCDs near a brick
# (lsstcoadd: a Rubin deep-coadd patch, 3300 x 3300 pixels of 0.2")
CCD_HALFDIAG_DEG = {'decam': 0.17, '90prime': 0.38, 'mosaic': 0.22, 'lsstcoadd': 0.13, 'megaprime': 0.14}
# Survey CCD tables for single-run (flat) layouts whose survey-ccds files live elsewhere (e.g. a
# legacypipe run on Rubin deep-coadd patches with forced bands), and the cameras to search there;
# set by set_survey_ccds().
SURVEY_CCDS_DIR = None
FLAT_CAMERAS = ()
# cameras whose CCD-table TAN WCS is exact (Rubin coadd patches: TAN, PC matrix with CDELT 1)
TAN_CAMERAS = ('lsstcoadd', 'megaprime')

# Canonical CCD table: the columns the PSF reconstruction needs, present in both the per-brick
# CCD tables and the survey-wide CCD tables.
CCD_DTYPE = np.dtype([
    ('image_filename', 'U256'), ('image_hdu', 'i4'), ('camera', 'U16'), ('expnum', 'i8'),   # 'lsstcoadd' > 8 chars
    ('ccdname', 'U8'), ('filter', 'U4'), ('exptime', 'f4'), ('mjd_obs', 'f8'),
    ('fwhm', 'f4'), ('width', 'i4'), ('height', 'i4'),
    ('crpix1', 'f8'), ('crpix2', 'f8'), ('crval1', 'f8'), ('crval2', 'f8'),
    ('cd1_1', 'f8'), ('cd1_2', 'f8'), ('cd2_1', 'f8'), ('cd2_2', 'f8'),
    ('ra', 'f8'), ('dec', 'f8'), ('sig1', 'f4'), ('ccdzpt', 'f4'), ('ccdskycounts', 'f4'),
    ('ccdraoff', 'f4'), ('ccddecoff', 'f4'), ('plver', 'U16'), ('ccd_cuts', 'i8'),
    # the CCD sub-image ("tim") legacypipe read for the brick, from brick CCD tables (-1: unknown)
    ('ccd_x0', 'i4'), ('ccd_x1', 'i4'), ('ccd_y0', 'i4'), ('ccd_y1', 'i4'),
])
TIM_BOX = ('ccd_x0', 'ccd_x1', 'ccd_y0', 'ccd_y1')
# CCD edge band excluded from the PSF per camera, where it differs from EDGE_MARGIN: HSC tim boxes
# already exclude the masked edges (18/22 columns, 8/21 rows on brick 1501p022)
EDGE_MARGIN_CAMERA = {'hsc': 0., 'lsstcoadd': 0.}


NORTH_SOUTH_DEC = 32.375   # DR resolve rule: north products are primary above this declination

# approximate central wavelengths (nm), only used to order bands
BAND_WAVELENGTH = {'u': 360, 'N395': 395, 'M411': 411, 'N419': 419, 'M438': 438, 'M464': 464, 'g': 480, 'M490': 490,
                   'N501': 501, 'M517': 517, 'N540': 540, 'r': 620, 'r2': 621, 'N673': 673, 'N708': 708, 'i': 770,
                   'i2': 771, 'z': 900, 'y': 1000, 'Y': 1001}
WISE_BANDS = ('W1', 'W2', 'W3', 'W4')


def band_sort_key(b):
    return (BAND_WAVELENGTH.get(b, 10000), b)


def is_flat(survey_dir):
    """True for a single-run directory (coadd/ directly under it) rather than a DR tree with regions."""
    return (os.path.isdir(os.path.join(survey_dir, 'coadd'))
            and not any(os.path.isdir(os.path.join(survey_dir, r, 'coadd')) for r in REGIONS))


def find_region(survey_dir, brick, prefer=None):
    """Region ('south' or 'north') whose tractor directory holds ``brick``.

    DR11 has a few thousand low-declination bricks in both trees (the north tree holds MzLS-only
    runs there). When both exist, the DR resolve rule decides: north above ``NORTH_SOUTH_DEC``,
    south below, unless ``prefer`` names a region.
    """
    found = [r for r in REGIONS if os.path.exists(_tractor_path(survey_dir, r, brick))]
    if len(found) == 1:
        return found[0]
    if not found:
        raise FileNotFoundError(f'brick {brick} not found in {survey_dir} (south or north)')
    if prefer in found:
        return prefer
    return 'north' if brick_dec(brick) >= NORTH_SOUTH_DEC else 'south'


def brick_dec(brick):
    """Declination of a brick centre from its name, e.g. '1501p022' -> 2.2, '0332m045' -> -4.5."""
    sign = 1. if brick[4] == 'p' else -1.
    return sign * float(brick[5:]) / 10.


def _tractor_path(survey_dir, region, brick):
    return os.path.join(survey_dir, region, 'tractor', brick[:3], f'tractor-{brick}.fits')


class BrickPaths:
    """File locations of one brick's products. ``region`` '' (or a flat survey_dir) means a
    single-run layout; ``catalog`` overrides the tractor catalog (a path, may contain {brick} and
    {bbb}), e.g. a forced-photometry catalog with the same rows."""

    def __init__(self, brick, survey_dir=None, region=None, catalog=None):
        self.brick = brick
        survey_dir = survey_dir or DEFAULT_SURVEY_DIR
        if not survey_dir:
            raise ValueError('no survey directory: pass survey_dir (gaap-brick --survey-dir) or set $LEGACY_SURVEY_DIR')
        self.survey_dir = survey_dir
        if region is None:
            region = '' if is_flat(survey_dir) else find_region(survey_dir, brick)
        self.region = region
        self.catalog = catalog

    def __repr__(self):
        return f'BrickPaths({self.brick!r}, region={self.region!r}, survey_dir={self.survey_dir!r})'

    @property
    def coadd_dir(self):
        return os.path.join(self.survey_dir, self.region, 'coadd', self.brick[:3], self.brick)

    @property
    def metrics_dir(self):
        return os.path.join(self.survey_dir, self.region, 'metrics', self.brick[:3])

    @property
    def tractor(self):
        if self.catalog:
            return self.catalog.format(brick=self.brick, bbb=self.brick[:3])
        return _tractor_path(self.survey_dir, self.region, self.brick)

    @property
    def ccds(self):
        return os.path.join(self.coadd_dir, f'legacysurvey-{self.brick}-ccds.fits')

    @property
    def maskbits(self):
        return os.path.join(self.coadd_dir, f'legacysurvey-{self.brick}-maskbits.fits.fz')

    def coadd(self, kind, band):
        return os.path.join(self.coadd_dir, f'legacysurvey-{self.brick}-{kind}-{band}.fits.fz')

    def bands(self):
        """Optical bands with an image coadd (any band name, WISE excluded), ordered by wavelength."""
        pre = f'legacysurvey-{self.brick}-image-'
        found = [os.path.basename(f)[len(pre):-len('.fits.fz')]
                 for f in glob.glob(os.path.join(self.coadd_dir, pre + '*.fits.fz'))]
        return sorted([b for b in found if b.upper() not in WISE_BANDS], key=band_sort_key)

    def exists(self):
        return os.path.exists(self.tractor) and os.path.exists(self.ccds)


def read_header(fn, ext=1):
    return fitsio.read_header(fn, ext=ext)


def read_coadd(paths, kind, band=None):
    """Coadd map as a numpy array (``kind`` in image, invvar, model, nexp, psfsize, ...; 'maskbits' needs no band)."""
    fn = paths.maskbits if kind == 'maskbits' else paths.coadd(kind, band)
    return fitsio.read(fn, ext=1)


def brick_wcs(paths, band=None):
    """TAN WCS of the brick's coadds (identical for all bands)."""
    band = band or paths.bands()[0]
    return rs.wcs_from_header(read_header(paths.coadd('image', band)))


def read_tractor(paths, columns=None, rows=None):
    return fitsio.read(paths.tractor, ext=1, columns=columns, rows=rows)


def col_str(table, name):
    """String column as stripped ``str`` array (fitsio may return bytes or padded unicode)."""
    v = table[name]
    if v.dtype.kind == 'S':
        v = np.char.decode(v)
    return np.char.strip(v.astype(str))


def psfex_path(survey_dir, image_filename):
    """Merged PsfEx file for an image: ``<survey_dir>/calib/psfex/<image dir>/<basename>-psfex.fits``
    (legacypipe's ``merged_psffn``)."""
    base = str(image_filename).strip()
    for suffix in ('.fits.fz', '.fits.gz', '.fits'):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    return os.path.join(survey_dir, 'calib', 'psfex', base + '-psfex.fits')


def set_images_root(root):
    """Where ``image_filename`` paths are resolved (CP headers for the WCS; HSC files for the PSF)."""
    global IMAGES_ROOT
    IMAGES_ROOT = root


# Optional replacement source for the CCD WCS. IBIS DR1 was built from CP V5.5.2LS images that are
# no longer on disk; the V6.0LS re-reduction of the same CCDs, with its own ccdraoff/ccddecoff,
# reproduces the DR1 astrometry to <= 0.04 px.
_WCS_OVERRIDE = {}
_WCS_OVERRIDE_CFG = {}


def set_wcs_override(table_fn, images_root, tag, expnums=None):
    """Take every CCD's WCS from the same (expnum, ccdname) in ``table_fn`` (image_filename,
    image_hdu, ccdraoff, ccddecoff) under ``images_root``; cached separately under ``tag``.
    Returns the number of CCDs loaded."""
    T = fitsio.read(table_fn, columns=['expnum', 'ccdname', 'image_filename', 'image_hdu', 'ccdraoff', 'ccddecoff'])
    if expnums is not None:
        T = T[np.isin(T['expnum'], np.asarray(list(expnums), dtype=T['expnum'].dtype))]
    _WCS_OVERRIDE.clear()
    names, fns = col_str(T, 'ccdname'), col_str(T, 'image_filename')
    for r, name, fn in zip(T, names, fns):
        _WCS_OVERRIDE[(int(r['expnum']), name)] = (fn, int(r['image_hdu']), float(r['ccdraoff']), float(r['ccddecoff']))
    _WCS_OVERRIDE_CFG.clear()
    _WCS_OVERRIDE_CFG.update(table=table_fn, images_root=images_root, tag=tag)
    return len(_WCS_OVERRIDE)


def clear_wcs_override():
    _WCS_OVERRIDE.clear()
    _WCS_OVERRIDE_CFG.clear()


def set_survey_ccds(ccds_dir, cameras):
    """Where survey-ccds-<camera>-*.kd.fits are found for bands missing from a brick's CCD table
    (legacypipe's forced bands), and which cameras to search in a flat layout."""
    global SURVEY_CCDS_DIR, FLAT_CAMERAS
    SURVEY_CCDS_DIR = ccds_dir
    FLAT_CAMERAS = tuple(cameras)


def _table_wcs_keys(ccd):
    """TAN WCS keywords from a CCD-table row (exact for TAN_CAMERAS)."""
    keys = {'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN', 'CUNIT1': 'deg', 'CUNIT2': 'deg',
            'NAXIS1': int(ccd['width']), 'NAXIS2': int(ccd['height'])}
    for k in ('crval1', 'crval2', 'crpix1', 'crpix2', 'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2'):
        keys[k.upper()] = float(ccd[k])
    return keys


def wcs_source(ccd):
    """(image file, HDU, ccdraoff, ccddecoff, images root, cache subdirectory) giving a CCD's WCS."""
    cam = str(ccd['camera']).strip()
    if _WCS_OVERRIDE:
        key = (int(ccd['expnum']), str(ccd['ccdname']).strip())
        if key not in _WCS_OVERRIDE:
            raise KeyError(f'no WCS override for expnum {key[0]} ccd {key[1]} in {_WCS_OVERRIDE_CFG.get("table")}')
        fn, hdu, dra, ddec = _WCS_OVERRIDE[key]
        return fn, hdu, dra, ddec, _WCS_OVERRIDE_CFG['images_root'], f"{cam}-{_WCS_OVERRIDE_CFG['tag']}"
    return (str(ccd['image_filename']).strip(), int(ccd['image_hdu']), float(ccd['ccdraoff']), float(ccd['ccddecoff']),
            IMAGES_ROOT, cam)


def ccd_wcs_keys(ccd, images_root=None, cache_dir=None):
    """WCS keywords of a CCD's image HDU, read once from the image file and cached as JSON.
    TAN_CAMERAS take them from the CCD table (Rubin patch headers use PC/CDELT, not CD)."""
    if str(ccd['camera']).strip() in TAN_CAMERAS:
        return _table_wcs_keys(ccd)
    fn, hdu, _, _, root, sub = wcs_source(ccd)
    cfn = os.path.join(cache_dir or WCS_CACHE, sub, f"{int(ccd['expnum']):08d}-{str(ccd['ccdname']).strip()}.json")
    if os.path.exists(cfn):
        with open(cfn) as f:
            return json.load(f)
    base = images_root or root
    if base is None:
        raise ValueError(f'no image root to read the WCS of {fn}: set $GAAP_IMAGES_ROOT or pass images_root')
    keys = rs.wcs_keys_from_header(fitsio.read_header(os.path.join(base, fn), ext=hdu))
    os.makedirs(os.path.dirname(cfn), exist_ok=True)
    tmp = cfn + f'.tmp{socket.gethostname()}-{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(keys, f)
    os.replace(tmp, cfn)
    return keys


def ccd_wcs_exact(ccd, apply_offsets=True, images_root=None, cache_dir=None):
    """Exact (TPV or SIP) WCS of a CCD as legacypipe uses it: header astrometry plus the per-CCD
    zero-point offsets ``ccdraoff``/``ccddecoff``. The table's TAN WCS (``resample.ccd_wcs``) is
    off by 10-20 pixels for DECam and is only good for coarse overlap tests."""
    keys = ccd_wcs_keys(ccd, images_root=images_root, cache_dir=cache_dir)
    if apply_offsets:
        _, _, dra, ddec, _, _ = wcs_source(ccd)
        return rs.wcs_from_keys(keys, dra=dra / 3600., ddec=ddec / 3600.)
    return rs.wcs_from_keys(keys)


# Per-CCD maps of the masked pixel fraction (cameras whose large masked regions matter: HSC bad
# amplifiers, ~10% of some CCDs' tim boxes on brick 1501p022). With ~6 HSC exposures per position a
# wrongly included CCD carries ~15-20% of the PSF weight; with ~30 IBIS/DECam exposures it is negligible.
MASK_CACHE = os.path.join(CACHE_DIR, 'mask_cache')
MASK_BIN = 16
MASK_CAMERAS = ('hsc',)
# legacypipe hsc.remap_hsc_bitmask: pixels with any of these bits get invvar = 0
HSC_MASK_BITS = ('BAD', 'SUSPECT', 'NO_DATA', 'CROSSTALK', 'UNMASKEDNAN', 'SAT', 'INTRP', 'CR', 'EDGE')


def ccd_mask_map(ccd, images_root=None, binsize=MASK_BIN, cache_dir=None):
    """Fraction of pixels legacypipe masks (invvar = 0) in ``binsize`` x ``binsize`` bins of the CCD,
    as float16 (shape ceil(H/bin), ceil(W/bin)); cached as .npy. None for cameras without support."""
    cam = str(ccd['camera']).strip()
    if cam != 'hsc':
        return None
    cfn = os.path.join(cache_dir or MASK_CACHE, cam, f"{int(ccd['expnum']):08d}-{str(ccd['ccdname']).strip()}-b{binsize}.npy")
    if os.path.exists(cfn):
        return np.load(cfn)
    fn = os.path.join(images_root or IMAGES_ROOT, str(ccd['image_filename']).strip())
    hdr = fitsio.read_header(fn, ext=2)
    bits = 0
    for name in HSC_MASK_BITS:
        if f'MP_{name}' in hdr:
            bits |= 1 << int(hdr[f'MP_{name}'])
    m = (fitsio.read(fn, ext=2) & bits) != 0
    H, W = m.shape
    ny, nx = -(-H // binsize), -(-W // binsize)
    pad = np.zeros((ny * binsize, nx * binsize), np.float32)
    pad[:H, :W] = m
    cnt = np.zeros_like(pad)
    cnt[:H, :W] = 1
    frac = pad.reshape(ny, binsize, nx, binsize).sum(axis=(1, 3)) / np.maximum(cnt.reshape(ny, binsize, nx, binsize).sum(axis=(1, 3)), 1)
    frac = frac.astype(np.float16)
    os.makedirs(os.path.dirname(cfn), exist_ok=True)
    tmp = cfn + f'.tmp{socket.gethostname()}-{os.getpid()}.npy'
    np.save(tmp, frac)
    os.replace(tmp, cfn)
    return frac


def flux_column(colnames, band):
    """Catalog column holding a band's model flux: tractor ``flux_<band>`` or, for bands measured by
    forced photometry on another survey's images, ``forced_flux_<band>``; None if neither exists."""
    lower = {c.lower(): c for c in colnames}          # forced-photometry outputs lower-case band names
    for c in (f'flux_{band}', f'forced_flux_{band}'):
        if c.lower() in lower:
            return lower[c.lower()]
    return None


DECAM_GLOW_EXPNUM = 298251


def good_subregion(ccd):
    """legacypipe's ``get_good_image_subregion`` for a CCD: (x0, x1, y0, y1) in 0-based pixels,
    x1/y1 exclusive, None = full extent. DECam: right half of S7 is always clipped; early DES
    r-band exposures (expnum < 298251) lose 100 rows at the glowing edge."""
    x0 = x1 = y0 = y1 = None
    if str(ccd['camera']).strip() == 'lsstcoadd':
        # Rubin deep-coadd patches overlap by 150 pixels (one cell); legacypipe keeps the inner
        # 3000 x 3000 (lsst.LsstCoaddImage.get_good_image_subregion)
        return (150, 3150, 150, 3150)
    if str(ccd['camera']).strip() == 'megaprime':
        # CFHT MegaCam Elixir images: DATASEC [33:2080,1:4612] of 2112 x 4644, taken by legacypipe
        # (cfht.MegaPrimeElixirImage.get_good_image_subregion) as x0, x1, y0, y1 = 33, 2080, 1, 4612
        return (33, 2080, 1, 4612)
    if str(ccd['camera']).strip() == 'decam':
        name = str(ccd['ccdname']).strip()
        if str(ccd['filter']).strip() == 'r' and int(ccd['expnum']) < DECAM_GLOW_EXPNUM:
            if 'N' in name:
                y0 = 100
            else:
                y1 = int(ccd['height']) - 100
        if name == 'S7':
            x1 = 1023
    return (x0, x1, y0, y1)


def ccd_bounds(ccd):
    """(x0, x1, y0, y1) of the CCD pixels that can reach the brick's coadd: legacypipe's good
    subregion intersected with the recorded tim box when the CCD table has one. legacypipe picks
    the tim box by clipping the CCD with the brick's four corners joined by straight lines in CCD
    pixels (inverse-SIP extrapolation for HSC), so it can be much smaller than the true overlap
    (up to ~200 px for HSC CCDs at brick corners); pixels outside it never entered the coadd."""
    x0, x1, y0, y1 = good_subregion(ccd)
    x0, y0 = x0 or 0, y0 or 0
    x1 = int(ccd['width']) if x1 is None else x1
    y1 = int(ccd['height']) if y1 is None else y1
    if 'ccd_x1' in ccd.dtype.names and int(ccd['ccd_x1']) > 0:
        x0, x1 = max(x0, int(ccd['ccd_x0'])), min(x1, int(ccd['ccd_x1']))
        y0, y1 = max(y0, int(ccd['ccd_y0'])), min(y1, int(ccd['ccd_y1']))
    return x0, x1, y0, y1


def ccd_brick_box(ccd, wcs, bwcs, n_edge=20):
    """Pixel box (x0, x1, y0, y1; 0-based, exclusive upper) of the part of the CCD that overlaps the
    brick: the bounding box of the brick outline in CCD coordinates, clipped to the CCD. This is the
    sub-image ("tim") legacypipe reads, whose centre is where it evaluates the per-CCD PSF norm."""
    poly = rs.footprint_pix(bwcs, wcs, n_edge=n_edge)
    W, H = int(ccd['width']), int(ccd['height'])
    x0 = int(np.clip(np.floor(poly[:, 0].min()), 0, W)); x1 = int(np.clip(np.ceil(poly[:, 0].max()) + 1, 0, W))
    y0 = int(np.clip(np.floor(poly[:, 1].min()), 0, H)); y1 = int(np.clip(np.ceil(poly[:, 1].max()) + 1, 0, H))
    return x0, x1, y0, y1


def ccd_footprint(ccd, wcs, bwcs, margin=0.):
    """Footprint polygon (brick pixels) of the good, edge-trimmed part of a CCD."""
    return rs.footprint_pix(wcs, bwcs, margin=margin, bounds=good_subregion(ccd))


def ccd_sbscale(ccd, target_pixscale, **kw):
    """legacypipe's surface-brightness scale for a CCD in a coadd of pixel scale ``target_pixscale``:
    (target / nominal CCD pixel scale)^2. legacypipe multiplies the CCD image by it and divides its
    inverse variance by its square (coadds.py, sbscale=True)."""
    return (target_pixscale / rs.nominal_pixel_scale(ccd_wcs_keys(ccd, **kw))) ** 2


def ccd_wcses(ccds, exact=True, **kw):
    return [ccd_wcs_exact(c, **kw) if exact else rs.ccd_wcs(c) for c in ccds]


def _encode(T):
    """Copy of a canonical CCD table with byte strings (for FITS writing)."""
    dt = [(n, f'S{T.dtype[n].itemsize // 4}' if T.dtype[n].kind == 'U' else T.dtype[n].str) for n in T.dtype.names]
    out = np.zeros(len(T), dtype=dt)
    for n in T.dtype.names:
        out[n] = np.char.encode(T[n]) if T.dtype[n].kind == 'U' else T[n]
    return out


def to_ccd_table(rows):
    """Copy the canonical columns out of any CCD table (brick or survey-wide)."""
    out = np.zeros(len(rows), dtype=CCD_DTYPE)
    for name in TIM_BOX:
        out[name] = -1
    for name in CCD_DTYPE.names:
        if name in rows.dtype.names:
            out[name] = col_str(rows, name) if rows[name].dtype.kind in 'SU' else rows[name]
    return out


def read_brick_ccds(paths):
    """The brick's CCD table (every CCD that went into the brick's coadds) in canonical form."""
    return to_ccd_table(fitsio.read(paths.ccds, ext=1))


def survey_ccds_file(survey_dir, camera):
    fns = sorted(glob.glob(os.path.join(survey_dir, f'survey-ccds-{camera}-*.kd.fits')))
    if not fns:
        raise FileNotFoundError(f'no survey-ccds-{camera}-*.kd.fits in {survey_dir}')
    return fns[0]


_INDEX_CACHE = {}


def _survey_index(fn):
    """(ra, dec) of all CCDs in a survey CCD table, cached per process (the decam table is 18M rows)."""
    if fn not in _INDEX_CACHE:
        t = fitsio.read(fn, ext=1, columns=['ra', 'dec'])
        _INDEX_CACHE[fn] = (t['ra'].astype(np.float64), t['dec'].astype(np.float64))
    return _INDEX_CACHE[fn]


def survey_ccds_near(survey_dir, camera, ra, dec, radius_deg, band=None, good_only=True):
    """CCDs (canonical table) whose centre is within ``radius_deg`` of (ra, dec)."""
    fn = survey_ccds_file(survey_dir, camera)
    ra_all, dec_all = _survey_index(fn)
    near = np.abs(dec_all - dec) <= radius_deg
    idx = np.flatnonzero(near)
    # exact angular separation for the declination-selected subset
    r1, d1 = np.deg2rad(ra_all[idx]), np.deg2rad(dec_all[idx])
    r0, d0 = np.deg2rad(ra), np.deg2rad(dec)
    cossep = np.sin(d0) * np.sin(d1) + np.cos(d0) * np.cos(d1) * np.cos(r1 - r0)
    idx = idx[cossep >= np.cos(np.deg2rad(radius_deg))]
    if len(idx) == 0:
        return np.zeros(0, dtype=CCD_DTYPE)
    rows = fitsio.read(fn, ext=1, rows=np.sort(idx))
    T = to_ccd_table(rows)
    keep = np.ones(len(T), dtype=bool)
    if band is not None:
        keep &= (T['filter'] == band)
    if good_only:
        keep &= (T['ccd_cuts'] == 0)
    return T[keep]


def candidate_ccds(paths, band, bwcs=None, cameras=None, step=8):
    """Survey CCDs of ``band`` (ccd_cuts == 0) whose footprint overlaps the brick."""
    bwcs = bwcs or brick_wcs(paths)
    ra0, dec0 = bwcs.wcs.crval
    cameras = cameras or CAMERAS.get(paths.region) or FLAT_CAMERAS
    sdir = SURVEY_CCDS_DIR or paths.survey_dir
    # brick half-diagonal: 0.19 deg for 3600 x 0.262" bricks, more for other brick geometries
    half = max(0.19, 0.5 * np.sqrt(2.) * rs.pixel_scale(bwcs) * max(bwcs.pixel_shape) / 3600.)
    parts = []
    for cam in cameras:
        radius = half + CCD_HALFDIAG_DEG.get(cam, 0.4) + 0.02
        T = survey_ccds_near(sdir, cam, ra0, dec0, radius, band=band)
        if len(T) == 0:
            continue
        shape = (bwcs.pixel_shape[1], bwcs.pixel_shape[0])
        overlaps = np.array([rs.rasterize(rs.footprint_pix(rs.ccd_wcs(c), bwcs), shape, step).any() for c in T])
        parts.append(T[overlaps])
    if not parts:
        return np.zeros(0, dtype=CCD_DTYPE)
    return np.concatenate(parts)


def select_ccds_by_nexp(ccds, ccd_wcses, bwcs, nexp, step=4, margin=EDGE_MARGIN, threshold=OVERCOUNT_THRESHOLD):
    """Keep the CCDs whose summed good-region footprints reproduce the pipeline's ``nexp`` map.

    ``nexp`` counts, per pixel, the CCDs with invvar > 0, so it is lower than the footprint count
    along the CP edge bands (``margin`` removes most of that), at clipped regions
    (``good_subregion``) and at masked pixels. Each candidate's footprint is rasterised on the
    ``step``-subsampled brick grid. Iteratively, the CCD with the largest fraction of its footprint
    over-counted (nexp minus count < 0) is removed while that fraction exceeds ``threshold``:
    an unused CCD is over-counted everywhere (fraction 1.0), a used one only where unused
    candidates or masked pixels overlap it. Returns (keep, count, nexp_ds, agreement, score);
    ``score`` holds the final over-counted fraction for kept CCDs, the fraction at removal for
    dropped ones, and 1.0 for candidates whose good region does not reach into the brick.
    """
    shape = nexp.shape
    masks = [rs.rasterize(ccd_footprint(c, w, bwcs, margin=margin), shape, step) for c, w in zip(ccds, ccd_wcses)]
    nexp_ds = rs.downsample(nexp, step).astype(np.int32)
    count = np.zeros_like(nexp_ds)
    for m in masks:
        count += m
    keep = np.array([m.any() for m in masks])
    score_out = np.zeros(len(masks))
    score_out[~keep] = 1.0     # no good pixel inside the brick: dropped, reported as fully over-counted
    resid = nexp_ds - count

    def scores():
        s = np.zeros(len(masks))
        for i, m in enumerate(masks):
            if keep[i]:
                s[i] = np.mean(resid[m] < 0)
        return s

    s = scores()
    while keep.any() and s.max() > threshold:
        i = int(np.argmax(s))
        keep[i] = False
        score_out[i] = s[i]
        resid += masks[i]
        count -= masks[i]
        s = scores()
    score_out[keep] = s[keep]
    agreement = float(np.mean(count == nexp_ds))
    return keep, count, nexp_ds, agreement, score_out


def ccds_for_brick(paths, band, bwcs=None, nexp_select=True, step=4, margin=EDGE_MARGIN, use_cache=True):
    """CCDs that contributed to the brick's ``band`` coadd, as a canonical table, plus an info dict.

    Bands present in the brick CCD table are taken from it. Other bands (the DR11 i band, which
    is a separate forced pass) are recovered from the survey CCD table and, when the nexp map
    exists, filtered with ``select_ccds_by_nexp``. The selected list is cached in data/ccd_cache
    (``use_cache=False`` reruns the selection and returns its diagnostics, score and keep).
    """
    bwcs = bwcs or brick_wcs(paths)
    T = read_brick_ccds(paths)
    info = {'band': band}
    if np.any(T['filter'] == band):
        T = T[T['filter'] == band]
        info.update(source='brick-table', n_candidates=len(T), n_selected=len(T))
        return T, info
    # flat layouts are keyed by their run directory (different runs share brick and band names)
    sub = paths.region or 'flat-' + os.path.basename(os.path.normpath(paths.survey_dir))
    cache = os.path.join(CCD_CACHE, sub, f'{paths.brick}-{band}.fits')
    if nexp_select and use_cache and os.path.exists(cache):
        T = fitsio.read(cache, ext=1)
        T = to_ccd_table(T)
        h = fitsio.read_header(cache, ext=1)
        info.update(source='survey-table', n_candidates=int(h['NCAND']), n_selected=len(T), agreement=float(h['AGREE']),
                    cached=True)
        return T, info
    T = candidate_ccds(paths, band, bwcs=bwcs)
    info.update(source='survey-table', n_candidates=len(T))
    if nexp_select and len(T) and os.path.exists(paths.coadd('nexp', band)):
        nexp = read_coadd(paths, 'nexp', band)
        keep, count, nexp_ds, agreement, score = select_ccds_by_nexp(T, ccd_wcses(T, exact=True), bwcs, nexp,
                                                                      step=step, margin=margin)
        T = T[keep]
        info.update(agreement=agreement, n_selected=len(T), score=score, keep=keep)
        try:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            tmp = cache + f'.tmp{socket.gethostname()}-{os.getpid()}'
            fitsio.write(tmp, _encode(T), header=dict(NCAND=info['n_candidates'], AGREE=agreement), clobber=True)
            os.replace(tmp, cache)
        except OSError:
            pass
    else:
        info.update(n_selected=len(T))
    return T, info
