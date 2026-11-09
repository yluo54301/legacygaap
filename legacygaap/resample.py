"""TAN WCS helpers (astropy), CCD footprint polygons and rasterisation on the brick grid.

Conventions: CRPIX values are FITS 1-based as stored in headers and CCD tables; every pixel
coordinate passed to or returned by these functions is 0-based (astropy ``origin=0``), matching
numpy array indices and the ``bx``, ``by`` columns of tractor catalogs.
Lanczos-3 resampling of PSF patches is added in milestone M1.
"""
import numpy as np
from astropy.wcs import WCS
from matplotlib.path import Path


def tan_wcs(crval1, crval2, crpix1, crpix2, cd1_1, cd1_2, cd2_1, cd2_2, width, height):
    """TAN WCS from FITS keywords; ``width``/``height`` are the image size in pixels."""
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [float(crval1), float(crval2)]
    w.wcs.crpix = [float(crpix1), float(crpix2)]
    w.wcs.cd = np.array([[cd1_1, cd1_2], [cd2_1, cd2_2]], dtype=float)
    w.pixel_shape = (int(width), int(height))
    return w


def wcs_from_header(hdr):
    """TAN WCS from a (fitsio or astropy) header of a legacypipe coadd (possibly fpack'ed)."""
    width = hdr.get('ZNAXIS1', None) or hdr['NAXIS1']
    height = hdr.get('ZNAXIS2', None) or hdr['NAXIS2']
    return tan_wcs(hdr['CRVAL1'], hdr['CRVAL2'], hdr['CRPIX1'], hdr['CRPIX2'],
                   hdr.get('CD1_1', 0.), hdr.get('CD1_2', 0.), hdr.get('CD2_1', 0.), hdr.get('CD2_2', 0.),
                   width, height)


def ccd_wcs(ccd, apply_offsets=False):
    """TAN WCS of one CCD from a row of a CCD table (same as legacypipe's ``get_approx_wcs``).

    With ``apply_offsets`` the astrometric zero-point offsets ``ccdraoff``/``ccddecoff`` (arcsec)
    are added to CRVAL, as legacypipe's full ``get_wcs`` does.
    """
    crval1, crval2 = float(ccd['crval1']), float(ccd['crval2'])
    if apply_offsets:
        dra, ddec = float(ccd['ccdraoff']) / 3600., float(ccd['ccddecoff']) / 3600.
        crval1 += dra / np.cos(np.deg2rad(crval2))
        crval2 += ddec
    return tan_wcs(crval1, crval2, ccd['crpix1'], ccd['crpix2'],
                   ccd['cd1_1'], ccd['cd1_2'], ccd['cd2_1'], ccd['cd2_2'], ccd['width'], ccd['height'])


WCS_KEY_PREFIXES = ('CTYPE', 'CRVAL', 'CRPIX', 'CD1_', 'CD2_', 'PV1_', 'PV2_', 'CUNIT', 'LONPOLE', 'LATPOLE',
                    'EQUINOX', 'RADESYS', 'A_', 'B_', 'AP_', 'BP_')   # A_/B_/AP_/BP_: SIP (HSC)


def wcs_keys_from_header(hdr):
    """The WCS keywords of an image header (TAN or TPV) plus the image size, as a plain dict."""
    keys = {}
    for k in hdr.keys():
        if isinstance(k, str) and k.startswith(WCS_KEY_PREFIXES):
            keys[k] = hdr[k]
    keys['NAXIS1'] = int(hdr.get('ZNAXIS1', None) or hdr['NAXIS1'])
    keys['NAXIS2'] = int(hdr.get('ZNAXIS2', None) or hdr['NAXIS2'])
    return keys


def wcs_from_keys(keys, dra=0., ddec=0.):
    """astropy WCS (TPV distortion included) from a keyword dict; ``dra``/``ddec`` (deg) shift CRVAL
    the way legacypipe applies the per-CCD astrometric offsets (dra divided by cos(dec))."""
    from astropy.io import fits
    h = fits.Header()
    h['NAXIS'] = 2
    for k, v in keys.items():
        if k not in ('NAXIS1', 'NAXIS2'):
            h[k] = v
    w = WCS(h)
    if dra or ddec:
        crval = w.wcs.crval.copy()
        crval[0] += dra / np.cos(np.deg2rad(crval[1]))
        crval[1] += ddec
        w.wcs.crval = crval
    w.pixel_shape = (int(keys['NAXIS1']), int(keys['NAXIS2']))
    return w


def nominal_pixel_scale(keys):
    """legacypipe's pixel scale of a CCD (``tim.subwcs.pixel_scale()``): astrometry.net's TPV->SIP
    conversion folds the linear TPV terms into CD, so it is 3600 sqrt(|det CD| |det L|) with
    L = [[PV1_1, PV1_2], [PV2_2, PV2_1]]. Verified against legacypipe to 1e-6 arcsec (M4)."""
    cd = np.array([[keys.get('CD1_1', 0.), keys.get('CD1_2', 0.)], [keys.get('CD2_1', 0.), keys.get('CD2_2', 0.)]], dtype=np.float64)
    L = np.array([[keys.get('PV1_1', 1.), keys.get('PV1_2', 0.)], [keys.get('PV2_2', 0.), keys.get('PV2_1', 1.)]], dtype=np.float64)
    return 3600. * np.sqrt(abs(np.linalg.det(cd) * np.linalg.det(L)))


def cd_pixel_scale(keys):
    """Pixel scale of the CD matrix alone (arcsec): older legacypipe versions (IBIS DR1's
    ibis-1.8) use it for the coadd sbscale and psfsize maps; for SIP/TAN WCSs it equals
    ``nominal_pixel_scale``."""
    return 3600. * np.sqrt(abs(float(keys.get('CD1_1', 0.)) * float(keys.get('CD2_2', 0.))
                               - float(keys.get('CD1_2', 0.)) * float(keys.get('CD2_1', 0.))))


def pixel_scale(wcs):
    """Pixel scale in arcsec/pixel."""
    return 3600. * np.sqrt(abs(np.linalg.det(wcs.wcs.cd)))


def footprint_pix(src_wcs, dst_wcs, n_edge=10, margin=0., bounds=None):
    """Boundary of the ``src_wcs`` image as an (N, 2) polygon in ``dst_wcs`` 0-based pixel coords.

    ``margin`` (source pixels) shrinks the footprint on every side (the CP masks an edge band).
    ``bounds`` = (x0, x1, y0, y1) restricts the footprint to a sub-rectangle of the image in
    0-based source pixels (``x1``/``y1`` exclusive; None entries mean the full extent), e.g.
    legacypipe's good image subregion that clips the bad half of DECam chip S7.
    """
    W, H = src_wcs.pixel_shape
    bx0, bx1, by0, by1 = bounds if bounds is not None else (None, None, None, None)
    x0 = -0.5 + margin + (bx0 or 0)
    x1 = (W if bx1 is None else bx1) - 0.5 - margin
    y0 = -0.5 + margin + (by0 or 0)
    y1 = (H if by1 is None else by1) - 0.5 - margin
    t = np.linspace(0., 1., n_edge, endpoint=False)
    xs = np.concatenate([x0 + (x1 - x0) * t, np.full(n_edge, x1), x1 - (x1 - x0) * t, np.full(n_edge, x0)])
    ys = np.concatenate([np.full(n_edge, y0), y0 + (y1 - y0) * t, np.full(n_edge, y1), y1 - (y1 - y0) * t])
    ra, dec = src_wcs.all_pix2world(xs, ys, 0)
    px, py = dst_wcs.wcs_world2pix(ra, dec, 0)   # destination is always a plain TAN
    return np.column_stack([px, py])


def sample_grid(shape, step=1):
    """0-based pixel coordinates (ys, xs) of the sample points of a ``step``-subsampled grid."""
    H, W = shape
    ys = step * np.arange(int(np.ceil(H / step))) + step // 2
    xs = step * np.arange(int(np.ceil(W / step))) + step // 2
    return ys[ys < H], xs[xs < W]


def downsample(arr, step=1):
    """Values of ``arr`` at the ``sample_grid`` points (plain subsampling, no averaging)."""
    ys, xs = sample_grid(arr.shape, step)
    return arr[ys[:, None], xs[None, :]]


def rasterize(poly, shape, step=1):
    """Boolean mask on the ``step``-subsampled grid of ``shape`` for points inside ``poly``."""
    ys, xs = sample_grid(shape, step)
    mask = np.zeros((len(ys), len(xs)), dtype=bool)
    xmin, ymin = poly.min(axis=0)
    xmax, ymax = poly.max(axis=0)
    ix = np.flatnonzero((xs >= xmin - step) & (xs <= xmax + step))
    iy = np.flatnonzero((ys >= ymin - step) & (ys <= ymax + step))
    if len(ix) == 0 or len(iy) == 0:
        return mask
    X, Y = np.meshgrid(xs[ix], ys[iy])
    inside = Path(poly).contains_points(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    mask[np.ix_(iy, ix)] = inside
    return mask


# ----------------------------------------------------------------------------- Lanczos resampling

def lanczos_filter(order, x):
    """Lanczos kernel L(x) = sinc(x) sinc(x/order) for |x| < order (astrometry.net's lanczos_filter)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)
    nz = (x != 0.) & (np.abs(x) < order)
    px = np.pi * x[nz]
    out[nz] = order * np.sin(px) * np.sin(px / order) / px ** 2
    out[x == 0.] = 1.
    return out


def lanczos_weights(order, d):
    """L(k - d) for k = -order..order and an array of offsets d: shape (N, 2*order+1).

    Same values as ``lanczos_filter(order, k - d)`` but with three transcendental calls per
    offset instead of 4*order+2, using sin(pi (k-d)) = -(-1)^k sin(pi d) and the angle-addition
    formula for sin(pi (k-d) / order)."""
    d = np.asarray(d, dtype=np.float64).ravel()
    ks = np.arange(-order, order + 1)
    x = ks[None, :] - d[:, None]
    sin_pix = -((-1.) ** ks)[None, :] * np.sin(np.pi * d)[:, None]
    a = np.pi * d / order
    sin_pixl = (np.sin(np.pi * ks / order)[None, :] * np.cos(a)[:, None]
                - np.cos(np.pi * ks / order)[None, :] * np.sin(a)[:, None])
    with np.errstate(divide='ignore', invalid='ignore'):
        out = order * sin_pix * sin_pixl / (np.pi ** 2 * x * x)
    out[x == 0.] = 1.
    out[np.abs(x) >= order] = 0.
    return out


def lanczos_shift(img, dx, dy, order=3):
    """Shift an image by (+dx, +dy) pixels with separable, flux-normalised Lanczos kernels
    (tractor's ``lanczos_shift_image``). Pixels outside the image count as zero."""
    from scipy.ndimage import correlate1d
    L = order
    Lx = lanczos_filter(L, np.arange(-L, L + 1) + dx)
    Ly = lanczos_filter(L, np.arange(-L, L + 1) + dy)
    Lx /= Lx.sum()
    Ly /= Ly.sum()
    sx = correlate1d(np.asarray(img, dtype=np.float64), Lx, axis=1, mode='constant')
    return correlate1d(sx, Ly, axis=0, mode='constant')


def lanczos_interpolate(img, fx, fy, order=3, normalize=True, edge='clamp'):
    """Values of ``img`` at float 0-based pixel positions (fx, fy) by Lanczos interpolation.

    Follows astrometry.net's ``resample_with_wcs``: the reference pixel is floor(f + 0.5), the
    kernel runs over ``order`` neighbours on each side, the 2-d kernel is normalised to unit sum
    (``normalize``), and neighbours outside the image take the nearest edge pixel
    (``edge='clamp'``) or zero (``edge='zero'``). Vectorised as one gather of
    (N, 2L+1, 2L+1) values. Returns an array shaped like ``fx``.
    """
    img = np.asarray(img, dtype=np.float64)
    H, W = img.shape
    shape = np.shape(fx)
    fx = np.asarray(fx, dtype=np.float64).ravel()
    fy = np.asarray(fy, dtype=np.float64).ravel()
    ix = np.floor(fx + 0.5).astype(np.int64)
    iy = np.floor(fy + 0.5).astype(np.int64)
    dx = (fx - ix).astype(np.float32).astype(np.float64)   # astrometry.net passes float32 offsets
    dy = (fy - iy).astype(np.float32).astype(np.float64)
    ks = np.arange(-order, order + 1)
    wx = lanczos_weights(order, dx)          # (N, 2L+1): neighbour ix+k sits at k - dx from the point
    wy = lanczos_weights(order, dy)
    xi = ix[:, None] + ks[None, :]
    yi = iy[:, None] + ks[None, :]
    if edge == 'clamp':
        vals = img[np.clip(yi, 0, H - 1)[:, :, None], np.clip(xi, 0, W - 1)[:, None, :]]
        out = np.einsum('nb,nba,na->n', wy, vals, wx)
        wsum = wy.sum(axis=1) * wx.sum(axis=1)
    else:
        okx = (xi >= 0) & (xi < W)
        oky = (yi >= 0) & (yi < H)
        ok = (oky[:, :, None] & okx[:, None, :]).astype(np.float64)
        vals = img[np.clip(yi, 0, H - 1)[:, :, None], np.clip(xi, 0, W - 1)[:, None, :]] * ok
        out = np.einsum('nb,nba,na->n', wy, vals, wx)
        wsum = np.einsum('nb,nba,na->n', wy, ok, wx)
    if normalize:
        nz = wsum != 0
        out[nz] /= wsum[nz]
    return out.reshape(shape)


def map_pixels(dst_wcs, xs, ys, src_wcs):
    """Destination 0-based pixel coordinates -> source 0-based pixel coordinates (TPV handled)."""
    ra, dec = dst_wcs.all_pix2world(xs, ys, 0)
    return src_wcs.all_world2pix(ra, dec, 0)


def resample_to_grid(img, x0, y0, src_wcs, dst_wcs, dst_x0, dst_y0, dst_h, dst_w, order=3, normalize=True,
                     edge='clamp'):
    """Lanczos-resample ``img``, whose pixel (0, 0) sits at source pixel (x0, y0) of ``src_wcs``,
    onto the destination pixels [dst_y0:dst_y0+dst_h, dst_x0:dst_x0+dst_w] of ``dst_wcs``."""
    ys, xs = np.mgrid[dst_y0:dst_y0 + dst_h, dst_x0:dst_x0 + dst_w]
    fx, fy = map_pixels(dst_wcs, xs.ravel().astype(float), ys.ravel().astype(float), src_wcs)
    vals = lanczos_interpolate(img, fx - x0, fy - y0, order=order, normalize=normalize, edge=edge)
    return vals.reshape(dst_h, dst_w)


class CcdMap:
    """Cubic-spline approximation of the brick-pixel -> CCD-pixel mapping over the part of the
    brick (plus ``pad`` pixels) that the CCD covers, with knots every ``step`` brick pixels.
    astrometry.net's ``resample_with_wcs``, which made the coadds, interpolates its pixel
    mapping with splines on 25-px steps in the same way; the spline error is < 1e-4 px (test)."""

    def __init__(self, src_wcs, dst_wcs, step=32, pad=96, bounds=None):
        from scipy.interpolate import RectBivariateSpline
        W, H = dst_wcs.pixel_shape
        poly = footprint_pix(src_wcs, dst_wcs, bounds=bounds)
        x0 = max(np.floor(poly[:, 0].min()) - pad, -pad)
        x1 = min(np.ceil(poly[:, 0].max()) + pad, W - 1 + pad)
        y0 = max(np.floor(poly[:, 1].min()) - pad, -pad)
        y1 = min(np.ceil(poly[:, 1].max()) + pad, H - 1 + pad)
        self.empty = (x1 <= x0) or (y1 <= y0)
        if self.empty:
            self.bbox = (np.inf, -np.inf, np.inf, -np.inf)
            return
        nx = max(int(np.ceil((x1 - x0) / step)) + 1, 4)
        ny = max(int(np.ceil((y1 - y0) / step)) + 1, 4)
        xs = x0 + step * np.arange(nx, dtype=np.float64)
        ys = y0 + step * np.arange(ny, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys)
        fx, fy = map_pixels(dst_wcs, X.ravel(), Y.ravel(), src_wcs)
        self.sx = RectBivariateSpline(ys, xs, np.asarray(fx).reshape(ny, nx), kx=3, ky=3)
        self.sy = RectBivariateSpline(ys, xs, np.asarray(fy).reshape(ny, nx), kx=3, ky=3)
        self.bbox = (xs[0], xs[-1], ys[0], ys[-1])

    def contains(self, x, y):
        b = self.bbox
        return (b[0] <= x <= b[1]) and (b[2] <= y <= b[3])

    def at(self, x, y):
        """Source (CCD) pixel coordinates of destination (brick) pixel (x, y)."""
        return float(self.sx.ev(y, x)), float(self.sy.ev(y, x))

    def grid(self, xs, ys):
        """Source coordinates on the destination grid xs (1-d) x ys (1-d): arrays (len(ys), len(xs))."""
        return self.sx(ys, xs), self.sy(ys, xs)
