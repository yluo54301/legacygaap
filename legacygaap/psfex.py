"""PsfEx model reader and evaluator (numpy only).

Reads the merged PsfEx tables that legacypipe writes per exposure (one row per CCD), the PsfEx
models the LSST stack stores inside HSC CALEXP/CORR files, and the cell PSFs of Rubin deep coadds
(``CellPsfModel``), and evaluates the spatially varying
PSF exactly as tractor's ``PsfExModel.at`` does: a polynomial in the CCD pixel coordinates (terms
ordered 1, x, x^2, y, xy, y^2 for degree 2, PsfEx convention) applied to the basis images.
Coordinates are tractor-style 0-based pixel coordinates; like tractor, no correction is made for
PsfEx's 1-based ``X_IMAGE`` zero points. Oversampled models (HSC: sampling 0.5) are sampled at the
image pixel centres by ``PsfExModel.native``.
"""
import numpy as np
import fitsio

_TABLE_CACHE = {}


class PsfExModel:
    """One CCD's PsfEx model: basis images and the spatial polynomial."""

    def __init__(self, bases, degree, polzero, polscal, sampling=1., fwhm=None, meta=None):
        bases = np.asarray(bases, dtype=np.float32)
        ne = (degree + 1) * (degree + 2) // 2
        assert bases.shape[0] == ne, (bases.shape, ne)
        self.bases = bases
        self.degree = int(degree)
        self.x0, self.y0 = float(polzero[0]), float(polzero[1])
        self.xscale, self.yscale = float(polscal[0]), float(polscal[1])
        self.sampling = float(sampling)
        self.fwhm = fwhm
        self.meta = meta or {}
        self._powers = self._term_powers(self.degree)
        self._b64 = self.bases.astype(np.float64).reshape(ne, -1)

    @staticmethod
    def _term_powers(degree):
        """(x power, y power) of each polynomial term in PsfEx/tractor order."""
        ne = (degree + 1) * (degree + 2) // 2
        xp, yp = np.zeros(ne, int), np.zeros(ne, int)
        for d in range(degree + 1):
            for j in range(d + 1):
                k = d - j
                ii = j + (degree + 1) * k - (k * (k - 1)) // 2
                xp[ii], yp[ii] = j, k
        return xp, yp

    @property
    def shape(self):
        return self.bases.shape[1:]

    def terms(self, x, y):
        dx = (x - self.x0) / self.xscale
        dy = (y - self.y0) / self.yscale
        xp, yp = self._powers
        return dx ** xp * dy ** yp

    def at(self, x, y, normalize=True):
        """PSF image (H, W) at CCD pixel position (x, y), unit sum if ``normalize``. For an
        oversampled model (sampling != 1) this is on the model's own grid; see ``native``."""
        t = self.terms(float(x), float(y))
        # einsum, not matmul: a tiny multithreaded-BLAS call costs ~3 ms of thread wake-up when
        # interleaved with other numpy work (M4 profiling)
        img = np.einsum('k,kn->n', t.astype(np.float64), self._b64).reshape(self.shape)
        if normalize:
            s = img.sum()
            if s != 0:
                img /= s
        return img

    def native(self, x, y, dx=0., dy=0., order=3):
        """Unit-sum PSF image on the CCD's own pixel grid at (x, y), centred on the central pixel
        shifted by (dx, dy) image pixels: the image tractor renders for a point source at
        (round(x) + dx, round(y) + dy). Native models are Lanczos-shifted (``resample.lanczos_shift``);
        oversampled ones (HSC: sampling 0.5) are Lanczos-3 sampled at the image pixel centres, as
        tractor's ``PixelizedPSF._sampleImage`` does. Size: ceil(model size x sampling)."""
        from . import resample as rs
        img = self.at(x, y, normalize=False)
        if self.sampling == 1.:
            out = rs.lanczos_shift(img, dx, dy, order=order) if (dx or dy) else img
        else:
            H, W = img.shape
            s = self.sampling
            nw, nh = int(np.ceil(W * s)), int(np.ceil(H * s))
            u = np.arange(nw) - nw // 2
            v = np.arange(nh) - nh // 2
            X = (W // 2) + (u[None, :] - dx) / s + np.zeros((nh, 1))
            Y = (H // 2) + (v[:, None] - dy) / s + np.zeros((1, nw))
            out = rs.lanczos_interpolate(img, X, Y, order=order, normalize=True, edge='zero')
        tot = out.sum()
        return out / tot if tot != 0 else out

    @classmethod
    def from_row(cls, row):
        """Build from one row of a merged PsfEx table (fitsio structured row)."""
        degree = int(row['poldeg1'])
        ne = (degree + 1) * (degree + 2) // 2
        nx, ny = int(row['psfaxis1']), int(row['psfaxis2'])
        bases = np.asarray(row['psf_mask'])[:ne, :ny, :nx]
        sampling = float(row['psf_samp']) or 1.0
        meta = {k: row[k] for k in ('expnum', 'ccdname', 'plver', 'legpipev', 'moffat_alpha', 'moffat_beta')
                if k in row.dtype.names}
        return cls(bases, degree, (row['polzero1'], row['polzero2']), (row['polscal1'], row['polscal2']),
                   sampling=sampling, fwhm=float(row['psf_fwhm']), meta=meta)


def read_merged_psfex(fn):
    """The merged PsfEx table of one exposure (cached per process)."""
    if fn not in _TABLE_CACHE:
        _TABLE_CACHE[fn] = fitsio.read(fn, ext=1)
    return _TABLE_CACHE[fn]


def psfex_row(table, expnum, ccdname):
    names = np.char.strip(table['ccdname'].astype(str))
    I = np.flatnonzero((table['expnum'] == int(expnum)) & (names == str(ccdname).strip()))
    if len(I) != 1:
        raise KeyError(f'{len(I)} rows for expnum {expnum} ccd {ccdname}')
    return table[I[0]]


def psfex_for_ccd(fn, expnum, ccdname):
    """PsfExModel for a CCD from the merged PsfEx file ``fn`` (clears cache entries lazily)."""
    row = psfex_row(read_merged_psfex(fn), expnum, ccdname)
    if 'failure' in row.dtype.names and row['failure']:
        raise RuntimeError(f'PsfEx fit flagged as failed for expnum {expnum} ccd {ccdname} in {fn}')
    return PsfExModel.from_row(row)


def hsc_psfex(fn):
    """PsfEx model stored in an HSC CALEXP/CORR file by the LSST stack: two BINTABLE HDUs with
    AR_NAME = 'PsfexPsf' (``_pixstep`` = sampling, ``_comp`` = basis images of shape ``_size``
    reversed, ``degree``, ``_context_first``/``_context_second`` = polynomial zero point and
    scale), read as legacypipe's ``HscImage.read_psf_model`` does. The persisted ``coeff`` array is
    PsfEx's last polynomial-fit scratch vector and is not applied (legacypipe multiplies
    ``ims[:, :, i]`` by it, i.e. image columns 0-5 of every basis, not the bases themselves)."""
    F = fitsio.FITS(fn)
    tabs = [i for i in range(1, len(F)) if F[i].read_header().get('AR_NAME') == 'PsfexPsf']
    if len(tabs) != 2:
        raise RuntimeError(f'{len(tabs)} PsfexPsf tables in {fn}')
    t1, t2 = F[tabs[0]].read()[0], F[tabs[1]].read()[0]
    size = [int(v) for v in t2['_size']]
    degree = int(t2['degree'])
    ne = (degree + 1) * (degree + 2) // 2
    assert size[2] == ne, (size, degree)
    bases = np.asarray(t2['_comp'], dtype=np.float64).reshape(list(reversed(size)))
    return PsfExModel(bases, degree, t2['_context_first'], t2['_context_second'], sampling=float(t1['_pixstep']),
                      meta={'file': fn})


class CellPsfModel:
    """Rubin/LSST deep-coadd PSF: one pixelized PSF per 150 x 150 pixel cell of the patch (the PSF
    HDU, a (22, 22, 35, 35) cube [cell y, cell x, y, x] of unit-sum images on the patch pixels),
    constant within a cell, as legacypipe's ``lsst.PiecewiseConstantPixelizedPsf``: the cell is
    floor(pixel / 150), clipped to the grid; a cell with NaNs uses the mean of the finite cells."""

    sampling = 1.

    def __init__(self, cube, cell=150, meta=None):
        cube = np.asarray(cube, dtype=np.float64)
        assert cube.ndim == 4, cube.shape
        self.cube = cube
        self.cell = int(cell)
        self.gh, self.gw = cube.shape[:2]
        good = np.all(np.isfinite(cube), axis=(2, 3))
        assert good.any(), 'no finite PSF cell'
        self.good = good
        self.avg = cube[good].mean(axis=0)
        self.meta = meta or {}
        self.fwhm = None

    @property
    def shape(self):
        return self.cube.shape[2:]

    def cell_of(self, x, y):
        cx = min(max(int(np.floor(float(x) / self.cell)), 0), self.gw - 1)
        cy = min(max(int(np.floor(float(y) / self.cell)), 0), self.gh - 1)
        return cy, cx

    def at(self, x, y, normalize=True):
        cy, cx = self.cell_of(x, y)
        img = (self.cube[cy, cx] if self.good[cy, cx] else self.avg).copy()
        if normalize:
            s = img.sum()
            if s != 0:
                img /= s
        return img

    def native(self, x, y, dx=0., dy=0., order=3):
        """Unit-sum PSF at patch pixel (x, y), Lanczos-shifted by (dx, dy) (cf. PsfExModel.native)."""
        from . import resample as rs
        img = self.at(x, y, normalize=False)
        out = rs.lanczos_shift(img, dx, dy, order=order) if (dx or dy) else img
        tot = out.sum()
        return out / tot if tot != 0 else out


def lsst_cell_psf(fn):
    """CellPsfModel from the PSF HDU of a Rubin deep-coadd file (4-d tile-compressed: fitsio cannot
    read it, astropy can)."""
    from astropy.io import fits
    with fits.open(fn, memmap=False) as f:
        cube = np.array(f['PSF'].data, dtype=np.float64)
    return CellPsfModel(cube, meta={'file': fn})


def neff_pixels(psf):
    """Noise-equivalent area in pixels of a unit-sum PSF image: 1 / sum(psf^2)."""
    return 1. / np.sum(np.asarray(psf, dtype=np.float64) ** 2)


def neff_fwhm(psf, pixscale):
    """FWHM (arcsec) of the Gaussian with the same noise-equivalent area as ``psf``
    (the definition behind legacypipe's psfsize maps)."""
    sigma_pix = np.sqrt(neff_pixels(psf) / (4. * np.pi))
    return 2. * np.sqrt(2. * np.log(2.)) * sigma_pix * pixscale
