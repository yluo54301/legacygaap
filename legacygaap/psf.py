"""Position-dependent coadd PSF reconstructed from the per-CCD PsfEx models.

A legacypipe coadd is the inverse-variance-weighted sum of the Lanczos-3-resampled CCD images, so
its PSF at a position is the same weighted sum of the per-CCD PSFs, each evaluated at that position
on its CCD, shifted to the sub-pixel position of the source on the CCD grid (as tractor renders a
point source) and resampled onto the brick grid. Weights are 1/sig1^2 per CCD, the constant
weights legacypipe uses for its psfsize map and brick-averaged PSF.
"""
import os
import numpy as np
from dataclasses import dataclass, field
from . import io, resample as rs, psfex


@dataclass
class PsfStamp:
    stamp: np.ndarray            # unit-sum PSF on the brick grid, centred on (x, y)
    x0: int                      # brick pixel coordinates of stamp pixel (0, 0)
    y0: int
    x: float                     # brick pixel position the PSF is centred on
    y: float
    band: str
    ccd_idx: np.ndarray          # indices into the band's CCD table
    weights: np.ndarray          # normalised 1/sig1^2 weights
    fwhm_ccd: np.ndarray         # per-CCD PsfEx FWHM (arcsec)
    psfsize_pipeline: float      # legacypipe's psfsize at this point, recomputed from our CCD list (arcsec)
    neff_fwhm: float             # FWHM of the Gaussian with the mixture's noise-equivalent area (arcsec)
    coadd_stamp: np.ndarray = None   # coadd image of a unit-flux star (legacypipe flux scaling; sum = flux_scale)
    flux_scale: float = np.nan   # coadd flux of a unit-flux star (legacypipe sbscale x local pixel area)
    ccd_weight: np.ndarray = None    # coadd weight of each CCD: (1/sig1^2) / sbscale^2
    sbscale: np.ndarray = None   # legacypipe sbscale of each CCD
    var_scale: float = np.nan    # true / invvar-predicted variance of smooth aperture sums: weighted mean
                                 # CCD-to-coadd pixel-area ratio (3.0 for BASS 0.454" pixels, ~1 for DECam)
    nexp_at: int = -1            # pipeline nexp map value at the centre pixel (-1 if unknown)
    flag_nexp: bool = False      # CCD count outside the nexp range within +-2 px by > 10% of the count
    flag_edge: bool = False      # stamp extends beyond the brick
    parts: list = field(default_factory=list)   # per-CCD resampled stamps (unit-sum models), if requested

    @property
    def nccd(self):
        return len(self.ccd_idx)


class CoaddPsfModel:
    """PSF of a brick's coadds at any position, per band."""

    def __init__(self, bwcs, ccds_by_band, wcs_by_band, psfex_loader, stamp=65, margin=io.EDGE_MARGIN,
                 nexp_by_band=None, order=3, normalize=True, edge='clamp', mapping='spline', sbscale_by_band=None,
                 nominal_scale_by_band=None):
        assert stamp % 2 == 1
        assert mapping in ('spline', 'exact')
        self.mapping = mapping
        self._maps = {}
        self.bwcs = bwcs
        self.ccds = ccds_by_band
        self.wcs = wcs_by_band
        self._loader = psfex_loader
        self._psfex = {}
        self.stamp = int(stamp)
        self.margin = float(margin)
        self._nexp = nexp_by_band or {}
        self.order = order
        self.normalize = normalize
        self.edge = edge
        self.pixscale = rs.pixel_scale(bwcs)
        # legacypipe's nominal CCD pixel scales (CD x linear TPV terms) and coadd sbscale factors;
        # without header keywords (synthetic CCDs) the CD scale and sbscale = 1
        self._ccd_scale = nominal_scale_by_band or {b: np.array([rs.pixel_scale(w) for w in ws]) for b, ws in wcs_by_band.items()}
        self.sbscale = sbscale_by_band or {b: np.ones(len(ws)) for b, ws in wcs_by_band.items()}
        self._norm_cache = {}
        self._masks = {}
        self._nexp_rng = {}
        self._scales = {'nominal': {b: np.array(v) for b, v in self._ccd_scale.items()}}
        self.use_masks = True
        self.mask_threshold = 0.5      # a CCD is dropped where more than this fraction of its bin is masked

    # ------------------------------------------------------------------ construction

    @classmethod
    def for_brick(cls, paths, bands=None, stamp=65, margin=io.EDGE_MARGIN, load_nexp=True, bwcs=None, **kw):
        """Build from the brick's products: CCD lists (M0), exact CCD WCSs and PsfEx files."""
        bwcs = bwcs or io.brick_wcs(paths)
        bands = bands or paths.bands()
        ccds, wcs, nexp = {}, {}, {}
        for band in bands:
            T, info = io.ccds_for_brick(paths, band, bwcs=bwcs, margin=margin)
            ccds[band] = T
            wcs[band] = io.ccd_wcses(T)
            if load_nexp:
                nexp[band] = io.read_coadd(paths, 'nexp', band)

        def loader(ccd):
            if str(ccd['camera']).strip() == 'hsc':     # PsfEx model inside the CALEXP/CORR file
                return psfex.hsc_psfex(os.path.join(io.IMAGES_ROOT, str(ccd['image_filename']).strip()))
            if str(ccd['camera']).strip() == 'lsstcoadd':   # cell PSFs inside the Rubin deep-coadd file
                return psfex.lsst_cell_psf(os.path.join(io.IMAGES_ROOT, str(ccd['image_filename']).strip()))
            return psfex.psfex_for_ccd(io.psfex_path(paths.survey_dir, ccd['image_filename']),
                                       ccd['expnum'], ccd['ccdname'])
        target = rs.pixel_scale(bwcs)
        nominal = {b: np.array([rs.nominal_pixel_scale(io.ccd_wcs_keys(c)) for c in ccds[b]]) for b in bands}
        sbscale = {b: (target / nominal[b]) ** 2 for b in bands}
        model = cls(bwcs, ccds, wcs, loader, stamp=stamp, margin=margin, nexp_by_band=nexp, sbscale_by_band=sbscale,
                    nominal_scale_by_band=nominal, **kw)
        model.paths = paths
        model._scales = {'nominal': nominal,
                         'cd': {b: np.array([rs.cd_pixel_scale(io.ccd_wcs_keys(c)) for c in ccds[b]]) for b in bands}}
        return model

    # legacypipe's CCD pixel scale (used for the coadd sbscale and the psfsize maps) depends on its
    # version: DR11 main folds the linear TPV terms into the CD matrix (astrometry.net pv2sip;
    # 'nominal'), the older ibis-1.8 branch of IBIS DR1 uses the CD matrix alone ('cd'): coadd
    # flux scale 1.004 instead of 0.982, psfsize maps 1.1% smaller (docs/method.md).
    pixscale_convention = 'nominal'

    def set_pixscale_convention(self, conv):
        assert conv in self._scales, conv
        for b in self.bands:
            self._ccd_scale[b] = self._scales[conv][b]
            self.sbscale[b] = (self.pixscale / self._scales[conv][b]) ** 2
        self.pixscale_convention = conv

    def detect_pixscale_convention(self, band=None, npos=40, seed=0, paths=None):
        """Pick the pixel-scale convention whose psfsize formula reproduces the pipeline's psfsize
        map (positions where our CCD count equals nexp). Returns (convention, {convention: median
        ratio}); keeps the current convention if there is no psfsize map or no usable position."""
        paths = paths or getattr(self, 'paths', None)
        band = band or self.bands[0]
        try:
            psz = io.read_coadd(paths, 'psfsize', band)
        except Exception:
            return self.pixscale_convention, {}
        W, H = self.bwcs.pixel_shape
        rng = np.random.default_rng(seed)
        pos = np.column_stack([rng.uniform(100, W - 100, npos), rng.uniform(100, H - 100, npos)])
        ratios = {}
        for conv in self._scales:
            self.set_pixscale_convention(conv)
            r = []
            for x, y in pos:
                ps = self.psf_at(x, y, band)
                v = psz[int(round(y)), int(round(x))]
                if ps.nexp_at == ps.nccd and ps.nccd > 0 and v > 0:
                    r.append(ps.psfsize_pipeline / v)
            ratios[conv] = float(np.median(r)) if r else np.nan
        good = {c: abs(v - 1) for c, v in ratios.items() if np.isfinite(v)}
        best = min(good, key=good.get) if good else 'nominal'
        self.set_pixscale_convention(best)
        return best, ratios

    @property
    def bands(self):
        return list(self.ccds.keys())

    def psfex_model(self, band, i):
        key = (band, int(i))
        if key not in self._psfex:
            self._psfex[key] = self._loader(self.ccds[band][i])
        return self._psfex[key]

    def ccd_maps(self, band):
        """Spline maps (brick pixel -> CCD pixel) for every CCD of the band, built on first use."""
        if band not in self._maps:
            self._maps[band] = [rs.CcdMap(w, self.bwcs, bounds=io.ccd_bounds(c))
                                for c, w in zip(self.ccds[band], self.wcs[band])]
        return self._maps[band]

    NEXP_HALO = 2
    NEXP_TOL = 0.1

    def _nexp_range(self, band):
        """Minimum and maximum of the nexp map within +-NEXP_HALO pixels (cached per band)."""
        if band not in self._nexp_rng:
            from scipy.ndimage import minimum_filter, maximum_filter
            nm = self._nexp[band]
            k = 2 * self.NEXP_HALO + 1
            self._nexp_rng[band] = (minimum_filter(nm, size=k), maximum_filter(nm, size=k))
        return self._nexp_rng[band]

    def mask_map(self, band, i):
        """Binned masked-pixel fraction of a CCD (io.ccd_mask_map; None if not used)."""
        key = (band, int(i))
        if key not in self._masks:
            ccd = self.ccds[band][i]
            self._masks[key] = io.ccd_mask_map(ccd) if (self.use_masks and str(ccd['camera']).strip() in io.MASK_CAMERAS) else None
        return self._masks[key]

    def preload(self, bands=None, clear_table_cache=True):
        """Build every spline map and load every PsfEx model now (call before forking workers, so
        they inherit them instead of re-reading the calibration files)."""
        for band in bands or self.bands:
            self.ccd_maps(band)
            for i in range(len(self.ccds[band])):
                self.psfex_model(band, i)
                self._ccd_psfnorm(band, i)
                self.mask_map(band, i)
            if band in self._nexp:
                self._nexp_range(band)
        if clear_table_cache:
            psfex._TABLE_CACHE.clear()

    def _in_good_region(self, ccd, px, py):
        """Whether CCD pixel (px, py) contributes to the coadd: inside legacypipe's good subregion
        less the masked edge band (``margin``; per camera, io.EDGE_MARGIN_CAMERA), and inside the
        recorded tim box (a hard limit: interior box edges are brick-boundary cuts, not masks)."""
        x0, x1, y0, y1 = io.good_subregion(ccd)
        m = io.EDGE_MARGIN_CAMERA.get(str(ccd['camera']).strip(), self.margin)
        lo_x, hi_x = (x0 or 0) + m, (int(ccd['width']) if x1 is None else x1) - m
        lo_y, hi_y = (y0 or 0) + m, (int(ccd['height']) if y1 is None else y1) - m
        if 'ccd_x1' in ccd.dtype.names and int(ccd['ccd_x1']) > 0:
            lo_x, hi_x = max(lo_x, int(ccd['ccd_x0'])), min(hi_x, int(ccd['ccd_x1']))
            lo_y, hi_y = max(lo_y, int(ccd['ccd_y0'])), min(hi_y, int(ccd['ccd_y1']))
        return lo_x - 0.5 <= px < hi_x - 0.5 and lo_y - 0.5 <= py < hi_y - 0.5

    def _masked(self, band, i, px, py):
        """True where the CCD's bin around pixel (px, py) is mostly masked (bad amplifiers etc.)."""
        mm = self.mask_map(band, i)
        if mm is None:
            return False
        iy, ix = int(round(float(py))) // io.MASK_BIN, int(round(float(px))) // io.MASK_BIN
        if not (0 <= iy < mm.shape[0] and 0 <= ix < mm.shape[1]):
            return True
        return float(mm[iy, ix]) > self.mask_threshold

    def covering_fast(self, band, x, y):
        """Like ``ccds_covering`` but from the spline maps and brick pixel coordinates (x, y).
        Returns (indices, {index: (ccd x, ccd y)})."""
        idx, pos = [], {}
        for i, m in enumerate(self.ccd_maps(band)):
            if m.empty or not m.contains(x, y):
                continue
            px, py = m.at(x, y)
            if self._in_good_region(self.ccds[band][i], px, py) and not self._masked(band, i, px, py):
                idx.append(i)
                pos[i] = (px, py)
        return np.array(idx, dtype=int), pos

    # ------------------------------------------------------------------ geometry

    def ccd_positions(self, band, ra, dec):
        """Position (ra, dec) in 0-based pixel coordinates of every CCD of ``band``."""
        xs, ys = np.zeros(len(self.wcs[band])), np.zeros(len(self.wcs[band]))
        for i, w in enumerate(self.wcs[band]):
            xs[i], ys[i] = w.all_world2pix(ra, dec, 0)
        return xs, ys

    def ccds_covering(self, band, ra, dec):
        """Indices of the CCDs whose good region (edge margin removed) contains the position."""
        xs, ys = self.ccd_positions(band, ra, dec)
        idx = [i for i, ccd in enumerate(self.ccds[band])
               if self._in_good_region(ccd, xs[i], ys[i]) and not self._masked(band, i, xs[i], ys[i])]
        return np.array(idx, dtype=int), xs, ys

    # ------------------------------------------------------------------ PSF

    def _ccd_psfnorm(self, band, i):
        """sqrt(sum psf^2) of the CCD's model at the centre of its overlap with the brick, which is
        where legacypipe evaluates the per-CCD psfnorm (the centre of the sub-image it reads)."""
        key = (band, int(i))
        if key not in self._norm_cache:
            ccd = self.ccds[band][i]
            x0, x1, y0, y1 = io.ccd_brick_box(ccd, self.wcs[band][i], self.bwcs)
            if x1 <= x0 or y1 <= y0:
                x0, x1, y0, y1 = 0, int(ccd['width']), 0, int(ccd['height'])
            m = self.psfex_model(band, i).native((x0 + x1) // 2, (y0 + y1) // 2)
            self._norm_cache[key] = float(np.sqrt(np.sum(m ** 2)))
        return self._norm_cache[key]

    def psf_at(self, x, y, band, return_parts=False):
        """Coadd PSF centred on brick pixel position (x, y) (0-based, fractional) in ``band``."""
        S = self.stamp
        R = S // 2
        cx, cy = int(round(float(x))), int(round(float(y)))
        sx0, sy0 = cx - R, cy - R
        T = self.ccds[band]
        if self.mapping == 'spline':
            idx, pos = self.covering_fast(band, float(x), float(y))
            xs = {i: pos[i][0] for i in idx}
            ys = {i: pos[i][1] for i in idx}
            gx = np.arange(sx0, sx0 + S, dtype=np.float64)
            gy = np.arange(sy0, sy0 + S, dtype=np.float64)
        else:
            ra, dec = self.bwcs.all_pix2world(float(x), float(y), 0)
            idx, xs, ys = self.ccds_covering(band, ra, dec)
        acc = np.zeros((S, S), dtype=np.float64)
        weights, cweights, sbs, parts, fwhms, inv_narcsec, areas = [], [], [], [], [], [], []
        for i in idx:
            model = self.psfex_model(band, i)
            ixc, iyc = round(float(xs[i])), round(float(ys[i]))
            if model.sampling == 1.:
                img = model.at(xs[i], ys[i])                 # unit sum, centred on its central pixel
                shifted = rs.lanczos_shift(img, xs[i] - ixc, ys[i] - iyc, order=self.order)
            else:                                            # oversampled (HSC): sampled at the pixel centres
                shifted = model.native(xs[i], ys[i], xs[i] - ixc, ys[i] - iyc, order=self.order)
            h, w = shifted.shape
            if self.mapping == 'spline':
                fxg, fyg = self.ccd_maps(band)[i].grid(gx, gy)
                part = rs.lanczos_interpolate(shifted, fxg - (ixc - w // 2), fyg - (iyc - h // 2), order=self.order,
                                              normalize=self.normalize, edge=self.edge)
            else:
                part = rs.resample_to_grid(shifted, ixc - w // 2, iyc - h // 2, self.wcs[band][i], self.bwcs,
                                           sx0, sy0, S, S, order=self.order, normalize=self.normalize, edge=self.edge)
            # legacypipe's coadd: image x sbscale, inverse variance / sbscale^2 (coadds.py, sbscale=True)
            f = float(self.sbscale[band][i])
            wgt = 1. / float(T['sig1'][i]) ** 2
            cw = wgt / f ** 2
            acc += cw * f * part
            areas.append(float(part.sum()))          # local CCD-to-coadd pixel-area ratio
            weights.append(wgt)
            cweights.append(cw)
            sbs.append(f)
            pixscale_i = self._ccd_scale[band][i]
            fwhms.append(float(T['fwhm'][i]) * pixscale_i)
            inv_narcsec.append(self._ccd_psfnorm(band, i) ** 2 / pixscale_i ** 2)   # 1 / (neff * pixscale^2)
            if return_parts:
                parts.append(part)
        weights = np.array(weights)
        cweights = np.array(cweights)
        total = acc.sum()
        coadd_stamp = acc / cweights.sum() if len(cweights) else acc.copy()
        var_scale = float(np.sum(cweights * np.array(areas)) / cweights.sum()) if len(cweights) else np.nan
        flux_scale = float(coadd_stamp.sum()) if len(cweights) else np.nan
        if total > 0:
            acc /= total
        if len(weights):
            wsum = weights.sum()
            # legacypipe: psfsizemap = sum(iv1/narcsec)/sum(iv1); fwhm = tofwhm * tosigma / sqrt(psfsizemap)
            psfsize = 2. * np.sqrt(2. * np.log(2.)) / (2. * np.sqrt(np.pi)) / np.sqrt(np.sum(weights * np.array(inv_narcsec)) / wsum)
            neff_fwhm = psfex.neff_fwhm(acc, self.pixscale)
            weights = cweights / cweights.sum()
        else:
            psfsize = neff_fwhm = np.nan
        nexp_at = -1
        flag_nexp = False
        if band in self._nexp:
            nm = self._nexp[band]
            if 0 <= cy < nm.shape[0] and 0 <= cx < nm.shape[1]:
                nexp_at = int(nm[cy, cx])
                # robust test: our CCD count must lie within the nexp range of the surrounding
                # +-NEXP_HALO pixels; single masked pixels (cosmic rays, bad columns) in one of many
                # exposures lower nexp locally without changing the PSF of an aperture measurement
                # and a mismatch only matters in proportion to the number of exposures: flag when the
                # count is outside that range by more than NEXP_TOL of it (1 of 6 HSC exposures: flagged;
                # 1-3 of ~30 IBIS exposures: not)
                lo, hi = self._nexp_range(band)
                n = len(idx)
                off = max(int(lo[cy, cx]) - n, n - int(hi[cy, cx]), 0)
                flag_nexp = off > self.NEXP_TOL * max(n, 1)
        W, H = self.bwcs.pixel_shape
        return PsfStamp(stamp=acc, x0=sx0, y0=sy0, x=float(x), y=float(y), band=band, ccd_idx=idx,
                        weights=weights, fwhm_ccd=np.array(fwhms), psfsize_pipeline=psfsize, neff_fwhm=neff_fwhm,
                        coadd_stamp=coadd_stamp, flux_scale=flux_scale, ccd_weight=cweights, sbscale=np.array(sbs),
                        var_scale=var_scale,
                        nexp_at=nexp_at, flag_nexp=flag_nexp,
                        flag_edge=(sx0 < 0 or sy0 < 0 or sx0 + S > W or sy0 + S > H), parts=parts)
