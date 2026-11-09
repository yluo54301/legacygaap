"""Brick driver: GAaP fluxes for the sources of a legacypipe brick, and at blank-sky positions.

``run_brick`` loads the brick's coadds, tractor catalog and CCD lists, builds the coadd PSF model
(preloading every PsfEx model and spline map so that forked workers inherit them), computes each
source's pre-seeing apertures (fixed circular ones and model-matched ellipses from the tractor
shape), measures every band in worker processes over spatially compact chunks of sources, and
writes one row per catalog row (plus an optional blank-sky table) with a self-describing header.
"""
import os
import time
import subprocess
import multiprocessing as mp
from dataclasses import dataclass, asdict
import numpy as np
import fitsio
from scipy.ndimage import distance_transform_edt

from . import __version__, io, psf as P, kernel as kn, aperture as ap, measure as ms, profiles, weights as wt

COPY_COLUMNS = ['release', 'brickid', 'brickname', 'objid', 'brick_primary', 'type', 'ra', 'dec', 'bx', 'by', 'maskbits']
SHAPE_COLUMNS = ['shape_r', 'shape_e1', 'shape_e2']
MODEL_COLUMNS = ['sersic']


def _fmt(v):
    """Aperture size for names: one decimal, two when needed (1.0 -> '1.0', 1.25 -> '1.25')."""
    t = f'{v:.2f}'.rstrip('0')
    return t + '0' if t.endswith('.') else t


# The measurement cutout of each source spans CUTOUT_NSIGMA times its largest pre-seeing aperture
# dispersion (pixels) on each side, and at least Settings.cutout: a Gaussian weight clipped at
# 4 sigma loses ~1e-4 of its integral (the fixed 97-px cutout clipped a 3" aperture at 2.7 sigma
# on the 0.168" HSC grid).
CUTOUT_NSIGMA = 4.0


def cutout_sizes(Wpix, base):
    """Per-source cutout size (odd, pixels) from the aperture matrices Wpix (n, nap, 2, 2)."""
    lam = np.linalg.eigvalsh(np.nan_to_num(np.asarray(Wpix, dtype=np.float64)))
    smax = np.sqrt(np.clip(lam.max(axis=(1, 2)), 0, None))
    return np.maximum(int(base), 2 * np.ceil(CUTOUT_NSIGMA * smax).astype(int) + 1)


@dataclass
class Settings:
    circ: tuple = (1.0, 1.5, 2.0)        # circular pre-seeing aperture dispersions (arcsec)
    min_apers: tuple = (1.0, 1.5)        # MIN_APER of the model-matched elliptical apertures (arcsec)
    max_aper: float = 2.0                # cap on the elliptical aperture axes (arcsec)
    kappa: float = 1.0                   # aperture axes = kappa * tractor axes (before MIN/MAX)
    scale: float = 1.15                  # target Gaussian dispersion / PSF Gaussian-equivalent dispersion
    ksize: int = 63                      # kernel size (pixels)
    lam: float = 1e-4                    # kernel regularisation
    stamp: int = 65                      # PSF stamp size (pixels)
    cutout: int = 97                     # measurement cutout (pixels)
    margin: float = io.EDGE_MARGIN       # CCD edge band excluded from the PSF (pixels)
    min_sigma_prime: float = kn.MIN_SIGMA_PRIME

    def aperture_names(self):
        return [f'circ{_fmt(s)}' for s in self.circ] + [f'ell{_fmt(m)}' for m in self.min_apers]

    @property
    def nap(self):
        return len(self.circ) + len(self.min_apers)


def source_apertures(st, shape_r, e1, e2, bwcs):
    """Per source: pre-seeing aperture matrices in brick pixels^2, shape (n, nap, 2, 2), and the
    sky-frame axes and position angle (arcsec, arcsec, degrees north through east), (n, nap, 3)."""
    n = len(shape_r)
    M = ap.sky_to_pixel_matrix(bwcs)
    Wpix = np.zeros((n, st.nap, 2, 2))
    abp = np.zeros((n, st.nap, 3))
    for k, s in enumerate(st.circ):
        Wpix[:, k] = M @ ap.circular(s) @ M.T
        abp[:, k] = (s, s, 0.)
    for j, mn in enumerate(st.min_apers):
        k = len(st.circ) + j
        for i in range(n):
            a, b, ang = ap.tractor_shape_axes(shape_r[i], e1[i], e2[i], st.kappa, mn, st.max_aper)
            Wpix[i, k] = M @ ap.aperture_matrix(a, b, ang) @ M.T
            abp[i, k] = (a, b, np.rad2deg(np.pi / 2 - ang) % 180.)
    return Wpix, abp


def blobs_file(paths):
    return os.path.join(paths.metrics_dir, f'blobs-{paths.brick}.fits.gz')


def read_blob_map(fn):
    """legacypipe's blob map (-1 outside source blobs): extension BLOB-MAP since DR11, the primary
    HDU in DR10."""
    with fitsio.FITS(fn) as F:
        for h in F:
            if h.get_extname().strip().upper() == 'BLOB-MAP':
                return h.read()
        return F[0].read()


def blank_positions(paths, n, seed=1, radius=15, margin=64, target_wcs=None):
    """``n`` random positions at least ``radius`` pixels from any source blob (legacypipe's blob
    map) and from any maskbits pixel, ``margin`` pixels inside the brick, drawn on ``paths``' grid.
    With ``target_wcs`` (another survey's coadd of the same brick, e.g. HSC on a 0.168" grid) the
    positions are returned in that grid's pixels, keeping those ``margin`` pixels inside it."""
    blobs = read_blob_map(blobs_file(paths))
    free = (blobs < 0) & (io.read_coadd(paths, 'maskbits') == 0)
    ok = distance_transform_edt(free) >= radius
    ok[:margin] = ok[-margin:] = False
    ok[:, :margin] = ok[:, -margin:] = False
    cand = np.flatnonzero(ok)
    rng = np.random.default_rng(seed)
    pick = rng.choice(cand, size=min(n, len(cand)), replace=False)
    y, x = np.divmod(pick, blobs.shape[1])
    x, y = x + rng.uniform(-0.5, 0.5, len(x)), y + rng.uniform(-0.5, 0.5, len(y))
    if target_wcs is None:
        return x, y
    ra, dec = io.brick_wcs(paths).all_pix2world(x, y, 0)
    tx, ty = target_wcs.all_world2pix(ra, dec, 0)
    W, H = target_wcs.pixel_shape
    keep = (tx > margin) & (tx < W - margin) & (ty > margin) & (ty < H - margin)
    return tx[keep], ty[keep]


PM_COLUMNS = ['pmra', 'pmdec', 'ref_epoch']


def band_epochs(model, bands):
    """Mean epoch (decimal year) of each band's CCDs, weighted like the coadd (1/sig1^2)."""
    out = {}
    for b in bands:
        C = model.ccds[b]
        w = 1. / np.asarray(C['sig1'], dtype=np.float64) ** 2
        mjd = float(np.sum(w * C['mjd_obs']) / np.sum(w))
        out[b] = 2000. + (mjd - 51544.5) / 365.25
    return out


def band_positions(T, bwcs, epochs):
    """Per band: brick pixel positions of the catalog rows at that band's epoch. legacypipe gives
    Gaia stars' ra/dec at ref_epoch (2016.0) with pmra/pmdec (mas/yr); IBIS images are ~8.5 yr later,
    so a star moving 20 mas/yr is 0.17" off and its 1" GAaP flux -4% low (M464, 1501p022).
    Returns ({band: (x, y)}, number of moved sources)."""
    ra, dec = np.asarray(T['ra'], dtype=np.float64), np.asarray(T['dec'], dtype=np.float64)
    have = all(c in T.dtype.names for c in PM_COLUMNS)
    if have:
        ep = np.nan_to_num(np.asarray(T['ref_epoch'], dtype=np.float64))
        pmra = np.nan_to_num(np.asarray(T['pmra'], dtype=np.float64))
        pmdec = np.nan_to_num(np.asarray(T['pmdec'], dtype=np.float64))
        move = (ep > 0) & ((pmra != 0) | (pmdec != 0))
    else:
        move = np.zeros(len(ra), bool)
    out = {}
    for b, year in epochs.items():
        if move.any():
            dt = np.where(move, year - ep, 0.)
            d = dec + pmdec * dt / 3.6e6
            r = ra + pmra * dt / 3.6e6 / np.cos(np.deg2rad(dec))
        else:
            r, d = ra, dec
        out[b] = tuple(np.asarray(v) for v in bwcs.all_world2pix(r, d, 0))
    return out, int(move.sum())


# ----------------------------------------------------------------------------- workers

_STATE = {}


def _empty_results(n, nap, bands):
    return {b: dict(flux=np.full((n, nap), np.nan, np.float32), ivar=np.zeros((n, nap), np.float32),
                    fracmasked=np.full((n, nap), np.nan, np.float32), flags=np.zeros((n, nap), np.int16),
                    psfcorr=np.full((n, nap), np.nan, np.float32), wcorr=np.full((n, nap), np.nan, np.float32),
                    fluxscale=np.full(n, np.nan, np.float32), escale=np.full(n, np.nan, np.float32),
                    sigma_t=np.full(n, np.nan, np.float32),
                    psffwhm=np.full(n, np.nan, np.float32), nccd=np.zeros(n, np.int16)) for b in bands}


def _work(chunk):
    S = _STATE
    st, model, bands = S['settings'], S['model'], S['bands']
    out = _empty_results(len(chunk), st.nap, bands)
    for j, i in enumerate(chunk):
        mixture = profiles.get_mixture(S['type'][i], S['sersic'][i]) if S['type'] is not None else None
        for b in bands:
            o = out[b]
            if S.get('xyb') is not None:
                x, y = float(S['xyb'][b][0][i]), float(S['xyb'][b][1][i])
            else:
                x, y = float(S['x'][i]), float(S['y'][i])
            if not (np.isfinite(x) and np.isfinite(y)):      # no sky position (legacypipe 'NUN' placeholder)
                o['flags'][j] = ms.FLAG_NOPSF
                continue
            ps = model.psf_at(x, y, b, return_parts=True)
            o['nccd'][j] = ps.nccd
            if ps.nccd == 0:
                o['flags'][j] = ms.FLAG_NOPSF
                continue
            sig_t = kn.target_sigma(ps.stamp, f=st.scale)
            o['sigma_t'][j] = sig_t * model.pixscale
            o['psffwhm'][j] = ps.neff_fwhm
            K = kn.gaussianization_kernel(ps.stamp, sig_t, lam=st.lam, ksize=st.ksize, cx=x - ps.x0, cy=y - ps.y0)
            d, v, x0, y0, trunc = ms.cutout(S['img'][b], S['ivar'][b], x, y, int(S['ncut'][i]))
            res, weff = ms.measure_source(d, v, x0, y0, x, y, ps.coadd_stamp, ps.x0, ps.y0, sig_t, K, S['Wpix'][i],
                                          truncated=trunc, min_sigma_prime=st.min_sigma_prime,
                                          extra_flags=ms.FLAG_NEXP if ps.flag_nexp else 0, return_weights=True,
                                          flux_scale=ps.flux_scale)
            flux = float(S['flux'][b][i]) if S['flux'] is not None else 0.
            wcorr = wt.correction(ps.parts, ps.ccd_weight, ps.sbscale, S['sky'][b][ps.ccd_idx], flux, mixture,
                                  S['Cpix'][i], weff, strength=S['strength'][b][ps.ccd_idx] if b in S['strength'] else 1.)
            o['wcorr'][j] = wcorr
            o['fluxscale'][j] = ps.flux_scale
            o['escale'][j] = np.sqrt(ps.var_scale)
            for k, r in enumerate(res):
                # calibrated GAaP flux: raw / (point-source response x coadd-weighting correction);
                # variance x var_scale for the pixel-scale change of the input CCDs (resampling correlations)
                c = r.psfcorr * wcorr[k]
                o['flux'][j, k] = r.flux / c
                o['ivar'][j, k] = r.ivar * c * c / ps.var_scale
                o['fracmasked'][j, k] = r.fracmasked
                o['flags'][j, k] = r.flags
                o['psfcorr'][j, k] = r.psfcorr
    return chunk, out


def _spatial_chunks(x, y, size=32, strip=256):
    order = np.lexsort((x, np.floor(y / strip)))
    return [order[i:i + size] for i in range(0, len(order), size)]


def measure_positions(model, imgs, ivars, x, y, Wpix, st, bands, threads=1, log=print, flux=None, typ=None, sersic=None,
                      Cpix=None, strength=None, xy_band=None):
    """Measure every band at positions (x, y) with per-position apertures Wpix; returns per-band dicts.
    ``flux`` (per band), ``typ``, ``sersic`` and ``Cpix`` (tractor model covariance, pixels^2) feed
    the coadd-weighting correction; without them it is 1 (blank sky)."""
    n = len(x)
    _STATE.clear()
    sky = {b: wt.sky_per_pixel(model.ccds[b]) for b in bands}
    _STATE.update(settings=st, model=model, bands=bands, img=imgs, ivar=ivars, x=x, y=y, xyb=xy_band, Wpix=Wpix, flux=flux,
                  ncut=cutout_sizes(Wpix, st.cutout),
                  type=typ, sersic=sersic if sersic is not None else np.full(n, np.nan),
                  Cpix=Cpix if Cpix is not None else np.zeros((n, 2, 2)), sky=sky, strength=strength or {})
    res = _empty_results(n, st.nap, bands)
    chunks = _spatial_chunks(x, y)
    t0 = time.time()

    def store(chunk, out):
        for b in bands:
            for key, arr in out[b].items():
                res[b][key][chunk] = arr

    if threads > 1:
        with mp.get_context('fork').Pool(threads) as pool:
            for k, (chunk, out) in enumerate(pool.imap_unordered(_work, chunks)):
                store(chunk, out)
                if (k + 1) % max(1, len(chunks) // 10) == 0:
                    log(f'  {k+1}/{len(chunks)} chunks ({time.time()-t0:.0f}s)')
    else:
        for chunk in chunks:
            store(*_work(chunk))
    _STATE.clear()
    return res


# ----------------------------------------------------------------------------- output

def _git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def build_table(base_cols, abp, res, st, bands):
    n = len(abp)
    nap = st.nap
    dt = []
    for name, arr in base_cols:
        dt.append((name, arr.dtype.str if arr.dtype.kind != 'U' else f'S{max(1, arr.dtype.itemsize // 4)}'))
    dt += [('gaap_a', 'f4', (nap,)), ('gaap_b', 'f4', (nap,)), ('gaap_pa', 'f4', (nap,))]
    for b in bands:
        dt += [(f'gaapflux_{b}', 'f4', (nap,)), (f'gaapflux_ivar_{b}', 'f4', (nap,)), (f'gaap_fracmasked_{b}', 'f4', (nap,)),
               (f'gaap_flags_{b}', 'i2', (nap,)), (f'gaap_psfcorr_{b}', 'f4', (nap,)), (f'gaap_wcorr_{b}', 'f4', (nap,)),
               (f'gaap_fluxscale_{b}', 'f4'), (f'gaap_escale_{b}', 'f4'), (f'gaap_sigma_t_{b}', 'f4'),
               (f'gaap_psffwhm_{b}', 'f4'), (f'gaap_nccd_{b}', 'i2')]
    T = np.zeros(n, dtype=dt)
    for name, arr in base_cols:
        T[name] = np.char.encode(arr.astype(str)) if arr.dtype.kind == 'U' else arr
    T['gaap_a'], T['gaap_b'], T['gaap_pa'] = abp[:, :, 0], abp[:, :, 1], abp[:, :, 2]
    for b in bands:
        r = res[b]
        T[f'gaapflux_{b}'] = r['flux']
        T[f'gaapflux_ivar_{b}'] = r['ivar']
        T[f'gaap_fracmasked_{b}'] = r['fracmasked']
        T[f'gaap_flags_{b}'] = r['flags']
        T[f'gaap_psfcorr_{b}'] = r['psfcorr']
        T[f'gaap_wcorr_{b}'] = r['wcorr']
        T[f'gaap_fluxscale_{b}'] = r['fluxscale']
        T[f'gaap_escale_{b}'] = r['escale']
        T[f'gaap_sigma_t_{b}'] = r['sigma_t']
        T[f'gaap_psffwhm_{b}'] = r['psffwhm']
        T[f'gaap_nccd_{b}'] = r['nccd']
    return T


def make_header(paths, st, bands, model, extra):
    hdr = [dict(name='GAAPVER', value=__version__, comment='legacygaap version'),
           dict(name='GAAPGIT', value=_git_hash(), comment='legacygaap git commit'),
           dict(name='BRICK', value=paths.brick), dict(name='REGION', value=paths.region),
           dict(name='SURVDIR', value=paths.survey_dir),
           dict(name='BANDS', value=','.join(bands)), dict(name='NAP', value=st.nap, comment='number of apertures')]
    for k, name in enumerate(st.aperture_names()):
        hdr.append(dict(name=f'AP{k}', value=name, comment='circ<sigma"> or ell<MIN_APER">'))
    for key, val in asdict(st).items():
        if not isinstance(val, tuple):
            hdr.append(dict(name=key[:8].upper(), value=val))
    for b in bands:
        hdr.append(dict(name=f'NCCD_{b.upper()}', value=len(model.ccds[b]), comment=f'CCDs in the {b} PSF model'))
    for bit, name in ms.FLAG_NAMES.items():
        hdr.append(dict(name=f'FLAG{bit}', value=name, comment='gaap_flags bit'))
    hdr.append(dict(name='CUTNSIG', value=CUTOUT_NSIGMA, comment='cutout half-width / largest aperture sigma (>= CUTOUT)'))
    hdr.append(dict(name='FLUXDEF', value='raw/(psfcorr*wcorr)', comment='gaapflux = raw coadd measurement / (psfcorr x wcorr)'))
    hdr.append(dict(name='IVARDEF', value='raw*(psfcorr*wcorr/escale)^2', comment='escale^2: mean CCD/coadd pixel-area ratio'))
    try:
        hdr.append(dict(name='LEGPIPEV', value=str(io.read_header(paths.coadd('image', bands[0])).get('LEGPIPEV', '')).strip()))
    except Exception:
        pass
    # legacypipe subtracts a per-brick median sky (COSKY_<B>, nmgy per CCD pixel) from every CCD
    # before fitting and coadding (docs/method.md). The GAaP flux of that constant is
    # COSKY * 2 pi a b / (psfcorr * wcorr), with a, b the aperture axes in coadd pixels.
    for b in bands:
        try:
            cs = io.read_header(paths.coadd('image', b)).get(f'COSKY_{b.upper()}')
            if cs is not None and str(cs).strip() not in ('', 'None'):
                hdr.append(dict(name=f'COSKY_{b.upper()}', value=float(cs), comment='legacypipe per-brick sky subtracted'))
        except Exception:
            pass
    for key, val in extra.items():
        hdr.append(dict(name=key, value=val))
    return hdr


def write_table(fn, T, hdr):
    os.makedirs(os.path.dirname(os.path.abspath(fn)), exist_ok=True)
    tmp = fn + '.tmp'
    if os.path.exists(tmp):
        os.remove(tmp)
    with fitsio.FITS(tmp, 'rw') as F:
        F.write(T, header=hdr, extname='GAAP')
    os.replace(tmp, fn)


# ----------------------------------------------------------------------------- driver

def calibration_stars_strength(paths, model, bands, ivars, bwcs=None, mags=(16.5, 20.5), log=print, epochs=None, iso_dmag=None):
    """Per band: shot-noise strength from the invvar dips of isolated bright stars
    (weights.calibrate_strength), as a per-CCD array. Per-CP-version strengths differ (DECam V5.0
    g weight maps: 0.78 against 0.96 for V4.8.2, measured on the weight maps in M4) but cannot be
    separated from coadd products; the per-band value leaves ~1% residuals for bright stars in
    bands that mix CP versions (docs/method.md). Stars are isolated (> 5.2" to the next source)
    unmasked PSF-type sources with catalog magnitude in ``mags`` (HSC Wide saturates near 18 mag).
    With ``iso_dmag`` only neighbours less than that many magnitudes fainter (in the band) count for the
    isolation (deep catalogues, e.g. on Rubin coadds, detect faint sources around every bright star).
    Returns {band: (per-CCD array, description)}."""
    from scipy.spatial import cKDTree
    bwcs = bwcs or model.bwcs
    names = fitsio.FITS(paths.tractor)[1].get_colnames()
    fcols = {b: io.flux_column(names, b) for b in bands}
    pmc = [c for c in PM_COLUMNS if c in names]
    C = io.read_tractor(paths, columns=['type', 'ra', 'dec', 'maskbits'] + pmc + sorted(set(c for c in fcols.values() if c)))
    typ = io.col_str(C, 'type')
    x, y = bwcs.all_world2pix(C['ra'], C['dec'], 0)
    # rows without a sky position (legacypipe 'NUN' placeholders at ra = dec = 0, e.g. 1513p000)
    # project to NaN; they are never calibration stars (nn = 0) and are measured as NOPSF
    fin = np.isfinite(x) & np.isfinite(y)
    nn = np.zeros(len(x))
    xy = np.column_stack([x, y])[fin]
    nn[fin] = cKDTree(xy).query(xy, k=2)[0][:, 1] * model.pixscale
    pos = band_positions(C, bwcs, epochs)[0] if epochs else {}
    W, H = bwcs.pixel_shape
    out = {}
    for b in bands:
        if fcols[b] is None:
            out[b] = (np.ones(len(model.ccds[b])), '1.000 (no catalog flux)')
            log(f'  shot-noise strength {b}: {out[b][1]}')
            continue
        f = np.asarray(C[fcols[b]], dtype=np.float64)
        m = 22.5 - 2.5 * np.log10(np.clip(f, 1e-6, None))
        iso = nn > 5.2
        if iso_dmag is not None:
            tree = cKDTree(np.column_stack([x, y])[fin])
            fi = np.flatnonzero(fin)
            cand = np.flatnonzero(fin & (typ == 'PSF') & (C['maskbits'] == 0) & (m > mags[0]) & (m < mags[1]))
            iso = np.zeros(len(x), bool)
            for i in cand:
                nb = fi[tree.query_ball_point([x[i], y[i]], r=5.2 / model.pixscale)]
                nb = nb[nb != i]
                iso[i] = not np.any(f[nb] > f[i] * 10 ** (-0.4 * iso_dmag))
        sel = np.flatnonzero((typ == 'PSF') & (C['maskbits'] == 0) & iso & (m > mags[0]) & (m < mags[1])
                             & (x > 40) & (x < W - 40) & (y > 40) & (y < H - 40))
        sel = sel[np.argsort(m[sel])]
        xyb = np.column_stack(pos[b]) if b in pos else xy
        s, n = wt.calibrate_strength(model, b, ivars[b], xyb[sel], f[sel])
        out[b] = (np.full(len(model.ccds[b]), s), f'{s:.3f} ({n} stars)')
        log(f'  shot-noise strength {b}: {out[b][1]}')
    return out


def output_paths(outdir, brick, tag=None):
    d = os.path.join(outdir, brick[:3])
    t = f'-{tag}' if tag else ''
    return os.path.join(d, f'gaap-{brick}{t}.fits'), os.path.join(d, f'gaap-sky-{brick}{t}.fits')


def run_brick(brick, survey_dir=None, region=None, bands=None, settings=None, threads=1, rows=None, limit=None,
              sky=0, outdir='.', image_kind='image', log=print, catalog=None, images_root=None, wcs_ccds=None,
              wcs_images_root=None, wcs_tag='alt', tag=None, star_mags=None, sky_ref_dir=None, pixscale='auto',
              survey_ccds=None, iso_dmag=None):
    """GAaP for one brick. Besides the DR layout, ``survey_dir`` may be a single-run directory
    (IBIS reduction, HSC coadds). ``catalog`` replaces the tractor catalog (same rows; band fluxes
    from ``flux_<band>`` or ``forced_flux_<band>``); ``images_root`` resolves image_filename;
    ``wcs_ccds`` takes every CCD's WCS from that CCD table's image of the same (expnum, ccdname)
    under ``wcs_images_root`` (IBIS DR1); ``sky_ref_dir`` supplies the blob map and maskbits for
    blank-sky positions when ``survey_dir`` has none; ``tag`` suffixes the output names;
    ``survey_ccds`` = (directory, cameras) of the survey CCD tables for bands missing from the brick
    CCD table (legacypipe's forced bands) in a flat layout (e.g. a legacypipe run on Rubin deep-coadd patches)."""
    t_start = time.time()
    st = settings or Settings()
    big = [m for m in st.min_apers if m >= st.max_aper]
    if big:
        log(f'  warning: MIN_APER {big} >= MAX_APER {st.max_aper}": those elliptical apertures are {st.max_aper}" circles')
    if images_root:
        io.set_images_root(images_root)
    if survey_ccds:
        io.set_survey_ccds(*survey_ccds)
    paths = io.BrickPaths(brick, survey_dir=survey_dir, region=region, catalog=catalog)
    survey_dir = paths.survey_dir
    default_root = io.IMAGES_ROOT is None          # image_filename relative to <survey_dir>/images (legacypipe)
    if default_root:
        io.set_images_root(os.path.join(survey_dir, 'images'))
    bands = bands or paths.bands()
    bwcs = io.brick_wcs(paths, band=bands[0])
    wcs_info = 'image headers'
    if wcs_ccds:
        nload = io.set_wcs_override(wcs_ccds, wcs_images_root or io.IMAGES_ROOT, wcs_tag,
                                    expnums=np.unique(io.read_brick_ccds(paths)['expnum']))
        wcs_info = f'{os.path.basename(wcs_ccds)} ({nload} CCDs)'
    names = fitsio.FITS(paths.tractor)[1].get_colnames()
    fcols = {b: io.flux_column(names, b) for b in bands}
    want = [c for c in COPY_COLUMNS + SHAPE_COLUMNS + MODEL_COLUMNS + PM_COLUMNS if c in names]
    T = io.read_tractor(paths, columns=want + sorted(set(c for c in fcols.values() if c)))
    sel = np.arange(len(T))
    if rows is not None:
        sel = np.asarray(rows, dtype=int)
    if limit is not None:
        sel = sel[:limit]
    T = T[sel]
    x, y = bwcs.all_world2pix(T['ra'], T['dec'], 0)
    log(f'{brick} ({paths.region or "flat"} {survey_dir}): {len(T)} sources, bands {bands}, {st.nap} apertures '
        f'{st.aperture_names()}, pixscale {3600 * np.sqrt(abs(np.linalg.det(bwcs.wcs.cd))):.3f}", WCS from {wcs_info}')
    log(f'  catalog {paths.tractor}; fluxes {fcols}')
    model = P.CoaddPsfModel.for_brick(paths, bands=bands, stamp=st.stamp, margin=st.margin, bwcs=bwcs)
    model.preload()
    if pixscale == 'auto':
        conv, ratios = model.detect_pixscale_convention()
        log(f'  CCD pixel-scale convention: {conv} (psfsize formula / pipeline map: '
            + ', '.join(f'{c} {v:.4f}' for c, v in ratios.items()) + ')')
    else:
        model.set_pixscale_convention(pixscale)
        ratios = {}
    imgs = {b: io.read_coadd(paths, image_kind, b) for b in bands}
    ivars = {b: io.read_coadd(paths, 'invvar', b) for b in bands}
    shape = [np.nan_to_num(np.asarray(T[c], dtype=np.float64)) for c in SHAPE_COLUMNS]
    typ = io.col_str(T, 'type')
    for c in shape:
        c[np.isin(typ, ['PSF', 'DUP'])] = 0.
    Wpix, abp = source_apertures(st, *shape, bwcs)
    M = ap.sky_to_pixel_matrix(bwcs)
    Cpix = np.array([M @ profiles.shape_covariance(r, e1, e2) @ M.T for r, e1, e2 in zip(*shape)])
    flux = {b: (np.nan_to_num(np.asarray(T[fcols[b]], dtype=np.float64)) if fcols[b] else np.zeros(len(T))) for b in bands}
    if star_mags is None:
        cams = set(str(c).strip() for b in bands for c in model.ccds[b]['camera'])
        star_mags = (18.5, 21.5) if cams == {'hsc'} else (16.5, 20.5)
    epochs = band_epochs(model, bands)
    xy_band, nmove = band_positions(T, bwcs, epochs)
    log(f'  band epochs ' + ', '.join(f'{b} {e:.2f}' for b, e in epochs.items()) + f'; {nmove} sources with proper motions moved')
    strength = calibration_stars_strength(paths, model, bands, ivars, bwcs=bwcs, mags=star_mags, log=log, epochs=epochs,
                                          iso_dmag=iso_dmag)
    t_setup = time.time() - t_start
    log(f'  setup {t_setup:.0f}s ({sum(len(v) for v in model.ccds.values())} CCDs); measuring with {threads} process(es)')
    t0 = time.time()
    res = measure_positions(model, imgs, ivars, x, y, Wpix, st, bands, threads=threads, log=log, flux=flux, typ=typ,
                            sersic=np.asarray(T['sersic'], dtype=np.float64) if 'sersic' in T.dtype.names else None,
                            Cpix=Cpix, strength={b: v[0] for b, v in strength.items()}, xy_band=xy_band)
    t_meas = time.time() - t0
    base = [(c, np.asarray(T[c])) for c in COPY_COLUMNS if c in T.dtype.names]
    for i, (c, _) in enumerate(base):
        if c in ('type', 'brickname'):
            base[i] = (c, io.col_str(T, c))
    out = build_table(base, abp, res, st, bands)
    fn, fn_sky = output_paths(outdir, brick, tag)
    if image_kind != 'image':
        fn, fn_sky = fn.replace('gaap-', f'gaap-{image_kind}-'), fn_sky.replace('gaap-sky-', f'gaap-sky-{image_kind}-')
    extra = dict(NSRC=len(T), NTHREADS=threads, TSETUP=round(t_setup, 1), TMEAS=round(t_meas, 1), IMGKIND=image_kind,
                 CATALOG=paths.tractor[-68:], WCSSRC=wcs_info[:68], IMGROOT=str(io.IMAGES_ROOT)[-68:],
                 PIXSCONV=model.pixscale_convention, NPMMOVE=nmove,
                 **{f'EPOCH_{b.upper()}': round(e, 3) for b, e in epochs.items()},
                 **{f'PSZRAT_{c.upper()[:3]}': round(v, 5) for c, v in ratios.items() if np.isfinite(v)},
                 **{f'FLUXC_{b.upper()}': (fcols[b] or 'none') for b in bands},
                 **{f'SHOTN_{b.upper()}': v[1][:68] for b, v in strength.items()},
                 SELECT='all' if (rows is None and limit is None) else f'rows={rows is not None} limit={limit}')
    write_table(fn, out, make_header(paths, st, bands, model, extra))
    log(f'  wrote {fn} ({len(out)} rows; measure {t_meas:.0f}s, {1e3*t_meas*max(threads,1)/max(1,len(T)*len(bands)):.0f} ms CPU per source-band)')
    if sky:
        ref = paths if os.path.exists(blobs_file(paths)) else (
            io.BrickPaths(brick, survey_dir=sky_ref_dir) if sky_ref_dir else None)
        if ref is None:
            log('  no blob map for blank-sky positions (give sky_ref_dir); skipped')
        else:
            sx, sy = blank_positions(ref, sky, target_wcs=None if ref is paths else bwcs)
            zeros = np.zeros(len(sx))
            Wsky, abp_sky = source_apertures(st, zeros, zeros, zeros, bwcs)
            t0 = time.time()
            rs_ = measure_positions(model, imgs, ivars, sx, sy, Wsky, st, bands, threads=threads, log=log,
                                    strength={b: v[0] for b, v in strength.items()})
            sra, sdec = bwcs.all_pix2world(sx, sy, 0)
            base = [('x', sx.astype(np.float32)), ('y', sy.astype(np.float32)), ('ra', np.asarray(sra)), ('dec', np.asarray(sdec))]
            sout = build_table(base, abp_sky, rs_, st, bands)
            write_table(fn_sky, sout, make_header(paths, st, bands, model, dict(NSKY=len(sx), TMEAS=round(time.time() - t0, 1))))
            log(f'  wrote {fn_sky} ({len(sout)} blank-sky positions)')
    io.clear_wcs_override()
    if default_root:
        io.set_images_root(None)
    log(f'  total {time.time()-t_start:.0f}s')
    return fn
