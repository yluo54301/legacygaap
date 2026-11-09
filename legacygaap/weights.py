"""Correction for the source-dependent pixel weights of legacypipe coadds.

legacypipe coadds weight each CCD pixel by its inverse variance, which includes the source's own
shot noise (DECam CP weight maps; BASS/MzLS via ``remap_invvar_shotnoise``). At the core of a
bright source the sharpest exposures are down-weighted, so the coadd profile is broader and holds
less flux than the constant-weight coadd the PSF model describes; GAaP of bright compact sources
is biased low (M4: -9% for 17-18 mag stars in g/r, -3% at 22 mag; GAaP on legacypipe's own model
coadd shows the same deficit to 0.1-0.2%).

Model: exposure e gets weight w_e(x) = omega_e / (1 + S_e(x)/sky_e) at brick pixel x, with
omega_e = 1/sig1_e^2, S_e the source (tractor model convolved with that CCD's PSF, nanomaggies
per pixel) and sky_e the sky level per pixel; for Poisson-limited sky the shot-noise variance is
S_e/sky_e times the sky variance (checked on DECam to 5%). The correction factor for an aperture
is the ratio of GAaP responses to the shot-noise-weighted and the constant-weight coadd of the
source model: wcorr = sum(w_eff C) / sum(w_eff C_mix). Corrected flux = gaapflux / wcorr.
"""
import numpy as np
import scipy.fft
from . import profiles, aperture as ap


def sky_per_pixel(ccds):
    """Sky level per CCD pixel (nanomaggies): ccdskycounts (counts/s) / zero-point scale."""
    return np.asarray(ccds['ccdskycounts'], dtype=np.float64) * 10 ** (-0.4 * (np.asarray(ccds['ccdzpt'], dtype=np.float64) - 22.5))


def model_transform(N, mixture, C_pix):
    """rfft2-layout transform of a unit-flux mixture of Gaussians with covariances var_k * C_pix (pixels^2)."""
    fy = np.fft.fftfreq(N)[:, None]
    fx = np.fft.rfftfreq(N)[None, :]
    q = C_pix[0, 0] * fx * fx + 2. * C_pix[0, 1] * fx * fy + C_pix[1, 1] * fy * fy
    amp, var = mixture
    return sum(a * np.exp(-2. * np.pi ** 2 * v * q) for a, v in zip(amp, var))


def source_images(parts, mixture, C_pix, flux):
    """Each CCD's image of the source on the stamp grid (nanomaggies per pixel): flux x (model (x) PSF_e)."""
    if mixture is None:
        return [flux * p for p in parts]
    S = parts[0].shape[0]
    N = scipy.fft.next_fast_len(2 * S - 1, real=True)
    Mhat = model_transform(N, mixture, C_pix)
    return [flux * scipy.fft.irfft2(scipy.fft.rfft2(p, s=(N, N)) * Mhat, s=(N, N))[:S, :S] for p in parts]


def coadd_images(images, cweight, sbscale, sky, strength=1.):
    """The source's coadd as legacypipe builds it: CCD image x sbscale, weighted by the CCD's coadd
    weight (1/sig1^2/sbscale^2) divided by (1 + shot-noise term); and the same with constant weights.
    ``images`` are in each CCD's own units (nanomaggies per CCD pixel)."""
    num = 0.
    den = 0.
    mix = 0.
    strength = np.broadcast_to(np.asarray(strength, dtype=np.float64), (len(images),))
    for img, cw, f, sk, s in zip(images, cweight, sbscale, sky, strength):
        w = cw / (1. + s * np.clip(img, 0., None) / max(sk, 1e-12))
        num = num + w * f * img
        den = den + w
        mix = mix + cw * f * img
    return num / den, mix / float(np.sum(cweight))


def correction(parts, cweight, sbscale, sky, flux, mixture, C_pix, w_effs, strength=1.):
    """wcorr per aperture (list of effective weights on the stamp grid; None for invalid apertures):
    GAaP response to the shot-noise-weighted coadd of the source model over the response to the
    constant-weight coadd. 1 for non-positive flux."""
    if not (flux > 0) or len(parts) == 0:
        return np.ones(len(w_effs))
    C, Cmix = coadd_images(source_images(parts, mixture, C_pix, flux), cweight, sbscale, sky, strength)
    out = np.ones(len(w_effs))
    for k, w in enumerate(w_effs):
        if w is None:
            out[k] = np.nan
            continue
        d = float(np.sum(w * Cmix))
        out[k] = float(np.sum(w * C)) / d if d > 0 else np.nan
    return out


def invvar_ratio(parts, cweight, sky, flux, mixture, C_pix, x, y, strength=1.):
    """Predicted coadd invvar at stamp pixel (x, y) relative to the sky-only invvar."""
    images = source_images(parts, mixture, C_pix, flux)
    ix, iy = int(round(x)), int(round(y))
    w = sum(cw / (1. + strength * max(img[iy, ix], 0.) / sk) for img, cw, sk in zip(images, cweight, sky))
    return w / float(np.sum(cweight))


def fit_strength(parts, cweight, sky, flux, x, y, measured, lo=0.2, hi=3.0, tol=1e-3):
    """Shot-noise strength s for which the predicted invvar dip at a point source equals the
    measured one (bisection; the predicted ratio decreases monotonically with s)."""
    f = lambda s: invvar_ratio(parts, cweight, sky, flux, None, None, x, y, s) - measured
    if f(lo) < 0 or f(hi) > 0:
        return np.nan
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate_strength(model, band, ivar, stars_xy, fluxes, nmax=40, ring=(22, 30)):
    """Strength of the shot-noise term for a band, from the pipeline invvar map: the median over up
    to ``nmax`` isolated bright stars of the per-star strength that reproduces the invvar at the
    star's central pixel relative to a surrounding ring. The sky variance contains non-Poisson
    terms (read noise etc.), so s < 1 (M4: 0.88-0.90 g, 0.94 r, 0.96 z on DECam).
    Returns (strength, number of stars used); (1.0, 0) if no usable star."""
    sky = sky_per_pixel(model.ccds[band])
    R = ring[1]
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(xx, yy)
    vals = []
    for (x, y), F in zip(stars_xy, fluxes):
        if len(vals) >= nmax:
            break
        cx, cy = int(round(x)), int(round(y))
        cut = ivar[cy - R:cy + R + 1, cx - R:cx + R + 1]
        if cut.shape != (2 * R + 1, 2 * R + 1) or cut[R, R] <= 0:
            continue
        rv = cut[(rr > ring[0]) & (rr < R) & (cut > 0)]
        if rv.size < 50:
            continue
        measured = float(cut[R, R]) / float(np.median(rv))
        if not (0.05 < measured < 0.9):
            continue
        ps = model.psf_at(x, y, band, return_parts=True)
        if ps.nccd == 0:
            continue
        s = fit_strength(ps.parts, ps.ccd_weight, sky[ps.ccd_idx], float(F), cx - ps.x0, cy - ps.y0, measured)
        if np.isfinite(s):
            vals.append(s)
    if not vals:
        return 1.0, 0
    return float(np.median(vals)), len(vals)


def ccd_groups(ccds):
    """Group key of each CCD for the shot-noise strength: camera and CP version (the CP's weight-map
    variance model differs between versions, e.g. DECam V5.0 g: 0.78 against 0.96 for V4.8.2, M4)."""
    return np.array([f"{str(c['camera']).strip()}/{str(c['plver']).strip()}" for c in ccds])
