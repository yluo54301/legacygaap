"""GAaP flux measurement on a coadd cutout.

The flux is a linear functional of the original pixels: with w_ap the post-seeing Gaussian
weight and K the Gaussianization kernel, F = pref * sum_i w_ap(i) (I (x) K)(i)
= pref * sum_j I(j) w_eff(j) with w_eff = w_ap correlated with K. The variance follows exactly
from the inverse-variance map, Var(F) = pref^2 sum_j w_eff(j)^2 / ivar(j), which equals
Kuijken+15 Eq. A17 for stationary noise and needs no autocorrelation function. Pixels with
ivar = 0 (masked) are excluded and their share of the weight is reported.
"""
import numpy as np
from dataclasses import dataclass
import scipy.fft
from scipy.signal import fftconvolve
from . import aperture as ap
from .kernel import MIN_SIGMA_PRIME

CUTOUT = 97          # cutout size (pixels): kernel reach (31) + 3 sigma of the widest aperture
FLAG_BIGPSF = 1      # aperture not larger than the target PSF (post-seeing axis < 1.5 px)
FLAG_EDGE = 2        # cutout truncated by the brick boundary
FLAG_MASKED = 4      # more than MASKED_LIMIT of the weight on masked (ivar = 0) pixels
FLAG_NOPSF = 8       # no PSF (no CCD covers the position)
FLAG_POORKERNEL = 16  # point-source response of the Gaussianized model PSF deviates from 1 by > POORKERNEL_LIMIT
FLAG_NEXP = 32       # CCDs used for the PSF differ in number from the pipeline nexp map (informational)
MASKED_LIMIT = 0.1
POORKERNEL_LIMIT = 0.01
FLAG_NAMES = {FLAG_BIGPSF: 'BIGPSF', FLAG_EDGE: 'EDGE', FLAG_MASKED: 'MASKED', FLAG_NOPSF: 'NOPSF',
              FLAG_POORKERNEL: 'POORKERN', FLAG_NEXP: 'NEXP'}


@dataclass
class GaapResult:
    flux: float
    ivar: float
    fracmasked: float
    flags: int
    prefactor: float
    nmasked: int = 0
    psfcorr: float = np.nan     # GAaP flux of a unit point source with the model PSF (ideally 1)


def cutout(img, ivar, x, y, size=CUTOUT):
    """Cutout of img and ivar centred on the pixel nearest (x, y), zero-padded (ivar = 0) outside
    the image. Returns (data, ivar, x0, y0, truncated)."""
    R = size // 2
    cx, cy = int(round(float(x))), int(round(float(y)))
    H, W = img.shape
    x0, y0 = cx - R, cy - R
    d = np.zeros((size, size), dtype=np.float64)
    v = np.zeros((size, size), dtype=np.float64)
    xa, xb = max(x0, 0), min(x0 + size, W)
    ya, yb = max(y0, 0), min(y0 + size, H)
    truncated = (xa != x0) or (ya != y0) or (xb != x0 + size) or (yb != y0 + size)
    if xb > xa and yb > ya:
        d[ya - y0:yb - y0, xa - x0:xb - x0] = img[ya:yb, xa:xb]
        v[ya - y0:yb - y0, xa - x0:xb - x0] = ivar[ya:yb, xa:xb]
    return d, v, x0, y0, truncated


def effective_weight(Wp, K, n, cx, cy):
    """w_eff = w_ap correlated with K on an n x n grid (w_ap centred on (cx, cy))."""
    w = ap.weight_image(Wp, n, cx, cy)
    return fftconvolve(w, K[::-1, ::-1], mode='same')


def measure_cutout(d, v, w_eff, prefactor, flags=0):
    """Flux, inverse variance and masked-weight fraction from a cutout and its effective weight."""
    good = v > 0
    flux = prefactor * float(np.sum(w_eff[good] * d[good]))
    var = prefactor ** 2 * float(np.sum(w_eff[good] ** 2 / v[good]))
    wabs = np.abs(w_eff)
    fracmasked = float(np.sum(wabs[~good]) / np.sum(wabs)) if np.sum(wabs) > 0 else 1.
    if fracmasked > MASKED_LIMIT:
        flags |= FLAG_MASKED
    return GaapResult(flux=flux, ivar=1. / var if var > 0 else 0., fracmasked=fracmasked, flags=flags,
                      prefactor=prefactor, nmasked=int(np.sum(~good)))


def gaap_flux(img, ivar, x, y, K, sigma_t, W, size=CUTOUT):
    """GAaP flux at brick pixel position (x, y) for a pre-seeing aperture matrix W (pixels^2),
    given the Gaussianization kernel K and target dispersion sigma_t (pixels) at that position."""
    Wp, pref, valid = ap.postseeing(W, sigma_t)
    if not valid:
        return GaapResult(flux=np.nan, ivar=0., fracmasked=np.nan, flags=FLAG_BIGPSF, prefactor=np.nan)
    d, v, x0, y0, truncated = cutout(img, ivar, x, y, size)
    w_eff = effective_weight(Wp, K, size, x - x0, y - y0)
    return measure_cutout(d, v, w_eff, pref, flags=FLAG_EDGE if truncated else 0)


def gaussianized_cutout(img, x, y, K, size=CUTOUT):
    """The convolved cutout (for display): I (x) K around (x, y)."""
    d, v, x0, y0, truncated = cutout(img, np.ones_like(img), x, y, size)
    return fftconvolve(d, K, mode='same'), x0, y0


class KernelFFT:
    """Correlation with a fixed kernel K on an n x n grid, with the kernel transform computed once
    (same result as ``fftconvolve(w, K[::-1, ::-1], mode='same')``)."""

    def __init__(self, K, n=CUTOUT):
        self.n = n
        self.k = K.shape[0]
        self.N = scipy.fft.next_fast_len(n + self.k - 1, real=True)
        self.off = (self.k - 1) // 2
        self.Kf = scipy.fft.rfft2(np.asarray(K, dtype=np.float64)[::-1, ::-1], s=(self.N, self.N))

    def correlate(self, w):
        full = scipy.fft.irfft2(scipy.fft.rfft2(w, s=(self.N, self.N)) * self.Kf, s=(self.N, self.N))
        return full[self.off:self.off + self.n, self.off:self.off + self.n]


def measure_source(d, v, x0, y0, x, y, stamp, sx0, sy0, sigma_t, K, Ws, truncated=False,
                   min_sigma_prime=MIN_SIGMA_PRIME, extra_flags=0, return_weights=False, flux_scale=1.):
    """GAaP fluxes of one source in one band for a list of pre-seeing aperture matrices ``Ws``
    (brick pixels^2), from the cutout (d, v) whose pixel (0, 0) is brick pixel (x0, y0).
    ``stamp`` is the coadd image of a unit-flux point source with pixel (0, 0) at brick pixel
    (sx0, sy0) (its sum is the coadd flux scale ``flux_scale``); it gives the point-source response
    ``psfcorr`` of each aperture, which includes that flux scale. Returns a list of GaapResult."""
    n = d.shape[0]
    cx, cy = x - x0, y - y0
    kfft = KernelFFT(K, n)
    S = stamp.shape[0]
    Pc = np.zeros((n, n))
    oy, ox = sy0 - y0, sx0 - x0
    ya, yb, xa, xb = max(oy, 0), min(oy + S, n), max(ox, 0), min(ox + S, n)
    Pc[ya:yb, xa:xb] = stamp[ya - oy:yb - oy, xa - ox:xb - ox]
    base = (FLAG_EDGE if truncated else 0) | extra_flags
    out, weights = [], []
    for W in Ws:
        Wp, pref, valid = ap.postseeing(W, sigma_t, min_sigma_prime)
        if not valid:
            out.append(GaapResult(flux=np.nan, ivar=0., fracmasked=np.nan, flags=FLAG_BIGPSF | base, prefactor=np.nan))
            weights.append(None)
            continue
        w_eff = kfft.correlate(ap.weight_image(Wp, n, cx, cy))
        r = measure_cutout(d, v, w_eff, pref, flags=base)
        r.psfcorr = pref * float(np.sum(w_eff * Pc))
        if not abs(r.psfcorr / flux_scale - 1.) <= POORKERNEL_LIMIT:
            r.flags |= FLAG_POORKERNEL
        out.append(r)
        weights.append(w_eff[ya:yb, xa:xb] if (yb - ya == S and xb - xa == S) else None)
    return (out, weights) if return_weights else out
