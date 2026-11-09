"""PSF Gaussianization: the kernel that turns the coadd PSF into a circular Gaussian, and the
point-source metric that measures how well it worked.

Method (docs/method.md): with P the unit-sum PSF stamp and G the target Gaussian, the
kernel is the regularised Fourier quotient K^ = G^ P^* / (|P^|^2 + lam max|P^|^2), cropped to
``ksize`` pixels and renormalised to unit sum. The default ``ksize`` = 63 spans the whole 65-px
stamp: a 41-px kernel leaves 0.5-1.5% of a heavy-wing (Moffat beta 2.5) PSF ungathered, 63 px
leaves < 0.25% (M2 test). The target width is ``f`` times the Gaussian-
equivalent sigma of P (from its noise-equivalent area, legacypipe's psfsize definition).
"""
import numpy as np
from scipy.signal import fftconvolve, correlate2d
from . import psfex

# Minimum dispersion (pixels) of the post-seeing aperture sqrt(sigma_ap^2 - sigma_t^2): below this
# the Gaussian weight is too narrow for the pixel sum to approximate the integral (aliasing error
# exp(-pi^2 w^2 / 2) with w the width of weight x image: 4e-4 at 1.5 px, 1.5% at 1 px) and the
# prefactor amplifies noise. Apertures closer to the PSF than this are flagged, as in KiDS/LSST.
MIN_SIGMA_PRIME = 1.5


def gaussian_image(sigma, n, cx=None, cy=None):
    """Unit-sum circular Gaussian on an n x n grid centred on (cx, cy) (default: the centre pixel)."""
    cx = n // 2 if cx is None else cx
    cy = n // 2 if cy is None else cy
    y, x = np.mgrid[:n, :n]
    g = np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / sigma ** 2)
    return g / g.sum()


def target_sigma(psf, f=1.15):
    """Dispersion (pixels) of the target Gaussian: f times the Gaussian-equivalent sigma of ``psf``."""
    return f * np.sqrt(psfex.neff_pixels(psf) / (4. * np.pi))


def gaussianization_kernel(psf, sigma_t, lam=1e-4, ksize=63, pad=True, cx=None, cy=None):
    """Kernel K (ksize x ksize, unit sum) such that psf (x) K ~ Gaussian(sigma_t) centred where the
    PSF is centred: (cx, cy) in stamp pixels (default: the centre pixel). Passing the true,
    fractional centre keeps the kernel free of a sub-pixel shift."""
    psf = np.asarray(psf, dtype=np.float64)
    n = psf.shape[0]
    assert psf.shape == (n, n) and n % 2 == 1 and ksize % 2 == 1 and ksize <= n
    cx = n // 2 if cx is None else cx
    cy = n // 2 if cy is None else cy
    N = 2 * n - 1 if pad else n
    P = np.zeros((N, N)); c = N // 2; r = n // 2
    P[c - r:c + r + 1, c - r:c + r + 1] = psf
    G = gaussian_image(sigma_t, N, c - r + cx, c - r + cy)
    Pf = np.fft.fft2(np.fft.ifftshift(P))
    Gf = np.fft.fft2(np.fft.ifftshift(G))
    Kf = Gf * np.conj(Pf) / (np.abs(Pf) ** 2 + lam * np.abs(Pf).max() ** 2)
    K = np.fft.fftshift(np.real(np.fft.ifft2(Kf)))
    h = ksize // 2
    K = K[c - h:c + h + 1, c - h:c + h + 1].copy()
    K /= K.sum()
    return K


def convolve(img, K):
    return fftconvolve(np.asarray(img, dtype=np.float64), K, mode='same')


def kernel_acf(K):
    """Auto-correlation function of the kernel (noise covariance of the convolved image for white noise)."""
    return correlate2d(K, K, mode='full')


def noise_factor(K):
    """Standard deviation of white noise after convolution with K, relative to before: sqrt(sum K^2)."""
    return float(np.sqrt(np.sum(np.asarray(K) ** 2)))


def aperture_valid(sigma_ap, sigma_t, min_sigma_prime=MIN_SIGMA_PRIME):
    """Whether a pre-seeing aperture of dispersion sigma_ap can be measured on an image whose PSF
    has dispersion sigma_t (pixels): sigma_ap^2 - sigma_t^2 >= min_sigma_prime^2."""
    return sigma_ap ** 2 - sigma_t ** 2 >= min_sigma_prime ** 2


def gaap_point_source_flux(img, sigma_ap, sigma_t, cx=None, cy=None, min_sigma_prime=MIN_SIGMA_PRIME):
    """GAaP flux (Kuijken+15 Eq. A16, circular aperture) of a unit point source whose image ``img``
    has a Gaussian PSF of dispersion ``sigma_t``; the pre-seeing aperture has dispersion
    ``sigma_ap``; both in pixels. The exact answer is 1 for any valid aperture. NaN if the
    aperture is not valid (``aperture_valid``)."""
    if not aperture_valid(sigma_ap, sigma_t, min_sigma_prime):
        return np.nan
    s2 = sigma_ap ** 2 - sigma_t ** 2
    n = img.shape[0]
    cx = n // 2 if cx is None else cx
    cy = n // 2 if cy is None else cy
    y, x = np.mgrid[:n, :n]
    w = np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / s2)
    return (sigma_ap ** 2 / s2) * float(np.sum(img * w))


def psf_quality(psf, K, sigma_t, apertures_pix, cx=None, cy=None):
    """How Gaussian psf (x) K is: point-source GAaP flux error per aperture (should be 0),
    the maximum residual against the target Gaussian in units of its peak, and the noise factor."""
    conv = convolve(psf, K)
    n = psf.shape[0]
    G = gaussian_image(sigma_t, n, cx, cy)
    out = dict(resid_peak=float(np.abs(conv - G).max() / G.max()), noise_factor=noise_factor(K), flux_err={})
    for a in apertures_pix:
        F = gaap_point_source_flux(conv, a, sigma_t, cx, cy)
        out['flux_err'][a] = F - 1. if np.isfinite(F) else np.nan
    return out
