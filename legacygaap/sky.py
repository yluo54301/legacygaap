"""Blank-sky statistics of a GAaP run: empirical error scales and depths.

``gaap-brick --sky N`` measures N blank-sky positions (away from every source blob and masked pixel)
with the same apertures, PSF model and code as the sources. Their scatter is the empirical noise of
each aperture. It includes what the inverse-variance errors leave out: noise correlated by the
coadd resampling, and background residuals on aperture scales. Typical error scales are 1.1-1.4
for DECam coadds and larger for deep stacks with background residuals.

    S = fitsio.read('gaap-sky-0001m002.fits')
    scale = error_scale(S, 'r')       # per aperture; multiply the GAaP errors by it
    depth5 = depth(S, 'r')            # per aperture, AB mag
"""
import numpy as np
from .measure import FLAG_NEXP


def nmad(x):
    """Normalised median absolute deviation of the finite values (NaN if there are none)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def valid(flags, ivar):
    """Usable measurements: no flag other than the informational FLAG_NEXP, and a positive ivar."""
    return ((np.asarray(flags) & ~FLAG_NEXP) == 0) & (np.asarray(ivar) > 0)


def _columns(sky, band):
    f = np.atleast_2d(np.asarray(sky[f'gaapflux_{band}'], dtype=np.float64).T).T
    iv = np.atleast_2d(np.asarray(sky[f'gaapflux_ivar_{band}'], dtype=np.float64).T).T
    fl = np.atleast_2d(np.asarray(sky[f'gaap_flags_{band}']).T).T
    return f, iv, fl


def error_scale(sky, band, floor=1.0):
    """Per aperture: the NMAD of flux x sqrt(ivar) at the valid blank-sky positions, at least ``floor``
    (NaN where no position is valid). Multiply ``1 / sqrt(gaapflux_ivar)`` by it."""
    f, iv, fl = _columns(sky, band)
    out = np.full(f.shape[1], np.nan)
    for k in range(f.shape[1]):
        m = valid(fl[:, k], iv[:, k])
        if m.any():
            out[k] = max(floor, nmad(f[m, k] * np.sqrt(iv[m, k])))
    return out


def depth(sky, band, nsigma=5., zeropoint=22.5):
    """Per aperture: the ``nsigma`` depth (AB mag) from the NMAD of the blank-sky fluxes (nanomaggies,
    zero point 22.5)."""
    f, iv, fl = _columns(sky, band)
    out = np.full(f.shape[1], np.nan)
    for k in range(f.shape[1]):
        m = valid(fl[:, k], iv[:, k])
        s = nmad(f[m, k]) if m.any() else np.nan
        if np.isfinite(s) and s > 0:
            out[k] = zeropoint - 2.5 * np.log10(nsigma * s)
    return out
