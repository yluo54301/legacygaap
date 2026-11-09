"""Gaussian apertures: aperture matrices (Kuijken+15 Eq. A15), the post-seeing shrink, and the
mapping from tractor model shapes to pre-seeing apertures.

Conventions: an aperture matrix W is the covariance-like 2x2 matrix of the Gaussian weight
exp(-1/2 x^T W^-1 x); ``a`` and ``b`` are the dispersions along the major and minor axes, ``pa``
the angle of the major axis from the +x axis towards +y (radians), in whichever frame W lives.
Sky-frame apertures live in (east, north) arcsec offsets; pixel-frame ones in brick pixels.
"""
import numpy as np
from .kernel import MIN_SIGMA_PRIME


def aperture_matrix(a, b, pa=0.):
    """2x2 aperture matrix for dispersions a (major) and b (minor) at angle pa (radians)."""
    c, s = np.cos(pa), np.sin(pa)
    return np.array([[a * a * c * c + b * b * s * s, (a * a - b * b) * s * c],
                     [(a * a - b * b) * s * c, a * a * s * s + b * b * c * c]], dtype=np.float64)


def circular(sigma):
    return np.diag([sigma * sigma, sigma * sigma]).astype(np.float64)


def axes_of(W):
    """(a, b, pa) of an aperture matrix."""
    vals, vecs = np.linalg.eigh(W)
    a, b = np.sqrt(vals[1]), np.sqrt(vals[0])
    v = vecs[:, 1]
    return a, b, np.arctan2(v[1], v[0]) % np.pi


def postseeing(W, sigma_t, min_sigma_prime=MIN_SIGMA_PRIME):
    """Post-seeing aperture W' = W - sigma_t^2 1 and the prefactor sqrt(det W / det W') (Eq. A16).
    Returns (W', prefactor, valid): valid requires the smallest axis of W' >= min_sigma_prime."""
    Wp = W - sigma_t ** 2 * np.eye(2)
    vals = np.linalg.eigvalsh(Wp)
    valid = bool(vals.min() >= min_sigma_prime ** 2)
    if not valid:
        return Wp, np.nan, False
    pref = float(np.sqrt(np.linalg.det(W) / np.linalg.det(Wp)))
    return Wp, pref, True


def weight_image(Wp, n, cx, cy):
    """exp(-1/2 x^T Wp^-1 x) on an n x n grid, x measured from (cx, cy) in pixels."""
    inv = np.linalg.inv(Wp)
    y, x = np.mgrid[:n, :n]
    dx, dy = x - cx, y - cy
    q = inv[0, 0] * dx * dx + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
    return np.exp(-0.5 * q)


def sky_to_pixel_matrix(bwcs):
    """Linear map M from (east, north) arcsec offsets to brick pixel offsets: x_pix = M @ xi."""
    cd = np.asarray(bwcs.wcs.cd, dtype=np.float64)     # deg per pixel, (xi, eta) = cd @ (dx, dy)
    return np.linalg.inv(cd) / 3600.


def to_pixel_frame(W_sky, bwcs):
    """Transform an aperture matrix from the sky frame (arcsec^2) to brick pixels^2."""
    M = sky_to_pixel_matrix(bwcs)
    return M @ W_sky @ M.T


def tractor_shape_axes(shape_r, e1, e2, kappa=1.0, min_aper=1.0, max_aper=2.0):
    """Pre-seeing aperture (a, b, angle) in the sky frame from a tractor galaxy model.

    tractor conventions (verified with ``EllipseE.getRaDecBasis`` and renderings in
    tests/test_shape_shifter.py): shape_r is the half-light radius along the MAJOR axis, the
    minor axis is shape_r * q with q = (1-|e|)/(1+|e|), and theta = atan2(e2, e1)/2 is the
    astronomical position angle of the major axis, north through east. The returned angle is
    measured from east towards north (``aperture_matrix`` convention in the (east, north)
    frame): pi/2 - theta. Aperture axes: a = kappa*shape_r and b = kappa*shape_r*q, each with
    min_aper added in quadrature (KiDS DR4 Eq. 8) and capped at max_aper.
    """
    e = float(np.hypot(e1, e2))
    e = min(e, 0.999)
    q = (1. - e) / (1. + e)
    theta = 0.5 * np.arctan2(e2, e1)
    r = kappa * max(float(np.nan_to_num(shape_r)), 0.)
    a = min(np.sqrt(r ** 2 + min_aper ** 2), max_aper)
    b = min(np.sqrt((r * q) ** 2 + min_aper ** 2), max_aper)
    return a, b, np.pi / 2 - theta


def tractor_shape_matrix(shape_r, e1, e2, bwcs, kappa=1.0, min_aper=1.0, max_aper=2.0):
    """Pre-seeing aperture matrix in brick pixels^2 from a tractor model shape."""
    a, b, theta = tractor_shape_axes(shape_r, e1, e2, kappa, min_aper, max_aper)
    return to_pixel_frame(aperture_matrix(a, b, theta), bwcs)


def circular_matrix(sigma_arcsec, bwcs):
    """Circular pre-seeing aperture of dispersion sigma (arcsec) in brick pixels^2."""
    return to_pixel_frame(circular(sigma_arcsec), bwcs)
