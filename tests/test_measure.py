import numpy as np
from legacygaap import aperture as ap, measure as ms, kernel as kn, resample as rs


def gauss2d(n, cov, flux, cx, cy):
    """flux x Gaussian with covariance ``cov`` (2x2 pixels^2) sampled at pixel centres."""
    inv = np.linalg.inv(cov)
    y, x = np.mgrid[:n, :n]
    dx, dy = x - cx, y - cy
    q = inv[0, 0] * dx * dx + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
    return flux * np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))


def setup(sigma_p=2.2, f=1.15):
    psf = kn.gaussian_image(sigma_p, 65)
    sigma_t = kn.target_sigma(psf, f=f)
    K = kn.gaussianization_kernel(psf, sigma_t)
    return psf, sigma_t, K


def test_aperture_matrix_axes_roundtrip():
    W = ap.aperture_matrix(5., 3., 0.7)
    a, b, pa = ap.axes_of(W)
    assert abs(a - 5.) < 1e-9 and abs(b - 3.) < 1e-9 and abs(pa - 0.7) < 1e-9
    assert np.allclose(ap.circular(4.), np.diag([16., 16.]))


def test_postseeing_validity_and_prefactor():
    W = ap.circular(4.)
    Wp, pref, valid = ap.postseeing(W, 3.)
    assert valid and abs(pref - 16. / 7.) < 1e-12 and np.allclose(Wp, np.diag([7., 7.]))
    _, pref, valid = ap.postseeing(ap.circular(3.2), 3.)      # post-seeing axis 1.1 px < 1.5
    assert not valid and np.isnan(pref)


def test_point_source_returns_total_flux():
    psf, sigma_t, K = setup()
    n, flux = 301, 1234.5
    for cx, cy in [(150., 150.), (150.3, 149.6)]:
        img = gauss2d(n, ap.circular(2.2), flux, cx, cy)
        ivar = np.ones_like(img)
        for sig_ap in [4., 6., 9.]:
            r = ms.gaap_flux(img, ivar, cx, cy, K, sigma_t, ap.circular(sig_ap))
            assert r.flags == 0 and abs(r.flux / flux - 1.) < 3e-3, (cx, sig_ap, r.flux / flux)
        W = ap.aperture_matrix(7., 4., 0.4)
        r = ms.gaap_flux(img, ivar, cx, cy, K, sigma_t, W)
        assert abs(r.flux / flux - 1.) < 3e-3


def test_gaussian_galaxy_matches_analytic():
    """F = flux * sqrt(det W / det(W + Sigma_g)) for a Gaussian galaxy (Kuijken+15 Eq. A14)."""
    psf, sigma_t, K = setup()
    n, flux = 301, 1000.
    cx, cy = 150.3, 150.7
    cases = [(ap.circular(3.), ap.circular(6.)), (ap.circular(5.), ap.circular(5.)),
             (ap.aperture_matrix(6., 3., 0.3), ap.circular(6.)), (ap.aperture_matrix(4., 2.5, 1.2), ap.aperture_matrix(7., 4., 0.4))]
    for Sg, W in cases:
        img = gauss2d(n, Sg + ap.circular(2.2), flux, cx, cy)        # galaxy (x) Gaussian PSF
        r = ms.gaap_flux(img, np.ones_like(img), cx, cy, K, sigma_t, W)
        expect = flux * np.sqrt(np.linalg.det(W) / np.linalg.det(W + Sg))
        assert r.flags == 0 and abs(r.flux / expect - 1.) < 5e-3, (r.flux / expect)


def test_error_model_against_noise_realisations():
    psf, sigma_t, K = setup()
    n, sig_n = 97, 0.7
    rng = np.random.default_rng(4)
    ivar = np.full((n, n), 1. / sig_n ** 2)
    W = ap.aperture_matrix(6., 4., 0.5)
    Wp, pref, valid = ap.postseeing(W, sigma_t)
    w_eff = ms.effective_weight(Wp, K, n, 48.2, 48.6)
    noise = rng.normal(0., sig_n, size=(3000, n, n))
    fluxes = pref * np.tensordot(noise, w_eff, axes=([1, 2], [0, 1]))
    pred = np.sqrt(1. / ms.measure_cutout(np.zeros((n, n)), ivar, w_eff, pref).ivar)
    assert abs(fluxes.std() / pred - 1.) < 0.04
    # non-uniform noise: variance follows the ivar map
    ivar2 = ivar * (1. + (np.mgrid[:n, :n][0] > 48))
    noise2 = rng.normal(0., 1., size=(3000, n, n)) / np.sqrt(ivar2)
    fluxes2 = pref * np.tensordot(noise2, w_eff, axes=([1, 2], [0, 1]))
    pred2 = np.sqrt(1. / ms.measure_cutout(np.zeros((n, n)), ivar2, w_eff, pref).ivar)
    assert abs(fluxes2.std() / pred2 - 1.) < 0.04


def test_masked_pixels_and_edges():
    psf, sigma_t, K = setup()
    n, flux = 301, 500.
    img = gauss2d(n, ap.circular(2.2), flux, 150., 150.)
    ivar = np.ones_like(img)
    ivar[148:153, 148:153] = 0.                     # masked patch on the source
    r = ms.gaap_flux(img, ivar, 150., 150., K, sigma_t, ap.circular(5.))
    assert r.fracmasked > 0.05 and r.flux < 0.9 * flux and r.nmasked == 25
    ivar = np.ones_like(img)
    ivar[200:260, 200:260] = 0.                     # masked patch far from the source
    r2 = ms.gaap_flux(img, ivar, 150., 150., K, sigma_t, ap.circular(5.))
    assert r2.fracmasked < 1e-6 and abs(r2.flux / flux - 1.) < 3e-3 and not (r2.flags & ms.FLAG_MASKED)
    img = gauss2d(n, ap.circular(2.2), flux, 20., 150.)
    r3 = ms.gaap_flux(img, np.ones_like(img), 20., 150., K, sigma_t, ap.circular(5.))
    assert (r3.flags & ms.FLAG_EDGE) and np.isfinite(r3.flux)


def test_tractor_shape_to_pixel_frame():
    """tractor's angle is the astronomical PA (north through east); in a brick WCS with x to
    the west and y to the north, e1 > 0 is elongated along y and e2 > 0 towards the north-east."""
    bwcs = rs.tan_wcs(33.25, -4.5, 1800.5, 1800.5, -0.262 / 3600, 0., 0., 0.262 / 3600, 3600, 3600)
    a, b, ang = ap.tractor_shape_axes(1.0, 0.3, 0.0, kappa=1., min_aper=0.5, max_aper=2.)
    q = (1 - 0.3) / (1 + 0.3)
    assert abs(ang - np.pi / 2) < 1e-12 and abs(a - np.sqrt(1. + 0.25)) < 1e-9 and abs(b - np.sqrt(q * q + 0.25)) < 1e-9
    W = ap.tractor_shape_matrix(1.0, 0.3, 0.0, bwcs, min_aper=0.5)
    ap_a, ap_b, pa = ap.axes_of(W)
    assert abs(ap_a * 0.262 - a) < 1e-9 and abs(ap_b * 0.262 - b) < 1e-9 and abs(pa - np.pi / 2) < 1e-9
    W = ap.tractor_shape_matrix(1.0, 0.0, 0.3, bwcs, min_aper=0.5)       # PA 45 deg N through E
    _, _, pa = ap.axes_of(W)
    assert abs(pa - 3 * np.pi / 4) < 1e-9                                 # north-east = up-left in pixels
    W = ap.tractor_shape_matrix(1.0, 0.0, -0.3, bwcs, min_aper=0.5)      # PA -45 deg: north-west = up-right
    _, _, pa = ap.axes_of(W)
    assert abs(pa - np.pi / 4) < 1e-9
    a, b, ang = ap.tractor_shape_axes(5.0, 0.0, 0.0, min_aper=1.0, max_aper=2.0)
    assert a == 2.0 and b == 2.0
    a, b, ang = ap.tractor_shape_axes(0.0, 0.0, 0.0, min_aper=1.0, max_aper=2.0)
    assert a == 1.0 and b == 1.0


def test_kernel_fft_matches_fftconvolve():
    from scipy.signal import fftconvolve
    rng = np.random.default_rng(1)
    K = rng.random((63, 63)); w = rng.random((97, 97))
    ref = fftconvolve(w, K[::-1, ::-1], mode='same')
    assert np.abs(ms.KernelFFT(K, 97).correlate(w) - ref).max() < 1e-9 * np.abs(ref).max()


def test_measure_source_matches_gaap_flux_and_psfcorr():
    psf, sigma_t, K = setup()
    n, flux = 301, 777.
    x, y = 150.3, 149.8
    img = gauss2d(n, ap.circular(2.2), flux, x, y)
    ivar = np.ones_like(img)
    Ws = [ap.circular(4.), ap.aperture_matrix(7., 4., 0.4), ap.circular(2.4)]       # the last is invalid (BIGPSF)
    d, v, x0, y0, trunc = ms.cutout(img, ivar, x, y)
    stamp = kn.gaussian_image(2.2, 65, 32 + (x - round(x)), 32 + (y - round(y)))      # model PSF centred on the source
    res = ms.measure_source(d, v, x0, y0, x, y, stamp, round(x) - 32, round(y) - 32, sigma_t, K, Ws)
    for W, r in zip(Ws[:2], res[:2]):
        ref = ms.gaap_flux(img, ivar, x, y, K, sigma_t, W)
        assert abs(r.flux / ref.flux - 1) < 1e-9 and abs(r.ivar / ref.ivar - 1) < 1e-9
        assert abs(r.psfcorr - 1) < 3e-3 and r.flags == 0
    assert res[2].flags & ms.FLAG_BIGPSF and np.isnan(res[2].flux)
