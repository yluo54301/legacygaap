import numpy as np
from legacygaap import kernel as kn


def moffat(n, fwhm, beta):
    alpha = fwhm / (2 * np.sqrt(2 ** (1. / beta) - 1))
    y, x = np.mgrid[:n, :n] - n // 2
    m = (1 + (x * x + y * y) / alpha ** 2) ** (-beta)
    return m / m.sum()


def test_target_sigma_of_gaussian():
    g = kn.gaussian_image(2.5, 65)
    assert abs(kn.target_sigma(g, f=1.) / 2.5 - 1) < 2e-3
    assert abs(kn.target_sigma(g, f=1.15) / (1.15 * 2.5) - 1) < 2e-3


def test_gaussian_to_gaussian_kernel_is_analytic():
    s_in, s_t = 2.0, 2.6
    P = kn.gaussian_image(s_in, 65)
    K = kn.gaussianization_kernel(P, s_t, lam=1e-6, ksize=41)
    expect = kn.gaussian_image(np.sqrt(s_t ** 2 - s_in ** 2), 41)
    # the regularisation reshapes the quotient only where the Gaussians' transforms are tiny
    assert np.abs(K - expect).max() < 2e-2 * expect.max()
    q = kn.psf_quality(P, K, s_t, [4., 6., 8.])
    assert q['resid_peak'] < 1e-3 and all(abs(e) < 1e-3 for e in q['flux_err'].values())


def test_point_source_flux_exact_for_gaussian_psf():
    for s_t in [2.0, 3.5]:
        img = kn.gaussian_image(s_t, 65)
        for a in [3.0, 4.0, 6.0, 8.0]:
            if a > s_t:
                assert abs(kn.gaap_point_source_flux(img, a, s_t) - 1.) < 1e-3, (s_t, a)
    assert np.isnan(kn.gaap_point_source_flux(img, 2.0, 3.5))


def test_moffat_psfs_gaussianize_to_half_percent():
    """Moffat PSFs like DECam's (beta 2.5-3.5, FWHM 1.0-2.3" at 0.262"/px): point-source GAaP
    error < 0.5% for apertures >= 1.0" with the default settings."""
    pixscale = 0.262
    for beta in [2.5, 3.5]:
        for fwhm in [1.0, 1.5, 2.3]:
            P = moffat(65, fwhm / pixscale, beta)
            s_t = kn.target_sigma(P, f=1.15)
            K = kn.gaussianization_kernel(P, s_t, lam=1e-4, ksize=63)
            aps = [a / pixscale for a in (1.0, 1.5, 2.0) if a / pixscale > s_t]
            q = kn.psf_quality(P, K, s_t, aps)
            errs = list(q['flux_err'].values())
            for a, e in q['flux_err'].items():
                assert abs(e) < 5e-3, (beta, fwhm, a * pixscale, e)
            # whatever is left is a normalisation effect (ungathered wings): aperture-independent
            assert max(errs) - min(errs) < 1e-3, (beta, fwhm, errs)
            assert q['noise_factor'] < 1.
            # a 41-px kernel is not enough for heavy wings: the metric must see it
            K41 = kn.gaussianization_kernel(P, s_t, lam=1e-4, ksize=41)
            q41 = kn.psf_quality(P, K41, s_t, aps)
            assert q41['resid_peak'] >= q['resid_peak'] * 0.5


def test_kernel_with_fractional_centre():
    P = kn.gaussian_image(2.0, 65, 32.3, 31.6)
    K = kn.gaussianization_kernel(P, 2.6, lam=1e-6, ksize=41, cx=32.3, cy=31.6)
    q = kn.psf_quality(P, K, 2.6, [5., 8.], 32.3, 31.6)
    assert q['resid_peak'] < 2e-3 and all(abs(e) < 1e-3 for e in q['flux_err'].values())
    # the kernel itself is centred (no built-in shift)
    y, x = np.mgrid[:41, :41]
    assert abs((x * K).sum() - 20) < 0.02 and abs((y * K).sum() - 20) < 0.02


def test_kernel_acf_and_noise_factor():
    K = kn.gaussian_image(1.5, 21)
    acf = kn.kernel_acf(K)
    assert acf.shape == (41, 41) and abs(acf.max() - np.sum(K ** 2)) < 1e-12
    assert abs(kn.noise_factor(K) - np.sqrt(np.sum(K ** 2))) < 1e-12
