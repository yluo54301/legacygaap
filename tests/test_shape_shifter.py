"""Position-angle convention check against tractor's own rendering (run via bin/sh.sh)."""
import numpy as np
import pytest
from legacygaap import aperture as ap, resample as rs

pytestmark = pytest.mark.legacypipe
pytest.importorskip('tractor')


def render(e1, e2, re=2.0, n=201, ra=33.25, dec=-4.5, pixscale=0.262):
    from tractor import Tractor, Image, ExpGalaxy, RaDecPos, NanoMaggies, ConstantSky, ConstantFitsWcs, LinearPhotoCal, NCircularGaussianPSF
    from tractor.ellipses import EllipseE
    from astrometry.util.util import Tan
    cd = pixscale / 3600.
    wcs = Tan(ra, dec, n / 2 + 0.5, n / 2 + 0.5, -cd, 0., 0., cd, n, n)
    tim = Image(data=np.zeros((n, n), np.float32), invvar=np.ones((n, n), np.float32), psf=NCircularGaussianPSF([1.0], [1.]),
                wcs=ConstantFitsWcs(wcs), sky=ConstantSky(0.), photocal=LinearPhotoCal(1., band='r'))
    gal = ExpGalaxy(RaDecPos(ra, dec), NanoMaggies(r=1000.), EllipseE(re, e1, e2))
    mod = Tractor([tim], [gal]).getModelImage(0)
    bwcs = rs.tan_wcs(ra, dec, n / 2 + 0.5, n / 2 + 0.5, -cd, 0., 0., cd, n, n)
    return np.asarray(mod, dtype=np.float64), bwcs


def moments_angle(img):
    y, x = np.mgrid[:img.shape[0], :img.shape[1]]
    w = img / img.sum()
    cx, cy = (w * x).sum(), (w * y).sum()
    mxx, myy, mxy = (w * (x - cx) ** 2).sum(), (w * (y - cy) ** 2).sum(), (w * (x - cx) * (y - cy)).sum()
    return (0.5 * np.arctan2(2 * mxy, mxx - myy)) % np.pi, mxx, myy, mxy


@pytest.mark.parametrize('e1,e2', [(0.4, 0.0), (0.0, 0.4), (0.3, 0.3), (0.2, -0.35)])
def test_aperture_axes_match_tractor_ellipse(e1, e2):
    """With kappa = 1 and no MIN/MAX, the aperture is tractor's ellipse: axes re and re*q."""
    from tractor.ellipses import EllipseE
    G = np.array(EllipseE(2.0, e1, e2).getRaDecBasis()) * 3600.        # (dRA*cos, dDec) per unit re
    C = G @ G.T
    a, b, ang = ap.tractor_shape_axes(2.0, e1, e2, kappa=1., min_aper=0., max_aper=100.)
    W = ap.aperture_matrix(a, b, ang)                                   # (east, north) frame
    assert np.allclose(W, C, atol=1e-9), (W, C)


@pytest.mark.parametrize('e1,e2', [(0.4, 0.0), (0.0, 0.4), (-0.4, 0.0), (0.3, 0.3), (0.2, -0.35)])
def test_aperture_orientation_matches_tractor_rendering(e1, e2):
    img, bwcs = render(e1, e2)
    pa_img, mxx, myy, mxy = moments_angle(img)
    W = ap.tractor_shape_matrix(2.0, e1, e2, bwcs, kappa=1., min_aper=0.0, max_aper=100.)
    _, _, pa_ap = ap.axes_of(W)
    d = abs(pa_img - pa_ap) % np.pi
    d = min(d, np.pi - d)
    assert d < np.deg2rad(1.0), (e1, e2, np.rad2deg(pa_img), np.rad2deg(pa_ap))
    # the aperture's elongation direction matches the rendered galaxy's (same sign of the moment asymmetry)
    assert np.sign(W[0, 0] - W[1, 1]) == np.sign(mxx - myy) or abs(mxx - myy) < 1e-3 * (mxx + myy)
    assert np.sign(W[0, 1]) == np.sign(mxy) or abs(mxy) < 1e-3 * (mxx + myy)


@pytest.mark.parametrize('kind,re,e1,e2', [('EXP', 1.5, 0.3, -0.2), ('DEV', 1.2, -0.1, 0.25), ('EXP', 0.6, 0.0, 0.0), ('DEV', 3.0, 0.2, 0.1)])
def test_model_gaap_fraction_matches_tractor_rendering(kind, re, e1, e2):
    """Render the tractor model through a Gaussian PSF, measure the GAaP flux with Eq. A16 directly
    (the PSF is already Gaussian), and compare with the analytic mixture prediction."""
    from tractor import Tractor, Image, ExpGalaxy, DevGalaxy, RaDecPos, NanoMaggies, ConstantSky, ConstantFitsWcs, LinearPhotoCal, NCircularGaussianPSF
    from tractor.ellipses import EllipseE
    from astrometry.util.util import Tan
    from legacygaap import profiles
    n, ra, dec, pix, sig_p = 301, 33.25, -4.5, 0.262, 2.5
    cd = pix / 3600.
    tim = Image(data=np.zeros((n, n), np.float32), invvar=np.ones((n, n), np.float32), psf=NCircularGaussianPSF([sig_p], [1.]),
                wcs=ConstantFitsWcs(Tan(ra, dec, n / 2 + 0.5, n / 2 + 0.5, -cd, 0., 0., cd, n, n)), sky=ConstantSky(0.),
                photocal=LinearPhotoCal(1., band='r'))
    cls = ExpGalaxy if kind == 'EXP' else DevGalaxy
    gal = cls(RaDecPos(ra, dec), NanoMaggies(r=1000.), EllipseE(re, e1, e2))
    img = np.asarray(Tractor([tim], [gal]).getModelImage(0), dtype=np.float64)
    bwcs = rs.tan_wcs(ra, dec, n / 2 + 0.5, n / 2 + 0.5, -cd, 0., 0., cd, n, n)
    for W_sky in [ap.circular(1.0), ap.circular(2.0), ap.aperture_matrix(2.0, 1.2, 0.5)]:
        W = ap.to_pixel_frame(W_sky, bwcs)
        Wp, pref, valid = ap.postseeing(W, sig_p)
        assert valid
        measured = pref * np.sum(img * ap.weight_image(Wp, n, n // 2, n // 2)) / img.sum()
        predicted = profiles.model_gaap_fraction(kind, re, e1, e2, W_sky)
        assert abs(measured / predicted - 1) < 3e-3, (kind, re, measured, predicted)


@pytest.mark.parametrize('n,re,e1,e2', [(0.8, 1.2, 0.2, 0.1), (2.5, 1.5, -0.1, 0.3), (5.0, 1.0, 0.0, 0.0)])
def test_sersic_model_gaap_fraction_matches_tractor_rendering(n, re, e1, e2):
    from tractor import Tractor, Image, RaDecPos, NanoMaggies, ConstantSky, ConstantFitsWcs, LinearPhotoCal, NCircularGaussianPSF
    from tractor.sersic import SersicGalaxy, SersicIndex
    from tractor.ellipses import EllipseE
    from astrometry.util.util import Tan
    from legacygaap import profiles
    npix, ra, dec, pix, sig_p = 301, 33.25, -4.5, 0.262, 2.5
    cd = pix / 3600.
    tim = Image(data=np.zeros((npix, npix), np.float32), invvar=np.ones((npix, npix), np.float32), psf=NCircularGaussianPSF([sig_p], [1.]),
                wcs=ConstantFitsWcs(Tan(ra, dec, npix / 2 + 0.5, npix / 2 + 0.5, -cd, 0., 0., cd, npix, npix)), sky=ConstantSky(0.),
                photocal=LinearPhotoCal(1., band='r'))
    gal = SersicGalaxy(RaDecPos(ra, dec), NanoMaggies(r=1000.), EllipseE(re, e1, e2), SersicIndex(n))
    img = np.asarray(Tractor([tim], [gal]).getModelImage(0), dtype=np.float64)
    bwcs = rs.tan_wcs(ra, dec, npix / 2 + 0.5, npix / 2 + 0.5, -cd, 0., 0., cd, npix, npix)
    for W_sky in [ap.circular(1.0), ap.aperture_matrix(2.0, 1.2, 0.5)]:
        Wp, pref, valid = ap.postseeing(ap.to_pixel_frame(W_sky, bwcs), sig_p)
        measured = pref * np.sum(img * ap.weight_image(Wp, npix, npix // 2, npix // 2)) / img.sum()
        predicted = profiles.model_gaap_fraction('SER', re, e1, e2, W_sky, sersic=n)
        assert abs(measured / predicted - 1) < 3e-3, (n, re, measured, predicted)
