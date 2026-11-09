import numpy as np
from legacygaap import resample as rs, psf as P, psfex, io


def _tan(ra, dec, crpix1, crpix2, scale, nx, ny, angle_deg=0.):
    cd = scale / 3600.
    t = np.deg2rad(angle_deg)
    c, s = np.cos(t), np.sin(t)
    return rs.tan_wcs(ra, dec, crpix1, crpix2, -cd * c, cd * s, cd * s, cd * c, nx, ny)


def _gauss(n, sig, cx=None, cy=None):
    cx = n // 2 if cx is None else cx
    cy = n // 2 if cy is None else cy
    y, x = np.mgrid[:n, :n]
    g = np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / sig ** 2)
    return g / g.sum()


def _centroid(img):
    y, x = np.mgrid[:img.shape[0], :img.shape[1]]
    return (x * img).sum() / img.sum(), (y * img).sum() / img.sum()


def test_lanczos_filter_values():
    L = rs.lanczos_filter(3, np.array([0., 1., 2., -1., 3., 0.5]))
    assert L[0] == 1. and abs(L[1]) < 1e-15 and abs(L[2]) < 1e-15 and abs(L[3]) < 1e-15 and L[4] == 0.
    assert abs(L[5] - np.sinc(0.5) * np.sinc(0.5 / 3)) < 1e-12


def test_lanczos_weights_match_filter():
    rng = np.random.default_rng(0)
    d = np.concatenate([rng.uniform(-0.5, 0.5, 500), [0., 0.5, -0.5, 1e-12]])
    ks = np.arange(-3, 4)
    ref = rs.lanczos_filter(3, ks[None, :] - d[:, None])
    assert np.abs(rs.lanczos_weights(3, d) - ref).max() < 1e-13


def _slow_interpolate(img, fx, fy, order=3):
    """Loop reference (the M1 implementation): clamped edges, normalised kernel."""
    H, W = img.shape
    ix = np.floor(fx + 0.5).astype(int); iy = np.floor(fy + 0.5).astype(int)
    dx = (fx - ix).astype(np.float32).astype(np.float64); dy = (fy - iy).astype(np.float32).astype(np.float64)
    ks = np.arange(-order, order + 1)
    wx = rs.lanczos_filter(order, ks[None, :] - dx[:, None]); wy = rs.lanczos_filter(order, ks[None, :] - dy[:, None])
    out = np.zeros(len(fx)); wsum = np.zeros(len(fx))
    for a, k in enumerate(ks):
        for b, l in enumerate(ks):
            w = wx[:, a] * wy[:, b]
            out += w * img[np.clip(iy + l, 0, H - 1), np.clip(ix + k, 0, W - 1)]
            wsum += w
    return out / wsum


def test_vectorised_interpolation_matches_loop():
    rng = np.random.default_rng(5)
    img = rng.random((63, 63))
    fx = rng.uniform(-2, 64, 3000); fy = rng.uniform(-2, 64, 3000)
    assert np.abs(rs.lanczos_interpolate(img, fx, fy) - _slow_interpolate(img, fx, fy)).max() < 1e-12


def test_lanczos_shift_moves_content_and_conserves_flux():
    img = np.zeros((21, 21))
    img[10, 10] = 1.
    s = rs.lanczos_shift(img, 0.3, -0.2)
    assert abs(s.sum() - 1.) < 1e-12
    cx, cy = _centroid(s)
    assert abs(cx - 10.3) < 0.02 and abs(cy - 9.8) < 0.02
    s1 = rs.lanczos_shift(img, 1., 0.)
    assert np.allclose(s1, np.roll(img, 1, axis=1), atol=1e-15)


def test_lanczos_interpolate_integer_positions_exact():
    rng = np.random.default_rng(2)
    img = rng.random((15, 17))
    xs = np.array([0., 3., 16., 7.]); ys = np.array([0., 5., 14., 7.])
    v = rs.lanczos_interpolate(img, xs, ys)
    assert np.allclose(v, img[ys.astype(int), xs.astype(int)], atol=1e-14)


def test_resample_identity_with_integer_offset():
    dst = _tan(33.25, -4.5, 1800.5, 1800.5, 0.262, 3600, 3600)
    src = _tan(33.25, -4.5, 1800.5 - 100, 1800.5 - 50, 0.262, 2046, 4094)    # src pixel (0,0) = dst (100, 50)
    img = _gauss(63, 2.5)
    out = rs.resample_to_grid(img, 500, 700, src, dst, 100 + 500 - 10, 50 + 700 - 10, 83, 83)
    expect = np.zeros((83, 83)); expect[10:73, 10:73] = img
    assert np.allclose(out, expect, atol=1e-10)


def test_resample_rotated_rescaled_conserves_flux_and_position():
    dst = _tan(33.25, -4.5, 1800.5, 1800.5, 0.262, 3600, 3600)
    src = _tan(33.26, -4.49, 1000.5, 2000.5, 0.2637, 2046, 4094, angle_deg=30.)
    cx_src, cy_src = 1200.3, 1500.7
    img = _gauss(63, 2.5, cx=31 + 0.3, cy=31 + 0.7)         # source centred at (1200.3, 1500.7) in src pixels
    ra, dec = src.all_pix2world(cx_src, cy_src, 0)
    xd, yd = (float(v) for v in dst.all_world2pix(ra, dec, 0))
    x0, y0 = int(round(xd)) - 25, int(round(yd)) - 25
    out = rs.resample_to_grid(img, 1200 - 31, 1500 - 31, src, dst, x0, y0, 51, 51)
    area_ratio = (0.2637 / 0.262) ** 2      # surface brightness is interpolated: more, smaller pixels
    assert abs(out.sum() / area_ratio - 1.) < 1e-3
    cx, cy = _centroid(out)
    assert abs(cx + x0 - xd) < 0.02 and abs(cy + y0 - yd) < 0.02


def _fake_model(sig):
    return psfex.PsfExModel(_gauss(63, sig)[None], 0, (0., 0.), (1., 1.), fwhm=2.3548 * sig)


def _fake_coadd_model(nexp_value=2, stamp=65):
    bwcs = _tan(33.25, -4.5, 1800.5, 1800.5, 0.262, 3600, 3600)
    ccds = np.zeros(2, dtype=io.CCD_DTYPE)
    ccds['camera'] = 'fake'; ccds['width'] = 2046; ccds['height'] = 4094
    ccds['sig1'] = [1., 2.]; ccds['fwhm'] = [2.3548 * 2., 2.3548 * 4.]
    wcs = [_tan(33.25, -4.5, 1800.5 - 800, 1800.5 - 300, 0.262, 2046, 4094),      # dst (800,300) = ccd (0,0)
           _tan(33.25, -4.5, 1800.5 - 1000, 1800.5 + 100, 0.262, 2046, 4094)]
    models = {0: _fake_model(2.), 1: _fake_model(4.)}
    nexp = np.full((3600, 3600), nexp_value, np.int16)
    return P.CoaddPsfModel(bwcs, {'r': ccds}, {'r': wcs}, lambda c: models[int(c['sig1'] > 1.5)], stamp=stamp,
                           nexp_by_band={'r': nexp})


def test_coadd_psf_weighted_mean_of_two_ccds():
    m = _fake_coadd_model()
    ps = m.psf_at(1800., 1800., 'r', return_parts=True)
    assert ps.nccd == 2 and not ps.flag_nexp and not ps.flag_edge and ps.nexp_at == 2
    assert np.allclose(ps.weights, [0.8, 0.2])
    expect = np.zeros((65, 65))
    expect[1:64, 1:64] = 0.8 * _gauss(63, 2.) + 0.2 * _gauss(63, 4.)
    assert np.allclose(ps.stamp, expect, atol=1e-9)
    assert abs(ps.stamp.sum() - 1.) < 1e-12 and ps.x0 == 1800 - 32 and ps.y0 == 1800 - 32
    # legacypipe-style psfsize: harmonic-type mean of the per-CCD Gaussian-equivalent sizes
    expect_size = 2.3548 * 0.262 / np.sqrt((0.8 / 2. ** 2 + 0.2 / 4. ** 2))
    assert abs(ps.psfsize_pipeline / expect_size - 1.) < 2e-3
    assert ps.neff_fwhm > 2.3548 * 2. * 0.262     # mixture is broader than the narrow component


def test_coadd_psf_nexp_flag_and_edge_membership():
    m = _fake_coadd_model(nexp_value=3)
    ps = m.psf_at(1800., 1800., 'r')
    assert ps.flag_nexp and ps.nexp_at == 3 and ps.nccd == 2
    # 5 px inside the second CCD's left edge (ccd x = 5): inside the 10-px margin -> that CCD is excluded
    ps = m.psf_at(1005., 1800., 'r')
    assert ps.nccd == 1 and ps.ccd_idx.tolist() == [0]
    ps = m.psf_at(1015., 1800., 'r')
    assert ps.nccd == 2
    ps = m.psf_at(100., 100., 'r')
    assert ps.nccd == 0 and ps.stamp.sum() == 0 and np.isnan(ps.psfsize_pipeline)
    ps = m.psf_at(20., 1800., 'r')
    assert ps.flag_edge
