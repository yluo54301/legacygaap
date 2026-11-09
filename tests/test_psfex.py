import numpy as np
from legacygaap import psfex


def _gauss(n, sig, dx=0., dy=0.):
    y, x = np.mgrid[:n, :n] - n // 2
    g = np.exp(-0.5 * ((x - dx) ** 2 + (y - dy) ** 2) / sig ** 2)
    return g / g.sum()


def test_term_order_matches_psfex_convention():
    xp, yp = psfex.PsfExModel._term_powers(2)
    assert list(zip(xp, yp)) == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)]
    xp, yp = psfex.PsfExModel._term_powers(0)
    assert list(zip(xp, yp)) == [(0, 0)]


def test_evaluation_is_polynomial_combination():
    rng = np.random.default_rng(0)
    bases = np.stack([_gauss(25, 2.0)] + [0.01 * rng.normal(size=(25, 25)) for _ in range(5)]).astype(np.float32)
    m = psfex.PsfExModel(bases, 2, polzero=(1000., 2000.), polscal=(2000., 4000.))
    x, y = 1500., 1000.
    dx, dy = (x - 1000.) / 2000., (y - 2000.) / 4000.
    expect = (bases[0] + dx * bases[1] + dx ** 2 * bases[2] + dy * bases[3] + dx * dy * bases[4] + dy ** 2 * bases[5]).astype(np.float64)
    got = m.at(x, y, normalize=False)
    assert np.allclose(got, expect, atol=1e-7)
    assert abs(m.at(x, y).sum() - 1.) < 1e-12
    m0 = psfex.PsfExModel(bases[:1], 0, polzero=(0., 0.), polscal=(1., 1.))
    assert np.allclose(m0.at(3., 4., normalize=False), bases[0])


def test_from_row_trims_padding():
    row = np.zeros(1, dtype=[('poldeg1', 'i4'), ('psfaxis1', 'i4'), ('psfaxis2', 'i4'), ('psf_mask', 'f4', (8, 31, 31)),
                             ('psf_samp', 'f4'), ('polzero1', 'f8'), ('polzero2', 'f8'), ('polscal1', 'f8'),
                             ('polscal2', 'f8'), ('psf_fwhm', 'f4')])[0]
    row['poldeg1'], row['psfaxis1'], row['psfaxis2'] = 2, 25, 25
    row['psf_mask'][:6, :25, :25] = 1.
    row['psf_samp'], row['polscal1'], row['polscal2'], row['psf_fwhm'] = 1., 1., 1., 4.
    m = psfex.PsfExModel.from_row(row)
    assert m.bases.shape == (6, 25, 25) and m.fwhm == 4. and m.sampling == 1.


def test_neff_fwhm_of_gaussian():
    sig = 3.0
    g = _gauss(101, sig)
    assert abs(psfex.neff_fwhm(g, 0.262) / (2.3548 * sig * 0.262) - 1) < 1e-3
