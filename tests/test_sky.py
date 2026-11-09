import numpy as np
from legacygaap import sky, measure as ms


def _table(n=4000, nap=3, sigma=(0.01, 0.02, 0.04), excess=(1.0, 1.3, 2.0), seed=3):
    rng = np.random.default_rng(seed)
    dt = [('gaapflux_r', 'f4', (nap,)), ('gaapflux_ivar_r', 'f4', (nap,)), ('gaap_flags_r', 'i2', (nap,))]
    T = np.zeros(n, dtype=dt)
    for k in range(nap):
        T['gaapflux_r'][:, k] = rng.normal(0., sigma[k] * excess[k], n)
        T['gaapflux_ivar_r'][:, k] = 1. / sigma[k] ** 2
    return T


def test_error_scale_recovers_excess_noise():
    T = _table()
    s = sky.error_scale(T, 'r')
    assert np.allclose(s, [1.0, 1.3, 2.0], rtol=0.06)


def test_error_scale_floor_and_flags():
    T = _table(excess=(0.5, 1.3, 2.0))
    T['gaap_flags_r'][:, 2] = ms.FLAG_BIGPSF
    s = sky.error_scale(T, 'r')
    assert s[0] == 1.0 and np.isnan(s[2])
    T['gaap_flags_r'][:, 1] = ms.FLAG_NEXP              # informational only: still valid
    assert np.isfinite(sky.error_scale(T, 'r')[1])


def test_depth_matches_gaussian_noise():
    T = _table(excess=(1.0, 1.0, 1.0))
    d = sky.depth(T, 'r')
    expect = 22.5 - 2.5 * np.log10(5 * np.array([0.01, 0.02, 0.04]))
    assert np.allclose(d, expect, atol=0.06)
