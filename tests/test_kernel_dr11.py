import numpy as np
import pytest
from legacygaap import io, psf as P, kernel as kn

pytestmark = pytest.mark.nersc
APERTURES = (1.0, 1.5, 2.0)


@pytest.mark.parametrize('brick,bands', [('0332m045', ['g', 'z']), ('1700p352', ['g', 'z'])])
def test_real_psfs_gaussianize(brick, bands):
    """Point-source GAaP flux error of the Gaussianized coadd PSF < 0.5% (99th percentile) for
    apertures >= 1.0" that are larger than the target PSF, at 20 random positions per band."""
    p = io.BrickPaths(brick)
    model = P.CoaddPsfModel.for_brick(p, bands=bands, load_nexp=False)
    rng = np.random.default_rng(9)
    for band in bands:
        errs = {a: [] for a in APERTURES}
        for _ in range(20):
            x, y = rng.uniform(100, 3500, 2)
            ps = model.psf_at(x, y, band)
            if ps.nccd == 0:
                continue
            s_t = kn.target_sigma(ps.stamp, f=1.15)
            K = kn.gaussianization_kernel(ps.stamp, s_t, cx=x - ps.x0, cy=y - ps.y0)
            q = kn.psf_quality(ps.stamp, K, s_t, [a / model.pixscale for a in APERTURES], x - ps.x0, y - ps.y0)
            assert q['resid_peak'] < 0.01 and q['noise_factor'] < 1.
            for a in APERTURES:
                e = q['flux_err'][a / model.pixscale]
                if np.isfinite(e):
                    errs[a].append(e)
        for a, e in errs.items():
            if len(e) >= 5:
                assert np.percentile(np.abs(e), 99) < 5e-3, (brick, band, a, np.percentile(np.abs(e), 99))
