"""GAaP flux fractions of tractor galaxy models (pre-seeing), for QA.

tractor renders EXP/REX and DEV galaxies as mixtures of Gaussians (Hogg & Lang 2013); the
amplitudes and variances below (units of shape_r^2) are copied from tractor.mixture_profiles
(exp_amp/exp_var, dev_amp/dev_var; DR11.2.0 image). Each component of a galaxy with shape
(r, e1, e2) has covariance var_k * C, with C the tractor ellipse covariance (axes r and r*q).
The GAaP flux of the pre-seeing model with aperture W is then analytic (Kuijken+15 Eq. A14):
F_W / F_total = sum_k amp_k sqrt(det W / det(W + var_k C)). SER galaxies are not covered.
"""
import os
import numpy as np
from . import aperture as ap

_SERSIC = None


def sersic_mixture(n):
    """tractor's SersicMixture for index n (nearest entry of a 0.05-step table, 0.5 <= n <= 6)."""
    global _SERSIC
    if _SERSIC is None:
        _SERSIC = dict(np.load(os.path.join(os.path.dirname(__file__), 'data', 'sersic_mixtures.npz')))
    i = int(np.argmin(np.abs(_SERSIC['n'] - float(n))))
    keep = _SERSIC['amp'][i] != 0
    return _SERSIC['amp'][i][keep], _SERSIC['var'][i][keep]


def get_mixture(typ, sersic=np.nan):
    """(amp, var) of the tractor profile of a catalog type, or None for point sources."""
    typ = str(typ).strip()
    if typ in ('PSF', 'DUP', ''):
        return None
    if typ in PROFILES:
        return PROFILES[typ]
    if typ == 'SER' and np.isfinite(sersic):
        return sersic_mixture(sersic)
    return PROFILES['EXP']

EXP_AMP = np.array([0.00019948597671798522, 0.0026161267931217196, 0.018972665516669755, 0.10018654376916275,
                    0.36853448400705735, 0.5094906939372705])
EXP_VAR = np.array([0.00120078965, 0.00884526493, 0.0391463084, 0.139976817, 0.4609625, 1.50159566])
DEV_AMP = np.array([0.010233, 0.0019977228467993897, 0.011251558613407773, 0.03213934311148659, 0.0711927728353375,
                    0.1328982628748762, 0.20919977462456607, 0.2682266735062975, 0.262860891587229])
DEV_VAR = np.array([0.0, 0.000223759216, 0.00100220099, 0.00418731126, 0.0169432589, 0.0684850479, 0.28720708,
                    1.33320254, 8.40215071])
PROFILES = {'EXP': (EXP_AMP, EXP_VAR), 'REX': (EXP_AMP, EXP_VAR), 'DEV': (DEV_AMP, DEV_VAR)}


def shape_covariance(shape_r, e1, e2):
    """tractor ellipse covariance C (arcsec^2, (east, north) frame): axes shape_r and shape_r*q."""
    a, b, ang = ap.tractor_shape_axes(shape_r, e1, e2, kappa=1., min_aper=0., max_aper=np.inf)
    return ap.aperture_matrix(a, b, ang)


def model_gaap_fraction(typ, shape_r, e1, e2, W_sky, sersic=np.nan):
    """GAaP flux / total flux of a tractor model for an aperture matrix W_sky (arcsec^2, (east,
    north) frame). 1 for point sources."""
    mix = get_mixture(typ, sersic)
    if mix is None:
        return 1.0
    if not np.isfinite(shape_r) or shape_r <= 0:
        return np.nan
    amp, var = mix
    C = shape_covariance(shape_r, e1, e2)
    detW = np.linalg.det(W_sky)
    return float(sum(a * np.sqrt(detW / np.linalg.det(W_sky + v * C)) for a, v in zip(amp, var)))
