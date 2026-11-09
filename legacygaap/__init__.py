"""legacygaap: Gaussian-Aperture-and-PSF (GAaP) photometry for legacypipe (Legacy Surveys) products.

GAaP (Kuijken 2008; Kuijken et al. 2015) measures fluxes through a Gaussian aperture defined before
seeing and identical in every band, so that colours do not depend on the PSF. legacygaap measures
it on legacypipe brick coadds, at the positions and shapes of a legacypipe (tractor) catalogue, with
a position-dependent coadd PSF rebuilt from the per-CCD PSF models.

The core depends only on numpy, scipy, astropy, fitsio and matplotlib. legacypipe, tractor and
astrometry.net are used only by ``compat_legacypipe``, for the comparison tests.
"""
__version__ = '0.1.0'
