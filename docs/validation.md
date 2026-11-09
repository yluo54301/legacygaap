# Validation

The test suite has three tiers. The unit tests use synthetic data and run anywhere, including CI.
The `nersc` tests use real legacypipe products. The `legacypipe` tests compare with the code that
made the coadds.

## Unit tests (synthetic)

| Area | What is checked |
|---|---|
| GAaP measurement (`test_measure.py`) | a point source returns its total flux for any aperture; a Gaussian galaxy matches the analytic GAaP flux (Kuijken et al. 2015 Eq. A14), circular and elliptical; the error model matches the scatter over noise realisations; masked pixels and brick edges set the right flags; tractor shapes map to the right pixel-frame orientation; the FFT and direct convolutions agree |
| Gaussianization (`test_kernel.py`) | the Gaussian-to-Gaussian kernel is analytic; Moffat PSFs Gaussianize to 0.5% in point-source flux; kernels centred on fractional positions; noise autocorrelation and smoothing factor |
| Resampling and WCS (`test_resample.py`, `test_io.py`) | Lanczos-3 weights, shifts and interpolation conserve flux and position; resampling through rotated, rescaled WCSs; TAN round trips; CCD footprints, good regions and edge margins; CCD selection from the `nexp` map |
| Coadd PSF (`test_resample.py`) | the coadd PSF is the weighted mean of the CCD PSFs; the `nexp` flag and edge membership |
| PsfEx (`test_psfex.py`, `test_ibis.py`) | polynomial term order and evaluation; padding; oversampled models sampled at native pixels; noise-equivalent FWHM |
| Blank-sky statistics (`test_sky.py`) | error scales and depths recovered from simulated blank-sky tables |

## On real data (`-m nersc`)

* The rebuilt coadd PSF reproduces legacypipe's `psfsize` maps (DR11 south and north bricks). The
  CCD-scale convention is detected from them, with a ratio of 1.0000 for the right one.
* The kernels of real coadd PSFs give point-source GAaP fluxes within 0.5% (99th percentile) for
  apertures of 1.0" and more.
* `gaap-brick` end to end: schema, finite fluxes wherever the aperture is valid, point-source
  response at the coadd flux scale, and bitwise-identical reruns.
* Single-run layouts (a medium-band reduction, HSC coadds): CCD coverage from replacement WCSs, SIP
  WCS and in-file PSF models, forced-catalogue rows.

## Against legacypipe and The Tractor (`-m legacypipe`)

* PsfEx evaluation and Lanczos-3 shifts match The Tractor's.
* Coadd PSF stamps match a reference built with legacypipe's own image and PSF objects.
* HSC SIP WCSs and oversampled PSFs match legacypipe's.
* Aperture axes and orientation match tractor's rendered ellipses. The GAaP fractions of rendered
  EXP, DEV and SER models match the analytic mixture prediction.

## Science validation (development repository)

The method was developed on Legacy Surveys DR10 and DR11 bricks. It was then applied to DECam
medium-band, HSC and other legacypipe reductions, where it gave these results:

* GAaP fluxes of isolated stars agree with their catalogue PSF fluxes to about 1%, the same in every
  aperture. Without the coadd-weighting correction (`wcorr`), 17–19 mag stars were up to 9% low.
* Sources in the overlap of neighbouring bricks agree to 0.03% (g, r) and 0.13% (z) once the
  per-brick sky (`COSKY`) is restored. Without it they differ by up to 0.6%.
* Blank-sky error scales are 1.1–1.4 for DECam coadds.
* Photometric redshifts from GAaP colours had clearly fewer outliers than those from forced
  (tractor-model) fluxes of the same images.
