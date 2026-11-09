# Changelog

## 0.1.0 (2026-11)

First standalone release, split out of the development repository in which the method was built
and validated on Legacy Surveys DR10/DR11, IBIS medium-band, HSC and other legacypipe reductions.

* `gaap-brick` afterburner: GAaP fluxes for every row of a legacypipe brick catalogue, and at
  blank-sky positions (`--sky`).
* Position-dependent coadd PSF rebuilt from per-CCD PsfEx models (DECam, 90Prime, Mosaic3,
  MegaCam), HSC CORR PsfEx models and Rubin deep-coadd cell PSFs.
* Regularised Fourier Gaussianization kernel; circular and shape-matched elliptical apertures;
  exact linear error propagation.
* legacypipe-specific corrections: coadd flux scale (`psfcorr`), source-dependent coadd weights
  (`wcorr`), correlated noise of resampled coarse pixels (`escale`), proper motions to each band's
  epoch; per-brick sky offset recorded (`COSKY_<B>`).
* `legacygaap.sky`: empirical error scales and depths from the blank-sky table.
* Reads blob maps in both layouts (DR10: primary HDU; DR11 and later: extension `BLOB-MAP`).
* Configuration through `LEGACY_SURVEY_DIR`, `GAAP_IMAGES_ROOT` and `GAAP_CACHE_DIR`.
