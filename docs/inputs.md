# Inputs

`gaap-brick` reads the products of a legacypipe run. It never writes to the survey directory. Its
caches go to `$GAAP_CACHE_DIR`, by default `~/.cache/legacygaap`.

## Survey directory layouts

**Data release** (DR10, DR11): `--survey-dir` contains `south/` and `north/`, each with `coadd/`,
`tractor/` and `metrics/`. The region is the one whose `tractor/` holds the brick. When both do, the
data-release rule applies: `north` at Dec ≥ 32.375°, `south` below, unless `--region` says otherwise.

**Single run** ("flat"): `coadd/`, `tractor/` and `metrics/` sit directly under `--survey-dir`, as
for a legacypipe run made with `--outdir`. The bands are those with an image coadd.

Per brick, with `<bbb>` the first three characters of the brick name:

| File | Used for |
|---|---|
| `coadd/<bbb>/<brick>/legacysurvey-<brick>-image-<b>.fits.fz` | the image measured (`--image-kind` picks another kind) |
| `coadd/<bbb>/<brick>/legacysurvey-<brick>-invvar-<b>.fits.fz` | errors and the masked-pixel fraction |
| `coadd/<bbb>/<brick>/legacysurvey-<brick>-nexp-<b>.fits.fz` | CCD selection checks (flag 32) |
| `coadd/<bbb>/<brick>/legacysurvey-<brick>-psfsize-<b>.fits.fz` | pixel-scale convention (`--pixscale auto`; optional) |
| `coadd/<bbb>/<brick>/legacysurvey-<brick>-maskbits.fits.fz` | blank-sky positions (`--sky`) |
| `coadd/<bbb>/<brick>/legacysurvey-<brick>-ccds.fits` | the brick's CCDs and the sub-images legacypipe read |
| `tractor/<bbb>/tractor-<brick>.fits` | positions, shapes, types and fluxes (or `--catalog`) |
| `metrics/<bbb>/blobs-<brick>.fits.gz` | blank-sky positions (`--sky`; or from `--sky-ref-dir`) |
| `calib/psfex/<image path>-psfex.fits` | merged PsfEx models, one row per CCD |
| `survey-ccds-<camera>-*.kd.fits` | CCDs of bands missing from the brick CCD table |
| `<images root>/<image_filename>` | image headers for the exact CCD WCS (read once, then cached) |

The images root is `--images-root`, else `$GAAP_IMAGES_ROOT`, else `<survey-dir>/images`. In the
Legacy Surveys data releases, `image_filename` is relative to the directory that holds `decam/`,
`90prime/` and `mosaic/`.

## Cameras

| Camera | PSF model | CCD WCS |
|---|---|---|
| `decam`, `90prime`, `mosaic` | merged PsfEx files in `calib/psfex/` | TPV/SIP from the image header |
| `hsc` | PsfEx model inside each CORR/CALEXP file (oversampled, sampled at the native pixels) | SIP from the image header |
| `lsstcoadd` | cell PSFs inside each Rubin deep-coadd patch file | TAN from the CCD table |
| `megaprime` | merged PsfEx files in `calib/psfex/` | TAN from the CCD table |

The good region of each CCD follows legacypipe's `get_good_image_subregion` (for example DECam S7,
the MegaCam data section, or the inner 3000 × 3000 pixels of a Rubin patch).

## Catalogue

By default each brick's tractor catalogue is measured. `--catalog PATTERN` measures another
catalogue instead. It must have the same rows; `PATTERN` may contain `{brick}` and `{bbb}`. A typical
use is a forced-photometry catalogue that puts one survey's models on another survey's coadds.
Positions come from `ra` and `dec`, shapes from `shape_r`, `shape_e1` and `shape_e2`, types from
`type` (and `sersic`). The coadd-weighting correction takes fluxes from `flux_<b>`, or else from
`forced_flux_<b>`. Gaia stars are moved to each band's epoch when `pmra`, `pmdec` and `ref_epoch`
are present.

`--rows FILE` (row indices) and `--limit N` measure a subset; the output then has those rows only.

## Options for special cases

| Option | When |
|---|---|
| `--images-root DIR` | the image files are not under `<survey-dir>/images` |
| `--wcs-ccds TABLE --wcs-images-root DIR --wcs-tag T` | the images the run was built from are gone and a re-reduction of the same CCDs exists. Each CCD's WCS and astrometric offsets then come from the same (expnum, ccdname) in `TABLE` (cached separately under `T`) |
| `--survey-ccds DIR:CAM[,CAM]` | a single run whose forced bands are not in the brick CCD tables and whose survey CCD tables live elsewhere |
| `--sky-ref-dir DIR` | measuring coadds of another survey on the same brick grid without its own blob map (blank-sky positions from `DIR`) |
| `--image-kind K` | measure `legacysurvey-<brick>-<K>-<b>.fits.fz`: `model` and `blobmodel` for tests, or your own processed coadd (e.g. background-subtracted) |
| `--pixscale nominal\|cd` | force legacypipe's CCD pixel-scale convention when there is no `psfsize` map |
| `--star-mags LO,HI`, `--iso-dmag D` | magnitude window and isolation rule of the stars that calibrate the shot-noise strength (deep catalogues need `--iso-dmag`) |
