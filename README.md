# legacygaap

Gaussian-Aperture-and-PSF (GAaP) photometry for [legacypipe](https://github.com/legacysurvey/legacypipe)
products: the Legacy Surveys data releases and any other legacypipe reduction.

GAaP (Kuijken 2008; Kuijken et al. 2015, 2019) measures each source through a Gaussian aperture that
is defined *before* seeing and is the same in every band. Each band's image is first convolved so
that its PSF becomes a circular Gaussian ("Gaussianized"). The aperture is then shrunk by that
Gaussian, so that the measured flux equals the flux through the pre-seeing aperture. Colours built
from GAaP fluxes therefore do not depend on the PSF of each band, which is what photometric redshifts
and SED fits need. For a point source the GAaP flux is the total flux, whatever the aperture.

legacygaap is to legacypipe what [meas_extensions_gaap](https://github.com/lsst/meas_extensions_gaap)
is to the LSST Science Pipelines. It runs after legacypipe, on the products of a run: the brick
coadds, the tractor catalogue, the CCD tables and the PSF models. It writes one row of GAaP fluxes
per catalogue row, so its output joins the tractor catalogue row by row.

## What it does

For every source and band, `gaap-brick`:

1. **Rebuilds the coadd PSF at the source.** A legacypipe coadd is an inverse-variance-weighted sum
   of resampled CCD images, so its PSF at a position is the same weighted sum of the per-CCD PSF
   models, each evaluated on its CCD and resampled with legacypipe's Lanczos-3 kernel. The CCD list,
   WCS, weights and flux scaling (`sbscale`) follow legacypipe's conventions.
2. **Gaussianizes it.** A regularised Fourier kernel turns the coadd PSF into a circular Gaussian
   1.15 times its width, and that kernel is applied to a cutout of the coadd.
3. **Measures the apertures.** By default: circular apertures of dispersion 1.0", 1.5" and 2.0", and
   shape-matched elliptical apertures from the tractor shape, with minimum axes of 1.0" and 1.5"
   (KiDS's MIN_APER rule), capped at 2.0". An aperture is valid only if it is larger than the target
   Gaussian PSF.
4. **Propagates the errors exactly.** The flux is a linear function of the coadd pixels, so its
   variance follows from the inverse-variance map without an autocorrelation model. Masked pixels
   are excluded and their share of the weight is reported.
5. **Corrects for how legacypipe builds its coadds.** It divides out the point-source response
   (which includes the coadd flux scale) and the effect of source-dependent pixel weights on bright
   sources. It scales the errors for the correlated noise of coarse-pixel cameras and moves stars
   with proper motions to each band's epoch. The per-brick sky level that legacypipe subtracts is
   recorded in the header.

With `--sky N` it also measures N blank-sky positions with the same code. `legacygaap.sky` turns that
table into empirical error scales and depths.

The core needs only numpy, scipy, astropy, fitsio and matplotlib. It does not import legacypipe or
The Tractor; it reproduces their conventions, and the `legacypipe` tests check it against them.

## Installation

```bash
pip install git+https://github.com/yluo54301/legacygaap
# or, for development
git clone https://github.com/yluo54301/legacygaap && cd legacygaap
pip install -e ".[test]"
```

Python 3.10 or later. No legacypipe installation is needed to run it.

## Quick start

```bash
export LEGACY_SURVEY_DIR=/path/to/legacysurvey/dr10     # as for legacypipe
export GAAP_IMAGES_ROOT=/path/to/images                  # where image_filename lives (default $LEGACY_SURVEY_DIR/images)
gaap-brick --brick 0001m002 --bands g,r,z --threads 32 --sky 1000 --outdir gaap
```

This writes `gaap/000/gaap-0001m002.fits` (one row per row of `tractor-0001m002.fits`) and
`gaap/000/gaap-sky-0001m002.fits`.

```python
import fitsio
from legacygaap import sky

T = fitsio.read('gaap/000/gaap-0001m002.fits')
S = fitsio.read('gaap/000/gaap-sky-0001m002.fits')
h = fitsio.read_header('gaap/000/gaap-0001m002.fits', ext='GAAP')
apertures = [h[f'AP{k}'] for k in range(h['NAP'])]      # circ1.0, circ1.5, circ2.0, ell1.0, ell1.5
k = apertures.index('circ1.0')
ok = (T['gaap_flags_r'][:, k] & ~32) == 0               # flag 32 is informational
flux_r = T['gaapflux_r'][:, k]                           # nanomaggies
err_r = sky.error_scale(S, 'r')[k] / T['gaapflux_ivar_r'][:, k] ** 0.5
```

`gaap-brick --help` lists every option. The main ones:

| Option | Default | Meaning |
|---|---|---|
| `--survey-dir` | `$LEGACY_SURVEY_DIR` | a data release (`south/`, `north/`) or a single legacypipe run |
| `--bands` | every optical band with a coadd | comma list |
| `--circ` | `1.0,1.5,2.0` | circular pre-seeing aperture dispersions (arcsec) |
| `--min-aper`, `--max-aper` | `1.0,1.5`, `2.0` | shape-matched elliptical apertures (arcsec) |
| `--scale` | `1.15` | target Gaussian PSF width / PSF width |
| `--catalog` | the run's tractor catalogue | any catalogue with the same rows, e.g. forced photometry |
| `--sky` | `0` | number of blank-sky positions to measure |
| `--threads` | `1` | worker processes |

## Output

One FITS table per brick (extension `GAAP`), row-aligned with the catalogue. Per band `<b>` and
aperture:

| Column | Content |
|---|---|
| `gaapflux_<b>`, `gaapflux_ivar_<b>` | GAaP flux (nanomaggies, AB zero point 22.5) and inverse variance |
| `gaap_flags_<b>` | 1 aperture not larger than the PSF, 2 cutout truncated by the brick edge, 4 > 10% of the weight on masked pixels, 8 no PSF, 16 poor Gaussianization, 32 CCD count differs from the nexp map (informational) |
| `gaap_fracmasked_<b>` | share of the aperture weight on masked pixels |
| `gaap_a`, `gaap_b`, `gaap_pa` | aperture axes (arcsec) and position angle (deg, north through east) |

Per band there are also the target PSF width, the coadd PSF FWHM, the number of CCDs and the
correction factors. Fluxes are not corrected for Galactic extinction. [docs/outputs.md](docs/outputs.md)
lists every column and header keyword.

## Supported inputs

* Legacy Surveys data releases (DR10 and DR11 layout, `south/` and `north/` trees): DECam,
  90Prime and Mosaic3, with the PsfEx models in `calib/psfex/`.
* Single legacypipe runs (`coadd/`, `tractor/` and `metrics/` directly under the run directory),
  such as DECam medium-band reductions. Also HSC CORR images (PsfEx model and SIP WCS inside each
  file), Rubin deep-coadd patches treated as CCDs (camera `lsstcoadd`, cell PSFs) and CFHT
  MegaCam (camera `megaprime`).
* Catalogues: the run's tractor catalogue, or any catalogue with the same rows (e.g. forced
  photometry of one survey's models on another's images) through `--catalog`.

[docs/inputs.md](docs/inputs.md) describes the layouts and the options for special cases.
[docs/nersc.md](docs/nersc.md) shows how to run at NERSC.

## Documentation

* [docs/method.md](docs/method.md): the measurement, the coadd PSF, the kernel, the errors and the
  legacypipe-specific corrections.
* [docs/outputs.md](docs/outputs.md): output columns, flags and header keywords.
* [docs/inputs.md](docs/inputs.md): survey layouts, cameras and catalogue options.
* [docs/nersc.md](docs/nersc.md): running at NERSC, batch jobs and the comparison tests.
* [docs/validation.md](docs/validation.md): what the tests check.

## Tests

```bash
python -m pytest            # unit tests on synthetic data run anywhere
```

Tests marked `nersc` read Legacy Surveys products on NERSC's file system and skip themselves
elsewhere. Tests marked `legacypipe` compare with legacypipe and The Tractor, and run inside the
`legacysurvey/legacypipe` container (see [docs/nersc.md](docs/nersc.md)).

## Citing

If you use legacygaap, please cite the GAaP papers:

* Kuijken K., 2008, A&A, 482, 1053 (the method)
* Kuijken K. et al., 2015, MNRAS, 454, 3500 (Appendix A: the pixel-space implementation)
* Kuijken K. et al., 2019, A&A, 625, A2 (Section 5: aperture rules)

and this software ([CITATION.cff](CITATION.cff)).

## License

BSD 3-Clause, as legacypipe ([LICENSE](LICENSE)). [NOTICE.md](NOTICE.md) lists the numerical tables
taken from other work.
