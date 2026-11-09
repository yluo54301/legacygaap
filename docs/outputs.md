# Output

`gaap-brick` writes `<outdir>/<bbb>/gaap-<brick>[-<tag>].fits`, one row per catalogue row in the same
order, in extension `GAAP`. With `--sky N` it also writes `gaap-sky-<brick>[-<tag>].fits`, the same
measurement at N blank-sky positions. Measuring another coadd (`--image-kind K`) inserts `K-` after
`gaap-`.

Fluxes are in nanomaggies (AB magnitude 22.5 − 2.5 log10 flux), like legacypipe's. They are not
corrected for Galactic extinction. Array columns have one entry per aperture, in the order of the
header keywords `AP0` … `AP<NAP-1>`: first the circular apertures (`circ<σ>`, e.g. `circ1.0`), then
the shape-matched elliptical ones (`ell<MIN>`).

## Columns

Copied from the catalogue, where present: `release`, `brickid`, `brickname`, `objid`,
`brick_primary`, `type`, `ra`, `dec`, `bx`, `by`, `maskbits`.

| Column | Shape | Content |
|---|---|---|
| `gaap_a`, `gaap_b` | nap | pre-seeing aperture dispersions along the major and minor axes (arcsec) |
| `gaap_pa` | nap | aperture position angle (deg, north through east) |

Per band `<b>`:

| Column | Shape | Content |
|---|---|---|
| `gaapflux_<b>` | nap | GAaP flux: raw measurement / (`psfcorr` × `wcorr`) |
| `gaapflux_ivar_<b>` | nap | inverse variance: raw × (`psfcorr` × `wcorr` / `escale`)² |
| `gaap_flags_<b>` | nap | bit flags (below) |
| `gaap_fracmasked_<b>` | nap | share of the aperture's effective weight on masked (ivar = 0) pixels |
| `gaap_psfcorr_<b>` | nap | point-source response of the Gaussianized coadd PSF, including the coadd flux scale |
| `gaap_wcorr_<b>` | nap | correction for legacypipe's source-dependent pixel weights (1 for faint sources) |
| `gaap_fluxscale_<b>` | | coadd flux of a unit-flux point source (legacypipe `sbscale` × local pixel-area ratio) |
| `gaap_escale_<b>` | | error scale for resampling correlations: √(weighted mean CCD-to-coadd pixel-area ratio) |
| `gaap_sigma_t_<b>` | | dispersion of the target Gaussian PSF (arcsec) |
| `gaap_psffwhm_<b>` | | FWHM of the Gaussian with the coadd PSF's noise-equivalent area (arcsec) |
| `gaap_nccd_<b>` | | number of CCDs in the PSF model at the source |

The blank-sky table has `x`, `y` (brick pixels), `ra` and `dec` in place of the catalogue columns,
and the same per-band columns. Its sources are measured as point sources: the elliptical apertures
are circles of dispersion MIN.

## Flags

| Bit | Name | Meaning |
|---|---|---|
| 1 | `BIGPSF` | aperture not larger than the target PSF (an axis of W − σ_t² below 1.5 pixels); flux NaN |
| 2 | `EDGE` | cutout truncated by the brick edge |
| 4 | `MASKED` | more than 10% of the effective weight on masked pixels |
| 8 | `NOPSF` | no CCD covers the position (or no sky position); flux NaN |
| 16 | `POORKERN` | `psfcorr / fluxscale` differs from 1 by more than 1% |
| 32 | `NEXP` | the CCD count of the PSF model disagrees with the pipeline `nexp` map near the source (informational) |

A measurement is normally usable when `(gaap_flags_<b> & ~32) == 0`.

## Header keywords

| Keyword | Content |
|---|---|
| `GAAPVER`, `GAAPGIT` | legacygaap version and git commit |
| `BRICK`, `REGION`, `SURVDIR` | brick, region (`south`, `north`, or empty for a single run), survey directory |
| `BANDS`, `NAP`, `AP<k>` | bands; number and names of the apertures |
| `MAX_APER`, `KAPPA`, `SCALE`, `KSIZE`, `LAM`, `STAMP`, `CUTOUT`, `MARGIN`, `MIN_SIGM` | settings (`gaap-brick` options; `MIN_SIGM` is the minimum post-seeing axis in pixels) |
| `CUTNSIG` | cutout half-width in units of the widest aperture dispersion |
| `NCCD_<B>` | CCDs in the band's PSF model for the brick |
| `FLAG<bit>` | flag names |
| `FLUXDEF`, `IVARDEF` | how the flux and its inverse variance were calibrated |
| `LEGPIPEV` | legacypipe version of the coadds |
| `COSKY_<B>` | per-brick sky level legacypipe subtracted (nanomaggies per CCD pixel), when recorded |
| `PIXSCONV`, `PSZRAT_*` | legacypipe's CCD pixel-scale convention, and the `psfsize` ratios that chose it |
| `EPOCH_<B>`, `NPMMOVE` | mean epoch of each band; number of sources moved for proper motion |
| `FLUXC_<B>` | catalogue flux column used for the coadd-weighting correction |
| `SHOTN_<B>` | shot-noise strength used, and how it was calibrated |
| `NSRC`, `SELECT`, `CATALOG`, `IMGKIND`, `IMGROOT`, `WCSSRC` | rows measured and selection, catalogue, coadd kind, image root, WCS source |
| `NTHREADS`, `TSETUP`, `TMEAS` | processes, setup and measurement time (s) |

The blank-sky table header has `NSKY` instead of the catalogue keywords.

## Using the fluxes

* Colours: use the same aperture in every band, e.g. `circ1.0` for faint galaxies or the `ell`
  apertures for resolved ones.
* Errors: multiply `1/sqrt(gaapflux_ivar_<b>)` by the blank-sky error scale
  (`legacygaap.sky.error_scale`) of the brick, band and aperture.
* Background continuity across bricks: see `COSKY` in [method.md](method.md#7-other-legacypipe-conventions).
