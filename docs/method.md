# Method

This page describes what `gaap-brick` measures and how, including the corrections that are specific
to legacypipe coadds. Module names refer to the `legacygaap` package.

## 1. The GAaP flux

Let $I(\mathbf{x})$ be the pre-seeing surface brightness of a source and $W$ a 2 × 2 aperture
matrix, built from the dispersions $a \ge b$ along the major and minor axes and the position angle.
The GAaP flux is the flux through the Gaussian aperture $\exp(-\tfrac12 \mathbf{x}^T W^{-1}
\mathbf{x})$ applied to the pre-seeing image. It is measured on the image $G$ whose PSF has been made
a circular Gaussian of dispersion $\sigma_t$ (Kuijken et al. 2015, Eq. A16):

$$
F_W = \int I(\mathbf{x})\, e^{-\frac12 \mathbf{x}^T W^{-1} \mathbf{x}}\, d\mathbf{x}
    = \sqrt{\frac{\det W}{\det W'}} \int G(\mathbf{x})\, e^{-\frac12 \mathbf{x}^T W'^{-1} \mathbf{x}}\, d\mathbf{x},
\qquad W' = W - \sigma_t^2 \mathbb{1} .
$$

Properties used throughout, and checked by the tests:

* For a point source, $F_W$ is the total flux for any $W$.
* For a circular Gaussian galaxy of flux $f$ and dispersion $s$ in a circular aperture $\sigma$,
  $F = f\,\sigma^2/(\sigma^2 + s^2)$, independent of the seeing.
* $W$ is defined before seeing and is the same in every band, so $F_W$ ratios (colours) do not depend
  on the PSF. GAaP fluxes of extended sources are not total fluxes.
* The measurement exists only if $W'$ is positive definite: the pre-seeing aperture must be larger
  than the target PSF. legacygaap requires both axes of $W'$ to be at least 1.5 pixels (flag 1
  otherwise). Close to that limit the prefactor $\sqrt{\det W / \det W'}$ grows and the flux becomes
  noisy.

## 2. The coadd PSF at each source (`psf.py`, `psfex.py`, `resample.py`)

legacypipe builds a brick coadd as the inverse-variance-weighted sum of the CCD images, each
resampled onto the brick grid with a Lanczos-3 kernel and scaled by `sbscale`, the ratio of the
brick and CCD pixel areas. Its PSF at a position is therefore the same weighted sum of the per-CCD
PSFs. legacygaap rebuilds it at every source, in every band:

1. **CCDs.** It uses the CCDs in the brick's CCD table that cover the position. It excludes a
   10-pixel edge band on DECam CCDs, the regions legacypipe never reads (DECam S7's bad half, the
   glowing edge of early DES r-band exposures), and anything outside the sub-image legacypipe read
   for the brick (the `ccd_x0..ccd_y1` box). For HSC it also drops a CCD where more than half of a
   16 × 16-pixel bin is masked. Bands missing from the brick CCD table (legacypipe's forced bands)
   come from the survey CCD tables, selected with the footprint and the coadd's `nexp` map.
2. **Per-CCD PSF.** The PsfEx model is evaluated at the source position on the CCD (polynomial in
   the CCD pixel coordinates applied to the basis images, as The Tractor does) and shifted to the
   source's sub-pixel phase with a Lanczos-3 kernel. Oversampled HSC models are sampled at the native
   pixel centres; Rubin deep-coadd patches use their cell PSFs.
3. **Resampling and stacking.** Each CCD stamp is resampled onto the brick grid through the exact
   CCD WCS (from the image header, cached as JSON; or from the CCD table for TAN cameras), multiplied
   by its `sbscale`, and added with weight $1/\mathrm{sig1}^2$, the constant per-CCD weight
   legacypipe uses for its `psfsize` maps.

The result is the coadd image of a unit-flux point source at that position, 65 × 65 pixels. Its sum
is the coadd flux scale (`gaap_fluxscale_<b>`, `sbscale` times the local pixel-area ratio), which is
0.98–1.00 for DECam.

legacypipe versions differ in how they compute the CCD pixel scale for `sbscale` and `psfsize`
(with or without the linear TPV distortion terms). `gaap-brick --pixscale auto` compares the
`psfsize` formula with the pipeline map at about 40 positions and uses the convention that matches
(header `PIXSCONV`, ratios `PSZRAT_*`).

## 3. Gaussianization (`kernel.py`)

The target is a circular Gaussian of dispersion $\sigma_t = f\,\sigma_{\rm eq}$, where $\sigma_{\rm eq}
= \sqrt{N_{\rm eff}/4\pi}$ is the Gaussian-equivalent dispersion of the coadd PSF $P$ (from its
noise-equivalent area, legacypipe's `psfsize` definition) and $f = 1.15$ (`--scale`), as in KiDS and
LSST. The kernel is the regularised Fourier quotient

$$
\hat K = \frac{\hat G\, \hat P^*}{|\hat P|^2 + \lambda \max |\hat P|^2},
$$

with $\lambda = 10^{-4}$ (`--lam`), cropped to 63 × 63 pixels (`--ksize`) and renormalised to unit
sum. The target Gaussian is centred on the source's sub-pixel position. For PSFs from Gaussian to
Moffat β = 2.5 this reproduces the point-source flux to better than 0.5% for apertures of 1" and
more. Smaller kernels leave the PSF wings ungathered (0.5–1.5% at 41 pixels).

## 4. Apertures (`aperture.py`)

* **Circular** apertures of dispersion $\sigma$ (`--circ`, default 1.0", 1.5", 2.0"), named
  `circ<σ>`.
* **Shape-matched elliptical** apertures from the tractor shape (`shape_r`, `shape_e1`,
  `shape_e2`), named `ell<MIN>` (Kuijken et al. 2019):
  $a^2 = (\kappa r_e)^2 + {\rm MIN}^2$ and $b^2 = (\kappa q r_e)^2 + {\rm MIN}^2$, with $r_e$ the
  major-axis half-light radius, $q = (1-|e|)/(1+|e|)$, $\kappa = 1$ (`--kappa`), MIN from
  `--min-aper` (default 1.0" and 1.5"), and each axis capped at `--max-aper` (2.0"). The position
  angle is $\theta = \tfrac12 \operatorname{atan2}(e_2, e_1)$, north through east. Point sources
  (`PSF`, `DUP`) get circular apertures of dispersion MIN.

Apertures are built in arcsec on the sky and mapped to brick pixels through the brick WCS. Their sky
axes and angle are written to `gaap_a`, `gaap_b` and `gaap_pa`.

Choose apertures larger than the target PSF in every band you want to compare. With $f = 1.15$,
$\sigma_t$ is 0.5–0.9" for typical DECam seeing and up to 1.1" for 90Prime g. Apertures of 1.0" are
safe in most DECam data; in poor seeing use 1.5".

## 5. Measurement and errors (`measure.py`)

With $w$ the post-seeing Gaussian weight of $W'$ and $K$ the kernel, the flux is a linear function
of the coadd pixels:

$$
F = p \sum_i w_i\,(I \ast K)_i = p \sum_j I_j\, w^{\rm eff}_j, \qquad
w^{\rm eff} = w \star K, \qquad p = \sqrt{\det W / \det W'} .
$$

Its variance follows exactly from the coadd inverse variance, $\mathrm{Var}(F) = p^2 \sum_j
(w^{\rm eff}_j)^2 / \mathrm{ivar}_j$. This equals Kuijken et al. (2015) Eq. A17 for stationary noise,
needs no noise autocorrelation function, and handles masked pixels. Pixels with ivar = 0 are left
out of both sums. Their share of $\sum |w^{\rm eff}|$ is `gaap_fracmasked_<b>`, and flag 4 is set
above 10%.

The cutout is at least 97 pixels wide, and wider for large apertures (4 σ of the widest aperture on
each side). Flag 2 is set when the brick edge truncates it.

## 6. Calibration of each measurement (`brick.py`, `weights.py`)

The flux written to the catalogue is

$$
\texttt{gaapflux} = \frac{F_{\rm raw}}{\texttt{psfcorr} \times \texttt{wcorr}}, \qquad
\texttt{gaapflux\_ivar} = \mathrm{ivar}_{\rm raw} \left(\frac{\texttt{psfcorr} \times \texttt{wcorr}}{\texttt{escale}}\right)^2 .
$$

* **`psfcorr`**: the same measurement applied to the coadd image of a unit-flux point source (the
  rebuilt PSF). It is 1 for a perfect kernel on a coadd with flux scale 1. Dividing by it removes
  both the coadd flux scale and any residual kernel error. Flag 16 is set if `psfcorr / fluxscale`
  differs from 1 by more than 1%.
* **`wcorr`**: legacypipe weights each CCD pixel by an inverse variance that includes the source's
  own shot noise. At the core of a bright source the sharpest exposures are therefore down-weighted,
  and the coadd profile is broader than the constant-weight PSF. GAaP of 17–19 mag stars comes out
  up to 9% low in DECam g and r without a correction. legacygaap renders each source (its tractor
  model convolved with each CCD's PSF), recomputes the per-CCD weights $w_e(x) = \omega_e / (1 + s
  S_e(x)/\mathrm{sky}_e)$, and takes `wcorr` as the ratio of the GAaP of the weighted and
  constant-weight model coadds. The strength $s$ (~0.9 for DECam) is calibrated per brick and band
  from the invvar dips of isolated bright stars (header `SHOTN_<B>`). For galaxies the median
  `wcorr` is 0.997 and 99% are above 0.98 (DECam g and r).
* **`escale`**: the coadd inverse variance ignores the correlations that resampling introduces when
  the CCD pixels are coarser than the brick pixels. The true variance of a smooth aperture sum is the
  invvar prediction times the weighted mean CCD-to-coadd pixel-area ratio: 1.0 for DECam, about 3
  for 90Prime. `gaap_escale_<b>` is its square root, and the inverse variance is divided by its
  square.

## 7. Other legacypipe conventions

* **Proper motions.** legacypipe catalogues give Gaia stars' positions at the Gaia epoch, with
  `pmra`, `pmdec` and `ref_epoch`. legacygaap measures each band at that band's mean epoch
  (1/sig1²-weighted CCD MJD, header `EPOCH_<B>`). Without this, a star moving 20 mas/yr loses about
  4% of a 1" GAaP flux in images taken 8 years after the Gaia epoch.
* **Per-brick sky.** legacypipe subtracts a constant sky level per brick and band from every CCD
  before fitting and coadding (`COSKY_<B>` in the coadd headers). Neighbouring bricks therefore have
  slightly different backgrounds. legacygaap measures the coadd as it is, consistent with the tractor
  catalogue, and copies `COSKY_<B>` into its header. To restore a background that is continuous
  across bricks, add `COSKY_<B> × 2π a b / (psfcorr × wcorr)` to each flux, with $a$, $b$ the aperture
  axes in coadd pixels. This is about 0.1% of a 21 mag source in a 1" aperture per $10^{-5}$
  nanomaggies per pixel.
* **Catalogue fluxes.** The coadd-weighting correction needs each source's flux. legacygaap reads
  `flux_<b>`, or `forced_flux_<b>` for forced-photometry catalogues (header `FLUXC_<B>`).

## 8. Empirical errors (`sky.py`)

Blank-sky aperture sums on legacypipe coadds scatter more than the inverse variance predicts. The
reasons are noise correlated by resampling and background residuals on aperture scales, and the
excess grows with aperture size and towards the blue. `gaap-brick --sky N` measures N random
positions at least 15 pixels from every source blob and masked pixel, with the same code. For each
band and aperture, `legacygaap.sky.error_scale` returns the NMAD of `gaapflux × sqrt(gaapflux_ivar)`
there (at least 1). Multiply the GAaP errors by it. `legacygaap.sky.depth` returns the blank-sky
5σ depth.

## 9. Limitations

* No deblending: neighbours inside the aperture contribute. Use the tractor `fracflux_<b>` or the
  `maskbits` to select clean sources. Alternatively, write a neighbour-subtracted coadd next to the
  run's coadds as `legacysurvey-<brick>-<kind>-<band>.fits.fz` and measure it with `--image-kind <kind>`.
* GAaP fluxes are aperture fluxes. They give colours, not total magnitudes, for extended sources.
* The shot-noise strength differs between CP versions. Where a band mixes them, bright stars keep
  about 1% residuals after `wcorr`.
* The PSF model assumes the constant per-CCD weights. That is right for the sky-dominated pixels of
  faint sources, which GAaP is for.

## References

* Kuijken K., 2008, A&A, 482, 1053
* Kuijken K. et al., 2015, MNRAS, 454, 3500 (Appendix A)
* Kuijken K. et al., 2019, A&A, 625, A2 (Section 5)
* Hogg D. W., Lang D., 2013, PASP, 125, 719 (Gaussian-mixture galaxy profiles)
* LSST meas_extensions_gaap: <https://github.com/lsst/meas_extensions_gaap>
