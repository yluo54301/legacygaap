# Third-party material

legacygaap is an independent implementation; it does not include code from legacypipe, The Tractor,
astrometry.net or the LSST Science Pipelines. It reproduces their conventions so that its
measurements match the legacypipe products, and it uses the following numerical tables.

* **Gaussian-mixture galaxy profiles** (`legacygaap/profiles.py`, exponential and de Vaucouleurs):
  the amplitudes and variances of Hogg & Lang (2013, PASP 125, 719), as used by The Tractor
  (`tractor.mixture_profiles`). They render tractor galaxy models for the coadd-weighting correction.
* **Sérsic mixtures** (`legacygaap/data/sersic_mixtures.npz`): Gaussian-mixture approximations of
  Sérsic profiles for 0.5 <= n <= 6 in steps of 0.05, tabulated from The Tractor's
  `tractor.sersic.SersicMixture.getProfile` (The Tractor, Lang, Hogg & Mykytyn 2016, ascl:1604.008;
  GPL-2.0). Used only for catalogue sources of type SER.

The method follows Kuijken (2008, A&A 482, 1053), Kuijken et al. (2015, MNRAS 454, 3500; Appendix A)
and Kuijken et al. (2019, A&A 625, A2; Section 5).
