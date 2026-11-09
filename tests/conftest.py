"""Test configuration.

Markers (registered in pyproject.toml):
  nersc       reads Legacy Surveys products at NERSC (DR11, IBIS DR1, HSC coadds). Skipped when
              /global/cfs/cdirs/cosmo ($GAAP_TEST_DATA_ROOT) is not mounted.
  legacypipe  compares with legacypipe, tractor and astrometry.net. Skipped when they cannot be
              imported; run these inside the legacysurvey/legacypipe Shifter image (docs/nersc.md).

When the NERSC data are mounted, the default survey directory is DR11 ($GAAP_TEST_DR11) and the
image root the CP staging area ($GAAP_TEST_IMAGES), since the nersc tests are written for DR11 bricks.
"""
import importlib.util
import os
import pytest
from legacygaap import io

DATA_ROOT = os.environ.get('GAAP_TEST_DATA_ROOT', '/global/cfs/cdirs/cosmo')
DR11 = os.environ.get('GAAP_TEST_DR11', '/global/cfs/cdirs/cosmo/work/legacysurvey/dr11')
IMAGES = os.environ.get('GAAP_TEST_IMAGES', '/global/cfs/cdirs/cosmo/staging')
HAVE_DATA = os.path.isdir(DATA_ROOT)
HAVE_LEGACYPIPE = all(importlib.util.find_spec(m) is not None for m in ('legacypipe', 'tractor', 'astrometry'))

if HAVE_DATA:
    io.DEFAULT_SURVEY_DIR = DR11
    if io.IMAGES_ROOT is None:
        io.IMAGES_ROOT = IMAGES


def pytest_collection_modifyitems(config, items):
    for item in items:
        if 'nersc' in item.keywords and not HAVE_DATA:
            item.add_marker(pytest.mark.skip(reason=f'needs Legacy Surveys data at NERSC ({DATA_ROOT})'))
        if 'legacypipe' in item.keywords and not HAVE_LEGACYPIPE:
            item.add_marker(pytest.mark.skip(reason='needs legacypipe, tractor and astrometry.net (Shifter image)'))
