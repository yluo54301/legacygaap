#!/bin/bash
# GAaP for one Legacy Surveys DR10 brick at NERSC. Run on a compute node:
#   salloc -N 1 -C cpu -q interactive -t 30:00 -A <account>
#   bash examples/run_brick.sh 0001m002
# Writes $OUTDIR/<bbb>/gaap-<brick>.fits and gaap-sky-<brick>.fits.
set -euo pipefail
BRICK=${1:-0001m002}
OUTDIR=${OUTDIR:-$PSCRATCH/gaap}
export LEGACY_SURVEY_DIR=${LEGACY_SURVEY_DIR:-/dvs_ro/cfs/cdirs/cosmo/data/legacysurvey/dr10}
export GAAP_IMAGES_ROOT=${GAAP_IMAGES_ROOT:-/dvs_ro/cfs/cdirs/cosmo/staging}
export GAAP_CACHE_DIR=${GAAP_CACHE_DIR:-$PSCRATCH/gaap-cache}
gaap-brick --brick "$BRICK" --threads 64 --sky 1000 --outdir "$OUTDIR"
