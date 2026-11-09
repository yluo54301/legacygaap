# Running at NERSC

The Legacy Surveys data releases are on NERSC's community file system. On compute nodes, read them
through the read-only DVS mount `/dvs_ro/cfs/...`, which is much faster than `/global/cfs/...`.

## Environment

legacygaap needs only numpy, scipy, astropy, fitsio and matplotlib, so any recent Python environment
works:

```bash
module load python
conda create -n gaap -c conda-forge python=3.12 numpy scipy astropy matplotlib fitsio pytest
conda activate gaap
pip install -e /path/to/legacygaap
```

Point it at the data and at a cache directory:

```bash
export LEGACY_SURVEY_DIR=/dvs_ro/cfs/cdirs/cosmo/data/legacysurvey/dr10   # DR10 (public)
export GAAP_IMAGES_ROOT=/dvs_ro/cfs/cdirs/cosmo/staging                    # CP images, for the CCD WCS
export GAAP_CACHE_DIR=$PSCRATCH/gaap-cache                                 # WCS headers and CCD lists
```

The first run on a brick reads one image header per CCD for the exact WCS, a few hundred per DECam
brick. The cache makes later runs on the same CCDs fast, and it can be shared between jobs.

## One brick

On a login node, only small tests (`--limit 100`). Full bricks belong on a compute node:

```bash
salloc -N 1 -C cpu -q interactive -t 30:00 -A <account>
gaap-brick --brick 0001m002 --threads 64 --sky 1000 --outdir $PSCRATCH/gaap
```

A brick spends a few minutes reading the PSF models of its CCDs, then about 10–20 ms of CPU per
source and band.

## Many bricks

[`examples/slurm_bricks.sbatch`](../examples/slurm_bricks.sbatch) deals a list of bricks
round-robin to the nodes of a job. Each node runs `BPN` bricks at a time with `128 / BPN` worker
processes each. Bricks whose output exists are skipped, so the job can simply be resubmitted.

```bash
OUTDIR=$PSCRATCH/gaap BPN=4 GAAP_ARGS="--sky 1000" \
    sbatch -A <account> -q regular -t 02:00:00 --nodes=4 examples/slurm_bricks.sbatch bricks.txt
```

Deep bricks (hundreds of exposures per band) need more memory per brick: use `BPN=1` or `2`.
[`examples/merge_bricks.py`](../examples/merge_bricks.py) merges the outputs into one catalogue. It
keeps the brick-primary rows and adds a table of per-brick blank-sky error scales and depths.

## Comparison tests against legacypipe

The tests marked `legacypipe` compare legacygaap's PsfEx evaluation, Lanczos resampling, coadd PSF,
WCS and aperture orientation with legacypipe and The Tractor. Run them inside the legacypipe
container. The image sets its own `PYTHONPATH`, so prepend the repository (and a directory holding
pytest) from inside the container:

```bash
cd /path/to/legacygaap
shifter --image=legacysurvey/legacypipe:DR11.2.0 python -m pip install --target $PSCRATCH/pylib pytest
shifter --image=legacysurvey/legacypipe:DR11.2.0 bash -c \
    'export PYTHONPATH=$PWD:$PSCRATCH/pylib:$PYTHONPATH; python -m pytest -m legacypipe'
```

The `nersc` tests read DR11, IBIS DR1 and HSC coadds where those are mounted
(`GAAP_TEST_DR11`, `GAAP_TEST_IBIS`, `GAAP_TEST_HSC` override the locations). Run them on a compute
node: `python -m pytest -m nersc`.
