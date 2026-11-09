#!/bin/bash
#SBATCH --constraint=cpu
#SBATCH --time=02:00:00
#SBATCH --job-name=gaap
#SBATCH --output=gaap-%j.log
# gaap-brick on a list of bricks (one per line). The bricks are dealt round-robin to the nodes; each
# node runs BPN bricks at a time with 128 / BPN worker processes each (Perlmutter CPU nodes have 128
# cores). Bricks whose output exists are skipped, so the job can be resubmitted.
#
#   export LEGACY_SURVEY_DIR=/dvs_ro/cfs/cdirs/cosmo/data/legacysurvey/dr10
#   export GAAP_IMAGES_ROOT=/dvs_ro/cfs/cdirs/cosmo/staging
#   export GAAP_CACHE_DIR=$PSCRATCH/gaap-cache
#   OUTDIR=$PSCRATCH/gaap BPN=4 GAAP_ARGS="--sky 1000" sbatch -A <account> -q regular --nodes=4 examples/slurm_bricks.sbatch bricks.txt
#
# Environment: OUTDIR (output directory), BPN (bricks per node, default 4; use 1-2 for very deep
# bricks), GAAP_ARGS (extra gaap-brick options), GAAP_TAG (passed as --tag), GAAP_BRICK (command).
set -u
LIST=$1
export OUTDIR=${OUTDIR:-$PWD/gaap}
BPN=${BPN:-4}
export NPROC=$((128 / BPN))
export GAAP=${GAAP_BRICK:-gaap-brick}
export GAAP_ARGS=${GAAP_ARGS:-}
export GAAP_TAG=${GAAP_TAG:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mapfile -t B < <(grep -v '^\s*#' "$LIST" | grep -v '^\s*$')
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
N=${#NODES[@]}
mkdir -p "$OUTDIR/logs"

one() {
    local b=$1 tag="" suffix=""
    if [ -n "$GAAP_TAG" ]; then tag="--tag $GAAP_TAG"; suffix="-$GAAP_TAG"; fi
    if [ -f "$OUTDIR/${b:0:3}/gaap-$b$suffix.fits" ]; then echo "$b exists"; return 0; fi
    if $GAAP --brick "$b" --threads "$NPROC" --outdir "$OUTDIR" $tag $GAAP_ARGS > "$OUTDIR/logs/$b$suffix.log" 2>&1; then
        echo "$b ok"
    else
        echo "$b FAILED (see $OUTDIR/logs/$b$suffix.log)"
    fi
}
export -f one

echo "=== ${#B[@]} bricks on $N nodes, $BPN per node at a time, $NPROC processes each: $(date)"
for ((k = 0; k < N; k++)); do
    SUB=()
    for ((i = k; i < ${#B[@]}; i += N)); do SUB+=("${B[$i]}"); done
    [ ${#SUB[@]} -gt 0 ] || continue
    srun --nodes=1 --ntasks=1 --cpus-per-task=256 --cpu-bind=none -w "${NODES[$k]}" \
        bash -c 'printf "%s\n" "${@:2}" | xargs -P "$1" -I{} bash -c "one {}"' _ "$BPN" "${SUB[@]}" &
done
wait
echo "=== done $(date)"
