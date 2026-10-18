# Sourced by the sbatch scripts. Edit these paths for another cluster.
export PY=${PY:-/scratch/$USER/envs/occwm/bin/python}
export RUNS=${RUNS:-/scratch/$USER/occwm_runs}
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export LC_ALL=C

notify() { python3 tools/notify.py "$@" || true; }
