#!/bin/bash
# Full evaluation of a trained world model + prior: forecasting (val) and planning (val).
#
#   WM=runs/wm/best.pt PRIOR=runs/prior/best.pt sbatch slurm/eval.sbatch
#
#SBATCH -J occwm_eval
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH -t 6:00:00
#SBATCH -o logs/slurm/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/env.sh
WM=${WM:-$RUNS/wm/best.pt}
PRIOR=${PRIOR:-$RUNS/prior/best.pt}

CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_forecast.py --method model --ckpt "$WM" --name ours --bs 4 > logs/eval_forecast.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/eval_planning.py \
    --wm "$WM" --prior "$PRIOR" --name planning > logs/eval_planning.log 2>&1 &
wait
notify "Evaluation finished.
\`\`\`$(tail -n 1 logs/eval_forecast.log | head -c 300)
$(tail -n 6 logs/eval_planning.log)\`\`\`" --title "job $SLURM_JOB_ID eval done" --color green
