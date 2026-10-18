#!/bin/bash
# Generic multi-GPU training job. Posts start / finish / failure to Discord.
#
#   STAGE=vae sbatch slurm/train.sbatch
#   STAGE=wm  ARGS="--vae runs/vae/best.pt" sbatch slurm/train.sbatch
#   sbatch --gres=gpu:2 slurm/train.sbatch      # fewer GPUs
#
#SBATCH -J occwm
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH -t 23:00:00
#SBATCH -o logs/slurm/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/env.sh

STAGE=${STAGE:-vae}
RUN=${RUN:-$RUNS/$STAGE}
NGPU=$(nvidia-smi -L | wc -l)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

notify "Training **$STAGE** started: ${NGPU}x $GPU, out: \`$RUN\`" --title "job $SLURM_JOB_ID started"
trap 'notify "Training **$STAGE** FAILED, see logs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out
\`\`\`$(tail -n 15 logs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out | cut -c1-180)\`\`\`" --title "job $SLURM_JOB_ID failed" --color red' ERR

$PY -m torch.distributed.run --standalone --nproc_per_node "$NGPU" \
    "scripts/train_${STAGE}.py" --out "$RUN" ${ARGS:-}

notify "Training **$STAGE** finished.
\`\`\`$(grep val_ "$RUN/log.jsonl" | tail -n 3)\`\`\`" "$RUN"/*.png --title "job $SLURM_JOB_ID done" --color green
