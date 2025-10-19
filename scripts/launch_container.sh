#!/usr/bin/env bash
set -euo pipefail

WORKDIR=/storage/openpsi/users/lichangye.lcy/VeRL_InternVL
IMAGE=/storage/openpsi/images/verl-1015-v1.sif

# 只启动一个交互容器；后续在容器内再算 IP、起 Ray
srun --mpi=pmi2 \
  --ntasks=1 \
  --gres=gpu:8 \
  --job-name=lcy \
  --chdir="$WORKDIR" \
  --cpus-per-task=64 \
  --mem=1500G \
  --nodes=1 \
  --pty singularity shell \
    --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    "$IMAGE"


