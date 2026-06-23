#!/bin/bash
# Stage 1 (Q1 gate): ray-only prior vs global-audio RayMLP + shuffled control.
# args: GPU RUN MODEL SEED EXTRA...
run () {
  GPU=$1; RUN=$2; MODEL=$3; SEED=$4; shift 4
  CUDA_VISIBLE_DEVICES=$GPU nohup python train.py --model $MODEL --run-name $RUN \
    --seed $SEED --epochs 25 --batch-size 64 --n-rays 2048 --num-workers 8 "$@" \
    > logs/$RUN.log 2>&1 &
  echo "launched $RUN on GPU$GPU (pid $!)"
}
run 1 A1_rayonly_s0 rayonly 0 --audio-mode none
run 2 A2_raymlp_s0  raymlp  0
run 3 A2_shuf_s0    raymlp  0 --shuffle-audio True
run 4 A1_rayonly_s1 rayonly 1 --audio-mode none
run 5 A2_raymlp_s1  raymlp  1
run 6 A2_shuf_s1    raymlp  1 --shuffle-audio True
wait
echo "ALL_STAGE1_DONE"
