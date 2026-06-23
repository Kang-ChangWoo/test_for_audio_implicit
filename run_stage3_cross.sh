#!/bin/bash
run () { GPU=$1; RUN=$2; SEED=$3; shift 3
  CUDA_VISIBLE_DEVICES=$GPU nohup python train.py --model cross --run-name $RUN \
    --seed $SEED --epochs 25 --batch-size 64 --n-rays 2048 --num-workers 8 --lr 3e-4 "$@" \
    > logs/$RUN.log 2>&1 &
  echo "launched $RUN on GPU$GPU (pid $!)"; }
run 1 A4_cross_s0      0
run 2 A4_cross_s1      1
run 3 A4_cross_shuf_s0 0 --shuffle-audio True
wait; echo "ALL_STAGE3_DONE"
