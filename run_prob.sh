#!/bin/bash
run () { GPU=$1; RUN=$2; shift 2
  CUDA_VISIBLE_DEVICES=$GPU nohup python train_prob.py --run-name $RUN --seed 0 \
    --epochs 25 --batch-size 64 --num-workers 6 --lr 2e-3 "$@" > logs/$RUN.log 2>&1 &
  echo "launched $RUN GPU$GPU pid $!"; }
run 0 P_k1   --prob-k 1
run 1 P_k5   --prob-k 5
run 2 P_k10  --prob-k 10
