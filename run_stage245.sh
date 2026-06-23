#!/bin/bash
# Stages 2/4/5 built on the cross-attn winner (lr 3e-4 + warmup + clip).
run () { GPU=$1; RUN=$2; shift 2
  CUDA_VISIBLE_DEVICES=$GPU nohup python train.py --run-name $RUN \
    --seed 0 --epochs 25 --batch-size 64 --n-rays 2048 --num-workers 8 --lr 3e-4 "$@" \
    > logs/$RUN.log 2>&1 &
  echo "launched $RUN on GPU$GPU (pid $!)"; }
run 1 A3_crossSH_s0    --model cross     --use-sh-pe True
run 2 A5_crossMic_s0   --model cross     --use-mic-pe True
run 3 A6_crossself_s0  --model crossself
run 4 A8_hybrid_s0     --model hybrid
run 5 A4_ffmask_s0     --model cross     --mask-farfield True
wait; echo "ALL_STAGE245_DONE"
