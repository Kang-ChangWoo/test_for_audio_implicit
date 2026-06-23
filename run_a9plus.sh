#!/bin/bash
run () { GPU=$1; RUN=$2; CORR=$3; SEED=$4
  CUDA_VISIBLE_DEVICES=$GPU nohup python train_fullmap.py --run-name $RUN --correction $CORR \
    --seed $SEED --epochs 25 --batch-size 64 --num-workers 8 --lr 2e-3 \
    > logs/$RUN.log 2>&1 &
  echo "launched $RUN ($CORR) GPU$GPU pid $!"; }
run 1 A9_fullmap_s0 none  0
run 2 A9_fullmap_s1 none  1
run 3 A10_cross_s0  cross 0
run 4 A11_shaux_s0  sh    0
run 5 A12_film_s0   film  0
wait; echo ALL_A9PLUS_DONE
