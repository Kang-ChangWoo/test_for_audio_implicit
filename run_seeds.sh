#!/bin/bash
# implicit (train.py)
imp () { GPU=$1; RUN=$2; MODEL=$3; SEED=$4; LR=$5
  CUDA_VISIBLE_DEVICES=$GPU nohup python train.py --model $MODEL --run-name $RUN --seed $SEED \
    --epochs 25 --batch-size 64 --n-rays 2048 --num-workers 6 --lr $LR > logs/$RUN.log 2>&1 &
  echo "launched $RUN GPU$GPU pid $!"; }
# full-map (train_fullmap.py)
fm () { GPU=$1; RUN=$2; CORR=$3; SEED=$4; shift 4
  CUDA_VISIBLE_DEVICES=$GPU nohup python train_fullmap.py --correction $CORR --run-name $RUN --seed $SEED \
    --epochs 25 --batch-size 64 --num-workers 6 --lr 2e-3 "$@" > logs/$RUN.log 2>&1 &
  echo "launched $RUN GPU$GPU pid $!"; }
imp 0 A2_raymlp_s2  raymlp    2 2e-3
imp 1 A4_cross_s2   cross     2 3e-4
imp 2 A6_crossself_s1 crossself 1 3e-4
imp 3 A6_crossself_s2 crossself 2 3e-4
fm  4 A9_fullmap_s2 none 2
fm  5 A11_shaux_s1  sh   1
fm  6 A11_shaux_s2  sh   2
fm  7 A13_ipd5_s2   none 2 --in-ch 5 --cache-dir cache_rir
wait; echo ALL_SEEDS_DONE
