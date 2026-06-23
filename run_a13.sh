#!/bin/bash
run () { GPU=$1; RUN=$2; INCH=$3
  CUDA_VISIBLE_DEVICES=$GPU nohup python train_fullmap.py --run-name $RUN --correction none \
    --in-ch $INCH --cache-dir cache_rir --epochs 25 --batch-size 64 --num-workers 8 --lr 2e-3 \
    > logs/$RUN.log 2>&1 &
  echo "launched $RUN (in_ch=$INCH) GPU$GPU pid $!"; }
run 1 A13_mag2_s0 2
run 2 A13_ild3_s0 3
run 3 A13_ipd5_s0 5
run 4 A13_ipd5_s1 5
wait; echo ALL_A13_DONE
