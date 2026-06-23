#!/bin/bash
fm () { GPU=$1; RUN=$2; SEED=$3; shift 3
  CUDA_VISIBLE_DEVICES=$GPU nohup python train_fullmap.py --correction cross_sup --run-name $RUN \
    --seed $SEED --epochs 25 --batch-size 64 --num-workers 6 "$@" > logs/$RUN.log 2>&1 &
  echo "launched $RUN GPU$GPU pid $!"; }
fm 0 A14_logmag_s0 0 --init-decoder A9_fullmap_s0 --lr 1e-3
fm 1 A14_logmag_s1 1 --init-decoder A9_fullmap_s1 --lr 1e-3
fm 4 A14_frozen_s0 0 --init-decoder A9_fullmap_s0 --freeze-decoder True --lr 1e-3
fm 5 A14_rir5_s0   0 --in-ch 5 --cache-dir cache_rir --chan-norm True --lr 2e-3
wait; echo ALL_A14_DONE
