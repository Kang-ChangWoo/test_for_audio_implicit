#!/bin/bash
run () { GPU=$1; RUN=$2; shift 2
  CUDA_VISIBLE_DEVICES=$GPU nohup python train.py --model cross --run-name $RUN --seed 0 \
    --epochs 25 --batch-size 64 --n-rays 2048 --num-workers 6 --lr 3e-4 "$@" > logs/$RUN.log 2>&1 &
  echo "launched $RUN GPU$GPU pid $!"; }
run 0 T_mlpskip   --ray-mlp-skip True
run 1 T_film      --ray-film True
run 2 T_progpe    --prog-pe True
run 3 T_sector    --sector-sample True
run 4 T_all       --ray-mlp-skip True --ray-film True --prog-pe True --sector-sample True
wait; echo ALL_TIPS_DONE
