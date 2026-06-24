#!/bin/bash
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$s nohup python train.py --model crossself --run-name A6sec_s$s \
    --sector-sample True --seed $s --epochs 25 --batch-size 64 --n-rays 2048 \
    --num-workers 6 --lr 3e-4 > logs/A6sec_s$s.log 2>&1 &
  echo "launched A6sec_s$s GPU$s pid $!"
done
wait; echo ALL_A6SEC_DONE
