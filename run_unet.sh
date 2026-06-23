#!/bin/bash
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$s nohup python train_fullmap.py --run-name Aunet_s$s --arch unet \
    --seed $s --epochs 25 --batch-size 64 --num-workers 6 --lr 2e-3 > logs/Aunet_s$s.log 2>&1 &
  echo "launched Aunet_s$s GPU$s pid $!"
done
wait; echo ALL_UNET_DONE
