#!/bin/bash

for I in  0.05 0.1 0.15 0.2
do
    python SSMLL-BAM/run_warmup.py --loss BAM --lb_ratio $I --warmup_epochs 12 --warmup_batch_size 16 --lr 5e-5 --net resnet50 --dataset_name VOC2012 --dataset_dir Dataset --cos-margin 0.4 --cos-norm 20
    python SSMLL-BAM/run_BAM.py --loss BAM --lb_ratio $I --warmup_epochs 12 --warmup_batch_size 16 --lr 5e-5 --net resnet50 --dataset_name VOC2012 --dataset_dir Dataset --init_pos_per 1.0 --init_neg_per 1.0 --cos-margin 0.4 --cos-norm 20
done
