#!/bin/bash

for I in 0.05 0.1 0.15 0.2
do
    python SSMLL-BAM/run_warmup_text.py --loss BAM --lb_ratio $I --warmup_epochs 12 --warmup_batch_size 16 --lr 1e-4 --net bert-base-uncased --dataset_name ohsumed --dataset_dir SSMLL-BAM/text_model/datasets --cos-margin 0.2 --cos-norm 10.0
    python SSMLL-BAM/run_BAM_text.py --loss BAM --lb_ratio $I --warmup_epochs 12 --warmup_batch_size 16 --lr 1e-4 --net bert-base-uncased --dataset_name ohsumed --dataset_dir SSMLL-BAM/text_model/datasets --init_pos_per 1.0 --init_neg_per 1.0 --cos-margin 0.2 --cos-norm 10.0
done


