# S$^2$ML$^2$-BBAM (NeurIPS2024) 

This is an official implementation of [Semi-supervised Multi-label Learning with Balanced Binary Angular Margin Loss](https://openreview.net/pdf?id=AqcPvWwktK), which is accepted by NeurIPS2024.

## Abstract

Semi-supervised multi-label learning (SSMLL) refers to inducing classifiers using a small number of samples with multiple labels and many unlabeled samples. The prevalent solution of SSMLL involves forming pseudo-labels for unlabeled samples and inducing classifiers using both labeled and pseudo-labeled samples in a self-training manner. Unfortunately, with the commonly used binary type of loss and negative sampling, we have empirically found that learning with labeled and pseudo-labeled samples can result in the variance bias problem between the feature distributions of positive and negative samples for each label. To alleviate this problem, we aim to balance the variance bias between positive and negative samples from the perspective of the feature angle distribution for each label. Specifically, we extend the traditional binary angular margin loss to a balanced extension with feature angle distribution transformations under the Gaussian assumption, where the distributions are iteratively updated during classifier training. We also suggest an efficient prototype-based negative sampling method to maintain high-quality negative samples for each label. With this insight, we propose a novel SSMLL method, namely Semi-Supervised Multi-Label Learning with Balanced Binary Angular Margin loss (S$^2$ML$^2$-BBAM). To evaluate the effectiveness of S$^2$ML$^2$-BBAM, we compare it with existing competitors on benchmark datasets. The experimental results validate that S$^2$ML$^2$-BBAM can achieve very competitive performance.

# Getting the Data
You can get the data according to the README.md in ./data

# Training the Model 
You can train the model according to the ./running 

For VOC2012:

```
for I in  0.05 0.1 0.15 0.2
 do
     python SSMLL-BBAM/run_warmup.py --loss BAM --lb_ratio $I --warmup_epochs 12 --warmup_batch_size 16 --lr 5e-5 --net resnet50 --dataset_name VOC2012 --dataset_dir Dataset --cos-margin 0.4 --cos-norm 20
     python SSMLL-BBAM/run_BAM.py --loss BAM --lb_ratio $I --warmup_epochs 12 --warmup_batch_size 16 --lr 5e-5 --net resnet50 --dataset_name VOC2012 --dataset_dir Dataset --init_pos_per 1.0 --init_neg_per 1.0 --cos-margin 0.4 --cos-norm 20
 done
```

