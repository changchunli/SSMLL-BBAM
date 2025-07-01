# Getting the Data

## VOC

1. Navigate to the VOC data directory:
```
cd ./voc
```
2. Download the data


## COCO

1. Navigate to the COCO data directory:
```
cd ./coco
```
2. Download the data


## AWA
1. Navigate to the COCO data directory:
cd ./awa

2. Download the data


# All urls can be found in the paper.


# Formatting the Data
The `preproc` folder contains a few scripts which can be used to produce uniformly formatted image lists and labels:
```
cd ../preproc
python format_coco.py
python format_voc.py
python format_awa.py
```
