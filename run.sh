#!/bin/sh
export CUDA_VISIBLE_DEVICES=7 
dl_predict --config /allen/aics/assay-dev/users/Dewen/project/code/aics-ml-segmentation/configs/baseline_testing_config.yaml
# python aicsmlsegment/bin/predict.py --config /allen/aics/assay-dev/users/Dewen/project/code/aics-ml-segmentation/configs/baseline_testing_config.yaml