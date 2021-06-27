#!/bin/sh
export CUDA_VISIBLE_DEVICES=3
dl_predict --config /allen/aics/assay-dev/users/Dewen/project/code/aics-ml-segmentation/configs/baseline_testing_config3.yaml
# python aicsmlsegment/bin/predict.py --config /allen/aics/assay-dev/users/Dewen/project/code/aics-ml-segmentation/configs/baseline_testing_config.yaml

# export CUDA_VISIBLE_DEVICES=1,2
# dl_train --config /allen/aics/assay-dev/users/Dewen/project/code/aics-ml-segmentation/configs/baseline_training_config.yaml