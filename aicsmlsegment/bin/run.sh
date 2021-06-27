#!/bin/sh
# export CUDA_VISIBLE_DEVICES=7 
# dl_predict --config /allen/aics/assay-dev/users/Dewen/project/code/aics-ml-segmentation/configs/baseline_testing_config.yaml --config2 /allen/aics/assay-dev/users/Dewen/project/code/aics-ml-segmentation/configs/baseline_testing_config2.yaml
# python aicsmlsegment/bin/predict.py --config /allen/aics/assay-dev/users/Dewen/project/code/aics-ml-segmentation/configs/baseline_testing_config.yaml

# export CUDA_VISIBLE_DEVICES=6,7
# python train_cv.py --config /allen/aics/assay-dev/users/Dewen/project/code/aics-ml-segmentation/configs/baseline_training_config3.yaml

export CUDA_VISIBLE_DEVICES=7
python predict_cv.py --config /allen/aics/assay-dev/users/Dewen/project/code/aics-ml-segmentation/configs/baseline_testing_config4.yaml