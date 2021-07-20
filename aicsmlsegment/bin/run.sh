#!/bin/sh

# dl_predict --config /allen/aics/assay-dev/users/Dewen/project/code/aics-ml-segmentation/configs/baseline_testing_config.yaml --config2 /allen/aics/assay-dev/users/Dewen/project/code/aics-ml-segmentation/configs/baseline_testing_config2.yaml
export CUDA_VISIBLE_DEVICES=4
dl_predict --config /allen/aics/assay-dev/users/Dewen/project/code/aics-ml-segmentation/configs/baseline_testing_config6.yaml

# export CUDA_VISIBLE_DEVICES=5,6
# python train_cv.py --config /allen/aics/assay-dev/users/Dewen/project/code/aics-ml-segmentation/configs/baseline_training_config3.yaml

# export CUDA_VISIBLE_DEVICES=2
# python predict_cv.py --config /allen/aics/assay-dev/users/Dewen/project/code/aics-ml-segmentation/configs/baseline_testing_config4.yaml

# export CUDA_VISIBLE_DEVICES=3
# python predict_qc.py --config /allen/aics/assay-dev/users/Dewen/project/code/aics-ml-segmentation/configs/baseline_testing_config5.yaml

# export CUDA_VISIBLE_DEVICES=2
# python train_qc.py --config /allen/aics/assay-dev/users/Dewen/project/code/aics-ml-segmentation/configs/baseline_training_config5.yaml

# export CUDA_VISIBLE_DEVICES=2,3
# dl_train --config /allen/aics/assay-dev/users/Dewen/project/code/aics-ml-segmentation/configs/baseline_full_train_config.yaml

# export CUDA_VISIBLE_DEVICES=4,5
# python train_probablistic.py --config /allen/aics/assay-dev/users/Dewen/project/code/aics-ml-segmentation/configs/baseline_train_probablistic_unet.yaml

# export CUDA_VISIBLE_DEVICES=6
# python predict_probablistic.py --config /allen/aics/assay-dev/users/Dewen/project/code/aics-ml-segmentation/configs/baseline_testing_probablistic_unet.yaml