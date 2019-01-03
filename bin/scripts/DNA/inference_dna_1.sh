#!/usr/bin/env bash

GPU=2
MODEL=unet_xy_p3
NCHANNEL=1
STATE=/allen/aics/assay-dev/Segmentation/DeepLearning/DNA_Labelfree_Evaluation/WOO/saved_model/20181017_01/unet_xy_p3-450-default.pth
NCLASS="2 2 2"
SIZEIN="76 420 420"
SIZEOUT="44 152 152"
OUTPUTCH="0 1"

INPUTDIR=/allen/aics/assay-dev/Segmentation/DeepLearning/DNA_Labelfree_Evaluation/final_evaluation/sample_original
OUTPUTDIR=/allen/aics/assay-dev/Segmentation/DeepLearning/DNA_Labelfree_Evaluation/final_evaluation/dna_out_1/

#INPUTDIR=//allen/aics/assay-dev/Segmentation/DeepLearning/TestData/caax/evaluation/raw/
#OUTPUTDIR=//allen/aics/assay-dev/Segmentation/DeepLearning/TestData/caax/evaluation/dna_seg/

DATATYPE=.czi
RESIZE="1.0 1.0 1.0"
TH=-1  #0.5 #0.95
INPUTCH=-2
NORM=12


python ../../exp_scheduler.py \
    --model $MODEL \
    --state  $STATE \
    --nchannel $NCHANNEL \
    --nclass $NCLASS \
    --size_in $SIZEIN \
    --size_out $SIZEOUT \
    --OutputCh $OUTPUTCH \
    --gpu $GPU \
    eval \
    --InputDir  $INPUTDIR \
    --OutputDir $OUTPUTDIR \
    --DataType $DATATYPE \
    --ResizeRatio $RESIZE \
    --InputCh $INPUTCH  \
    --Normalization $NORM \
    --Threshold $TH
