# Building Block 2: **Curator**

The **Curator** is used to prepare training data for **Trainer**. This is the core of the "iterative" nature of the iterative DL workflow we presented in the [paper](https://www.biorxiv.org/content/10.1101/491035v1). Namely, you have some preliminary segmentation and you can improve the performance by training a deep learning model on curated segmentations, iteratively. 

![segmenter pic](./bb2_pic.png)

There are three scenarios that we currently support. But the same "curation" spirit can be adapted based on current scripts for your special needs. 

1. Sorting: Suppose your segmentation algorithm works well on a subset of images that you need to analyze (maybe due to unstability of the algorithm or variations between images). In this senario, you can sort out the successful cases and train your model on them.
2. Merging: Suppose objects to be segmented in each image form two sub-populations (e.g., mitotic cells vs. interphase cells) and different algorithms are needed to segment each sub-population well. In this senario, you can merge the two segmentation versions and train your model on the merged ground truth. 
3. Take-All: You may already have ground truth ready to use (e.g., by manual annotation or you are using simulated images with known ground truth). In this scenrio, we have a simple script to convert your data into the format compatible with **Trainer**.


## Sorting:

Suppose you have a set of images, saved at `/home/data/original/`. In each image, the structure channel is the third one (so, index=2, zero-base). The preliminary structure segmentations are saved at `/home/data/segmentation_v1/`. You want to save the training data at `/home/data/training_v1/`. A `.csv` file will be generated to track this process and resume the process when necessary. We need to give it a name, say `/home/data/curator_v1_tracker.csv`

(Note: Sometimes, an image is almost segmented perfectly, just a small area is not good. We may not want to simply say the segmentation of this image failed. So, we have an optional step to draw a mask (polygons) on a 2D image (max z-projection) to indicate the areas to be excluded. We can, for example, save them at `/home/data/sorting_mask_v1/`. This is optional and only meant to include more training data.)


```bash
curator_sorting --raw_path /home/data/original/ --input_channel 2 --seg_path /home/data/segmentation_v1/ --train_path /home/data/training_v1/ --csv_name /home/data/curator_v1_tracker.csv --mask_path  /home/data/sorting_mask_v1/  --Normalization 15
```

## Merging: 

Suppose you have a set of images, saved at `/home/data/original/`. In each image, the structure channel is the third one (so, index=2, zero-base). Two different versions of preliminary structure segmentations are saved at `/home/data/segmentation_v1/` and `/home/data/segmentation_v2/`. The mask for merging the two versions will be saved at `/home/data/merging_mask_v1/`. For each mask, you will draw polygons on a 2D image (max z-projection) to indicate the areas to use `segmentation_v1`, while the areas outside the polygons will use `segmentation_v2`. You want to save the training data at `/home/data/training_v1/`. A `.csv` file will be generated to track this process and resume the process when necessary. We need to give it a name, say `/home/data/curator_v1_tracker.csv`

```bash
curator_sorting --raw_path /home/data/original/ --input_channel 2 --seg1_path /home/data/segmentation_v1/ --seg2_path /home/data/segmentation_v2/ --train_path /home/data/training_v1/ --csv_name /home/data/curator_v1_tracker.csv --mask_path  /home/data/merging_mask_v1/ --Normalization 15
```

## Take-All:

```bash
curator_takeall --raw_path /home/data/original/ --input_channel 2 --seg_path /home/data/segmentation_v1/ --train_path /home/data/training_v1/ --Normalization 15 
```

=======================

Special note about "--Normalization"

List of all recipes:

* 1
* 3
* 5
* 15

