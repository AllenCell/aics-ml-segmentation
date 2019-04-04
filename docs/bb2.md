# Building Block 2: **Curator**

The **Curator** is used to prepare training data for **Trainer**. This is the core of the "iterative" nature of the iterative DL workflow we presented in the [paper](https://www.biorxiv.org/content/10.1101/491035v1). Namely, you have some preliminary segmentation and you can improve the performance by training a deep learning model on curated segmentations, iteratively. 

![segmenter pic](./bb2_pic.png)

There are three scenarios that we currently support. But the same "curation" spirit can be adapted based on current scripts for your special needs. 

1. Sorting: Suppose your segmentation algorithm works well on a subset of images that you need to analyze (maybe due to unstability of the algorithm or variations between images). In this senario, you can sort out the successful cases and train your model on them.
2. Merging: Suppose objects to be segmented in each image form two sub-populations (e.g., mitotic cells vs. interphase cells) and different algorithms are needed to segment each sub-population well. In this senario, you can merge the two segmentation versions and train your model on the merged ground truth. 
3. You may already have ground truth ready to use (e.g., by manual annotation or you are using simulated images with known ground truth). In this scenrio, we have a simple script to convert your data into the format compatible with **Trainer**.


## Sorting:



