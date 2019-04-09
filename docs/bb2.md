# Building Block 2: **Curator**

The **Curator** is used to prepare training data for **Trainer**. This is the core of the "iterative" nature of the iterative DL workflow we presented in the [paper](https://www.biorxiv.org/content/10.1101/491035v1). Namely, you have some preliminary segmentation and you can improve the performance by training a deep learning model on curated segmentations, iteratively. 

![segmenter pic](./bb2_pic.png)

There are three scenarios that we currently support. But the same "curation" spirit can be adapted based on current scripts for your special needs. 

1. Sorting: Suppose your segmentation algorithm works well on a subset of images that you need to analyze (maybe due to unstability of the algorithm or variations between images). In this senario, you can sort out the successful cases and train your model on them.
2. Merging: Suppose objects to be segmented in each image form two sub-populations (e.g., mitotic cells vs. interphase cells) and different algorithms are needed to segment each sub-population well. In this senario, you can merge the two segmentation versions and train your model on the merged ground truth. 
3. Take-All: You may already have ground truth ready to use (e.g., by manual annotation or you are using simulated images with known ground truth). In this scenrio, we have a simple script to convert your data into the format compatible with **Trainer**.


## Sorting:

Suppose you have a set of images, saved at `/home/data/original/`. In each image, the structure channel is the third one (so, index=2, zero-base). The current binary segmentations are saved at `/home/data/segmentation_v1/`. You want to save the training data at `/home/data/training_v1/`. A `.csv` file will be generated to track this process and resume the process when necessary. We need to give it a name, say `/home/data/curator_v1_tracker.csv`. We also want to save the images masking areas to be excluded (see special note 1 below) in each image at `/home/data/sorting_excluding_mask_v1/`. Also, check special note 2 for input image normalization, we use recipe 15 here.

### How to run?

```bash
curator_sorting --raw_path /home/data/original/ --input_channel 2 --seg_path /home/data/segmentation_v1/ --train_path /home/data/training_v1/ --csv_name /home/data/curator_v1_tracker.csv --mask_path  /home/data/sorting_excluding_mask_v1/  --Normalization 15
```

### How to use?

A side-by-side view of the orginal image and the segmentation is first pop up. 

* left mouse click = 'Bad'
* right mouse click = 'Good'

If an image is labeled as 'Good' (i.e, after a right mouse click), users will be asked if an excluing mask is needed. If so, type in `y` in the command line. Otherwise, type in `n`. When selecting `y`, a second window will pop up for drawing polygons on the image as the mask.

* To add a polygon, left mouse click will be recorded as the vertices of the polygon. After the last vertex, a right mouse click will close the polygon (connecting the last vertex to the first vertex). **Make sure you only draw within the upper left pannel, i.e., the orginal image** 
* Users can add multiple polygons in one image
* After finishing one image, press `d` to close the window and move on the next image.



## Merging: 

Suppose you have a set of images, saved at `/home/data/original/`. In each image, the structure channel is the third one (so, index=2, zero-base). Two different versions of preliminary structure segmentations are saved at `/home/data/segmentation_v1/` and `/home/data/segmentation_v2/`. The mask for merging the two versions will be saved at `/home/data/merging_mask_v1/`. For each mask, you will draw polygons on a 2D image (max z-projection) to indicate the areas to use `segmentation_v1`, while the areas outside the polygons will use `segmentation_v2`. You want to save the training data at `/home/data/training_v1/`. A `.csv` file will be generated to track this process and resume the process when necessary. We need to give it a name, say `/home/data/curator_v1_tracker.csv`. We also want to save the images masking areas to be excluded (see special note 1 below) in each image at `/home/data/merging_excluding_mask_v1/`. Also, check special note 2 for input image normalization, we use recipe 15 here.

### How to run?

```bash
curator_sorting --raw_path /home/data/original/ --input_channel 2 --seg1_path /home/data/segmentation_v1/ --seg2_path /home/data/segmentation_v2/ --train_path /home/data/training_v1/ --csv_name /home/data/curator_v1_tracker.csv --mask_path  /home/data/merging_mask_v1/ --ex_mask_path  /home/data/merging_excluding_mask_v1/ --Normalization 15
```

### How to use?

A side-by-side view of the orginal image and the two versions of segmentation is first pop up. Uses can first determine this image worth being included or not. If it is bad and not worth being used for training, simple press `b` to label it as "bad" and move on to the next image. Otherwise, users can start drawing polygons. **Make sure you only draw within the upper left pannel, i.e., the orginal image. For each polygon you draw, the segmentation on the rightmost panel will be used to replace the corresponding part in the middel panel.**

* To add a polygon, left mouse click will be recorded as the vertices of the polygon. After the last vertex, a right mouse click will close the polygon (connecting the last vertex to the first vertex). **Make sure you only draw within the upper left pannel, i.e., the orginal image** 
* Users can add multiple polygons in one image
* After finishing one image, press `d` to close the window and move on the next image.

After each merging mask, users will be asked if an excluing mask is needed. If so, type in `y` in the command line. Otherwise, type in `n`. When selecting `y`, a second window will pop up for drawing polygons on the image as the excluding mask (usuage is the same as for mering mask).


## Take-All:

```bash
curator_takeall --raw_path /home/data/original/ --input_channel 2 --seg_path /home/data/segmentation_v1/ --train_path /home/data/training_v1/ --Normalization 15 
```

=======================

### Special note 1: Masking areas to be excluded

Sometimes, in all three scenarios above (sorting/merging/take-all), there could be a small area in a particular image needs to be excluded from training. In the context of sorting, for example, an image is almost segmented well except a small area. We may not want to simply say the segmentation of this image failed. So,  we always have an option to include such masking areas in all three scenarios above. This is optional and only meant to include more data for training. For sorting/merging, an optional step can be triggered to draw a mask (polygons) on a 2D image (max z-projection) to indicate the areas to be excluded. For take-all, an optional folder can be used saved all mask images.


### Special note 2: "--Normalization"

It is important to normalize your image before fed into the deep learning model. For example, if your model is trained on image with values from 300 to 400 with mean intensity 310, it can hardly be applied a new image with values from 360 to 480 with mean intensity 400, even the actual content looks very similar. Currently, all of our "recipes" are based on two basic functions: min-max, auto contrast and background subtraction. Min-max and auto contrast are the same as the one we explained [here](). `suggest_normalization` in the `aicssegmentation` package can help to detemine propoer parameters for your data. Background subtraction is implemented as subtracting gaussian smoothed image from the original image and recale to [0, 1]. This may be used to correct uneven intensity. The only parameter is the gaussian kernel size. As a rule of thumb, one can use half the size of average uneven areas. An optional parameter is to set an upper bound intensity, so that any intensity value over the upper bound will be considered as outliers and re-assign to the min intensity of the image. IN case you need to add your own recipes, you can simply modify the function `input_normalization` in `utils.py` (e.g., cope and past one of current recipes and change the parameters, and give it a new recipe index).

List of current pre-defined recipes:

* 0: min-max
* 1: auto contrast [mean - 2 * std, mean + 11 * std]
* 2: auto contrast [mean - 2.5 * std, mean + 10 * std]
* 7: auto contrast [mean - 1 * std, mean + 6 * std]
* 10: auto contrast [min, mean + 25 * std] with upper bound intensity 4000
* 12: background subtraction (kernel=50) + auto contrast [mean - 2.5 * std, mean + 10 * std]
* 15: background subtraction (kernel=50) with uppder bound intensity 4000

### Special note 3: Interface design

Currently, the interface of merging/sorting is implemented purely with `matplotlib`, without any advanced packages for interface building. Our goal is to keep it simple and make it easy to setup, robust across different platforms, and easy to be hacked so that users may customize their own curation interface when necessary. 
