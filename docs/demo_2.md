# Demo 2: Segmentation of Lamin B1 in 3D fluorescent microscopy images of hiPS cells 

In this demo, we will show how we get the segmentation of Lamin B1 in 3D fluorescent microscopy images of hiPS cells. 

## Stage 1: Run **Binarizer** (a classic image segmentation workflow)

We refer [demo 1](./demo_1.md) for how to develop a classic image segmentation workflow. Suppose we already have work out a workflow for it and save it as `seg_lmnb1_interphase.py` (i.e., `workflow_name=lmnb1_interphase`). So, we can run 

```bash
batch_processing --workflow_name lmnb1_interphase --struct_ch 0 --output_dir /allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_classic_workflow_segmentation_iter_1 per_dir --input_dir  /allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_fluorescent --data_type .tiff
```

> Are they good?

After looking at the result, we find some results are good and some have errors like the picture below (left: original; right: binary image from **Binarizer**). 

![wf1 pic](./wf_pic.png)

Some objects are missed in the segmentation due to the failure of an automatic seeding step (see the yellow arrow). Also, this workflow has a poor performance on mitotic cells (see the blue arrow). In short, the segmentation on some images are good, but fails on others. So, we want to leverage the successful ones to build a DL model.

## Stage 2: Run **Curator** (sorting)

The curation goal of this step is to collect those images have been successfully segmented. So, we chose to use the "sorting" strategy in **Curator**. 

```bash
curator_sorting \
    --raw_path /allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_fluorescent \
    --input_ch 0 \
    --seg_path /allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_classic_workflow_segmentation_iter_1 \
    --mask_path /allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_mask_iter_1 \
    --csv_name /allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/sorting_test.csv \
    --train_path /allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_training_data_iter_1 \
    --Normalization 10
```

## Stage 3: Run **Trainer** 

After clicking through all images, the training data is saved in `/allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_training_data_iter_1`. Manually updating the paths (path to training data and path to save the trained model) the training configuration file in 'train.yaml'. Then, simply run

```bash
dl_train --config /home/config_files/train_lab.yaml
```
Depending on the size of your training data, the training process may take 8~32 hours

## Stage 4: Run **Binarizer**

The trained model is saved at `/allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_saved_model_iter_1/checkpoint_epoch_400.pytorch`. After updating the paths in prediction configuration (`predict_folder_config.yaml`), we can simply run

```bash
dl_predict --config ../config/predict_folder_config.yaml
```

Looking at the results, we find that Lamin B1 in all interphase cells are segmented very well, but still need improvement for mitotic cells. 

How can we segment Lamin B1 in mitotic cells better? We can quickly develop a workflow to get reasonable segmentation of Lamin B1 in mitotic cells. Again, we refer [demo 1](./demo_1.md) for how to develop a classic image segmentation workflow. Suppose we already have work out this workflow and call it `lmnb1_mitotic`.

For convenice, we use a set of samples from mitotic enriched experiments, where there are usually at least one mitotic cell in each FOV. Suppose the images are saved at `/allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_fluorescent_mitosis`. Then, we run the **Binarizer** twices

* first run with the DL model and save at `/allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_DL_iter_2`

```bash
dl_predict --config ../config/predict_folder_config.yaml
```

* second run with the `lmnb1_mitotic` workflow, and save at `/allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_classic_workflow_segmentation_iter_2`

```bash
batch_processing --workflow_name lmnb1_mitotic --struct_ch 0 --output_dir /allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_classic_workflow_segmentation_iter_2 per_dir --input_dir /allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_fluorescent_mitosis --data_type .tiff
```

## Stage 5: Run **Curator**

The curation goal of this step is for each image merge the two segmentation versions (one for interphase, one for mitotic). So, we chose to use the "merging" strategy in **Curator**.

```bash
curator_merging \
    --raw_path /allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_fluorescent_mitosis/  \
    --input_ch 0  \
    --seg1_path /allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_DL_iter_2/ \
    --seg2_path /allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_classic_workflow_segmentation_iter_2 \
    --mask_path /allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_mask_iter_2   \
    --ex_mask_path /allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_excluding_mask_iter_2 \
    --csv_name /allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/merging_test.csv  \
    --train_path /allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_training_data_iter_2 \
    --Normalization 10
```

## Stage 6: Run **Trainer**

After clicking through all images, the training data is saved in `/allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_training_data_iter_2`. Manually updating the paths (path to training data and path to save the trained model) the training configuration file in 'train.yaml'. Then, simply run

```bash
dl_train --config /home/config_files/train_lab.yaml
```
## Stage 7: Run *Binarizer*

The trained model is saved at `/allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_saved_model_iter_1/checkpoint_epoch_400.pytorch`. After updating the paths in prediction configuration (`predict_folder_config.yaml`), we can simply run

```bash
dl_predict --config ../config/predict_folder_config.yaml
```

Looking at the results, Lamin B1 in both interphase cells and mitotic cells are segmented well. 
