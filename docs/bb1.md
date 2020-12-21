# Building Block 1: **Binarizer**


The **Binarizer** is the core building block actually doing the computation for getting segmentation by either a classic image segmentation workflow or a model trained by an iterative deep learning workflow. We refer [this documentation](./demo_1.md) for a demo on how to develope a classic image segmentation workflow for a specific cell structure, and [Curator tutorial](./bb2.md) + [Trainer tutorial](./bb3.md) for how to train a deep learning based segmentation model. 

![segmenter pic](./bb1_pic.png)

## Option 1: Classic image segmentation 

Suppose you already build a classic image segmentation workflow for your data and you call this workflow, for example "FBL_HIPSC". Assume the original images are multi-channel with structure channel in the second (so use`--struct_ch 1`, sine python is zero-based).

### Apply on one image


```bash
batch_processing \
    --workflow_name FBL_HIPSC \
    --struct_ch 0 \
    --output_dir /path/to/save/segmentation/ \
    per_img \
    --input /path/to/image_test.tiff 
```

### Apply on a folder of images 

Suppose we want to segment all `.tiff` files in one folder, we can do

```bash
batch_processing \
    --workflow_name FBL_HIPSC \
    --struct_ch 0 \
    --output_dir /path/to/save/segmentation/ \
    per_dir \
    --input_dir /path/to/raw_images/ \
    --data_type .tiff
```


## Option 2: Deep learning segmentation model

### Understanding model output

The actual prediction from a deep learning based segmentation model is not binary. The value of each voxel is a real number between 0 and 1. To make it binary, we usually apply a cutoff value, i.e., the `Threshold` parameter in the [configuration file](./doc_pred_yaml.md). For each model, a different cutoff value may be needed. To determine a proper cutoff value, you can use `-1` for `Threshold` on sample images and open the output in ImageJ (with [bio-formats importer](https://imagej.net/Bio-Formats#Bio-Formats_Importer)) and try out different threshold values. Then, you can set `Threshold` as the new value and run on all images. Now, the results will be binary.


### Apply on one image

Find/build a `.yaml` file for processing a single file (e.g., `./config/predict_file.yaml`) and make sure to follow the list [**here**](./doc_pred_yaml.md) to change the parameters, such as the image file path, the output path, the model path, etc..

```bash
dl_predict --config /path/to/predict_file.yaml
```

### Apply on a folder of images 

Find/build a `.yaml` file for processing a folder of images (e.g., `./config/predict_folder.yaml`) and make sure to follow the list [**here**](./doc_pred_yaml.md) to change the parameters, such as the image folder path, the output path, the model path, etc..

```bash
dl_predict --config /path/to/predict_folder.yaml
```