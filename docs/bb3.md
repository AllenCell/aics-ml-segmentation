# Building Block 3: **Trainer**

**Trainer** is used to train deep learning-based segmentation models. The input for **Trainer** are usually data prepared by **Curator** (see [documentation](./bb2.md)) and the output should be a model that can be used in **Segmenter**. 

In case you don't want to use **Curator** to generate the training data, the images should follow the naming convention: the data folder should contain `XYZ.tiff` (raw image, float, properly normalized to [0,1]), `XYZ_GT.tiff` (ground truth, uint8, 0=background, 1=foreground of class 1, can have more integers if more classes are needed), `XYZ_CM.tiff` (cost map, float, the importance of each pixel, usually 0 for areas to be excluded and 1 for normal areas). 

![segmenter pic](./bb3_pic.png)

Find/build the `.yaml` file for training (e.g, './config/train.yaml') and make sure to following the list [**here**](./doc_train_yaml.md) to change the parameters, such as the training data path, the path for saving the model, etc.. 

```bash
dl_train --config /home/config_files/train_lab.yaml
```

### When multiple GPUs are available

By default, **Trainer** will use the first available GPU for computation. If there are multiple GPUs on your machine, you can choose which GPU to use by setting `CUDA_VISIBLE_DEVICES` before running **Trainer**.

```bash
CUDA_VISIBLE_DEVICES=2  dl_train --config /home/config_files/train_lab.yaml
```
