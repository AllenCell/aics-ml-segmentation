# Configuration for Running a DL Model in **Segmenter**

This is a detailed description of the prediction configuration for running a DL model in **Segmenter** to generate the segmentation. There are a lot of parameters in the configuration file, which can be categorized into three types:

1. Parameters specific to each running (need to change every time), marked by :pushpin: 
2. Parameters specific to each machine (only need to change once on a particular machine), marked by :computer:
3. Parameters pre-defined for the general training scheme (no need to change for most problems and need basic knowledge of deep learning to adjust), marked by :ok:


### model-related parameters

1. choose which model to use (:pushpin:)
```yaml
model: 
  name: unet_xy_zoom
  zoom_ratio: 3
```
or
```yaml
model: 
  name: unet_xy
```
see model parameters in [training configuration](./doc_train_yaml.md)

2. input and output type (:ok:)
```yaml
nchannel: 1
nclass: [2, 2, 2]
OutputCh: [0, 1]
```
These are related to the model architecture and fixed by default. 

3. patch size (:computer:)

```yaml 
size_in: [48, 148, 148] 
size_out: [20, 60, 60]
```
see patch size parameters in [training configuration](./doc_train_yaml.md)

4. model directory (:pushpin:)
```yaml
model_path: '/home/model/checkpoint_epoch_300.pytorch'
```
This the place to specify which trained model to run. 


### Data Info (:pushpin:)
```yaml
OutputDir: '//allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/'
InputCh: [0]
ResizeRatio: [1.0, 1.0, 1.0]
Threshold: 0.75
RuntimeAug: False
Normalization: 10
mode:
  name: folder
  InputDir: '/allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_fluorescent'
  DataType: tiff
```

`DataType` is the type of images to be processed in `InputDir`, which the `InputCh`'th (keep the [ ]) channel of each images will be segmented. If your model is trained on images of a certain resolution and your test images are of different resolution `ResizeRatio` needs to be set as [new_z_size/old_z_size, new_y_size/old_y_size, new_x_size/old_x_size]. The acutal output is the likelihood of each voxels being the target structure. A `Threshold` between 0 and 1 needs to be set to generate the binary mask. We recommend to use 0.6 ~ 0.9. When `Threshold` is set as `-1`, the raw prediction from the model will be saved, for users to determine a proper binary cutoff. `Normalization` is the index of a list of pre-defined normalization recipes and should be the same index as generating training data (see [Curator](./bb2.md) for the full list of normalization recipes).
  
  
  