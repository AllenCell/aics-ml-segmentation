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
See model parameters in [training configuration](./doc_train_yaml.md)

2. input and output type (:ok:)
```yaml
nchannel: 1
nclass: [2, 2, 2]
OutputCh: 1
```
These are related to the model architecture and fixed by default. See model parameters in [training configuration](./doc_train_yaml.md)

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

### Prediction 
```yaml
OutputCh: 1 
precision: 16 
batch_size: 1
inference_mode: 'fast'
gpus: -1 
dist_backend: ddp 
NumWorkers: 8 
segmentation_name: "name_of_type_of_segmentation" 
``` 
`OutputCh` specifies which channel should be extracted from the raw output image. `precision` specifies whether 16- or 32-bit model weights should be used. `inference_mode` has two options: `fast` or `efficient`. `fast` mode increases speed at the cost of GPU memory usage, while `efficient` mode decreases GPU memory usage at the cost of speed. Multi-GPU training using ddp is supported. If `gpus` = -1, all available GPUs will be used. Otherwise, `gpus` must be a positive integer and up to that many GPUs will be used depending on GPU availability. If multiple GPUs are used, a `dist_backend` can be specified. Currently, ddp is the only supported backend. `NumWorkers` specifies how many workers should be spawned for loading and normalizing images. Additional workers increase CPU memory usage. `segmentation_name` will be saved in the image metadata of the output image to identify the segmentation. 


### Data Info (:pushpin:)
```yaml
mode:
  name: folder
  InputDir: "path/to/input/input/img"
  DataType: .tiff

OutputDir:  'path/to/save/results'
InputCh: [0]
ResizeRatio: [1.0, 1.0, 1.0]
large_image_resize: [1,1,1]
Threshold: 0.75
RuntimeAug: False
Normalization: 10
uncertainty: 'entropy'

```

`DataType` is the type of images to be processed in `InputDir`, which the `InputCh`'th (keep the [ ]) channel of each image will be segmented. If your model is trained on images of a certain resolution and your test images are of different resolution `ResizeRatio` needs to be set as [new_z_size/old_z_size, new_y_size/old_y_size, new_x_size/old_x_size]. The actual output is the likelihood of each voxels being the target structure. `large_image_resize` can be set if large images cause GPU out of memory during prediction. This parameter specifies how many patches in the `ZYX` axes each image should be split into. After patch-wise prediction, the final image is reconstructed based on overlap between patches.  A `Threshold` between 0 and 1 needs to be set to generate the binary mask. We recommend to use 0.6 ~ 0.9. When `Threshold` is set as `-1`, the raw prediction from the model will be saved, for users to determine a proper binary cutoff. `Normalization` is the index of a list of pre-defined normalization recipes and should be the same index as generating training data (see [Curator](./bb2.md) for the full list of normalization recipes). If `RuntimeAug` is `True`, the model will predict on the original image and three flipped versions of the image. The final prediction is then averaged across each flipped prediction. This increases prediction quality, but takes ~4x longer. `Uncertainty` is an optional argument that can be used with dropout models. Possible values area `entropy`, `softmax`, `variance`, or `mutual_information`. These uncertainty estimation techniques run inference 10 times on each image and calculate the uncertainty based on differences between runs. 
  
  
  