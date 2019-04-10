# Configuration for Training a DL Model in **Trainer**

This is a detailed description of the configuration for training a DL model in **Trainer**. There are a lot of parameters in the configuration file, which can be categorized into three types:

1. Parameters specific to each training (need to change/check every time), marked by :pushpin:
2. Parameters specific to each machine (only need to change once on a particular machine), marked by :computer:
3. Parameters pre-defined for the general training scheme (no need to change for most problems and need basic knowledge of deep learning to adjust), marked by :ok:


### Model related parameters

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
There may be probably more than 100 models in the literature for 3D image segmentation. The two models we implemented here are carefully designed for cell structure segmentation in 3D microscopy images. Model `unet_xy` is suitable for smaller-scale structures, like severl voxels thick (e.g., tubulin, lamin b1). Model `unet_xy_zoom` is more suitable for larger-scale structures, like more than 100 voxels in diameter (e.g., nucleus), while the `zoom_ratio` is an integer (e.g., 2 or 3) and can be estimated by average diameter of target object in voxels divided by 150. 

2. input and output type (:ok:)
```yaml
nchannel: 1
nclass: [2, 2, 2]
```
These are related to the model architecture and fixed by default. 

3. patch size (:computer:)

```yaml 
size_in: [48, 148, 148] 
size_out: [20, 60, 60]
```
In most situations, we cannot fit the entire image into the memory of a single GPU. These are also related to `batch_size` (an data loader parameter), which will be discussed shortly. Here are some pre-calculated values for different models on different types of GPUs.

|                           | size_in           | size_out          |  batch_size   |
| --------------------------|:-----------------:|:-----------------:|:-------------:|
| unet_xy on 8GB GPU        |                   |                   |               |
| unet_xy on 32GB GPU       | [48, 148, 148]    | [20, 60, 60]      |       8       |
| unet_xy_zoom on 8GB GPU   |                   |                   |               |
| unet_xy_zoom on 32GB GPU  | [52, 420, 420]    | [20, 152, 152]    |       8       |

4. model directory
```yaml
checkpoint_dir:  /home/model/xyz/
resume: null
```
This is the directory to save the trained model. If you want to start this training from a previous saved model, you may add the path to `resume`.

### Training scheme realted parameters
1. optimization parameters (:ok:)
```yaml
learning_rate: 0.00001
weight_decay: 0.005
loss:
  name: Aux
  loss_weight: [1, 1, 1]
  ignore_index: null
```

2. training epochs (:pushpin:)
```yaml
epochs: 400
save_every_n_epoch: 40
```
`epochs` controls how many iterations in the training. We suggest to use values between 300 and 600. A model will be saved on every `save_every_n_epoch` epochs.


### Data realted parameters 

```yaml
loader:
  name: default
  datafolder: '/home/data/train/'
  batch_size: 8
  PatchPerBuffer: 200
  epoch_shuffle: 5
  NumWorkers: 1
```
`datafolder` and `PatchPerBuffer` (:pushpin:) need to check in each training. `datafolder` is the directory of training data. `PatchPerBuffer` is the number of sample patches randomly drawn in each epoch, which can be set as *number of patches to draw from each data* **x** *number of training data*. `name`, `epoch_shuffle` and `NumWorkers` (:ok:) are fixed by default. `batch_size` is related to GPU memory and patch size (see values presented with patch size).

### Validation related parameter

In machine learning studies, we usually do a validation after every few epochs to make sure things are not going wrong. For most 3d microscopy image segmentation problems, the training data is very limited. We cannot save a portion from the training data for validation purpose. So, by default, validation is turn-off (:ok:) and may only be used for advanced users. 