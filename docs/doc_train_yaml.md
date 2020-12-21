# Configuration for Training a DL Model in **Trainer**

This is a detailed description of the configuration for training a DL model in **Trainer**. There are a lot of parameters in the configuration file, which can be categorized into three types:

1. Need to change on every run (i.e., parameters specific to each execution), marked by :warning:
    * `checkpoint_dir` (where to save trained models), `datafolder` (where are training data), `resume` (whether to start from a previous model)
2. Need to change for every segmentation problem (i.e., parameters specific to one problem), marked by :pushpin:
    * `model`, `epochs`, `save_every_n_epoch`, ``PatchPerBuffer``
3. Only need to change once on a particular machine (parameters specific to each machine), marked by :computer:
4. No need to change for most problems (parameters pre-defined as a general training scheme and requires advacned deep learning knowledge to adjust), marked by :ok:


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

2. start from an existing model? (:warning:)

```yaml
resume: null
```

When doing iterative deep learning, it may be useful to start from the model trained in the previous step. The model can be specified at `resume`. If `null`, a new model will be trained from scratch. 

3. input and output type (:ok:)
```yaml
nchannel: 1
nclass: [2, 2, 2]
```
These are related to the model architecture and fixed by default. We assume the input image has only one channel.

4. patch size (:computer:) (:pushpin:)

```yaml 
size_in: [50, 156, 156] 
size_out: [22, 68, 68]
```
In most situations, we cannot fit the entire image into the memory of a single GPU. These are also related to `batch_size` (an data loader parameter), which will be discussed shortly. `size_in` is the actual size of each patch fed into the model, while `size_out` is the size of the model's prediction. The prediction size is smaller than the input size is because the multiple convolution operations. The equation for calculating `size_in` and `size_out` is as follows.

> For unet_xy, `size_in` = `[z, 8p+60, 8p+60]`, `size_out` = `[z-28, 8p-28, 8p-28]`

> For unet_xy_zoom, with `zoom_ratio`=`k`, `size_in` = `[z, 8kp+60k, 8kp+60k]` and `size_out` = `[z-32, 8kp-28k-4, 8kp-28k-4]`

Here, `p` and `z` can be any positive integers that make `size_out` has all positive values.

Here are some pre-calculated values for different models on different types of GPUs.

|                                       | size_in           | size_out          |  batch_size   |
| --------------------------------------|:-----------------:|:-----------------:|:-------------:|
| unet_xy on 12GB GPU                   |  [44, 140, 140]   | [16, 52, 52]      |       4       |
| unet_xy on 33GB GPU                   |  [50, 156, 156]   | [22, 68, 68]      |       8       |
| unet_xy_zoom (ratio=3) on 12GB GPU    |  [52, 372, 372]   | [20, 104, 104]    |       4       |
| unet_xy_zoom (ratio=3) on 33GB GPU    |  [52, 420, 420]   | [20, 152, 152]    |       8       |

5. model directory (:warning:)
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
`datafolder` (:warning:) and `PatchPerBuffer` (:pushpin:) need to be specified for each problem. `datafolder` is the directory of training data. `PatchPerBuffer` is the number of sample patches randomly drawn in each epoch, which can be set as *number of patches to draw from each data* **x** *number of training data*. `name`, `epoch_shuffle` and `NumWorkers` (:ok:) are fixed by default. `batch_size` is related to GPU memory and patch size (see values presented with patch size).

### Validation related parameter

In machine learning studies, we usually do a validation after every few epochs to make sure things are not going wrong. For most 3d microscopy image segmentation problems, the training data is very limited. We cannot save a big portion (e.g., 20%) from the training data for validation purpose. So, by default, we use leave-one-out for validation (:ok:) and may only need to adjust for advanced users. 