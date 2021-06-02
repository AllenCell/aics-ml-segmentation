# Configuration for Training a DL Model in **Trainer**

This is a detailed description of the configuration for training a DL model in **Trainer**. There are a lot of parameters in the configuration file, which can be categorized into three types:

1. Need to change on every run (i.e., parameters specific to each execution), marked by :warning:
    * `checkpoint_dir` (where to save trained models), `datafolder` (where are training data), `resume` (whether to start from a previous model)
2. Need to change for every segmentation problem (i.e., parameters specific to one problem), marked by :pushpin:
    * `model`, `epochs`, `save_every_n_epoch`, ``PatchPerBuffer``
3. Only need to change once on a particular machine (parameters specific to each machine), marked by :computer:
4. No need to change for most problems (parameters pre-defined as a general training scheme and requires advacned deep learning knowledge to adjust), marked by :ok:

********************************************************************
Here is a list of example yaml configuration files:
- [basic](../configs/train_config.yaml)
- [unet_xy_zoom_0pad using tensorboard]()
- [TBA]()
- [TBA]()
- [TBA]()
- [TBA]()
********************************************************************


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
There may be probably more than 100 models in the literature for 3D image segmentation. The two models we implemented here are carefully designed for cell structure segmentation in 3D microscopy images. Model `unet_xy` is suitable for smaller-scale structures, like severl voxels thick (e.g., tubulin, lamin b1). Model `unet_xy_zoom` is more suitable for larger-scale structures, like more than 100 voxels in diameter (e.g., nucleus), while the `zoom_ratio` is an integer (e.g., 2 or 3). Larger `zoom_ratio` allows the model to take more neighbor information into account, but reduce the resolution. 

There are a few more variations of `unet_xy_zoom`: `unet_xy_zoom_0pad` (add 0 padding after all convolutions, so `size_in = size_out`), `unet_xy_zoom_dilated` (use dilated convolution), `unet_xy_zoom_stridedconv` (replace max pooling with strided convolutions), `unet_xy_zoom_0pad_stridedconv` (both 0 padding and strided convolution), etc. 

Now, we can also support most of the baseline models implemented in [MONAI](https://docs.monai.io/en/latest/networks.html#nets). All the parameters in those models can be passed in here.


2. input and output type (:ok:)
```yaml 
  nchannel: 1
  nclass: [2, 2, 2]
```
These are related to the model architecture and fixed by default. We assume the input image has only one channel. For `nclass`, it could be `[2, 2, 2]` (for `unet_xy`, `unet_xy_zoom`) or `2` (for other monai models). You can also use more classes than 2, which will also need to have corresponding values in the ground truth images.

3. patch size (:computer:) (:pushpin:)

```yaml 
  size_in: [50, 156, 156] 
  size_out: [22, 68, 68]
```
In most situations, we cannot fit the entire image into the memory of a single GPU. These are also related to `batch_size` (a data loader parameter), which will be discussed shortly. `size_in` is the actual size of each patch fed into the model, while `size_out` is the size of the model's prediction. The prediction size is smaller than the input size is because the multiple convolution operations. The equation for calculating `size_in` and `size_out` is as follows.

> For unet_xy, `size_in` = `[z, 8p+60, 8p+60]`, `size_out` = `[z-28, 8p-28, 8p-28]`

> For unet_xy_zoom, with `zoom_ratio`=`k`, `size_in` = `[z, 8kp+60k, 8kp+60k]` and `size_out` = `[z-32, 8kp-28k-4, 8kp-28k-4]`

> For unet_xy_zoom_0pad with `zoom_ratio`=`k`, `size_in` = `size_out` and `x` and `y` must be divisible by `8k`.

Here, `p` and `z` can be any positive integers that make `size_out` has all positive values.

Here are some pre-calculated values for different models on different types of GPUs.

|                                       | size_in           | size_out          |  batch_size   |
| --------------------------------------|:-----------------:|:-----------------:|:-------------:|
| unet_xy on 12GB GPU                   |  [44, 140, 140]   | [16, 52, 52]      |       4       |
| unet_xy on 33GB GPU                   |  [50, 156, 156]   | [22, 68, 68]      |       8       |
| unet_xy_zoom (ratio=3) on 12GB GPU    |  [52, 372, 372]   | [20, 104, 104]    |       4       |
| unet_xy_zoom (ratio=3) on 33GB GPU    |  [52, 420, 420]   | [20, 152, 152]    |       8       |

4. start from an existing model? (:warning:)

```yaml
resume: null
```

When doing iterative deep learning, it may be useful to start from the model trained in the previous step. The model can be specified at `resume`. If `null`, a new model will be trained from scratch. 


5. model directory (:warning:)
```yaml
checkpoint_dir:  /home/model/xyz/
resume: null
```
This is the directory to save the trained model. If you want to start this training from a previous saved model, you may add the path to `resume`.

6. Precision
```yaml
precision: 16 
```
Optional. If not specified, defaults to 32-bit model weights. 16-bit model weights can be specified as shown to decrease model memory usage and increase batch size with the possible tradeoff of reduced model performance. 

### Training scheme related parameters
1. optimization parameters (:ok:)
```yaml
learning_rate: 0.00001
weight_decay: 0.005
loss:
  name: Aux
  loss_weight: [1, 1, 1]
  ignore_index: null
```

2. Learning Rate Scheduler
```yaml
scheduler:
  name: ExponentialLR 
  gamma: 0.85
  verbose: True
```
Supported schedulers and associated parameters are as follows:

|    Scheduler Name                |          Parameters                   |
|----------------------------------|:-------------------------------------:|
|ExponentialLR                     | gamma, verbose                        |
|CosineAnnealingLR                 | T_max, verbose                        |
|StepLR                            | step_size, gamma, verbose             |
|ReduceLROnPlateau                 | mode, factor, patience, verbose       |
|1cycle                            |max_lr, total_steps, pct_start, verbose|

Explanation of these parameters can be found in the [Pytorch documentation](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate). Specifying a scheduler is optional.

3. Stochastic Weight Averaging
```yaml
SWA: 
  swa_start: 1 
  swa_lr: 0.001
  annealing_epochs: 3 
  annealing_strategy: cos 

```

Stochastic Weight Averaging  combines high learning rate and weight averaging across epochs to help model generalization by taking advantage of the shape of the loss minima found by stochastic gradient descent. A more in-depth explanation can be found in the [Pytorch Documentation] (https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/). If used, all parameters must be specified, else this option can be omitted from the config file or specified as `SWA: `.

|    Parameter                |          Options                   | Explanation |
|-----------------------------|:----------------------------------:|-------------|
|swa_start                    | positive integer or between 0 and 1|Epoch number or percentage of total epochs to start SWA|
|swa_lr                       | float > 0                          |Relatively high learning rate to anneal to|
|annealing_epochs             | integer > 0                        |Number of epochs to attain swa_lr|
|annealing_strategy           | `cos` or `linear`                  |Whether to perform linear or cosine ramp to swa_lr|

4. training epochs (:pushpin:)
```yaml
epochs: 400
save_every_n_epoch: 40
```
`epochs` controls how many iterations in the training. We suggest to use values between 300 and 600. A model will be saved on every `save_every_n_epoch` epochs.

5. Early Stopping
```yaml
callbacks:
  name: EarlyStopping
  monitor: val_loss 
  min_delta: 0.01 
  patience: 10
  verbose: True 
  mode: min
```
Early stopping can avoid overfitting on the training set by halting training when validation performance stops improving. `monitor` specifies which metric (either `val_loss` or `val_iou`) the decision to stop should be based on. Changes in `monitor` less than `min_delta` will not count as improvement. `patience` specifies how many epochs to wait for an improvement in a metric before stopping. `verbose` controls whether to print updates to the commandline. `mode` specifies whether the monitored value should be minimized or maximized. Early stopping does not have to be specified in the config file. 

6. GPU configuration
```yaml
gpus: -1 
dist_backend: ddp 
```
Multi-GPU training using ddp is supported. If `gpus` = -1, all available gpus will be used. Otherwise, `gpus` must be a positive integer and up to that many gpus will be used depending on gpu availability. If multiple gpus are used, a `backend` can be specified. Currently, ddp is the only supported backend. 

7. Tensorboard
```yaml
tensorboard: "path/to/logdir" 
```
Tensorboard can be used to track the progression of training and compare experiments. The `tensorboard` argument is a path to a directory containing tensorboard logs.



### Data related parameters 

```yaml
loader:
  name: default
  datafolder: '/home/data/train/'
  batch_size: 8
  PatchPerBuffer: 200
  epoch_shuffle: 5
  NumWorkers: 1
  Transforms: ['RR']
```
`datafolder` (:warning:) and `PatchPerBuffer` (:pushpin:) need to be specified for each problem. `datafolder` is the directory of training data. `PatchPerBuffer` is the number of sample patches randomly drawn in each epoch, which can be set as *number of patches to draw from each data* **x** *number of training data*. `name`, `epoch_shuffle` and `NumWorkers` (:ok:) are fixed by default. `batch_size` is related to GPU memory and patch size (see values presented with patch size).
`Transforms` is a list of random transformations to apply to the training data. 

|    Transform Name    |          Description                       |
|----------------------|:------------------------------------------:|
|`RR`                  | Random rotation from 1-180 degrees         |
|`RF`                  | 50% probability of left/right fliping image|
|`RN`                  | Addition of 0-mean Gaussian random noise  with standard deviation ~U(0, 0.25)|
|`RI`                  | Random shift in intensity of up to 0.1 with probability 0.2       |
|`RBF`                 |Application of [Random Bias Field] (https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomBiasField)|


### Validation related parameter

In machine learning studies, we usually do a validation after every few epochs to make sure things are not going wrong. For most 3d microscopy image segmentation problems, the training data is very limited. We cannot save a big portion (e.g., 20%) from the training data for validation purpose. So, by default, we use leave-one-out for validation (:ok:) and may only need to adjust for advanced users. 