# Building Block 3: **Trainer**

**Trainer** is used to train deep learning-based segmentation models. The input for **Trainer** should be data prepared by **Curator** (see [documentation](./bb2.md)) and the output should be a model that can be used in **Segmenter**.

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
