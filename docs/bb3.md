# Building Block 3: **Trainer**

The **Trainer** is used to train a deep learning based segmentation model. The input is the trained data prepared by **Curator** (see [documentation](./bb2.md)) and the output will be a model can be used in **Segmenter**.

![segmenter pic](./bb3_pic.png)

Copy and paster the configuration file at './config/train.yaml' into a new one, for example '/home/config_files/train_lab.yaml' and check the parameters and make changes as needed. In general, you only need to change (1) choose the model from two options, (2)the path to data, (3) the output path for saveing the model. A detailed explaination of all parameters can be found [here](./doc_train_yaml.md). 

```bash
dl_train --config /home/config_files/train_lab.yaml
```

### When multiple GPUs are available

By default, the **Trainer** takes the first available GPU to run the computation. If there are multiple GPUs on your machine, you can choose which GPU to use. Simply set `CUDA_VISIBLE_DEVICES` before the command, like 

```bash
CUDA_VISIBLE_DEVICES=2  dl_train --config /home/config_files/train_lab.yaml
```