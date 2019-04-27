## Overview

The Allen Cell Structure Segmenter is a Python-based open source toolkit developed for 3D segmentation of intracellular structures in fluorescence microscope images, developed at the Allen Institute for Cell Science. This toolkit consists of two complementary elements, a classic image segmentation workflow with a restricted set of algorithms and parameters and an iterative deep learning segmentation workflow. We created a collection of 20 classic image segmentation workflows based on 20 distinct and representative intracellular structure localization patterns as a lookup table reference and starting point for users. The iterative deep learning workflow can take over when the classic segmentation workflow is insufficient. Two straightforward human-in-the-loop curation strategies convert a set of classic image segmentation workflow results into a set of 3D ground truth images for iterative model training without the need for manual painting in 3D. The Allen Cell Structure Segmenter thus leverages state of the art computer vision algorithms in an accessible way to facilitate their application by the experimental biology researcher. More details including algorithms, validations, and examples can be found in our [bioRxiv paper](https://www.biorxiv.org/content/10.1101/491035v1) or [allencell.org/segmenter](allencell.org/segmenter).

**Note: This repository is only for the "Iterative Deep Learning Workflow". The classic part can be found at [https://github.com/AllenInstitute/aics-segmentation](https://github.com/AllenInstitute/aics-segmentation)**

## Installation:

0. prerequisite:

To use the deep learning part of this package, we assume an [NVIDIA GPU](https://www.nvidia.com/en-us/deep-learning-ai/developer/) has been set up properly on a machine running a Linux operating system (we recommend Ubuntu).  

Also, make sure to check if your GPU supports at least CUDA 8.0 (CUDA 9.0 and up is preferred): [NVIDIA Driver check](https://www.nvidia.com/Download/index.aspx?lang=en-us).

The GPUs we used to develop and test our package are two types: (1) GeForce GTX 1080 Ti GPU (about 11GB GPU memory), (2) Tesla V100 for PCIe (with about 33GB memory), two common chips for personal workstation and data center, respectively.

> We are investigating alternative cloud computing service to deploy our package and will have updates in the next few months. Stay tuned :)  


1. create a conda environment: 

```bash
conda create --name mlsegmenter python=3.6
```

2. activate your environment and do the installation within the environment:

```bash 
conda activate mlsegmenter 
```

(Note: always check out [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment) for updates. If you are using an older version of conda, you may need to activate the environment by `source activate mlsegmenter`.)

3. Install Pytorch

Go to [PyTorch website](https://pytorch.org/get-started/locally/), and find the right installation command for you. 

* we use version 1.0 (which is the stable version at the time of our development)
* we use Linux (OS), Conda (package), python 3.6 (Language), CUDA=9.0 (Question about CUDA? see [setup CUDA](./docs/check_cuda.md)). So, the installation command for us is

```bash
conda install pytorch torchvision -c pytorch
```

4. Install Allen Cell Segmenter (deep learning part)

```bash
git clone https://github.com/AllenInstitute/aics-ml-segmentation.git
cd ./aics-ml-segmentation
pip install -e .
```

## Level of Support
We are offering it to the community AS IS; we have used the toolkit within our organization. We are not able to provide guarantees of support. However, we welcome feedback and submission of issues. Users are encouraged to sign up on our [Allen Cell Discussion Forum](https://forum.allencell.org/) for community quesitons and comments.


# Link to [Documentations and Tutorials](./docs/overview.md)