### Installation 

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
git clone https://aicsbitbucket.corp.alleninstitute.org/scm/assay/aics-ml-segmentation.git
cd ./aics-ml-segmentation
pip install -e .
```

External resource:

[Python Development in Visual Studio Code](https://realpython.com/python-development-visual-studio-code/)
[Which GPU to buy? A guide in 2018](https://blog.slavv.com/picking-a-gpu-for-deep-learning-3d4795c273b9)