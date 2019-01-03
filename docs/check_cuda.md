# How to setup CUDA and check the right CUDA version?

What is [CUDA](https://developer.nvidia.com/cuda-toolkit)? It is a package for utilizing NVDIA GPU for high performance computing.  

First of all, you need an NVIDIA GPU card properly installed on your machine. Suppose the GPU card has been in place and the driver has been installed. 

1. check your GPU driver version

Running command  ` nvidia-smi` in your terminal will give an overview of your GPU cards, for example

![nvidia_smi](./nvidia_smi.png)

So, we have Driver Version: 390.87

2. Determine which CUDA version fits your GPU

Find your drive version in this [chart](https://stackoverflow.com/questions/30820513/what-is-the-correct-version-of-cuda-for-my-nvidia-driver/30820690#30820690). For our case, CUDA 9.0 or higher is good for us. 

Go to [CUDA website](https://developer.nvidia.com/cuda-toolkit) and follow the download and installation instruction to install CUDA

