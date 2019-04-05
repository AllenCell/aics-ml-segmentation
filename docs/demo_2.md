# Demo 1: Segmentation of Lamin B1 in 3D fluorescent microscopy images of hiPS cells 

In this demo, we will show how we get the segmentation of Lamin B1 in 3D fluorescent microscopy images of hiPS cells. 

## Stage 1: Run **Segmenter** (a classic image segmentation workflow)

We refer [demo 1](./demo_1.md) for how to develop a classic image segmentation workflow. Suppose we already have work out a workflow for it and save it as `seg_lmnb1_interphase.py`.

## Stage 2: Visual evaluation 

We apply the workflow `lmnb1_interphase` on a set of images, we find some results are good and some have errors like the image below. 


[pic]


Some objects are missed in the segmentation due to the failure of an automatic seeding step. Also, this workflow has a poor performance on mitotic cells. In short, the segmentation on some images are good, but fails on others. So, we want to leverage the successful ones to build a DL model model.

## Stage 3: **Curator** (sorting)

