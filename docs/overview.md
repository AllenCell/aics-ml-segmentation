# Tutorial:

This is a tutorial on how to use the *Allen Cell Structure Segmenter* including both the classic image segmentation workflow and the iterative deep learning workflow. Please refer

Assume we have one or a set of 3D microscopy images tagging a specific intracellular structure. the goal of the *Allen Cell Structure Segmenter* is to generate a binary image of the structures segmentation as accurately as possible for each input image. Details, such as the underlying algorithms, validation, data, etc., can be found in our [bioRxiv paper](https://www.biorxiv.org/content/10.1101/491035v1). The tutorial will focus on how to run the *Allen Cell Structure Segmenter* (including both classic workflows and iterative DL workflows) to get an acuurate segmentation. The execution is based on three building blocks: **Segmenter**, **Curator** and **Trainer**. We will first explain each building blocks and demonstrate the actual execution in a few applications to generate accurate segmentations.

## Understanding each building blocks:

* **Segmenter**: [documentation](./bb1.md)
* **Curator**: [documentation](./bb2.md)
* **Trainer**: [documentation](./bb3.md)

## Demos on real examples:

### Case 1: Segmentation of ATP2A2 in 3D fluorescent microscopy images of hiPS cells 

(pic here)

[Link to the demo documentation](./demo_1.md)
Link to the demo video

### Case 2: Segmentation of Lamin B1 in 3D fluorescent microscopy images of hiPS cells 

(pic here)

[Link to the demo documentation](./demo_2.md)
Link to the demo video



