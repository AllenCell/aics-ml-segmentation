# Tutorial:

This is a tutorial on how to use the *Allen Cell Structure Segmenter* including both the classic image segmentation workflow and the iterative deep learning workflow.

Assume we have one or a set of 3D microscopy images tagging a specific intracellular structure. the goal of the *Allen Cell Structure Segmenter* is to generate a binary image of the structures segmentation as accurately as possible for each input image. Details, such as the underlying algorithms, validation, data, etc., can be found in our [bioRxiv paper](https://www.biorxiv.org/content/10.1101/491035v1). The tutorial will focus on how to run the *Allen Cell Structure Segmenter* (including both classic workflows and iterative DL workflows) to get an acuurate segmentation. The execution is based on three building blocks: **Binarizer**, **Curator** and **Trainer**. We will first explain each building blocks and demonstrate the actual execution in a few applications to generate accurate segmentations.

*Note: Our image reader used in our package supports images in common formats, such as `.tiff`, `.tif`, `.ome.tif`. The only vendor specific format our can package can handle is `.czi` (the file format for ZEISS microscope). For other format, images have to be converted to `.tiff` or `.ome.tif` in advance.* 

## Understanding each building blocks:

* **Binarizer**: [documentation](./bb1.md)
* **Curator**: [documentation](./bb2.md)
* **Trainer**: [documentation](./bb3.md)

## Demos on real examples:

![overview pic](./overview_pic.png)

Above picture shows a flowchart of the actual steps in practice. **Curator** and **Trainer** are used to improve the segentation from **Binarizer** when necessary. Here, we will have demonstration on two examples: first with one **Binarizer** solving the problem and another with more iterations needed.

### Case 1: Segmentation of ATP2A2 in 3D fluorescent microscopy images of hiPS cells 

![demo1 pic](./demo1_pic.png)

[Link to the demo documentation](./demo_1.md)

Link to the demo video

### Case 2: Segmentation of Lamin B1 in 3D fluorescent microscopy images of hiPS cells 

![demo2 pic](./demo2_pic.png)

[Link to the demo documentation](./demo_2.md)

Link to the demo video



