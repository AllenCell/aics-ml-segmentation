# Instruction for using the iterative ML workflow of *Allen Cell Structure Segmenter*

In general, the iterative ML workflow is designed to improve the segmentation from preliminary results by supervised deep learning. The motivation is that we need ground truth segmentation to train the deep learning model, but the ground truth is difficult to collect. Manually painting on the 3D images as ground truth is not only very time-consuming, but also intrisically difficult (you may be able to draw a line in 3D, but not a spider web). So, we provide two strategies to semi-automatically generate high-quality training data from reasonable preliminary segmentations with very little human effort. Technically, if you already have some ground truth images, you can still easily use this repository for building your deep learning models for 3D image segmentation.

Main steps of the iterative ML workflow:

1. Get preliminary segmentation: This may come from classic image processing workflows of *Allen Cell Structure Segmenter* (See XXXX for more details about the classic workflows), another deep learning model, or any other algorithms. 
2. Prepare training data from preliminary results.
3. Train the model
4. Apply the model on more images. If more improvement is still needed, go back to step 1. Otherwise, the segmentation results are ready for your analysis.


Here, we demonstrate this workflow with an example of Lamin B1 segmentation. Similar procedure can be applied on your own data.

### Step 1: Get preliminary segmentation

Common questions:

1. How many ground truth images do I need?