# Demo 1: Segmentation of ATP2A2 in 3D fluorescent microscopy images of hiPS cells 

In this demo, we will demonstrate how to get the segmentation of ATP2A2 in 3D fluorescent microscopy images of hiPS cells. 

# Also available in [a video tutorial](https://youtu.be/Ynl_Yt9N8p4)

*Note: This demo only uses the classic segmentation workflow and thus does not require a GPU. See package [aics-segmentation](https://github.com/AllenInstitute/aics-segmentation).

## Stage 1: Develop a classic image segmentation workflow

We recommend users starting by identifying a structure in the [lookup table](https://www.allencell.org/segmenter.html) that looks the most similar to the segmentation task that you have. Once you have identified a structure, open the corresponding Jupyter Notebook and follow the instructions in the notebook to tune the workflow. After finalizing the algorithms and parameters in the workflow, modify batch_processing.py to batch process all images (file by file or folder by folder).

#### Step 1: Find the structure in the lookup table with the most similar morphology to your data

List of "playgrounds" for the lookup table:

1. playground_st6gal.ipynb: workflow for Sialyltransferase 1
2. playground_spotty.ipynb: workflow for Fibrillarin, Beta catenin
3. playground_npm1.ipynb: workflow for Nucleophosmin
4. playground_curvi.ipynb: workflows for Sec61 beta, Tom 20, Lamin B1 (mitosis-specific)
5. playground_lamp1.ipynb: workflow for LAMP-1
6. playground_dots.ipynb: workflows for Centrin-2, Desmoplakin, and PMP34
7. playground_gja1.ipynb: workflow for Connexin-43
8. playground_filament3d.ipynb: workflows for Tight junction protein ZO1, Beta actin, Non-muscle myosin IIB, Alpha-actinin-1, Alpha tubulin, Troponin I, and Titin
9. playground_shell.ipynb: workflow for Lamin B1 (Interphase-specific)

In this example, ATP2A2 localizes to the nuclear periphery and ER tubules, very similar to Sec61B. Therefore we are starting with `playground_curvi.ipynb`.

#### Step 2: Go to Jupyter Notebook and tune the workflow

First, start your Jupyter Notebook App (make sure to activate your conda environment, see package [aics-segmentation](https://github.com/AllenInstitute/aics-segmentation) for details).

```bash
jupyter notebook
```

Now, Jupyter Notebook should have opened in your default browser and you can make a copy of `playground_curvi.ipynb` to start working. Simply follow the instructions embedded in the notebook to tune the workflow for your image. ([how to use a Jupyter Notebook?](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html#executing-a-notebook))

#### Step 3: Batch run 

You can easily test your workflow on multiple images with batch processing following the steps below.

1. Duplicate the template file in `/aics-segmentation/aicssegmentation/structure_wrapper/seg_template.py` and change name to `/aics-segmentation/aicssegmentation/structure_wrapper/seg_atp2a2.py`
2. Open `seg_atp2a2.py`
3. Change the function name from `Workflow_template()` to `Workflow_atp2a2()`
4. insert parameters and functions at the placeholders. Meanwhile, make sure you `import` all the functions you want to use. You can check `seg_lamin_interphase.py` under structure_wrapper to see examples.
5. Save the file
6. Run (make sure to use your own path and structure channel index)

```bash
batch_processing --workflow_name atp2a2 --struct_ch 1 --output_dir /path/to/output per_dir --input_dir /path/to/raw --data_type .czi
```
Or, you can also use these scripts (`aicssegmentation/bin/run_toolkit.sh` for linux/mac, `aicssegmentation/bin/run_toolkit.bat`)

## Stage 2: Evaluation

The goal of the Jupyter Notebook "playground" is to design and assess the overall workflow on one or a couple of images. After applying on more images (ideally representing possible variations in the full dataset to be analyzed), we want to make sure the workflow works well on different examples. 

### Case 1:

If everything looks good, the script is ready for processing new data for analysis.

### Case 2: 

If the results are okay/aceptable on all images, but may need a little tuning (e.g., decreasing the cutoff value of `filament_3d_wrapper` to be more permissive), you can adjust the paramters in `seg_atp2a2.py`. You may need several rounds of finetuning of the parameters in batch mode to finally achieve the most satisfactory results on all representative images. You may not have to go back to the Jupyter Notebook file since the notebook is only meant to help you quickly test out the overall workflow and get reasonable parameters.

### Case 3: 

If the results are good on some images, but bad on others, or the results are only good on certain cells, you may consider use the iterative deep learning workflow to improve the segmentation quality. See [demo 2](./demo_2.md) for details.


