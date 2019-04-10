# Demo 1: Segmentation of ATP2A2 in 3D fluorescent microscopy images of hiPS cells 

In this demo, we will show how we get the segmentation of ATP2A2 in 3D fluorescent microscopy images of hiPS cells. 

## Stage 1: Develop a classic image segmentation workflow

We recommend that users start by identifying a structure in the [lookup table](https://www.allencell.org/segmenter.html) that looks most similar to the segmentation task that you are faced with. Once you have identified this structure, open the correspoinding Jupyter Notebook and follow the instructions in the notebook to tune the workflow for your particular task. Finally, after finalizing the algorithms and parameters in the workflow, modify batch_processing.py to batch process all data (file by file or folder by folder).

#### Step 1: Find the entry in the lookup table with most similar morphology to your data

List of "playgrounds" accomoanying the lookup table:

1. playground_st6gal.ipynb: workflow for Sialyltransferase 1
2. playground_spotty.ipynb: workflow for Fibrillarin, Beta catenin
3. playground_npm1.ipynb: workflow for Nucleophosmin
4. playground_curvi.ipynb: workflows for Sec61 beta, Tom 20, Lamin B1 (mitosis-specific)
5. playground_lamp1.ipynb: workflow for LAMP-1
6. playground_dots.ipynb: workflows for Centrin-2, Desmoplakin, and PMP34
7. playground_gja1.ipynb: workflow for Connexin-43
8. playground_filament3d.ipynb: workflows for Tight junction protein ZO1, Beta actin, Non-muscle myosin IIB, Alpha-actinin-1, Alpha tubulin, Troponin I, and Titin
9. playground_shell.ipynb: workflow for Lamin B1 (Interphase-specific)

In this example, ATP2A2 tags the same structure as Sec61B and thus looks very similar to Sec61B. So, we pick `playground_curvi.ipynb`.

#### Step 2: Go to the Jupyter Notebook and tune the workflow

First, start your Jupyter Notebook App (make sure you change the conda environment name and path accordingly).

```bash
source activate segmentation
cd PATH/TO/aics-segmentation/lookup_table_demo
jupyter notebook
```

Now, your Jupyter should have started in your default brower and you can open `playground_curvi.ipynb`, then simply follow the instruct embedded in the notebook to tune the workflow for segmenting your images. ([how to use a jupyter notebook?](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html#executing-a-notebook))

#### Step 3: Batch run 

You can easily test your workflow on a folder of images following the steps below.

1. duplicate the template file in `/aics-segmentation/aicssegmentation/structure_wrapper/seg_template.py` as `/aics-segmentation/aicssegmentation/structure_wrapper/seg_atp2a2.py`
2. Open `seg_atp2a2.py`
3. Change the function name from `Workflow_template()` to `Workflow_atp2a2()` on line 12
4. insert you parameters and functions at the placeholders (searching `#ADD-HERE` in the code). Meanwhile, make sure you `import` all the functions you want to use. You can check the `seg_lamin_interphase.py` under structure_wrapper to see examples.
5. Save the file
6. run (make sure to use your own path and structure channel index)
```bash
batch_processing --workflow_name atp2a2 --struct_ch 1 --output_dir /path/to/output per_dir --input_dir /path/to/raw --data_type .czi
```
Or, you can also use the scripts (`aicssegmentation/bin/run_toolkit.sh` for linux/mac, `aicssegmentation/bin/run_toolkit.bat`)

## Stage 2: Evaluation

After batch running on several images, you can roughly know how well the current **Binarizer** works. In our case, we find it works pretty well. We are all done!

In some situations, after closely inspecting on several images, you may find that some parameters may need to be adjusted a little bit low or a little bit higher. Then, you can go back to your structure wrapper file, e.g., `seg_atp2a2.py`, to make the changes and re-run it. You may not have to go back to the jupyter notebook. The notebook is only meant to help you test out the overall workflow and a quick test to get reasonable parameters.



