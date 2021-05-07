aicsmlsegment
.
|-- bin
    |--curator
        |-- curator_merging.py
        |-- curator_sorting.py
        |-- curator_takeall.py
    |-- predict.py
    |-- train.py 
|-- DataUtils
    |-- DataMod.py
    |-- Universal_Loader.py
|-- NetworkArchitecture
    |-- unet_xy_zoom.py
    |-- unet_xy_zoom_0pad.py
    |-- ...
|-- tests
    | -- ... (dummy unit test)
|-- custom_loss.py
|-- custom_metric.py
|-- fnet_prediction_torch.py
    (core inference function for new Segmenter models, which have same input and output size)
|-- model_utils.py
|-- Model.py
    (core function defining a PLT model class)
|-- multichannel_sliding_window.py
    (core inference function for old Segmenter models, which have different input and output size)
|-- training_utils.py
|-- utils.py
|-- version.py

