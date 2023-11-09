# 3D Landmark Detection
An extention of the 2D networks to 3D networks based on paper [Influence of growth structures and fixed appliances on automated cephalometric landmark recognition with a customized convolutional neural network](https://bmcoralhealth.biomedcentral.com/articles/10.1186/s12903-023-02984-2)

## Installation
create your own virtual environment and install the requirements with

`pip install -r requirements.txt`


## Structure
The structure of the project is as follows:

```
├── config: configs in yaml format
│   ├── 2d.yaml: 2d models
│   ├── 3d.yaml: 3d models, support backbone in both 2d and 3d types
|
├── data: some real data from the Open Data LMU platform
    ├── ...

├── dummydata: the generated 2D and 3D image data and the corresponding labels for each point
    ├── images
        ├── 2d
        ├── 3d
    
    ├── S_Point
        ├── 2d
            ├── train.csv
            ├── val.csv
            ├── test.csv

        ├── 3d
            ├── train.csv
            ├── val.csv
            ├── test.csv
|
├── main.py: the main file to run the training and testing
├── ...

```

## Models
The project support data in both 2D and 3D dimensions. For 3D data, we can treat the slice dimension as either channels (thus use normal 2D CNN layers as backbone) or the third dimension (thus use 3D CNN layers are backbone). Particularly:

#### For 2D data
- custom: the same model as proposed in the paper. Structure and number of parameters are the same as the original model.
- resnet18: the well-established resnet18 model. The number of parameters is similar to the custom model.
- for more details please refer to `cnn2d.py`

#### For 3D data
- custom 2d backbone: use the same 2D model as proposed in the paper. Only modify the input layer to adjust channels
- resnet18 2d backbone: use the same 2D model as resnet18. Only modify the input layer to adjust channels
- custom 3d backbone: modify the 2D CNNs in the custom model to 3D CNNs. Structures are maintained
- resnet18 3d backbone: modify the 2D CNNs in the resnet18 model to 3D CNNs. Structures are maintained

## How to use
First create your own dummy data

```python create_dummy_data.py```

Adjust the config in `config` folder, also modify the corrsponding config_name in `main.py`.


Then run the training and testing. 

```python main.py hydra.job.chdir=False```

Model checkpoints are saved at `checkpoints` folder. Prediction results are saved at `predictions` folder. 
