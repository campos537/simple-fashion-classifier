# Simple Fashion Classifier

## Summary

This repository is just a simple example of how to run an ONNX classification model trained with FashionMNIST using OpenCV

## Model 

The model was trained with LeNet5 in order to be fast and also have great accuracy due to the low resolution of the training dataset

Download the [model](https://drive.google.com/file/d/1EdIyDQIeioFH_cJ2b25kD5R_oDg7X7Fi/view?usp=sharing)
Download MO version for InferenceEngine and ForwardAsync [model](https://drive.google.com/file/d/18TO7oapcS1H4jIe96mWCJ-8VQNCU4-Ad/view?usp=sharing)

## Usage

To use this repository please follow the steps below:

```
cd simple-fashion-classifier
python3.7 -m pip install -r requirements
python3.7 main.py path/to/model/file path/to/image_or_image_folder
```

## Extra

To convert the FashionMnist csv files into images create a folder called data and use:

```
python3.7 scripts/fashion_mnist_from_csv.py
```
