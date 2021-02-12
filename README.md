# Simple Fashion Classifier

## Summary

This repository is just a simple example of how to run an ONNX classification model trained with FashionMNIST using OpenCV

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