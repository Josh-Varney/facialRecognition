# Facial Recognition System

This repository contains a facial recognition system implemented using TensorFlow, focusing on training and evaluating models like ResNet50 and MobileNetV2.

## Dataset

### Overview
The dataset consists of HDF5 files containing RGB images and binary classification labels. The files are stored in LFS with details about their sizes and locations.

1. **train_data.h5**: HDF5 file format containing denoised mixture of WIDERFACE and MAFA train dataset, with 15,000 RGB images of dimensions (244, 244).
2. **train_labels.h5**: HDF5 file format containing binary classification data labels corresponding to the images in train_data.h5, with 15,000 labels.
3. **normalised_train**: HDF5 format containing the normalised 5000 images used to train the images accordingly, This also contains the equivalent data labels for the normalised images.

## Notebooks

### Contents

1. **preprocess.ipynb**: Demonstrates data preprocessing and noise reduction for downloaded datasets, along with combining and randomizing train and test datasets.
2. **extra_preprocess.ipynb**: Validates the correctness of preprocessed MAFA and WIDERFACE Datasets, including visualizations.
3. **resnet50.ipynb**: Illustrates the training and testing process of the ResNet50 TensorFlow model.
4. **mobile_net_v2.ipynb**: Describes the training and testing process of the MobileNetV2 TensorFlow model.
5. **masked_model.ipynb**: Describes the training and testing process of the Custom Made TensorFlow model combined with KNNClassifier.
6. **all .py files**: Demonstrates the functional programming for all elements used in the notebook. Furthermore, this covers code documentation aspects of the project.

## Configurations

### Software

- **Python Version**: 3.9.13
- **CUDA Toolkit**: 11.2
- **cuDNN**: 11.2
- **Packages**: TensorFlow v2, Matplotlib, Scipy, Numpy, Pandas, Scikit-learn, H5py, PIL, OpenCV

### System Specifications

- **Operating System**: Windows 11
- **Processor**: AMD Ryzen 5 3600 6-Core Processor 3.60 GHz
- **RAM**: 16.0 GB
- **Architecture**: 64-bit (x64)

### Utilisation 

- **Use of h5py files**: run `extra_preprocess.ipynb` to access the preprocessed data as a result from `preprocess.ipynb` this will normalize the data for usage within the models.
- **Ensure specification match or run on Colab**: Ensure System Specifications are met.  

## Models

### MobileNetV2-based Classification Model (`mobilenetV2.ipynb`)

This script defines a function to create a custom classification model based on the MobileNetV2 architecture for binary classification tasks.

#### Model Architecture

- Base Model: MobileNetV2 pretrained on ImageNet.
- Global Average Pooling: Reduces spatial dimensions of feature maps.
- Batch Normalization: Normalizes activations.
- Dense Layers with Regularization and Dropout: ReLU activation, L2 regularization, and dropout layers.
- Output Layer: Dense layer with sigmoid activation for binary classification.

### VGG19 Transfer Learning Model with Dense Layers (`VGG19.ipynb`)

This repository contains a deep learning model implemented using TensorFlow and Keras based on the VGG19 architecture pretrained on the ImageNet dataset. It is adapted for binary classification tasks.

#### Model Overview

The model consists of two main parts:

1. **Feature Extraction Base**: Utilizes the VGG19 convolutional base pretrained on ImageNet.
2. **Custom Dense Layers**: Adds custom dense layers on top of the VGG19 base for classification.

### CNN Network Model (`masked_model.ipynb`)

This repository contains code for constructing a custom convolutional neural network (CNN) model using TensorFlow and Keras for image classification tasks.

#### Model Architecture

- Input Layer: Specifies input shape of images.
- Rescaling Layer: Normalizes pixel values to the range [0, 1].
- Convolutional Layers: Stack of convolutional layers with increasing filters and downsampling.
- Batch Normalization: Normalizes activations.
- Activation (ReLU): Applies ReLU activation function.
- Max Pooling: Performs downsampling.
- Global Average Pooling: Aggregates spatial information.
- Dropout: Introduces regularization.
- Output Layer: Dense layer with sigmoid activation for binary classification.

1. **Intermediate Layer Extraction**: This implements the use of KNNClassification to utilize the custom model as hybridized.

### Code Documentation

- See all original python files for code documentation.
- Else, then markdown will be provided within all notebook `.ipynb files`

## References

- MobileNetV2 Paper: [Sandler et al., 2018](https://arxiv.org/abs/1801.04381)
- VGG19 Paper: [Simonyan & Zisserman, 2014](https://arxiv.org/abs/1409.1556)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf)
- [Keras Documentation](https://keras.io)

 
 
