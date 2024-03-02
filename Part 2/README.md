# Walkthrough of MNIST digit Classifier

Note: This is a development of [s5-pytorch-tutorial](https://github.com/aakashvardhan/s5-pytorch-tutorial)

- [Motivation](#motivation)
- [Model Architecture](#model-architecture)
- [Data Loading and Transformation](#data-loading-and-transformation)
- [Train the Model](#train-the-model)
- [Evaluation](#evaluation)
- [Summary](#summary)


## Motivation

The motivation of this is to improve the model's architecture and performance by adding the following features:
- Batch Normalization (`nn.BatchNorm2d`)
- Dropout (`nn.Dropout`)
- Global Average Pooling (`nn.AdaptiveAvgPool2d`): this is used to replace the fully connected layer at the end of the model, reducing the number of parameters and improving generalization

The model's constraints are:
- Less than 20k parameters
- Less than 20 epochs

Target:
- Achieve at least 99.4% accuracy

## Model Architecture

This section outlines the architecture of a Convolutional Neural Network (CNN) model designed for digit classification, leveraging custom convolutional and pooling blocks to achieve efficient learning with less than 20k parameters. The model architecture is modular, consisting of reusable `ConvBlock` and `MaxPoolingBlock` components, allowing for clear structure and easy experimentation.

### Convolutional Blocks

The `ConvBlock` class is a fundamental building block of the model, encapsulating a convolutional layer followed by batch normalization and dropout. This design enhances the model's ability to generalize by reducing internal covariate shift (thanks to batch normalization) and preventing overfitting (via dropout). Each `ConvBlock` performs the following operations in sequence:
- 2D Convolution (`nn.Conv2d`)
- Batch Normalization (`nn.BatchNorm2d`)
- ReLU Activation (`F.relu`)
- Dropout (`nn.Dropout`)

### MaxPooling Block

The `MaxPoolingBlock` integrates a max pooling operation with a 1x1 convolution. This structure is utilized for downsampling the feature maps, thereby reducing the dimensions while allowing the model to retain essential features. The 1x1 convolution, in particular, helps in adjusting the number of feature maps, enabling flexible model depth configuration.

### `Net` Architecture

The `Net` class defines the overall CNN model, incorporating multiple `ConvBlock` and a `MaxPoolingBlock` for an effective yet compact architecture.

- **Initial Convolutional Layers**: The model begins with a series of `ConvBlock` layers, gradually building a rich feature representation of the input images. Each block increases the receptive field while preserving spatial dimensions, thanks to padding.
  
- **Transition with MaxPooling**: A `MaxPoolingBlock` follows, serving as a transition that reduces the spatial dimensions of the feature maps. The inclusion of a 1x1 convolution in this block further enables adjustment of feature map depth without losing critical information.
  
- **Further Convolutional Layers**: After downsampling, additional `ConvBlock` layers are applied. These layers continue to refine the features, preparing the model for classification. The dropout rate is slightly increased in these layers to further encourage generalization.
  
- **Output Block**: The model concludes with a global average pooling (GAP) layer, reducing each feature map to a single value. This reduction simplifies the model by eliminating the need for fully connected layers, thus reducing the total number of parameters. The output of the GAP layer is fed into a linear layer (`nn.Linear`), which maps the reduced feature representation to the class probabilities.

### Forward Pass

The `forward` method of the `Net` class defines the data flow through the model. Starting from the initial `ConvBlock` layers, through the max pooling transition, and concluding with the GAP and final classification layer, this method orchestrates how an input tensor is transformed into a prediction.

By employing a combination of convolutional layers, batch normalization, dropout, and pooling strategies, this model aims to efficiently learn from the MNIST dataset for digit classification with a minimal parameter count.

### Usage

```python
# Example instantiation of the model
model = Net(in_channels=1, n_channels=32)
```

## Data Loading and Transformation

Explained in [s5-pytorch-tutorial](https://github.com/aakashvardhan/s5-pytorch-tutorial#prepare-the-dataset)

## Train the Model

Explained in [s5-pytorch-tutorial](https://github.com/aakashvardhan/s5-pytorch-tutorial?tab=readme-ov-file#train-the-model)

## Evaluation

### Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             160
       BatchNorm2d-2           [-1, 16, 26, 26]              32
           Dropout-3           [-1, 16, 26, 26]               0
         ConvBlock-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 16, 24, 24]           2,320
       BatchNorm2d-6           [-1, 16, 24, 24]              32
           Dropout-7           [-1, 16, 24, 24]               0
         ConvBlock-8           [-1, 16, 24, 24]               0
            Conv2d-9           [-1, 16, 22, 22]           2,320
      BatchNorm2d-10           [-1, 16, 22, 22]              32
          Dropout-11           [-1, 16, 22, 22]               0
        ConvBlock-12           [-1, 16, 22, 22]               0
           Conv2d-13           [-1, 32, 20, 20]           4,640
      BatchNorm2d-14           [-1, 32, 20, 20]              64
          Dropout-15           [-1, 32, 20, 20]               0
        ConvBlock-16           [-1, 32, 20, 20]               0
           Conv2d-17           [-1, 16, 20, 20]             528
        MaxPool2d-18           [-1, 16, 10, 10]               0
  MaxPoolingBlock-19           [-1, 16, 10, 10]               0
           Conv2d-20             [-1, 16, 8, 8]           2,320
      BatchNorm2d-21             [-1, 16, 8, 8]              32
          Dropout-22             [-1, 16, 8, 8]               0
        ConvBlock-23             [-1, 16, 8, 8]               0
           Conv2d-24             [-1, 16, 6, 6]           2,320
      BatchNorm2d-25             [-1, 16, 6, 6]              32
          Dropout-26             [-1, 16, 6, 6]               0
        ConvBlock-27             [-1, 16, 6, 6]               0
           Conv2d-28             [-1, 32, 4, 4]           4,640
      BatchNorm2d-29             [-1, 32, 4, 4]              64
          Dropout-30             [-1, 32, 4, 4]               0
        ConvBlock-31             [-1, 32, 4, 4]               0
        AvgPool2d-32             [-1, 32, 1, 1]               0
           Linear-33                   [-1, 10]             330
================================================================
Total params: 19,866
Trainable params: 19,866
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.38
Params size (MB): 0.08
Estimated Total Size (MB): 1.46
----------------------------------------------------------------

```

The model consists of 19,866 parameters, well within the specified constraint of less than 20k parameters. The model's architecture is designed to efficiently learn from the MNIST dataset, leveraging convolutional layers, batch normalization, dropout, and global average pooling.

### Loss and Accuracy Plots

```
Adjusting learning rate of group 0 to 5.0000e-02.
Epoch 1
Train: Loss=0.0675 Batch_id=117 Accuracy=89.19: 100%|██████████| 118/118 [00:32<00:00,  3.58it/s]
Test set: Average loss: 0.0001, Accuracy: 9870/10000 (98.70%)
Adjusting learning rate of group 0 to 5.0000e-02.
Epoch 2
Train: Loss=0.1117 Batch_id=117 Accuracy=97.78: 100%|██████████| 118/118 [00:20<00:00,  5.89it/s]
Test set: Average loss: 0.0001, Accuracy: 9883/10000 (98.83%)
Adjusting learning rate of group 0 to 5.0000e-02.
Epoch 3
Train: Loss=0.0661 Batch_id=117 Accuracy=98.17: 100%|██████████| 118/118 [00:20<00:00,  5.71it/s]
Test set: Average loss: 0.0001, Accuracy: 9903/10000 (99.03%)
Adjusting learning rate of group 0 to 5.0000e-02.
Epoch 4
Train: Loss=0.0122 Batch_id=117 Accuracy=98.40: 100%|██████████| 118/118 [00:21<00:00,  5.47it/s]
Test set: Average loss: 0.0001, Accuracy: 9919/10000 (99.19%)
Adjusting learning rate of group 0 to 5.0000e-02.
Epoch 5
Train: Loss=0.0082 Batch_id=117 Accuracy=98.58: 100%|██████████| 118/118 [00:21<00:00,  5.55it/s]
Test set: Average loss: 0.0001, Accuracy: 9916/10000 (99.16%)

.......................
.......................
.......................
.......................

Epoch 15
Train: Loss=0.0088 Batch_id=117 Accuracy=99.26: 100%|██████████| 118/118 [00:22<00:00,  5.34it/s]
Test set: Average loss: 0.0000, Accuracy: 9955/10000 (99.55%)
Adjusting learning rate of group 0 to 5.0000e-03.
Epoch 16
Train: Loss=0.0227 Batch_id=117 Accuracy=99.22: 100%|██████████| 118/118 [00:20<00:00,  5.64it/s]
Test set: Average loss: 0.0000, Accuracy: 9955/10000 (99.55%)
Adjusting learning rate of group 0 to 5.0000e-03.
Epoch 17
Train: Loss=0.0294 Batch_id=117 Accuracy=99.30: 100%|██████████| 118/118 [00:20<00:00,  5.63it/s]
Test set: Average loss: 0.0000, Accuracy: 9956/10000 (99.56%)
Adjusting learning rate of group 0 to 5.0000e-03.
Epoch 18
Train: Loss=0.0284 Batch_id=117 Accuracy=99.29: 100%|██████████| 118/118 [00:21<00:00,  5.41it/s]
Test set: Average loss: 0.0000, Accuracy: 9959/10000 (99.59%)
Adjusting learning rate of group 0 to 5.0000e-03.
Epoch 19
Train: Loss=0.0075 Batch_id=117 Accuracy=99.28: 100%|██████████| 118/118 [00:22<00:00,  5.28it/s]
Test set: Average loss: 0.0000, Accuracy: 9958/10000 (99.58%)
Adjusting learning rate of group 0 to 5.0000e-03.

```

The model achieves a validation accuracy of 99.58% in just 19 epochs, demonstrating the effectiveness of the architecture in learning from the MNIST dataset, including less overfitting of the model. The loss and accuracy plots illustrate the model's learning progress, with the validation accuracy consistently improving over the training epochs.

![Loss and Accuracy Plots](https://github.com/aakashvardhan/s6-backprop/blob/main/Part%202/model-performance.png)

