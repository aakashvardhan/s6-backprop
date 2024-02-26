# How Backpropagation Works 

## Introduction

Backpropagation is a method used in artificial neural networks to calculate the error contribution of each neuron after a batch of data. It is a supervised learning algorithm, and is used to train the weights of a neural network. It is based on the chain rule of calculus, which allows us to calculate the gradient of a composite function.

## How it Works

The backpropagation algorithm works by computing the gradient of the loss function with respect to the weights of the network. To do this, we need the derivative of the loss function with respect to the output of the network, and the derivative of the output of the network with respect to the weights. Using the chain rule, we can compute the derivative of the loss function with respect to the weights.


## The Backpropagation Algorithm

The backpropagation algorithm consists of two main steps. In the forward pass, the input is passed through the network, and the output of the network is computed. In the backward pass, the gradient of the loss function with respect to the output of the network is computed, and then the gradient of the loss function with respect to the weights of the network is computed using the chain rule. The weights of the network are then updated using gradient descent.

## Tutorial

To understand backpropagation better, let's work through an example. We'll use a simple neural network with one input layers, two hidden layer, and one output layers. The input layer has two neurons, the hidden layer has two neurons, and the output layer has two neurons. We'll use the sigmoid activation function for the hidden layer, and the identity activation function for the output layer. We'll use mean squared error as the loss function.

<p align="center">
  <img src="https://github.com/aakashvardhan/s6-backprop/blob/main/Part%201/simple-nn.png" width="60%" />
</p>

- $i_{1}$ and $i_{2}$ are the inputs to the network, which are connected to hidden layer neurons $h_{1}$ and $h_{2}$
- $h_{1}$ neuron is connected to both inputs by weights $w_{1}$ and $w_{2}$
- $h_{2}$ neuron is connected to both inputs by weights $w_{3}$ and $w_{4}$
- $h_{1}$ and $h_{2}$ are connected to output layer neurons $o_{1}$ and $o_{2}$ by weights $w_{5}$, $w_{6}$, $w_{7}$, and $w_{8}$
- Activation function $a_{h_{1}}$ and $a_{h_{2}}$ is applied to the output of the hidden layer neurons
- Activation function $a_{o_{1}}$ and $a_{o_{2}}$ is applied to the output of the output layer neurons
- Error is calculated using mean squared error loss function $E_{total} = E_{1} + E_{2} = \frac{1}{2}(target_{1} - output_{1})^{2} + \frac{1}{2}(target_{2} - output_{2})^{2}$
- $t_{1}$ and $t_{2}$ are the target values for the output layer neurons

To make the explanation of backpropagation and its mathematical expressions clearer and easier to recall, let's break it down into more detailed steps and organize the information more coherently:

### Forward Pass
1. **Input to Hidden Layer Calculations**:
   - Calculate hidden layer inputs: 
     - \(h_1 = w_1 \cdot i_1 + w_2 \cdot i_2\)
     - \(h_2 = w_3 \cdot i_1 + w_4 \cdot i_2\)
   - Apply the activation function (σ, a sigmoid function in this case) to get the hidden layer activations:
     - \(a\_h_1 = \sigma(h_1) = \frac{1}{1 + \exp(-h_1)}\)
     - \(a\_h_2 = \sigma(h_2)\)

2. **Hidden Layer to Output Layer Calculations**:
   - Calculate output layer inputs:
     - \(o_1 = w_5 \cdot a\_h_1 + w_6 \cdot a\_h_2\)
     - \(o_2 = w_7 \cdot a\_h_1 + w_8 \cdot a\_h_2\)
   - Apply the activation function to get the output activations:
     - \(a\_o_1 = \sigma(o_1)\)
     - \(a\_o_2 = \sigma(o_2)\)

3. **Error Calculation**:
   - Compute the total error for both output neurons:
     - \(E_{total} = E_1 + E_2\)
     - \(E_1 = \frac{1}{2} \cdot (t_1 - a\_o_1)^2\)
     - \(E_2 = \frac{1}{2} \cdot (t_2 - a\_o_2)^2\)

### Backward Pass (Backpropagation)
The goal is to compute how much each weight contributes to the error and adjust accordingly.

1. **Gradient Calculation for Output Layer Weights (\(w_5, w_6, w_7, w_8\))**:
   - The derivative of the error with respect to each weight is calculated using the chain rule. For example, for \(w_5\):
     - \(\frac{\partial E_{total}}{\partial w_5} = (a\_o_1 - t_1) \cdot a\_o_1 \cdot (1 - a\_o_1) \cdot a\_h_1\)
   - Similarly, calculate gradients for \(w_6, w_7\), and \(w_8\).

2. **Gradient Calculation for Input Layer Weights (\(w_1, w_2, w_3, w_4\))**:
   - First, compute the gradient of the error with respect to the activations of the hidden layer. Then, apply the chain rule to find the gradient with respect to each weight. For \(w_1\):
     - \(\frac{\partial E_{total}}{\partial w_1} = ((a\_o_1 - t_1) \cdot a\_o_1 \cdot (1 - a\_o_1) \cdot w_5 + (a\_o_2 - t_2) \cdot a\_o_2 \cdot (1 - a\_o_2) \cdot w_7) \cdot a\_h_1 \cdot (1 - a\_h_1) \cdot i_1\)
   - Repeat this process for \(w_2, w_3\), and \(w_4\) using their respective paths through the network.

### Important Notes
- **Activation Function Derivative**: The derivative of the sigmoid function, \(\sigma(x)\), is \(\sigma(x) \cdot (1 - \sigma(x))\), crucial for calculating gradients.
- **Chain Rule Application**: The gradients of the weights are computed by applying the chain rule of derivatives, taking into account the path of each weight's influence on the total error.




## Graph Results (Error vs Epochs)

Shown below are the effects of learning rate on the convergence of the backpropagation algorithm. The learning rate is a hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated. A smaller learning rate requires more training epochs given the smaller changes made to the weights each update, and a larger learning rate may cause the model to converge too quickly, which may result in the model overshooting the optimal weights.

<table>
  <tr>
    <td><img src="https://github.com/aakashvardhan/s6-backprop/blob/main/Part%201/error_graph_lr/0.1.png" alt="Plot 1" style="width: 100%;"/></td>
    <td><img src="https://github.com/aakashvardhan/s6-backprop/blob/main/Part%201/error_graph_lr/0.2.png" alt="Plot 2" style="width: 100%;"/></td>
    <td><img src="https://github.com/aakashvardhan/s6-backprop/blob/main/Part%201/error_graph_lr/0.5.png" alt="Plot 3" style="width: 100%;"/></td>
  </tr>
  <tr>
    <td align="center">LR = 0.1</td>
    <td align="center">LR = 0.2</td>
    <td align="center">LR = 0.5</td>
  </tr>
  <tr>
    <td><img src="https://github.com/aakashvardhan/s6-backprop/blob/main/Part%201/error_graph_lr/0.8.png" alt="Plot 4" style="width: 100%;"/></td>
    <td><img src="https://github.com/aakashvardhan/s6-backprop/blob/main/Part%201/error_graph_lr/1.0.png" alt="Plot 5" style="width: 100%;"/></td>
    <td><img src="https://github.com/aakashvardhan/s6-backprop/blob/main/Part%201/error_graph_lr/2.0.png" alt="Plot 6" style="width: 100%;"/></td>
  </tr>
  <tr>
    <td align="center">LR = 0.8</td>
    <td align="center">LR = 1.0</td>
    <td align="center">LR = 2.0</td>
  </tr>
</table>



## Model Training Summary

This document provides an overview of the training parameters, final weights, errors, and outputs for a machine learning model experiment.

### Training Parameters

The model was trained with various learning rates to observe the effect on final weights, error rates, and output values.

#### Learning Rates

The learning rates tested were: 0.1, 0.2, 0.5, 0.8, 1.0, and 2.0.

## Results

Below are the results obtained for each learning rate.

### Final Weights

- **Learning Rate 0.1:** `[0.14759022399188299, 0.19518044798376574, 0.24728683077459623, 0.29457366154919246, 0.28734129806637, 0.3365100307893313, 0.36374351855021364, 0.4127381404881795]`
- **Learning Rate 0.2:** `[0.14559145085691055, 0.1911829017138207, 0.2448975840546724, 0.2897951681093448, 0.1410041832582521, 0.18909909371537825, 0.18299706819313005, 0.2306653858518859]`
- **Learning Rate 0.5:** `[0.14493121614580018, 0.1898624322916001, 0.2438902879470082, 0.2877805758940164, 0.00889564912556632, 0.05602699181151564, 0.01757388497347127, 0.06403565048852086]`
- **Learning Rate 0.8:** `[0.1449365537903099, 0.18987310758061965, 0.24381759450331297, 0.28763518900662594, -0.021168320583130288, 0.025744527764154968, -0.02019565573140353, 0.025991614080182808]`
- **Learning Rate 1.0:** `[0.14493932892629438, 0.1898786578525883, 0.2438144791499831, 0.2876289582999662, -0.02344408094753957, 0.02345223991324796, -0.023055049813298808, 0.023111453929145206]`
- **Learning Rate 2.0:** `[0.14493944357831537, 0.1898788871566308, 0.24381436483683838, 0.28762872967367675, -0.023532558630870247, 0.02336311970345274, -0.02316621855801369, 0.022999477875086095]`

### Final Error

- **Learning Rate 0.1:** `0.00789511004702777`
- **Learning Rate 0.2:** `0.0022794087728258373`
- **Learning Rate 0.5:** `9.113838254513836e-05`
- **Learning Rate 0.8:** `4.94363163780147e-07`
- **Learning Rate 1.0:** `7.033390093100831e-10`
- **Learning Rate 2.0:** `1.1870055777625926e-15`

### Final Output

- **Learning Rate 0.1:** `[0.5789273692584652, 0.5977787833632298]`
- **Learning Rate 0.2:** `[0.5421529528784338, 0.5527441571103399]`
- **Learning Rate 0.5:** `[0.5084083716344767, 0.5105629565722294]`
- **Learning Rate 0.8:** `[0.5006192119541653, 0.5007780121357531]`
- **Learning Rate 1.0:** `[0.500023355957549, 0.5000293458219444]`
- **Learning Rate 2.0:** `[0.5000000303417227, 0.5000000381233658]`


