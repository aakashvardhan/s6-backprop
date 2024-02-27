# How Backpropagation Works 

The backpropagation algorithm consists of two main steps. In the forward pass, the input is passed through the network, and the output of the network is computed. In the backward pass, the gradient of the loss function with respect to the output of the network is computed, and then the gradient of the loss function with respect to the weights of the network is computed using the chain rule. The weights of the network are then updated using gradient descent.


To understand backpropagation better, let's work through an example. We'll use a simple neural network with one input layers, two hidden layer, and one output layers. The input layer has two neurons, the hidden layer has two neurons, and the output layer has two neurons. We'll use the sigmoid activation function for the hidden layer, and the identity activation function for the output layer. We'll use mean squared error as the loss function.

<p align="center">
  <img src="https://github.com/aakashvardhan/s6-backprop/blob/main/Part%201/simple-nn.png" width="60%" />
</p>

## Neural Network Backpropagation Explained

Understanding the intricacies of backpropagation in neural networks is crucial for understanding how these models learn from data. This guide provides a detailed breakdown of the key components and processes involved.

### Network Inputs and Connections

### Inputs

- **`i_1` and `i_2`**: Initial data points fed into the network, representing features such as pixels in an image or attributes of a dataset.

### Hidden Layer Neurons

- **`h_1` and `h_2`**: Neurons that capture complex patterns by combining inputs through weighted connections.
  - `h_1` connects to `i_1` and `i_2` through weights `w_1` and `w_2`.
  - `h_2` connects to `i_1` and `i_2` through weights `w_3` and `w_4`.

## Processing and Activation Functions

### Activation Functions for Hidden Layer

- **`a_h_1` and `a_h_2`**: Non-linear activation functions applied to the output of `h_1` and `h_2`, enabling the network to model complex relationships.

### Activation Functions for Output Layer

- **`a_o_1` and `a_o_2`**: Activation functions applied to the output neurons `o_1` and `o_2`, shaping the final predictions of the network.

## Weighted Connections to Output Layer

- Outputs from `h_1` and `h_2` connect to `o_1` and `o_2` via weights `w_5`, `w_6`, `w_7`, and `w_8`.

## Error Calculation and Targets

### Mean Squared Error Loss Function

- **`E_total = E_1 + E_2`**: Performance evaluated by calculating the mean squared error between predictions and actual target values `t_1` and `t_2`.

## Backpropagation and Learning

Adjusting weights (`w_1` through `w_8`) based on the calculated error, using the derivative of the error with respect to each weight and the chain rule for derivatives.

## Step-by-Step Backpropagation Process

### Forward Pass

1. **Input to Hidden Layer Calculations**:
   - `h_1 = w_1 * i_1 + w_2 * i_2`
   - `h_2 = w_3 * i_1 + w_4 * i_2`
   - Activation functions applied to hidden layer outputs:
     - `a_h_1 = σ(h_1) = 1 / (1 + exp(-h_1))`
     - `a_h_2 = σ(h_2)`

2. **Hidden to Output Layer Calculations**:
   - `o_1 = w_5 * a_h_1 + w_6 * a_h_2`
   - `o_2 = w_7 * a_h_1 + w_8 * a_h_2`
   - Activation functions applied to output layer outputs:
     - `a_o_1 = σ(o_1)`
     - `a_o_2 = σ(o_2)`

3. **Error Calculation**:
   - Total error calculation:
     - `E_total = E_1 + E_2`
     - `E_1 = 1/2 * (t_1 - a_o_1)^2`
     - `E_2 = 1/2 * (t_2 - a_o_2)^2`

### Backward Pass (Backpropagation)

Goal: Compute how much each weight contributes to the error and adjust accordingly.

1. **Gradient Calculation for Output Layer Weights (`w_5`, `w_6`, `w_7`, `w_8`)**:
   - `∂E_total/∂w_5 = (a_o_1 - t_1) * a_o_1 * (1 - a_o_1) * a_h_1`
   - Similar calculations for `w_6`, `w_7`, and `w_8`.

2. **Gradient Calculation for Input Layer Weights (`w_1`, `w_2`, `w_3`, `w_4`)**:
   - `∂E_total/∂w_1 = ((a_o_1 - t_1) * a_o_1 * (1 - a_o_1) * w_5 + (a_o_2 - t_2) * a_o_2 * (1 - a_o_2) * w_7) * a_h_1 * (1 - a_h_1) * i_1`
   - Similar calculations for `w_2`, `w_3`, and `w_4` using their respective paths through the network.

### Important Notes

- **Activation Function Derivative**: The derivative of the sigmoid function `σ(x)` is crucial for calculating gradients and is given by `σ(x) * (1 - σ(x))`.
- **Chain Rule Application**: The gradients of the weights are computed by applying the chain rule of derivatives. This takes into account the path of each weight's influence on the total error, allowing for the precise adjustment of weights to minimize error in the network's predictions.



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


