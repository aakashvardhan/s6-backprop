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

## Backpropagation Mathematical Expressions

### Forward Pass:
Calculate the input to the hidden layer neurons:
- $h_{1} = w_{1} \cdot i_{1} + w_{2} \cdot i_{2}$
- $h_{2} = w_{3} \cdot i_{1} + w_{4} \cdot i_{2}$

Apply the activation function (like sigmoid etc.) to the hidden layer inputs:
- $a_{h_{1}} = activation(h_{1})$
- $a_{h_{2}} = activation(h_{2})$

Calculate the input to the output layer neurons:
- $o_{1} = w_{5} \cdot a_{h_{1}} + w_{6} \cdot a_{h_{2}}$
- $o_{2} = w_{7} \cdot a_{h_{1}} + w_{8} \cdot a_{h_{2}}$

Apply the activation function to the output layer inputs to get the actual outputs:
- $a_{o_{1}} = σ(o_{1})$
- $a_{o_{2}} = σ(o_{2})$

### Loss Calculation:
Calculate the error for each output neuron (assuming a mean squared error function):
- $E_{1} = \frac{1}{2} (target_{1} - a_{o_{1}})^{2}$
- $E_{2} = \frac{1}{2} (target_{2} - a_{o_{2}})^{2}$

Total error for the network:
- $E_{Total} = E_{1} + E_{2}$

### Backward Pass (assuming sigmoid activation function for simplicity):
Calculate the gradient of the error with respect to the output activations:
- $\frac{\partial E_{Total}}{\partial a_{o_{1}}} = -(target_{1} - a_{o_{1}})$
- $\frac{\partial E_{Total}}{\partial a_{o_{2}}} = -(target_{2} - a_{o_{2}})$

Calculate the gradient of the error with respect to the net input of the output neurons (derivative of the activation function):
- $\frac{\partial E_{Total}}{\partial o_{1}} = \frac{\partial E_{Total}}{\partial a_{o_{1}}} \cdot \frac{\partial a_{o_{1}}}{\partial o_{1}}$
- $\frac{\partial E_{Total}}{\partial o_{2}} = \frac{\partial E_{Total}}{\partial a_{o_{2}}} \cdot \frac{\partial a_{o_{2}}}{\partial o_{2}}$

Update the weights between hidden and output layers:
- $\Delta w_{5} = -\eta \cdot \frac{\partial E_{Total}}{\partial o_{1}} \cdot a_{h_{1}}$
- $\Delta w_{6} = -\eta \cdot \frac{\partial E_{Total}}{\partial o_{1}} \cdot a_{h_{2}}$
- $\Delta w_{7} = -\eta \cdot \frac{\partial E_{Total}}{\partial o_{2}} \cdot a_{h_{1}}$
- $\Delta w_{8} = -\eta \cdot \frac{\partial E_{Total}}{\partial o_{2}} \cdot a_{h_{2}}$

Calculate the gradients for the hidden layer weights by propagating the errors back through the network (not explicitly shown here for brevity).

### Weights Update:
The weights are then updated by subtracting the product of the learning rate (η) and the calculated deltas:
- $w_{5} = w_{5} + \Delta w_{5}$
- $w_{6} = w_{6} + \Delta w_{6}$
- $w_{7} = w_{7} + \Delta w_{7}$
- $w_{8} = w_{8} + \Delta w_{8}$


## Results

Shown below are the effects of learning rate on the convergence of the backpropagation algorithm. The learning rate is a hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated. A smaller learning rate requires more training epochs given the smaller changes made to the weights each update, and a larger learning rate may cause the model to converge too quickly, which may result in the model overshooting the optimal weights.

<table>
  <tr>
    <td><img src="https://github.com/aakashvardhan/s6-backprop/blob/main/Part%201/error_graph_lr/0.1.png" alt="Plot 1" style="width: 100%;"/></td>
    <td><img src="https://github.com/aakashvardhan/s6-backprop/blob/main/Part%201/error_graph_lr/0.2.png" alt="Plot 2" style="width: 100%;"/></td>
    <td><img src="https://github.com/aakashvardhan/s6-backprop/blob/main/Part%201/error_graph_lr/0.5.png" alt="Plot 3" style="width: 100%;"/></td>
  </tr>
  <tr>
    <td align="center">Caption for Plot 1</td>
    <td align="center">Caption for Plot 2</td>
    <td align="center">Caption for Plot 3</td>
  </tr>
  <tr>
    <td><img src="https://github.com/aakashvardhan/s6-backprop/blob/main/Part%201/error_graph_lr/0.8.png" alt="Plot 4" style="width: 100%;"/></td>
    <td><img src="https://github.com/aakashvardhan/s6-backprop/blob/main/Part%201/error_graph_lr/1.0.png" alt="Plot 5" style="width: 100%;"/></td>
    <td><img src="https://github.com/aakashvardhan/s6-backprop/blob/main/Part%201/error_graph_lr/2.0.png" alt="Plot 6" style="width: 100%;"/></td>
  </tr>
  <tr>
    <td align="center">Caption for Plot 4</td>
    <td align="center">Caption for Plot 5</td>
    <td align="center">Caption for Plot 6</td>
  </tr>
</table>