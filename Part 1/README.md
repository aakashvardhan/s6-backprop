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
$h1 = w1 \cdot i1 + w2 \cdot i2$
$h2 = w3 \cdot i1 + w4 \cdot i2$

Apply the activation function (like sigmoid etc.) to the hidden layer inputs:
$a_{h1} = activation(h1)$
$a_{h2} = activation(h2)$

Calculate the input to the output layer neurons:
$o1 = w5 \cdot a_{h1} + w6 \cdot a_{h2}$
$o2 = w7 \cdot a_{h1} + w8 \cdot a_{h2}$

Apply the activation function to the output layer inputs to get the actual outputs:
$a_{o1} = σ(o1)$
$a_{o2} = σ(o2)$

### Loss Calculation:
Calculate the error for each output neuron (assuming a mean squared error function):
$E1 = \frac{1}{2} (target1 - a_{o1})^2$
$E2 = \frac{1}{2} (target2 - a_{o2})^2$

Total error for the network:
$E_{Total} = E1 + E2$

### Backward Pass (assuming sigmoid activation function for simplicity):
Calculate the gradient of the error with respect to the output activations:
$\frac{\partial E_{Total}}{\partial a_{o1}} = -(target1 - a_{o1})$
$\frac{\partial E_{Total}}{\partial a_{o2}} = -(target2 - a_{o2})$

Calculate the gradient of the error with respect to the net input of the output neurons (derivative of the activation function):
$\frac{\partial E_{Total}}{\partial o1} = \frac{\partial E_{Total}}{\partial a_{o1}} \cdot \frac{\partial a_{o1}}{\partial o1}$
$\frac{\partial E_{Total}}{\partial o2} = \frac{\partial E_{Total}}{\partial a_{o2}} \cdot \frac{\partial a_{o2}}{\partial o2}$

Update the weights between hidden and output layers:
$\Delta w5 = -\eta \cdot \frac{\partial E_{Total}}{\partial o1} \cdot a_{h1}$
$\Delta w6 = -\eta \cdot \frac{\partial E_{Total}}{\partial o1} \cdot a_{h2}$
$\Delta w7 = -\eta \cdot \frac{\partial E_{Total}}{\partial o2} \cdot a_{h1}$
$\Delta w8 = -\eta \cdot \frac{\partial E_{Total}}{\partial o2} \cdot a_{h2}$

Calculate the gradients for the hidden layer weights by propagating the errors back through the network (not explicitly shown here for brevity).

### Weights Update:
The weights are then updated by subtracting the product of the learning rate (η) and the calculated deltas:
$w5 = w5 + \Delta w5$
$w6 = w6 + \Delta w6$
$w7 = w7 + \Delta w7$
$w8 = w8 + \Delta w8$
