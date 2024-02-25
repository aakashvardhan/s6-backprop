# How Backpropagation Works 

## Introduction

Backpropagation is a method used in artificial neural networks to calculate the error contribution of each neuron after a batch of data. It is a supervised learning algorithm, and is used to train the weights of a neural network. It is based on the chain rule of calculus, which allows us to calculate the gradient of a composite function.

## How it Works

The backpropagation algorithm works by computing the gradient of the loss function with respect to the weights of the network. To do this, we need the derivative of the loss function with respect to the output of the network, and the derivative of the output of the network with respect to the weights. Using the chain rule, we can compute the derivative of the loss function with respect to the weights.

## The Chain Rule

The chain rule is a formula to compute the derivative of a composite function. If we have a function `y = f(u)` and `u = g(x)`, then the derivative of `y` with respect to `x` is given by `dy/dx = dy/du * du/dx`. In the context of backpropagation, `y` is the loss function, `u` is the output of the neural network, and `x` is the weights of the network.

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

This can be mathematically represented as:

- $h_{1} = i_{1}w_{1} + i_{2}w_{2}$
- $h_{2} = i_{1}w_{3} + i_{2}w_{4}$

- $a_{h_{1}} = σ(h_{1}) = \frac{1}{(1 + exp(-1*h_{1}))}$
