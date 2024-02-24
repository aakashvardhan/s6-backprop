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

![Neural Network](simple_nn.png)