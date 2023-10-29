# ECE 533 - Paper implementation project

Implementation of the paper "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization" ECE 533 - Advanced Computer Communication Networks, University of Illinois Chicago, fall 2023.

## Project structure

* <./model> contains the neural network model used.

## Model overview

The considered neural network is used for function interpolation.

It is a single-input, single-output network, composed of two layers, with the hidden layer using $N=24$ nodes.
The function to be approximated is: $y = \sin{20x} + 3x$ over the domain $x \in [0, 1]$.

The training points are uniformly-distributed over the domain, with the addition of uniform noise (in $[-0.1, 0.1]$) on the y coordinate.

The objective function to be minimized during training is the mean squared error (MSE) on the $y$ coordinate.
The values of the gradient are evaluated through backpropagation.

