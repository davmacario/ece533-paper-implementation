# ECE 533 - Paper implementation project

![Python Version](https://img.shields.io/badge/python-3.10-informational?style=flat&logo=python&logoColor=white)

<!-- markdownlint-disable MD033 -->
<img src="./assets/University_of_Illinois_at_Chicago_circle_logo.png" alt="drawing" width="150"/>
<!-- markdownlint-enable MD033 -->

Implementation of the paper "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization" ECE 533 - Advanced Computer Communication Networks, University of Illinois Chicago, fall 2023.

## Project structure

* The [model folder](./model) contains the neural network model used.
* The [server folder](./server) contains the server program.
* The [client folder](./client) contains the client program.

## Model Overview

The considered neural network performs function interpolation via a single-input, single-output network, composed of two layers, with the hidden layer using $N=30$ nodes.
The approximated function is: $y = \sin{4x} + 2x$ over the domain $x \in [0, 1]$.

The training points are uniformly distributed over the domain, with the addition of uniform noise (in $[-0.1, 0.1]$) on the y coordinate.

The objective function to be minimized during training is the mean squared error (MSE) on the $y$ coordinate.
The values of the gradient are evaluated through backpropagation.

## Server

The server is the central authority in the network of nodes and it has the task to gather all local gradient updates from each client at each iteration, and then put all changes together using the *FedNova* algorithm.

### Working principle & mathematical model

The server has to maintain network information, split the data set based on the node capabilities, and gather results (local model weights after each round), updating them using the *FedNova* update rule:

$$\textbf{x}^{(t+1, 0)} - \textbf{x}^{(t, 0)} = -\tau_{eff}^t \sum_{i=1}^m p_i \cdot\eta\cdot \textbf{d}_i^{(t)}\left(\textbf{x}_i^{(t, 0)}, \ldots, \textbf{x}_i^{(t, \tau_i - 1)}\right)$$

Where:

* $m$: number of clients
* $\textbf{x}_i^{(t, k)}$: model parameters of client $i$ at local epoch $k$, (global) algorithm iteration $t$
* $\eta$: learning rate (hyperparameter)
* $p_i = \frac{n_i}{n}$: local data set ratio - number of local training elements/number of total training elements for client $i$
* $\tau_i = \left\lceil \frac{E n_i}{B} \right\rceil$: number of local iterations performed at each round by client $i$
  * $E$: number of epochs performed locally
  * $B$: local batch size for training
* $\textbf{d}_i^{(t)}$: normalized local gradient
  
  $$\textbf{d}_i^{(t)} = \frac{\textbf{G}_i^{(t)} \textbf{a}_i}{\lvert\lvert \textbf{a}_i \rvert\rvert^2}$$

  With $\textbf{G}_i^{(t)} = \left[\textbf{g}_i(\textbf{x}_i^{(t, 0)}), \ldots, \textbf{g}_i(\textbf{x}_i^{(t, \tau_i - 1)})\right]$ being the matrix containing all stochastic gradients at each local epoch for global algorithm iteration $t$, and $\textbf{a}_i$ being the $\tau_i$-dimensional vector of weights that indicates how gradients $\textbf{g}_i^{(t, k)}$ are accumulated;
  In the case of vanilla SGD, $\textbf{a}_i = \left[1, 1, \ldots, 1\right]$.
* $\tau_{eff}$: effective number of steps; it is a hyperparameter of the global model, to be tuned over values close to $\bar{\tau} = \frac{1}{m}\sum_{i=1}^m \tau_i$

#### Important notices

* The different clients have different values of $\tau_i$ (local iterations - gradient evaluations), $n_i$ (number of training items), $E$ (local number of epochs), $B$ (local batch size).
  Different values correspond to different capabilities.
* After each global update, the new weights (that result from the contribution of all clients) are transmitted back to every client, which will then perform the next round starting from their value.

### Server Operation

The server provides a JSON-based REST API, with support for the following operations:

* GET
  * `http://<server-ip>:<server-port>/dataset?id=<client_pid>` - Fetch the data set, given the client PID, assigned at registration.
  * `http://<server-ip>:<server-port>/weights` - Fetch the current most recent global weights model.
* POST + `http://<server-ip>:<server-port>/register`: used as a client to register to the server, providing in the body of the message the client information (JSON format); *the server knows the number of clients*, and will be able to provide the data set to each once all clients have been correctly registered.
* PUT + `http://<server-ip>:<server-port>/updated_params?id=<client_pid>`: used as a client to upload to the server the updated information (accumulated gradient matrix + training parameters).

## Client

Clients perform local updates and forward update information to the central server.

The local updates are transmitted as matrix $\textbf{G}_i^{(t)}$.

$$
\textbf{G}_i^{(t)} = \begin{bmatrix}
\textbf{g}_i(\textbf{x}_i^{(t, 0)}), \ldots, \textbf{g}_i(\textbf{x}_i^{(t, \tau_i - 1)})
\end{bmatrix}
$$

where each column contains the average gradient of the objective function for batch $k\in \left\{0, \ldots, \tau_i - 1\right\}$.

### Client Operation

1. Each client should first send an HTTP POST to the server registering itself as a node
2. Once the client has registered it will send a blocking GET for data, waiting for the server to start training.
3. The client will then at some point in the future receive a block of data from the server. It will use this data to update its own model parameters.
4. The client will go through all of the data it receives and send POST the updated weights back to the server along with the amount of time it took to learn.
5. Repeat this process until the client requests data and receives a stop command instead.
