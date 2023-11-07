# Server

The server is the central authority in the network of nodes and it has the task to gather all local gradient updates from each client at each iteration, and then put all changes together using the *FedNova* algorithm.

## Working principle & mathematical model

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

### Important notices

* The different clients have different values of $\tau_i$ (local iterations - gradient evaluations), $n_i$ (number of training items), $E$ (local number of epochs), $B$ (local batch size).
  Different values correspond to different capabilities.
* After each global update, the new weights (that result from the contribution of all clients) are transmitted back to every client, which will then perform the next round starting from their value.

## Operation

The server provides a JSON-based REST API, with support for the following operations:

* GET
  * `http://<server-ip>:<server-port>/dataset?id=<client_pid>` - Fetch the data set, given the client PID, assigned at registration.
  * `http://<server-ip>:<server-port>/weights` - Fetch the current most recent global weights model.
* POST + `http://<server-ip>:<server-port>/register`: used as a client to register to the server, providing in the body of the message the client information (JSON format); *the server knows the number of clients*, and will be able to provide the data set to each once all clients have been correctly registered.
* PUT + `http://<server-ip>:<server-port>/updated_params?id=<client_pid>`: used as a client to upload to the server the updated information (accumulated gradient matrix + training parameters).

## Message format

The general format of messages to and from the server is JSON.

In particular, the server requires the following structure for the client information (sent at registration):
[TODO - decide]

* `"pid"`: PID of the client (int)
* `"capabilities"` (dict)
  * `"n_epochs"`: number of epochs executed locally by the client
  * `"batch_size"`: batch size used locally
  * `"cli_class"`: [FIX] integer in the range $[0, 10]$ which indicates the "capabilities" of the client (high value means "better" client); when splitting the data set among the clients, the server assigns a number of elements proportional to this value to each client

Concerning the model variables and the data sets, the structures of the JSONs are the following.
Model:

* `"weights"`: list of weights - to convert them to Numpy arrays, simply plug them in `np.array()`, and they will be converted to a column vector (ready to be used in the model)
* `"last_update"`: timestamp (int) used by the client to understand whether these weights are the "updated" ones - **the client should keep track of the last timestamp of the weights**

Dataset:

* `"x_tr"`: list containing the training $x$ values - can be converted back to Numpy array as the weights
* `"y_tr"`: list containing the training $y$ values - can be converted back to Numpy array as the weights
