import random
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Callable
import warnings


def tanh_prime(x):
    """
    Derivative of tanh function
    """
    return 1 - (np.tanh(x)) ** 2


def mse(d: np.ndarray, y: np.ndarray) -> float:
    """
    Mean square error evaluation

    ### Input parameters
    - d: (nx1) vector of training values
    - y: (nx1) vector of neural network outputs associated with training inputs
    """
    assert d.shape == y.shape, "The provided vectors don't have the same dimensions!"
    n = d.shape[0]
    return (1 / n) * np.sum((d - y) ** 2)


def targetFunction(x: float) -> float:
    """
    targetFunction
    ---
    Sample function to be approximated by the model.

    $ y = \sin{(20x)} + 3x$
    """
    return np.sin(20 * x) + 3 * x


class CurveFitter:
    """
    CurveFitter
    ---

    """

    def __init__(
        self,
        n_neurons_hidden: int,
        target_function: Callable = targetFunction,
        act_func: Callable = np.tanh,
        act_func_deriv: Callable = tanh_prime,
    ):
        """
        Initialize CurveFitter object.

        The implemented neural network is single-input, single-output, with one hidden layer.

        ### Input parameters
        - n_neurons_hidden: number of neurons in the hidden layer.
        """

        self.n_hidden = n_neurons_hidden
        self.n_params = 3 * self.n_hidden + 1  # Number of network parameters

        # Initialized weights:
        self.w = np.random.normal(0, 1, (self.n_params))
        # Activation function:
        self.phi = act_func

        # Target function - to be approximated
        self.target = target_function

        self._train_init = False  # True if the training set was initialized

        # ----- Placeholders -----
        # Training elements (and noise) need to be created with the
        # `createTrainSet` method
        self.n_train = None
        self.x_train = None
        self._noise = None
        self.y_train = None
        self.batch_size = 0  # If set to 1, the update is 'online'

        self.eta = None

    def createTrainSet(self, n_train: int) -> [np.ndarray, np.ndarray]:
        """
        createTrainSet
        ---
        Initialize the training set.

        The values of x are uniformly-distributed in [0, 1], while the values of
        y are given by self.target, plus the addition of uniform random noise in
        [-0.1, 0.1].

        The function updates the class attributes, but it also returns the
        generated x_train and y_train vectors.

        ### Input parameters
        - n_train: number of training elements.

        ### Output parameters:
        - self.x_train
        - self.y_train
        """
        if self.target is None:
            raise ValueError(
                "No target function was provided - unable to generate training set!"
            )

        self.n_train = n_train
        self.x_train = np.random.uniform(0, 1, (self.n_train, 1))
        self._noise = np.random.uniform(-0.1, 0.1, (self.n_train, 1))

        self.y_train = self.target(self.x_train) + self.noise
        self._train_init = True

        return self.x_train, self.y_train

    def assignTrainSet(self, x_train, y_train):
        """
        assignTrainSet
        ---
        Pass training values to the model.

        This method should be used when the target function is not known
        and only the training points are given.

        Note that this method will overwrite the current training values.

        ### Input parameters
        - x_train
        - y_train
        """
        if self._train_init:
            warnings.warn("Overwriting existing training set!")

        self.x_train = x_train
        self.y_train = y_train
        self._train_init = True

    def checkTrainInit(self) -> int:
        """Check whether the training set is assigned"""
        if not self._train_init:
            raise ValueError("The training set is not defined!")
        else:
            return 1

    def setLearningRate(self, learn_rate: float):
        """
        setLearningRate
        ---
        Assign the learning rate.
        """
        if learn_rate > 0:
            self.eta = learn_rate
        else:
            raise ValueError("The learning rate must be strictly positive")

    def train(self, n_epochs: int, batch_size: int = 1, norm: bool = False):
        """
        train
        ---
        Launch the training algorithm, via backpropagation.

        The only stopping condition is given by the number of epochs.

        ### Input parameters
        - n_epochs: number of epochs
        - batch_size: default = 1
        - norm: if true, normalize inputs

        ### Output parameters
        - grad_matrix: (n_parameters x n_updates) matrix having as columns the gradient
        values at each update; the number of updates is: floor(n_epochs * n_train / batch_size)
        """
        # TODO: checks
        self.checkTrainInit()

        if self.batch_size != 0 and batch_size != self.batch_size:
            warnings.warn("The batch size changed!")

        self.batch_size = batch_size

        y_curr = np.zeros((self.n_train, 1))
        v_curr = np.zeros(
            (self.n_train, self.n_hidden)
        )  # Intermediate values (for backpropagation)

        # Number of iterations per epoch is given by the batch size
        n_iter_epoch = np.ceil(self.n_train / batch_size)
        # Tau: number of total iterations = floor(epochs * n_train / batch size)
        tau = np.ceil(n_epochs * self.n_train / batch_size)

        # matrix to store (in each column) the average gradient over each batch
        batch_gradients = np.zeros((self.n_params, n_iter_epoch))

        for i in range(self.n_train):
            y_curr[i], v = self.forward(self.x_train[i])
            v_curr[i, :] = v.T

        mse_curr = mse(self.y_train, y_curr)  # Epoch 0 MSE
        curr_grad = np.zeros(self.w.shape)

        for epoch in range(n_epochs):
            # For each epoch:
            for i in range(n_iter_epoch):
                # Iterate over batches (last one may not be full)
                grad_sum = np.zeros(self.w.shape)
                self.mse_per_epoch.append(mse_curr)
                actual_batch_size = 0
                for j in range(batch_size):
                    # curr_index points at the j-th element of batch i
                    curr_index = i * batch_size + j
                    if (
                        curr_index < self.x_train.shape[0]
                    ):  # Check is needed because last batch may not be full
                        actual_batch_size += 1
                        y, v = self.forward(self.x_train[curr_index])
                        y_curr[curr_index] = y
                        grad_sum = self.grad(
                            self.x_train[curr_index], self.y_train[curr_index], y, v
                        )

                avg_grad_batch = grad_sum / actual_batch_size
                batch_gradients[:, i] = grad_sum

                self.w = self.w - self.eta * avg_grad_batch

                # NOTE: HERE

    def grad(self, x: float, d: float, y: float, v: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient of the MSE, key step of backpropagation algorithm

        ### Input parameters
        - x: individual training input
        - d: individual training output
        - y: output associated with weights w
        - v: array of intermediate values (N x 1) associated with input x - dimensions are checked

        ### Output variables
        - grad_mse: (3N+1 x 1) vector containing the gradient of MSE wrt each element of w
        """
        if len(v) != self.n_hidden:
            raise ValueError(
                f"The length of the vector of local fields should be {self.n_hidden}."
            )
        v = v.reshape((self.n_hidden, 1))

        b_i = self.w[: self.n_hidden]  # Biases of 1st layer
        w_ij = self.w[self.n_hidden : 2 * self.n_hidden]  # Weights of 1st layer
        w_prime_1j = self.w[
            2 * self.n_hidden : 3 * self.n_hidden
        ]  # Weights of 2nd layer (to output)
        b_prime = self.w[-1]  # Weight of output

        grad_mse = np.zeros((3 * self.n_hidden + 1, 1))
        # NOTE: derivative of output activation function is 1 (linear)
        for i in range(3 * self.n_hidden + 1):
            if i in range(0, self.n_hidden):
                # Gradient wrt biases of neurons in 1st layer
                grad_mse[i] = -1 * (d - y) * self.phi_prime(v[i]) * w_prime_1j[i]
            elif i in range(self.n_hidden, 2 * self.n_hidden):
                # Gradient wrt weights of neurons in second layer
                grad_mse[i] = (
                    -1
                    * x
                    * (d - y)
                    * self.phi_prime(v[i - self.n_hidden])
                    * w_prime_1j[i - self.n_hidden]
                )
            elif i in range(2 * self.n_hidden, 3 * self.n_hidden):
                # Gradient wrt weights of output neuron
                grad_mse[i] = -1 * self.phi(v[i - 2 * self.n_hidden]) * (d - y)
            else:
                # Gradient wrt bias of output neuron
                assert i == 3 * self.n_hidden
                grad_mse[i] = -1 * (d - y)

        return grad_mse

    def forward(self, x: float) -> [float, np.ndarray]:
        """
        Evaluate the output of the neural network

        ### Input parameters
        - x: input of NN (single value, in this case)

        ### Output values
        - y: output of NN
        - v: intermediate local fields at central layer (before activation)
        """
        # Isolate elements in array w
        b_i = self.w[: self.n_hidden]  # Biases of 1st layer
        w_ij = self.w[self.n_hidden : 2 * self.n_hidden]  # Weights of 1st layer
        w_prime_ij = self.w[
            2 * self.n_hidden : 3 * self.n_hidden
        ]  # Weights of 2nd layer (to output)
        b_prime = self.w[-1]  # Weight of output

        # Evaluate intermediate values:
        v = x * w_ij + b_i
        # Pass them through activation function:
        z = self.phi(v)
        # Evaluate output (y)
        y = sum(z * w_prime_ij) + b_prime

        return y, v


########################################### To be reviewed:


def grad_mse(
    x: float,
    d: float,
    y: float,
    v: np.ndarray,
    w: np.ndarray,
    N: int,
    phi: Callable,
    phi_prime: Callable,
) -> np.ndarray:
    """
    Evaluate the gradient of the MSE, key step of backpropagation algorithm

    ### Input parameters
    - x: individual training input
    - d: individual training output
    - y: output associated with weights w
    - v: array of intermediate values (N x 1) associated with input x - dimensions are checked
    - w: current weight vector (3N+1 x 1)
    - N: number of neurons in the central layer
    - phi: activation function, central layer
    - phi_prime: derivative of activation function, central layer

    ### Output variables
    - grad_mse: (3N+1 x 1) vector containing the gradient of MSE wrt each element of w
    """
    assert (
        w.shape[0] == 3 * N + 1
    ), f"The array of weights has the wrong size; it should be {3 * N + 1} x 1"
    try:
        assert v.shape == (N, 1)
    except:
        v = v.reshape((N, 1))

    b_i = w[:N]  # Biases of 1st layer
    w_ij = w[N : 2 * N]  # Weights of 1st layer
    w_prime_1j = w[2 * N : 3 * N]  # Weights of 2nd layer (to output)
    b_prime = w[-1]  # Weight of output

    grad_mse = np.zeros((3 * N + 1, 1))
    # NOTE: derivative of output activation function is 1 (linear)
    for i in range(3 * N + 1):
        if i in range(0, N):
            # Gradient wrt biases of neurons in 1st layer
            grad_mse[i] = -1 * (d - y) * phi_prime(v[i]) * w_prime_1j[i]
        elif i in range(N, 2 * N):
            # Gradient wrt weights of neurons in second layer
            grad_mse[i] = -1 * x * (d - y) * phi_prime(v[i - N]) * w_prime_1j[i - N]
        elif i in range(2 * N, 3 * N):
            # Gradient wrt weights of output neuron
            grad_mse[i] = -1 * phi(v[i - 2 * N]) * (d - y)
        else:
            # Gradient wrt bias of output neuron
            assert i == 3 * N
            grad_mse[i] = -1 * (d - y)

    return grad_mse


def backpropagation(
    x_init: np.ndarray,
    d: np.ndarray,
    eta: float,
    N: int,
    w: np.ndarray = None,
    max_epoch: int = None,
    img_folder: str = None,
    plots: bool = False,
) -> [np.ndarray, float]:
    """
    backpropagation
    ---
    Backpropagation algorithm on 1 x N x 1 neural network, with
    training set elements (x_i, d_i), starting with weights w.
    This function performs centering of the training inputs (i.e.,
    it removes the mean value before training), and returns the
    mean among the outputs.

    ### Input parameters
    - x_init: training set inputs, non-centered
    - d: training set outputs
    - eta: learning coefficient
    - N: number of perceptron in central layer (number of weights is 3N + 1)
    - w: initial weights (if None, inintialized uniformly in [-1,1])
    - max_epoch: maximum number of training epochs (if None, no maximum)
    - img_folder: folder where to store images
    - plots: flag indicating whether to display plots

    ### Output parameters
    - w: nDarray containing the trained model parameters
    - mu_x: mean value of the provided x_init (will be needed at test)
    """
    assert eta > 0, "Eta must be a strictly positive value!"
    phi = np.tanh  # Activation function of central layer
    phi_prime = tanh_prime  # Derivative of activation function, central layer
    n = x_init.shape[0]

    # CENTER INPUTS
    mu_x = np.mean(x_init)  # Mean value of x
    x = x_init - mu_x  # Center training elements

    if w is None:
        w = np.random.uniform(-1, 1, (3 * N + 1, 1))

    if max_epoch is None:
        max_ind = 2
    else:
        max_ind = max_epoch

    mse_per_epoch = []

    y_curr = np.zeros((n, 1))
    v_curr = np.zeros((n, N))  # Row contains values for element x_i
    for i in range(n):
        y_curr[i], v = forward(x[i], w, N, phi)
        v_curr[i, :] = v.T

    mse_curr = mse(d, y_curr)
    epoch = 0
    last_eta_update = 130  # Makes the 1st eta update after 200 iterations minimum

    while mse_curr >= 0.005 and epoch < max_ind - 1:
        print(f"Epoch: {epoch} - MSE: {mse_curr}")
        mse_per_epoch.append(mse_curr)

        ## Tuning learning rate
        # Idea: perform first 200 iterations with initial eta, then try to increase it
        # if the value of mse between the current epoch and ~70 epochs before has
        # decreased by less than 3%
        # Update condition (at least every 70 epochs from last):
        update_cond = False
        if epoch >= 70 + last_eta_update:
            # Update eta if:
            # 1. The MSE increases in the last iterations (take mean to
            # prevent singular values)
            c1 = mse_per_epoch[-1] > np.mean(mse_per_epoch[-6:-2])

            # 2. The MSE does not decrease by at least 3%
            c2 = mse_per_epoch[-1] > 0.97 * np.mean(mse_per_epoch[-75:-66])

            update_cond = c1 or c2

        if update_cond and eta >= 5e-4:
            eta *= 0.95
            last_eta_update = epoch + 1
            print(f"> Eta decreased ({eta})")

        epoch += 1
        if max_epoch is None:  # Case no max. epochs
            max_ind += 1

        # Update weights - BP
        for i in range(n):
            # Update weights for every element in training set
            y, v = forward(x[i], w, N, phi)
            y_curr[i] = y
            grad_mse_curr = grad_mse(
                x[i],
                d[i],
                y,
                v,
                w,
                N,
                phi,
                phi_prime,
            )
            w = w - eta * grad_mse_curr

        mse_curr = mse(d, y_curr)

    print(f"Epoch: {epoch} - MSE: {mse_curr}")
    epoch += 1
    mse_per_epoch.append(mse_curr)
    if epoch == max_ind:
        print("Early stopping")

    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    ax.plot(list(range(epoch)), mse_per_epoch)
    ax.grid()
    plt.title(f"MSE vs. epoch, eta = {eta}, final MSE = {mse_per_epoch[-1]}")
    ax.set_xlabel(r"epoch")
    ax.set_ylabel(r"MSE")
    if img_folder is not None:
        plt.savefig(os.path.join(img_folder, "mse_per_epoch.png"))
    if plots:
        plt.show()

    # Plot derivative of mse per epoch (to find regions of max variation)
    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    ax.plot(list(range(epoch)), np.gradient(mse_per_epoch))
    ax.grid()
    plt.title(r"$\frac{dMSE}{dt}$ vs. epoch")
    ax.set_xlabel(r"t")
    ax.set_ylabel(r"$\frac{dMSE}{dt}$")
    if img_folder is not None:
        plt.savefig(os.path.join(img_folder, "grad_mse_per_epoch.png"))
    if plots:
        plt.show()

    return w, mu_x


def main(n: int, N: int, img_folder: str, plots: bool = False):
    """
    Main function of the program.

    ### Input parameters
    - n: number of random (training) points
    - N: number of neurons (middle layer) - the network is 1xNx1
    - img_folder: path of the folder where to store images
    - plots: flag for displaying plots
    """
    np.random.seed(660603047)
    # np.random.seed(0)

    # Draw random training elements:
    x = np.random.uniform(0, 1, (n, 1))
    nu = np.random.uniform(-0.1, 0.1, (n, 1))  # Random uniform noise

    d = np.sin(20 * x) + 3 * x + nu

    # Plot points
    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    ax.plot(x, d, "or")
    ax.grid()
    plt.title(f"Training points, n={n}")
    ax.set_xlabel(r"$x_i$")
    ax.set_ylabel(r"$d_i$")
    if img_folder is not None:
        plt.savefig(os.path.join(img_folder, "training_points.png"))
    if plots:
        plt.show()

    # Launch BP algorithm
    eta = 5e-2
    w = np.random.normal(0, 1, (3 * N + 1, 1))  # Gaussian initialization of weights

    w_0, mean_x = backpropagation(
        x, d, eta, N, w, max_epoch=15000, img_folder=img_folder, plots=plots
    )

    print("BP terminated!")

    x_plot = np.linspace(0, 1, 1000)
    y_plot_est = np.zeros((1000, 1))
    for i in range(len(x_plot)):
        # Need to center the test elements
        y_plot_est[i] = forward(x_plot[i] - mean_x, w_0, N, np.tanh)[0]

    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    ax.plot(x_plot, y_plot_est, "b", label="NN output")
    ax.plot(x, d, "or", label="Training set")
    ax.grid()
    ax.legend()
    plt.title(f"Result, n={n}")
    ax.set_xlabel(r"$x_i$")
    ax.set_ylabel(r"$d_i$, $y_i$")
    if img_folder is not None:
        plt.savefig(os.path.join(img_folder, "result.png"))
    if plots:
        plt.show()


if __name__ == "__main__":
    script_folder = os.path.dirname(__file__)
    imgpath = os.path.join(script_folder, "img")
    if not os.path.exists(imgpath):
        os.makedirs(imgpath)

    main(300, 24, imgpath, True)
