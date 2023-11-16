import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Callable
import warnings

try:
    from sub.utilities import loadingBar
    from sub.config import VERB
except:
    from model.sub.utilities import loadingBar
    from model.sub.config import VERB


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


class CurveFitter:
    """
    CurveFitter
    ---

    """

    def __init__(
        self,
        n_neurons_hidden: int,
        target_function: Callable = None,
        act_func: Callable = np.tanh,
        act_func_deriv: Callable = tanh_prime,
    ):
        """
        Initialize CurveFitter object.

        The implemented neural network is single-input, single-output, with one hidden layer.

        ### Input parameters
        - n_neurons_hidden: number of neurons in the hidden layer.
        - target_function: function to be approximated; can be None, but the training cannot
        be generated automatically
        - act_func: activation function used in the intermediate layer; default: tanh
        - act_func_deriv: derivative of the activation function
        """

        self.n_hidden = n_neurons_hidden
        self.n_params = 3 * self.n_hidden + 1  # Number of network parameters

        # Initialized weights:
        self.w = np.random.normal(0, 1, (self.n_params, 1))
        # Activation function:
        self.phi = act_func
        self.phi_prime = act_func_deriv

        # Target function - to be approximated
        # NOTE: it can be None
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
        self.mse_per_epoch = None

    def createTrainSet(
        self, n_train: int, range: tuple = (0, 1)
    ) -> [np.ndarray, np.ndarray]:
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
        - range: tuple indicating the range of the training 'x' values; default
        (0, 1)

        ### Output parameters:
        - self.x_train
        - self.y_train
        """
        if self.target is None:
            raise ValueError(
                "No target function was provided - unable to generate training set!"
            )

        self.n_train = n_train
        self.x_train = np.random.uniform(range[0], range[1], (self.n_train, 1))
        self._noise = np.random.uniform(-0.1, 0.1, (self.n_train, 1))

        self.y_train = self.target(self.x_train) + self._noise
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

        assert len(x_train) == len(y_train), "x and y must be equal length"
        self.n_train = len(x_train)
        self.x_train = x_train
        self.y_train = y_train
        self._train_init = True

    def checkTrainInit(self) -> int:
        """Check whether the training set is assigned"""
        if not self._train_init:
            raise ValueError("The training set is not defined!")
        else:
            return 1

    def assignParameters(self, w: np.ndarray):
        """
        assignParameters
        ---
        Assign specific values to the model parameters.

        ### Input parameters
        - w: ndarray containing the new model parameters; needs to be of
        the right length
        """
        if len(w) != self.n_params:
            raise ValueError(
                f"The new vector of parameters has the wrong length!\nExpected: {self.n_params}, got: {len(w)}"
            )
        self.w = w.reshape((self.n_params, 1))

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

    def train(
        self, n_epochs: int, learn_rate: float, batch_size: int = 1, norm: bool = False
    ) -> tuple[np.ndarray, list]:
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
        - mse_per_epoch: MSE values at each training epoch
        """
        # TODO: checks
        self.checkTrainInit()

        self.setLearningRate(learn_rate)

        if self.batch_size != 0 and batch_size != self.batch_size:
            warnings.warn("The batch size changed!")

        self.batch_size = batch_size

        y_curr = np.zeros((self.n_train, 1))
        v_curr = np.zeros(
            (self.n_train, self.n_hidden)
        )  # Intermediate values (for backpropagation)

        # Number of iterations per epoch (= number of batches)
        n_iter_epoch = int(np.ceil(self.n_train / batch_size))
        # Tau: number of total iterations = floor(epochs * n_train / batch size)
        tau = np.ceil(n_epochs * self.n_train / batch_size)

        # matrix to store (in each column) the average gradient over each batch
        batch_gradients = np.zeros((self.n_params, n_epochs * n_iter_epoch))

        for i in range(self.n_train):
            y_curr[i], v = self.forward(self.x_train[i])
            v_curr[i, :] = v.T

        mse_curr = mse(self.y_train, y_curr)  # Epoch 0 MSE
        curr_grad = np.zeros(self.w.shape)
        self.mse_per_epoch = []

        for epoch in range(n_epochs):
            # For each epoch:
            if VERB:
                prt_str = (
                    f"{loadingBar(epoch, n_epochs, 30)} {epoch} / {n_epochs} epochs"
                )
                print(
                    prt_str,
                    end="\r",
                )
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
                        grad_sum += self.grad(
                            self.x_train[curr_index], self.y_train[curr_index], y, v
                        )

                avg_grad_batch = grad_sum / actual_batch_size
                batch_gradients[:, epoch * n_iter_epoch + i] = grad_sum.squeeze()
                self.w = self.w - self.eta * avg_grad_batch

            mse_curr = mse(self.y_train, y_curr)

        if VERB:
            print(" " * (len(prt_str) + 1), end="\r")
        return batch_gradients, self.mse_per_epoch

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

        grad_mse = np.zeros(self.w.shape)
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

    def plotTrainingStats(self, img_path: str = None) -> int:
        """
        plotTrainingStats
        ---
        Plot MSE vs. epoch for last training.

        Can provide a path for the image to save it.
        """
        if self.mse_per_epoch is None:
            warnings.warn("No training was launched yet!")
            return 0

        fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
        ax.plot(list(range(len(self.mse_per_epoch))), self.mse_per_epoch)
        ax.grid()
        plt.title(
            f"MSE vs. epoch, eta = {self.eta}, final MSE = {self.mse_per_epoch[-1]}"
        )
        ax.set_xlabel(r"epoch")
        ax.set_ylabel(r"MSE")
        if img_path is not None:
            plt.savefig(os.path.join(img_path))
        plt.show()
        return 1

    def plotTrainSet(self, img_path: str = None) -> int:
        """
        Plot the training set points if already assigned
        """
        if not self._train_init:
            raise ValueError("Training set points have not been assigned!")

        fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
        ax.plot(self.x_train, self.y_train, "or")
        ax.grid()
        plt.title(f"Training points, n={self.n_train}")
        ax.set_xlabel(r"$x_{i, \text{tr}}$")
        ax.set_ylabel(r"$y_{i, \text{tr}}$")
        if img_path is not None:
            plt.savefig(img_path)
        plt.show()

        return 1


# +------------------------ Target Function ---------------------------+


def targetFunction(x: float) -> float:
    """
    targetFunction
    ---
    Sample function to be approximated by the model.

    $ y = \sin{(20x)} + 3x$
    """
    return np.sin(4 * x) + 2 * x


# +--------------------------------------------------------------------+


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

    myNN = CurveFitter(N, targetFunction)

    # Draw random training elements:
    myNN.createTrainSet(n)

    # Plot points

    # Launch BP algorithm
    eta = 5e-3
    if len(sys.argv) == 3:
        n_epochs = int(sys.argv[1])
        batch_size = int(sys.argv[2])
    else:
        n_epochs = 800
        batch_size = 5
    myNN.train(n_epochs, eta, batch_size)

    print("BP terminated!")

    x_plot = np.linspace(0, 1, 1000)
    y_plot_est = np.zeros((1000, 1))
    for i in range(len(x_plot)):
        # Need to center the test elements
        y = myNN.forward(x_plot[i])[0]
        y_plot_est[i] = y

    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    ax.plot(x_plot, y_plot_est, "b", label="NN output")
    ax.plot(myNN.x_train, myNN.y_train, "or", label="Training set")
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
