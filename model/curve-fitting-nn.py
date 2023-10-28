import random
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Callable


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


def nn_output(x: float, w: np.ndarray, N: int, phi: Callable) -> [float, np.ndarray]:
    """
    Evaluate the output of the neural network

    ### Input parameters
    - x: input of NN (single value, in this case)
    - w: weights of NN (3N+1 elements)
    - N: number of perceptron in central layer
    - phi: activation function in central layer (output is linear)

    ### Output values
    - y: output of NN
    - v: intermediate values at central layer (before activation)
    """
    assert (
        w.shape[0] == 3 * N + 1
    ), f"The array of weights has the wrong size; it should be {3 * N + 1} x 1"

    # Isolate elements in array w
    b_i = w[:N]  # Biases of 1st layer
    w_ij = w[N : 2 * N]  # Weights of 1st layer
    w_prime_ij = w[2 * N : 3 * N]  # Weights of 2nd layer (to output)
    b_prime = w[-1]  # Weight of output

    # Evaluate intermediate values:
    v = x * w_ij + b_i
    # Pass them through activation function:
    z = phi(v)
    # Evaluate output (y)
    y = sum(z * w_prime_ij) + b_prime

    return y, v


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
        y_curr[i], v = nn_output(x[i], w, N, phi)
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
            y, v = nn_output(x[i], w, N, phi)
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
        y_plot_est[i] = nn_output(x_plot[i] - mean_x, w_0, N, np.tanh)[0]

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
