import requests
import threading
from os import getpid
import numpy as np
import json
import time
import sys

from model import CurveFitter, targetFunction
from .config import *


class ClientNode:
    PID = str(getpid())
    server_port = ""
    addr = "http://127.0.0.1:"
    client_model = CurveFitter(N_NEURONS_HIDDEN)
    num_rounds = 0
    # The following value can be overwritten by the configuration JSON
    n_epochs = 100
    batch_size = 12
    cli_class = 5
    learning_rate = 0.0000028
    current_tx = np.array([])
    current_ty = np.array([])
    current_grad_matrix = np.array([])
    last_params_update = 0

    def __init__(self, server_port, cli_info_path: str = None):
        """tell the new client node what port
        the server is active on"""
        self.server_port = server_port
        self.addr += self.server_port + "/"
        print(f"starting client: PID {self.PID}")

        if cli_info_path is not None:
            with open(cli_info_path) as f:
                self.cli_info = json.load(f)
            self.cli_info["pid"] = int(self.PID)
            self.n_epochs = self.cli_info["capabilities"]["n_epochs"]
            self.batch_size = self.cli_info["capabilities"]["batch_size"]
            self.cli_class = self.cli_info["capabilities"]["cli_class"]
            self.learning_rate = self.cli_info["capabilities"]["learning_rate"]
        else:
            self.cli_info = {
                "pid": int(self.PID),
                "capabilities": {
                    "n_epochs": self.n_epochs,
                    "batch_size": self.batch_size,
                    "cli_class": self.cli_class,
                    "learning_rate": self.learning_rate,
                },
            }

    def register_with_server(self):
        """send post request to register as a client
        with the current open server"""
        time.sleep(1)
        if DEBUG:
            print(f"{self.PID}: registering with server")
        json_to_send = json.dumps(self.cli_info)
        r = requests.post(self.addr + "register", data=json_to_send)
        if r.status_code != 200 and r.status_code != 201:
            raise Exception(
                f"{self.PID}: bad response from server on client registration"
            )

    def request_data_from_server(self):
        """send a get request to receive allocated data
        to do local training on from the server"""
        time.sleep(1)
        if DEBUG:
            print(f"{self.PID}: requesting data - round {self.num_rounds}")
        # blocks until we get data from server
        r = requests.get(self.addr + f"dataset?pid={self.PID}")
        if r.status_code != 201 and r.status_code != 200:
            raise Exception(
                f"{self.PID}: bad response from server on request data - round {self.num_rounds}"
            )
        data = r.json()
        t_x = np.array(data["x_tr"])
        t_x = t_x.reshape((len(t_x), 1))
        t_y = np.array(data["y_tr"])
        t_y = t_y.reshape((len(t_y), 1))
        self.current_tx = t_x
        self.current_ty = t_y
        if DEBUG:
            print("N. train elements: ", len(t_x))
        return t_x, t_y

    def request_updated_weights_from_server(self):
        """after sending gradient matrix to server
        we then request the updated global weights before
        starting the process over again so all clients are
        synced"""
        if DEBUG:
            print(f"{self.PID}: requesting global weights - round {self.num_rounds}")
        # blocks until we get the data from server
        r = requests.get(self.addr + f"weights")
        if r.status_code != 201 and r.status_code != 200:
            raise Exception(
                f"{self.PID}: bad response from server on request weights - round {self.num_rounds}"
            )

        # New global params should replace local only if timestamp is new!
        model_params = r.json()
        if model_params["last_update"] > self.last_params_update:
            self.client_model.assignParameters(np.array(model_params["weights"]))
            self.last_params_update = model_params["last_update"]

    def train_with_new_data(self, t_x, t_y):
        """class method to do the training of current
        local model and store the local gradients"""
        current_weights = self.client_model.w
        self.client_model.assignTrainSet(t_x, t_y)
        start_time = time.time()
        # if (self.num_rounds%10 == 0 and self.num_rounds != 0):
        #    self.learning_rate /= 2
        self.current_grad_matrix, mse_lst = self.client_model.train(
            self.n_epochs,
            self.learning_rate,
            self.batch_size,
        )
        if DEBUG:
            print(f"MSE before: {mse_lst[0]}; MSE after: {mse_lst[-1]}")
            print("Shape of gradient matrix: ", self.current_grad_matrix.shape)
        end_time = time.time()
        train_time = end_time - start_time
        json_dict = {"gradients": self.current_grad_matrix.tolist(), "time": train_time}
        json_obj = json.dumps(json_dict)
        return json_obj

    def send_data_to_server(self, json_obj):
        """send a put request to give the local gradients
        back to the server"""
        r = requests.put(self.addr + f"updated_params?pid={self.PID}", data=json_obj)
        if r.status_code != 200:
            raise Exception(
                f"{self.PID} bad response from server on send data - round {self.num_rounds}"
            )
        self.num_rounds += 1


def main():
    server_port = "9099"
    if len(sys.argv) > 1:
        # Argv[1] contains path with client settings
        my_node = ClientNode(server_port, str(sys.argv[1]))
    else:
        # If no path is passed, default client is instantiated
        my_node = ClientNode(server_port)

    registered = False
    att = 1
    while not registered:
        try:
            my_node.register_with_server()
            registered = True
        except:
            if DEBUG:
                print(f"Registration attempt {att} failed")
            att += 1
            time.sleep(1)

        if att >= 50:
            return 0

    my_node.request_updated_weights_from_server()
    t_x, t_y = my_node.request_data_from_server()
    while 1:
        try:
            json_obj = my_node.train_with_new_data(t_x, t_y)
            my_node.send_data_to_server(json_obj)
            my_node.request_updated_weights_from_server()
        except KeyboardInterrupt:
            return 1

    return


if __name__ == "__main__":
    main()
