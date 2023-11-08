import requests
import threading
from os import getpid
from curvefittingnn import CurveFitter, targetFunction
import numpy as np
import json
import time

class ClientNode:
    PID = str(getpid())
    server_port = ""
    addr = "http://localhost:"
    client_model = CurveFitter(24, targetFunction)
    num_rounds = 0
    n_epochs = 200
    current_tx = np.array([])
    current_ty = np.array([])
    current_grad_matrix = np.array([])

    def __init__(self, server_port):
        """ tell the new client node what port 
            the server is active on """
        self.server_port = server_port
        self.addr += (self.server_port + "/")
        print(f"starting client: PID {self.PID}")

    def register_with_server(self):
        time.sleep(1)
        """ send post request to register as a client
            with the current open server """
        print(f"{self.PID}: registering with server")
        json_to_send = json.dumps({"pid":int(self.PID), "capabilities":{"n_epochs":self.n_epochs,"batch_size":24,"cli_class":8}})
        r = requests.post(self.addr + "register", data=json_to_send)
        if r.status_code != 200 and r.status_code != 201:
            raise Exception(f"{self.PID}: bad response from server on client registration")
    def request_data_from_server(self):
        time.sleep(1)
        """ send a get request to receive allocated data 
            to do local training on from the server """
        print(f"{self.PID}: requesting data - round {self.num_rounds}")
        # blocks until we get data from server
        r = requests.get(self.addr + f"dataset?pid={self.PID}")
        if r.status_code != 201 and r.status_code != 200:
            raise Exception(f"{self.PID}: bad response from server on request data - round {self.num_rounds}")
        try:
            data = r.json()
            t_x = np.array(data["x_tr"])
            t_y = np.array(data["y_tr"])
            self.current_tx = t_x
            self.current_ty = t_y
        except Exception as e:
            raise e
        return t_x, t_y
    def train_with_new_data(self, t_x, t_y):
        """ class method to do the training of current 
            local model and store the local gradients """
        current_weights = self.client_model.w
        self.client_model.assignTrainSet(t_x, t_y)
        start_time = time.time()
        self.current_grad_matrix = self.client_model.train(self.n_epochs, 5e-3, 1)
        end_time = time.time()
        train_time = end_time - start_time
        json_dict = {"gradients":self.current_grad_matrix.tolist(),"time":train_time}
        json_obj = json.dumps(json_dict)
        return json_obj
    def send_data_to_server(self, json_obj):
        """ send a put request to give the local gradients 
            back to the server """
        r = requests.put(self.addr + f"updated_params?pid={self.PID}",data=json_obj)
        if r.status_code != 200:
            raise Exception(f"{self.PID} bad response from server on send data - round {self.num_rounds}")
        self.num_rounds += 1
        return True

def main():
    server_port = '9099'
    my_node = ClientNode(server_port)
    my_node.register_with_server()

    while(1):
        t_x, t_y = my_node.request_data_from_server()
        json_obj = my_node.train_with_new_data(t_x, t_y)
        success = my_node.send_data_to_server(json_obj)
        if success is True:
            continue
        else:
            break
    return

if __name__=="__main__":
    main()
