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
    current_tx = np.array([])
    current_ty = np.array([])
    current_grad_matrix = np.array([])

    def __init__(self, server_port):
        self.server_port = server_port
        self.addr += (self.server_port + "/")
    def register_with_server(self):
        print(f"{self.PID}: registering with server")
        r = requests.post(self.addr, data={'pid':self.PID})
        if r.status_code != 200:
            raise Exception(f"{self.PID}: bad response from server on client registration")
    def request_data_from_server(self):
        print(f"{self.PID}: requesting data - round {self.num_rounds}")
        # blocks until we get data from server
        r = requests.get(self.addr + f"dataset&pid={self.PID}")
        if r.status_code != 200:
            raise Exception(f"{self.PID}: bad response from server on request data - round {self.num_rounds}")
        try:
            data = r.json()
            t_x = np.array(data["t_x"])
            t_y = np.array(data["t_y"])
            self.current_tx = t_x
            self.current_ty = t_y
        except Exception as e:
            raise e
        return t_x, t_y
    def train_with_new_data(self, t_x, t_y):
        current_weights = self.client_model.w
        self.client_model.assignTrainSet(t_x, t_y)
        start_time = time.time()
        self.current_grad_matrix = self.client_model.train(100, 5e-3, 1)
        end_time = time.time()
        train_time = end_time - start_time
        json_dict = {"grad_matrix":self.current_grad_matrix.tolist(),"time":train_time}
        json_obj = json.dumps(json_dict)
        return json_obj
    def send_data_to_server(self, json_obj):
        r = requests.post(self.addr + f"datacollect&pid={self.PID}",json=json_obj)
        if r.status_code != 200:
            raise Exception(f"{self.PID} bad response from server on send data - round {self.num_rounds}")
        self.num_rounds += 1
        return True

def main():
    server_port = input("input central server port: ").strip()
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
