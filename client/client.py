import requests
import threading
from os import getpid
from curvefittingnn import CurveFitter, targetFunction
import numpy as np
import json
import time

num_rounds = 0
localhost = 'http://localhost:5555/'

local_model = CurveFitter(24, targetFunction)

def request_data_thread():
    global num_rounds
    while(1):
        print(f"requesting data: round {num_rounds}")
        ## blocks until server has data for client
        r = requests.get(localhost + 'data')
        if r.status_code != 200:
            print(f"uh oh. bad response from server [2]: round {num_rounds}")
            return
        ## get json data from server response
        ## will contain test data
        try:
            data = r.json()
            t_x = np.array(data["t_x"])
            t_y = np.array(data["t_y"])
            print(f"successfully received training data: round {num_rounds}")
        except Exception as e:
            print(f"failed to parse JSON: {e}")
            ## if json fails to parse, go to next server round and wait
            continue

        ## now train model with new data
        w_n0 = local_model.w
        local_model.assignTrainSet(t_x, t_y)
        start_time = time.time()
        grad_matrix_train = local_model.train(100, 5e-3, 1)
        end_time = time.time()
        ## send gradients back to server
        ## create a json with the requisite information
        w_n1 = (w_n0 - local_model.w).tolist()
        train_time = end_time - start_time
        json_dict = {"new_weights":w_n1,"time":train_time}
        json_obj = json.dumps(json_dict)
        r = requests.post(localhost + 'grads',json=json_obj)
        if r.status_code != 200:
            print(f"uh oh. bad response from server [3]: round {num_rounds}")
            return
        num_rounds += 1


def main():
    pid = str(getpid())
    print(f"starting client node-- PID {pid}")

    try:
        r = requests.post(localhost + 'post', data={'pid':pid})
    except:
        print("uh oh. must start server before clients!")
        return
    if r.status_code != 200:
        print("uh oh. bad response from server [1]")
        return
    rdt = threading.Thread(target=request_data_thread)
    rdt.start()


if __name__=="__main__":
    main()
