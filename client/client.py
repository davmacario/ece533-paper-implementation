import requests
import threading
from os import getpid
from curvefittingnn import CurveFitter, targetFunction
import numpy as np

num_rounds = 0
localhost = 'http://localhost:5555/'

local_model = CurveFitter(24, targetFunction)

def request_data_thread():
    global num_rounds
    while(1):
        print(f"requesting data: round {num_rounds}")
        r = requests.get(localhost + 'data')
        if r.status_code != 200:
            print(f"uh oh. bad response from server [2]: round {num_rounds}")
            return
        ## response will be json
        try:
            data = r.json()
            t_x = np.array(data["t_x"])
            t_y = np.array(data["t_y"])
            print(f"successfully received training data: round {num_rounds}")
        except Exception as e:
            print(f"failed to parse JSON: {e}")
            return
        ## now do something with the data i.e. train model
        local_model.assignTrainSet(t_x, t_y)
        grad_matrix_train = local_model.train(600, 5e-3, 1)
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
