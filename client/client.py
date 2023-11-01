import requests
import threading
from os import getpid

num_rounds = 0
localhost = 'http://localhost:5555/'

def request_data_thread():
    global num_rounds
    while(1):
        print(f"requesting data: round {num_rounds}")
        r = requests.get(localhost + 'get_data')
        if r.status_code != 200:
            print(f"uh oh. bad response from server [2]:round {num_rounds}")
            return
        ## response will be json
        try:
            data = r.json()
        except:
            print("failed to parse JSON")
        ## now do something with the data i.e. train model
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
