import cherrypy
from datetime import datetime
import json
import numpy as np
import os
import sys
import time
import warnings

import threading
import matplotlib.pyplot as plt

from .config import *
from model import mse, CurveFitter, targetFunction


first = 1

fig = None
ax = None


def plot_current_model(webserver, pause: bool, new_fig: bool = False):
    """Display the current model results"""
    global fig, ax, first

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
        plt.pause(0.001)
    x_plot = np.linspace(0, 1, 1000)
    y_plot_est = np.zeros((1000, 1))
    for i in range(len(x_plot)):
        # Need to center the test elements
        y = webserver.serv.model.forward(x_plot[i])[0]
        y_plot_est[i] = y

    ax.plot(x_plot, y_plot_est, color=(0, 0, 1), label="NN output")
    if first:
        ax.grid()
        ax.plot(
            webserver.serv.model.x_train,
            webserver.serv.model.y_train,
            "or",
            label="Training set",
        )
        first = 0
    if pause:
        plt.show()
    else:
        plt.pause(0.001)


class FedNovaServer:
    """
    FedNovaServer class
    ---

    ### Attributes

    (Private)
    - _cli_registered

    (Public)
    - n_clients
    - cli_last_grad: Vector containing the update term from each client at every iteration;
    TODO
    """

    def __init__(
        self,
        n_clients: int,
        in_json_path: str,
        out_json_path: str,
        update_type: str = "FedNova",
    ):
        """
        FedNovaServer
        ---
        Server for federated learning using the FedNova paradigm described
        in "Tackling the Objective Inconsistency Problem in Heterogeneous
        Federated Optimization", by <>.

        The server is supposed to know the number of clients it will manage
        and the settings will be imported from the config.py file in the same
        folder.

        ### Input parameters

        - n_clients: number of federated learning clients to be managed
        by the server
        - in_json_path: path of the json template maintained by the server,
        which will be used to store the global information (clients,
        iterations, hyperparameters, ...)
        - out_json_path: path where to store updated information
        - update_type: string indicating the update rule to be used; default
        "FedNova", "FedAvg".

        ### Exposed API

        The server exposes a HTTP-based REST API, allowing for GET, POST
        and PUT requests.

        - GET: used for fetching the data set as a client (only when all
        clients have successfully registered) and updated weights at each
        algorithm iteration
        - POST: used for registration, providing the information in JSON
        format
        - PUT: used for uploading the local model parameter at each
        algorithm iteration
        """
        if n_clients <= 0:
            raise ValueError("The number of clients needs to be >= 1")

        self.n_clients = n_clients
        self.cli_map_PID = []  # elem. i contains PID of client i - append as they come
        self._cli_registered = [
            False
        ] * n_clients  # elem. i true if client i registered

        # Import the server information dict
        self._server_info = json.load(open(in_json_path))
        self.updateTimestamp()
        self._out_info_path = out_json_path
        self.saveStateJson()

        # Keys of the client dictionary that need to be provided at registration
        self._cli_params_input = ["pid", "capabilities"]
        # Keys of the client dictionary after registration (add "id")
        self._cli_params = ["id", "pid", "capabilities"]
        self._cli_capabilities_params = [
            "n_epochs",
            "batch_size",
            "cli_class",
            "learning_rate",
        ]

        # Model
        self._valid_update_rules = ["FedNova", "FedAvg"]
        if not update_type in self._valid_update_rules:
            raise ValueError(
                f"Specified update rule {update_type} is not valid!\nValid rules are: {self._valid_update_rules}"
            )
        self.update_rule = update_type
        self.n_neurons_hidden = self._server_info["parameters"]["n_neurons_hidden"]
        self.n_train = self._server_info["parameters"]["n_train"]
        self.n_tr_cli = [0] * self.n_clients
        self.p_i = [0] * self.n_clients
        self.tau_i = [0] * self.n_clients
        self.tau_eff = 0
        self.model = CurveFitter(self.n_neurons_hidden, targetFunction)
        # Learning rate for local update (NOT USED...)
        # Clients use local learning rate!
        self.eta = self._server_info["parameters"]["learning_rate"]
        self.model.setLearningRate(self.eta)
        # NOTE: maybe need to change range in training set 'x'
        self.train_x, self.train_y = self.model.createTrainSet(self.n_train)
        self.train_split = []  # Will contain dicts with "x_tr" and "y_tr"
        self.train_split_send = []  # Will contain the 'list' version of the datasets
        self.n_model_parameters = self.model.n_params
        # Model parameters vector - all clients should start from the same!
        self.model_params = {}
        self.model_params["weights"] = self.model.w.tolist()
        self.model_params["last_update"] = time.time()  # Used to track parameter age
        self.n_update_iterations = 0  # Number of global model parameter updates

        # Placeholders for the data set sections:
        # Each element is the dataset for the corresponding client in JSON format
        self.cli_data_sets = []
        self.cli_last_grad = [0] * n_clients
        self.cli_last_update = [0] * n_clients
        # This will store the update parameters of the calling client
        # p_i is the proportion of data given to client i and tau_i is the number of updates
        self.cli_last_update_params = [
            {"p_i": 0, "tau_i": 0} for n in self.cli_last_update
        ]
        self.cli_last_tau = [0] * n_clients

        # Store stats
        self.mse_per_global_iter = []

    def updateTimestamp(self) -> str:
        """
        Update the 'last_update' field in the server info dict.
        """
        self._last_update_info = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._server_info["last_update"] = self._last_update_info
        return self._last_update_info

    def saveStateJson(self) -> int:
        """
        Save the current server state as JSON at the output path passed at
        instantiation.
        This function needs to be called every time the state is updated.

        The function returns 1 for success, 0 for failure.
        """
        try:
            with open(self._out_info_path, "w") as f:
                json.dump(self._server_info, f)
                return 1
        except:
            return 0

    def searchClient(self, key: str, value: any) -> dict:
        """
        Search for a registered client by parameter.

        ### Input parameters
        - key: string indicating the parameter to be looked for
        (needs to be included in self._cli_params_input)
        - value: the value associated with the key to be looked for

        ### Output parameters
        - elem: if client with matching value was found, it contains
        the client information (as dictionary), else it is an empty
        dict
        """
        if key not in self._cli_params:
            raise KeyError(f"Invalid client parameter name {key}")
        elem = {}
        for cli in self._server_info["clients"]:
            if cli[key] == value:
                elem = cli.copy()
                return elem
        # If here, client was not found
        return {}

    def addClient(self, cl_info: dict) -> int:
        """
        Add a new client - registration.
        The client will only be added if the PID is different from any
        of the ones already in the system.
        The new client is assigned a unique ID, that will overwrite its
        current one, if present.

        ### Return values
        - [>= 0]: unique ID of the new client
        - -1: no free IDs available (max. n. of clients has registered)
        - -2: wrong client format
        - -3: the client already registered
        """
        new_id = self.getFreeID()

        # Check for info validity
        if not self.checkValidClient(cl_info):
            new_id = -2
        elif self.searchClient("pid", cl_info["pid"]) != {}:
            # if here, the client is valid, i.e., it contains "PID", but
            # the value results already registered
            new_id = -3

        if new_id >= 0:
            # Set the ID as occupied
            self._cli_registered[new_id] = True
            # Only keep the keys specified in _cli_params_input:
            new_cli_info = {}
            self.cli_map_PID.append(cl_info["pid"])
            for k in self._cli_params_input:
                new_cli_info[k] = cl_info[k]
            new_cli_info["id"] = new_id
            new_cli_info["last_update"] = self.updateTimestamp()
            self._server_info["clients"].append(new_cli_info)

            self.updateTimestamp()
            self.saveStateJson()

            # Since the new client was added, if the required number of
            # clients was reached, split the data set
            if self.allCliRegistered():
                self.splitTrainSet()
        return new_id

    def getFreeID(self) -> int:
        """
        Return the first unused ID that can be assigned.

        If the return value is -1, there are no more free IDs.
        """
        i = 0
        while i < self.n_clients:
            if not self._cli_registered[i]:
                return i
            i += 1
        return -1

    def checkValidClient(self, cl_info: dict) -> int:
        """
        Check whether the new client information is in the valid format,
        i.e., all the required keys are present.
        (Attribute: self._cli_params_input)

        ### Output parameters
        - 1 if good syntax
        - 0 if missing keys
        """
        for k in self._cli_params_input:
            if k not in list(cl_info.keys()):
                return 0
        for k in self._cli_capabilities_params:
            # Check all required capabilities are present
            if k not in cl_info["capabilities"].keys():
                return 0
        return 1

    def allCliRegistered(self):
        """
        Return true if all the expected clients have registered
        """
        if DEBUG and len(self.cli_map_PID) == self.n_clients:
            print(f"All {self.n_clients} clients have registered!")
        return len(self.cli_map_PID) == self.n_clients

    def splitTrainSet(self):
        """
        Divide the training set among the different clients.
        The function will raise an error if not all the required clients
        have registered!
        """
        if not self.allCliRegistered():
            raise ValueError(
                f"Not all clients have registered! {len(self.cli_map_PID)}/{self.n_clients}"
            )

        capabilities = [
            int(c["capabilities"]["cli_class"]) for c in self._server_info["clients"]
        ]
        tot_c = sum(capabilities)
        for i in range(self.n_clients):
            if i < self.n_clients - 1:
                self.n_tr_cli[i] = round(self.n_train * capabilities[i] / tot_c)
            else:
                # This is necessary due to rounding - may lose some training elements...
                self.n_tr_cli[i] = self.n_train - sum(self.n_tr_cli[:-1])
            # FIXME: the following assumes clients are added in the order of their
            # ID to the server info
            cli_curr = self._server_info["clients"][i]
            cli_curr["capabilities"]["n_train"] = self.n_tr_cli[i]
            self.tau_i[i] = np.ceil(
                cli_curr["capabilities"]["n_epochs"]
                * self.n_tr_cli[i]
                / cli_curr["capabilities"]["batch_size"]
            )
            cli_curr["capabilities"]["tau"] = self.tau_i[i]
            # TODO: confirm this code works - are the updates on cli_curr reflected on the
            # values in self._server_info?
        self.p_i = [n / self.n_train for n in self.n_tr_cli]

        assert sum(self.n_tr_cli) == self.n_train

        ind_ds = 0
        for i in range(self.n_clients):
            tr_set_curr = {}
            tr_set_curr["x_tr"] = self.train_x[ind_ds : ind_ds + self.n_tr_cli[i]]
            tr_set_curr["y_tr"] = self.train_y[ind_ds : ind_ds + self.n_tr_cli[i]]
            self.train_split.append(tr_set_curr)
            # Create the "JSON-friendly" version of the dataset
            tr_set_curr_send = {}
            tr_set_curr_send["x_tr"] = tr_set_curr["x_tr"].tolist()
            tr_set_curr_send["y_tr"] = tr_set_curr["y_tr"].tolist()
            self.train_split_send.append(tr_set_curr_send)
            ind_ds += self.n_tr_cli[i]

        self.updateTimestamp()
        self.saveStateJson()

        if DEBUG:
            print("Training set was split!")

        return 1

    def addGradientMatrix(
        self, grad_mat: np.ndarray, user_val: int, attr_key: str = "pid"
    ) -> int:
        """
        Add the result of the last iteration of the client identified by
        the specific PID.
        The information is stored in attribute `cli_last_grad` in the position
        associated with the client ID.

        ### Input parameters
        - grad_mat: matrix of gradients resulting from the last update
        at the client
        - user_val: value identifying the client
        - attr_key: key associated with the value (default: "pid")

        ### Return value(s)
        - 1 if successful update
        """

        ## in this function we want to store the relevant information
        ## needed for updateweights so that all the clients data is in
        ## the correct place

        if attr_key == "pid" and user_val not in self.cli_map_PID:
            raise ValueError(f"PID {user_val} not found among registered clients!")
        elif attr_key == "id" and (user_val < 0 or user_val >= self.n_clients):
            raise ValueError(f"Invalid user ID {user_val}")
        if DEBUG:
            print("Shape of gradient matrix: ", grad_mat.shape)

        cli = self.searchClient(attr_key, user_val)
        if cli == {}:
            warnings.warn(f"Unable to find client with {attr_key} = {user_val}")
            return 0
        cli_id = cli["id"]
        if DEBUG:
            print("storing values for [" + str(cli_id) + "]")
        ## get number of update steps for client i
        self.cli_last_update_params[cli_id]["tau_i"] = grad_mat.shape[1]
        ## get proportion of data given to client i
        self.cli_last_update_params[cli_id]["p_i"] = self.p_i[cli_id]
        self.cli_last_grad[cli_id] = grad_mat

        return 1

    def get_a(self, tau_i: int, method: str = "Vanilla SGD") -> np.ndarray:
        """
        Evaluate the vector 'a', used to compute the update term for each client
        from the matrix of gradients.

        ### Input parameters
        - tau_i: number of local iterations at current (i-th) client; it is the
        length of vector 'a'
        - method: string indicating the update method; based on this, the output
        vector will be built in different ways

        ### Output
        - a: column ndarray
        """
        if tau_i <= 0 or not isinstance(tau_i, int):
            raise ValueError("Unsupported value for tau_i")

        ## a is always a vector of tau_i ones no matter the update type
        a = np.ones((tau_i, 1))
        return a

    def updateWeights(self) -> int:
        """
        Update the model parameters using the specified rule.
        This requires that every client has concluded its local iteration,
        i.e., that `cli_last_grad` is filled.

        Once the update is performed, the weights of the model are updated
        and the clients can get them to proceed with the next iteration.

        ### Ouptut value
        - 0 if not ready (missing local updates), 1 if success
        """
        assert self.update_rule in self._valid_update_rules  # Shouldn't need it

        if self.update_rule.lower() == "fedavg":
            ## this will be used to show bad convergence using fedavg
            ## calculate tau_eff
            if DEBUG:
                print("[1] updating weights fedavg style")
            tau_eff = 0
            for i in range(self.n_clients):
                tau_eff += (
                    self.cli_last_update_params[i]["p_i"]
                    * self.cli_last_update_params[i]["tau_i"]
                )
            ## calculate weightings
            final_sum = 0
            for i in range(self.n_clients):
                ## for each sum we calculate w_i and d_i and multiply together
                final_sum += (
                    (
                        self.cli_last_update_params[i]["p_i"]
                        * self.cli_last_update_params[i]["tau_i"]
                        / tau_eff
                    )
                    * np.dot(
                        self.cli_last_grad[i],
                        self.get_a(self.cli_last_update_params[i]["tau_i"]),
                    )
                    / self.cli_last_update_params[i]["tau_i"]
                )
            final_sum *= self.eta
            final_sum *= tau_eff
            self.model.w = self.model.w - final_sum
            self.model_params["weights"] = self.model.w.tolist()
            self.model_params["last_update"] = time.time()

        elif self.update_rule.lower() == "fednova":
            ## the only difference with fednova is that w_i is replaced by p_i
            if DEBUG:
                print("[1] updating weights fednova style")
            tau_eff = 0
            for i in range(self.n_clients):
                tau_eff += (
                    self.cli_last_update_params[i]["p_i"]
                    * self.cli_last_update_params[i]["tau_i"]
                )
            ## calculate weightings
            final_sum = 0
            for i in range(self.n_clients):
                ## for each sum we calculate w_i and d_i and multiply together
                final_sum += (
                    self.cli_last_update_params[i]["p_i"]
                    * np.dot(
                        self.cli_last_grad[i],
                        self.get_a(self.cli_last_update_params[i]["tau_i"]),
                    )
                    / self.cli_last_update_params[i]["tau_i"]
                )
            final_sum *= self.eta
            final_sum *= tau_eff
            self.model.w = self.model.w - final_sum
            self.model_params["weights"] = self.model.w.tolist()
            self.model_params["last_update"] = time.time()
        else:
            raise ValueError(f"Unsupported update rule {self.update_rule}!")

        # Evaluate the MSE after the update
        curr_mse = self.getMSE()
        self.mse_per_global_iter.append(curr_mse)
        if DEBUG:
            print(f"Iteration {self.n_update_iterations}:")
            print(f"> Current MSE: {curr_mse}")

        self.n_update_iterations += 1
        return 1

    def learningRate(self) -> float:
        """Fix the learning rate - depending on the global iterations."""
        if self.n_update_iterations >= 200:
            return 0.1
        else:
            return 1

    def getMSE(self):
        """
        Evaluate the Mean Squared Error on the whole training set with the
        current model parameters.
        """
        if DEBUG:
            print(self.model.w.shape)
            print("Input: ", self.train_x[0])
            print("Output: ", self.model.forward(self.train_x[0])[0])
        y_est = np.zeros((self.train_y.shape[0], 1))
        for i in range(len(self.train_x)):
            y_est[i], _ = self.model.forward(self.train_x[i])

        return mse(self.train_y, y_est)

    def getIP(self) -> str:
        """Return IP address contained in the server information"""
        return self._server_info["ip"]

    def getPort(self) -> int:
        """Return port number contained in the server information"""
        return self._server_info["port"]


class FedNovaWebServer:
    """
    FedNovaWebServer
    ---
    This class implements the HTTP APIs of the FedNova server.
    """

    exposed = True

    def __init__(
        self,
        n_clients: int,
        in_json_path: str = os.path.join(os.path.dirname(__file__), "serv_info.json"),
        out_json_path: str = os.path.join(
            os.path.dirname(__file__), "serv_info_updated.json"
        ),
        cmd_list_path: str = os.path.join(
            os.path.dirname(__file__), "fednova_API.json"
        ),
        public: bool = False,
        update_type="FedNova",
    ):
        """
        FedNovaWebServer
        ---
        Create web server for Federated Learning using FedNova paradigm.

        ### Input parameters
        - n_clients: number of required clients
        - in_json_path: path of the input JSON containing server information
        - out_json_path: path of the output JSON
        - cmd_list_path: list of the JSON file containing the API definitions
        - public: flag to choose whether to make server publicly accessible (from
        any network interface of the host)
        """
        self.serv = FedNovaServer(
            n_clients, in_json_path, out_json_path, update_type=update_type
        )
        with open(cmd_list_path) as f:
            self.API = json.load(f)

        self.ip = self.serv.getIP()
        # Webserver will accept connections from any address
        if public:
            self.ip_out = "0.0.0.0"
        else:
            self.ip_out = self.ip
        self.port = self.serv.getPort()
        self.ws_config = {
            "/": {
                "request.dispatch": cherrypy.dispatch.MethodDispatcher(),
                "tools.sessions.on": True,
            }
        }

        # Default messages:
        self.msg_ok = {"status": "SUCCESS", "msg": "", "params": {}}
        self.msg_ko = {"status": "FAILURE", "msg": "", "params": {}}

        # For synchronizing clients
        self.clients_requesting = []
        self.response_ready = threading.Event()
        self.clients_done_training = []
        self.ready_to_continue = threading.Event()

    def GET(self, *uri, **params):
        """
        GET
        ---
        ### Syntax

        - GET + `http://<server-ip>:<server-port>/dataset&pid=<client-pid>` - retrieve the
        data set portion assigned to the specific client PID.
        - GET + `http://<server-ip>:<server-port>/weights` - retrieve the most recent version
        of the global weights; the returned JSON-formatted string also contains the last
        update timestamp to help comparing with the current local version at the client

        FIXME: clients are identified by their PID
        """
        if len(uri) >= 1:
            if str(uri[0]) == "dataset" and "pid" in params:
                # Ensure the user is registered
                cli_pid = int(params["pid"])
                client_info = self.serv.searchClient("pid", cli_pid)
                if client_info == {}:
                    # Not found!:
                    raise cherrypy.HTTPError(
                        404, f"Client with PID = {cli_pid} not found"
                    )
                # Get the data set associated with the client
                cli_id = client_info["id"]

                # Dont send data until all clients have requested
                if cli_id not in self.clients_requesting:
                    self.clients_requesting.append(cli_id)

                if len(self.clients_requesting) < self.serv.n_clients:
                    self.response_ready.clear()
                else:
                    self.response_ready.set()
                # Wait here
                self.response_ready.wait()
                # If here, response_ready has been set, meaning that all clients
                # have been added to the "clients_requesting" list
                # Now clear this counter
                time.sleep(1)  # do this so that clients dont get stuck
                self.clients_requesting = []
                return json.dumps(self.serv.train_split_send[cli_id])

            elif str(uri[0]) == "dataset" and "id" in params:
                # Ensure the user is registered
                cli_id = int(params["id"])
                # Necessary to prevent wrong ID range
                client_info = self.serv.searchClient("id", cli_id)
                if client_info == {}:
                    # Not found!:
                    raise cherrypy.HTTPError(
                        404, f"Client with ID = {cli_id} not found"
                    )
                # Get the data set associated with the client
                return json.dumps(self.serv.train_split_send[cli_id])

            elif str(uri[0]) == "weights":
                # This is where clients are requesting the updated model params
                # We want this also to block until all clients have finished training
                # and also have sent in their data
                if len(self.clients_done_training) < self.serv.n_clients:
                    self.ready_to_continue.clear()
                else:
                    self.ready_to_continue.set()
                # Wait here until Event is set()
                self.ready_to_continue.wait()
                # If here, ready_to_continue has been set, i.e., all clients
                # have successfully registered (1st call) or updated their model
                # changes
                time.sleep(1)  # Leave some time for all cli to sync
                self.clients_done_training = []

                # We can be certain that the new model has been
                # updated by all client models
                return json.dumps(self.serv.model_params)
            else:
                raise cherrypy.HTTPError(
                    404, "Available commands:\n" + json.dumps(self.API["methods"][0])
                )
        else:
            # Default return value: dump of API definition JSON file
            return "Available commands:\n" + json.dumps(self.API["methods"][0])

    def POST(self, *uri, **params):
        """
        POST
        ---
        ### Syntax

        POST + http://<server-ip>:<server-port>/register - register to the server
        as a new client; the server will return the client ID.
        """
        body = json.loads(cherrypy.request.body.read())
        if DEBUG:
            print(body)
        if len(uri) >= 1:
            if str(uri[0]) == "register":
                ret_code_add = self.serv.addClient(body)
                if ret_code_add >= 0:
                    out = self.msg_ok.copy()
                    out["msg"] = f"Client {ret_code_add} was added"
                    # The new client ID is returned in the response message
                    out["params"]["id"] = ret_code_add
                    out["params"]["pid"] = body["pid"]
                    self.serv.saveStateJson()
                    cherrypy.response.status = 201
                    # Fix blocking before first iteration - registered client is ready for
                    # receiving weights!
                    self.clients_done_training.append(ret_code_add)
                    return json.dumps(out)
                elif ret_code_add == -1:
                    # Required number reached
                    out = self.msg_ko.copy()
                    out["msg"] = f"Unable to add client! Max. number of clients reached"
                    out["params"]["ret_code"] = ret_code_add  # Help in debug
                    cherrypy.response.status = 400
                    return json.dumps(out)
                elif ret_code_add == -2:
                    # Invalid info syntax
                    out = self.msg_ko.copy()
                    out["msg"] = f"Unable to add client! Invalid client information"
                    out["params"]["ret_code"] = ret_code_add  # Help in debug
                    cherrypy.response.status = 400
                    return json.dumps(out)
                elif ret_code_add == -3:
                    # Client already registered
                    out = self.msg_ko.copy()
                    out["msg"] = f"Unable to add client! Client already registered"
                    out["params"]["ret_code"] = ret_code_add  # Help in debug
                    cherrypy.response.status = 400
                    return json.dumps(out)

            if str(uri[0]) == "update" and "pid" in params:
                # TODO: Add possibility to update records - use HTTP code 200
                pass
            elif str(uri[0]) == "update" and "id" in params:
                pass

    def PUT(self, *uri, **params):
        """
        PUT
        ---
        ### Syntax

        http://<server-ip>:<server-port>/updated_params&id=<client_id> - upload the
        updated parameters (in the message body) to the server after a local iteration.
        """
        # Body should contain attribute "gradients", with list of
        # all values (columns are gradients)
        body = json.loads(cherrypy.request.body.read())
        gradients_mat = np.array(body["gradients"])
        if len(uri) >= 1:
            if str(uri[0]) == "updated_params" and "pid" in params:
                pid_cli = int(params["pid"])
                client_info = self.serv.searchClient("pid", pid_cli)
                if client_info == {}:
                    raise cherrypy.HTTPError(
                        404, f"Client with pid {pid_cli} not found!"
                    )
                id_cli = client_info["id"]
                ## every client calls addGradientMatrix
                res = self.serv.addGradientMatrix(gradients_mat, pid_cli)
                if res == 1:
                    # Success
                    out = self.msg_ok.copy()
                    out[
                        "msg"
                    ] = f"Updated gradients matrix for client with pid = {pid_cli}!"
                    cherrypy.response.status = 200
                    # a single client has contributed to the model
                    if id_cli not in self.clients_done_training:
                        self.clients_done_training.append(id_cli)

                    if len(self.clients_done_training) >= self.serv.n_clients:
                        ## only ONE client (the last client) thread updates all the weights
                        res2 = self.serv.updateWeights()

                    return json.dumps(out)
                else:
                    # Fail
                    out = self.msg_ko.copy()
                    out["msg"] = f"Client with pid = {pid_cli} does not exist!"
                    cherrypy.response.status = 400
            elif str(uri[0]) == "updated_params" and "id" in params:
                id_cli = int(params(["id"]))
                client_info = self.serv.searchClient("id", id_cli)
                if client_info == {}:
                    raise cherrypy.HTTPError(404, f"Client with id {id_cli} not found!")
                res = self.serv.addGradientMatrix(gradients_mat, id_cli, "id")
                if res == 1:
                    # Success
                    out = self.msg_ok.copy()
                    out[
                        "msg"
                    ] = f"Updated gradients matrix for client with pid = {id_cli}!"
                    cherrypy.response.status = 200
                    # a single client has contributed to the model
                    if id_cli not in self.clients_done_training:
                        self.clients_done_training.append(id_cli)
                    if len(self.clients_done_training) == self.serv.n_clients:
                        res2 = self.serv.updateWeights()
                    return json.dumps(out)
                else:
                    # Fail
                    out = self.msg_ko.copy()
                    out["msg"] = f"Client with pid = {id_cli} does not exist!"
                    cherrypy.response.status = 400


def main():
    if len(sys.argv) > 1:
        # Argv[1] contains path with client settings
        upd_type = sys.argv[1]
    else:
        # If no path is passed, default client is instantiated
        upd_type = "FedNova"
    print(f"Update type: {upd_type}")
    webserver = FedNovaWebServer(N_CLIENTS, update_type=upd_type)

    cherrypy.tree.mount(webserver, "/", webserver.ws_config)
    cherrypy.config.update(
        {"server.socket_host": webserver.ip_out, "server.socket_port": webserver.port}
    )
    cherrypy.engine.start()

    try:
        while True:
            time.sleep(1)
            if webserver.serv.n_update_iterations >= 100:
                cherrypy.engine.stop()
                # plot_current_model(webserver, pause=True, new_fig=True)
                total_mse_plot = np.array(webserver.serv.mse_per_global_iter)
                np.save(
                    "./mse_arrays/" + upd_type.lower() + "/" + str(os.getpid()),
                    total_mse_plot,
                )
                break
            # plot_current_model(webserver, pause=False)
    except KeyboardInterrupt:
        cherrypy.engine.stop()
        # plot_current_model(webserver, pause=True, new_fig=True)


if __name__ == "__main__":
    pass
