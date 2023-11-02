import cherrypy
from datetime import datetime
import json
import numpy as np
import os
import sys
import time

from .config import *
from model import CurveFitter


class FedNovaServer:
    """
    FedNovaServer class
    """

    def __init__(
        self,
        n_clients: int,
        in_json_path: str,
        out_json_path: str,
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

        ### Exposed API

        The server exposes a HTTP-based REST API, allowing for GET, POST
        and PUT requests.

        - GET: used for fetching the data set as a client (only when all
        clients have successfully registered)
        - POST: used for registration, providing the information in JSON
        format
        - PUT: used for uploading the local model parameter at each
        algorithm iteration
        """
        if n_clients <= 0:
            raise ValueError("The number of clients needs to be >= 1")

        self.n_clients = n_clients
        self.cli_map_PID = [] # elem. i contains PID of client i - append as they come
        self.cli_registered = [False] * n_clients  # elem. i true if client i registered

        # Import the server information dict
        self._server_info = json.load(open(in_json_path))
        self.updateTimestamp()
        self._out_info_path = out_json_path
        self.saveStateJson()

        # Keys of the client dictionary that need to be provided at registration
        self._cli_params = ["PID", "capabilities"]  # TODO: decide the syntax

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
        (needs to be included in self._cli_params)
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
        return elem

    def addClient(self, cl_info: dict) -> int:
        """
        Add a new client - registration.
        The new client is assigned a unique ID, that will overwrite its
        current one, if present.

        The return code will be the ID of the new client if success, -1
        if no free IDs are left, -2 if the client info is in the wrong
        format.
        """
        new_id = self.getFreeID()

        # Check for info validity
        if not self.checkValidClient(self, cl_info):
            new_id = -2

        # TODO: add check to prevent duplicate clients registering at server

        if new_id > -1:
            # Set the ID as occupied
            self.cli_registered[new_id] = True
            # Only keep the keys specified in _cli_params:
            new_cli_info = {}
            self.cli_map_PID.append(cl_info["PID"])
            for k in self._cli_params:
                new_cli_info[k] = cl_info[k]
            new_cli_info["id"] = new_id
            new_cli_info["last_update"] = self.updateTimestamp()
            self._server_info["clients"].append(new_cli_info)
        return new_id

    def getFreeID(self) -> int:
        """
        Return the first unused ID that can be assigned.

        If the return value is -1, there are no more free IDs.
        """
        i = 0
        while i < self.n_clients:
            if not self.cli_registered[i]:
                return i
        return -1

    def checkValidClient(self, cl_info: dict) -> int:
        """
        Check whether the new client information is in the valid format,
        i.e., all the required keys are present.

        ### Output parameters
        - 1 if good syntax
        - 0 if missing keys
        """
        for k in self._cli_params:
            if k not in list(cl_info.keys()):
                return 0
        return 1


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
    ):
        self.serv = FedNovaServer(n_clients, in_json_path, out_json_path)
        with open(cmd_list_path) as f:
            self.api = json.load(f)

        # Default messages:
        self.msg_ok = {"status": "SUCCESS", "msg": "", "params": {}}
        self.msg_ko = {"status": "FAILURE", "msg": "", "params": {}}

    def GET(self, *uri, **params):
        """
        GET
        ---
        ### Syntax

        GET + http://<server-ip>:<server-port>/dataset&id=<client-id> - retrieve the
        data set portion assigned to the specific client ID.
        FIXME: clients are identified by their PID
        FIXME: clients are identified by their PID
        """
        if len(uri) >= 1:
            if str(uri[0]) == "dataset" and "id" in params:
                # Ensure the user is registered
                cli = int(params["id"])
                if self.serv.searchClient("id", cli) == {}:
                    # Not found!:
                    raise cherrypy.HTTPError(f"Client {cli}")
                # Get the data set associated with the client
                # TODO

    def POST(self, *uri, **params):
        """
        POST
        ---
        ### Syntax

        POST + http://<server-ip>:<server-port>/register - register to the server
        as a new client; the server will return the client ID.
        """
        body = json.loads(cherrypy.request.body.read())
        if len(uri) >= 1:
            if str(uri[0]) == "register":
                ret_code_add = self.serv.addClient(body)
                if ret_code_add >= 0:
                    out = self.msg_ok.copy()
                    out["msg"] = f"Client {ret_code_add} was added"
                    # The new client ID is returned in the response message
                    out["params"]["id"] = ret_code_add
                    self.serv.saveStateJson()
                    cherrypy.response.status = 201
                    return json.dumps(out)
                else:
                    # Failed to add client
                    out = self.msg_ko.copy()
                    out["msg"] = f"Unable to add client!"
                    cherrypy.response.status = 400
                    return json.dumps(out)

    def PUT(self, *uri, **params):
        """
        PUT
        ---
        ### Syntax

        http://<server-ip>:<server-port>/updated_params&id=<client_id> - upload the
        updated parameters (in the message body) to the server after a local iteration.
        """
        body = json.loads(cherrypy.request.body.read())
        if len(uri) >= 1:
            if str(uri[0]) == "updated_params" and "id" in params:
                


if __name__ == "__main__":
    pass
