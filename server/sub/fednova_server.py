import cherrypy
from datetime import datetime
import json
import numpy as np
import os
import sys
import time

from config import *


class FedNovaServer:
    """
    FedNovaServer class
    """

    def __init__(
        self,
        n_clients: int,
        in_json_path: str = os.path.join(
            os.path.getdirname(__file__), "serv_info.json"
        ),
        out_json_path: str = os.path.join(
            os.path.getdirname(__file__), "serv_info_updated.json"
        ),
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
        self.cli_registered = [False] * n_clients  # elem. i true if client i registered

        # Import the server information dict
        self._server_info = json.load(open(in_json_path))
        self._last_update_info = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._server_info["last_update"] = self._last_update_info
        self._out_info_path = out_json_path

        self._cli_params = ["ip", "port", "capabilities"]  # TODO: decide the syntax

    def saveStateJson(self) -> int:
        """
        Save the current server state as JSON at the output path passed at
        instantiation.

        The function returns 1 for success, 0 for failure.
        """
        try:
            with open(self._out_info_path, "w") as f:
                json.dump(self._server_info, f)
                return 1
        except:
            return 0

    def searchClient(self, cl_id: int) -> dict:
        """
        Search for a registered client given its ID.
        """
        pass

    def addClient(self, cl_info: dict) -> int:
        """
        Add a new client - registration.
        The new client is assigned a unique ID.

        The return code will be the ID of the new client if success, -1 
        if no free IDs are left, -2 if the client info is in the wrong \
        format.
        """
        new_id = self.getFreeID()

        # Check for info validity
        if not self.checkValidClient(self, cl_info):
            new_id = -2

        if new_id > -1:
            # Set the ID as occupied
            self.cli_registered[new_id] = True

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
        Check whether the new client information is in the valid format.

        ### Output parameters
        - 1 if good syntax
        - 0 if missing keys
        """
        for k in self._cli_params:
            if k not in list(cl_info.keys()):
                return 0
        return 1
