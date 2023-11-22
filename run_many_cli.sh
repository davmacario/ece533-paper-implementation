#!/bin/bash
python ./main_serv.py FedNova &
python3 ./main_client.py ./client/settings/cli_info1.json &
python3 ./main_client.py ./client/settings/cli_info2.json &
python3 ./main_client.py ./client/settings/cli_info3.json &
python ./main_client.py ./client/settings/cli_info4.json &
