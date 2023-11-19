#!/bin/bash
python ./main_serv.py FedNova &
pid_serv=$!

python3 ./main_client.py ./client/settings/cli_info1.json &
pid_a=$!

python3 ./main_client.py ./client/settings/cli_info2.json &
pid_b=$!

python3 ./main_client.py ./client/settings/cli_info3.json &
pid_c=$!

python ./main_client.py ./client/settings/cli_info4.json &
pid_d=$!

wait $pid_serv
wait $pid_a
wait $pid_b
wait $pid_c
wait $pid_d

