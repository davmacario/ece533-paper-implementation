#!/bin/bash

run_fednova() {
    python ./main_serv.py FedNova &
    pid_a=$!

    python ./main_client.py ./client/settings/cli_info1.json &
    pid_b=$! 

    python ./main_client.py ./client/settings/cli_info2.json &
    pid_c=$!

    python ./main_client.py ./client/settings/cli_info3.json &
    pid_d=$!

    python ./main_client.py ./client/settings/cli_info4.json &
    pid_e=$!

    # Wait for all programs to finish
    wait $pid_a
    wait $pid_b
    wait $pid_c
    wait $pid_d
    wait $pid_e
}

run_fedavg() {
    python ./main_serv.py FedAvg &
    pid_a=$!

    python ./main_client.py ./client/settings/cli_info1.json &
    pid_b=$! 

    python ./main_client.py ./client/settings/cli_info2.json &
    pid_c=$!

    python ./main_client.py ./client/settings/cli_info3.json &
    pid_d=$!

    python ./main_client.py ./client/settings/cli_info4.json &
    pid_e=$!

    # Wait for all programs to finish
    wait $pid_a
    wait $pid_b
    wait $pid_c
    wait $pid_d
    wait $pid_e
}

# Number of times to repeat
num_repeats=10  # Change this number to your desired repetition count

# Loop to run the programs multiple times
for ((i = 0; i < num_repeats; i++)); do
    echo "Running iteration $i fednova"
    run_fednova
done

echo "done running fednova tests" 

for ((i = 0; i < num_repeats; i++)); do
    echo "Running iteration $i fedavg"
    run_fedavg
done

echo "All iterations completed."
