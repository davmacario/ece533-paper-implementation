program to run client node

run multiple instances of the program to have multiple nodes

1. Each client should first send an http POST to the server registering itself as a node
2. The server may want to wait until it has found >4 nodes to start distributing data
3. Once client has registered it will send a blocking GET for data, waiting for server to start training. 
4. Client will then at some point in the future receive a block of data from the server. Client will use this data to update its own model parameters.
5. Client will go through all of the data it receives and send POST the updated weights back to the server along with the amount of time it took to learn.
6. repeat this process until client requests data and receives instead of data a stop command
