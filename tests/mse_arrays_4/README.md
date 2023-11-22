these tests were done with the following parameters:
clients: 4
    - client 1:     "capabilities": {
                    "n_epochs": 4,
                    "batch_size": 6,
                    "cli_class": 8,
                    "learning_rate": 0.0001
                }
    - client 2:     "capabilities": {
                    "n_epochs": 40,
                    "batch_size": 6,
                    "cli_class": 8,
                    "learning_rate": 0.0001
                }
    - client 3:     "capabilities": {
                    "n_epochs": 100,
                    "batch_size": 6,
                    "cli_class": 8,
                    "learning_rate": 0.0001
                }
    - client 4:     "capabilities": {
                    "n_epochs": 20,
                    "batch_size": 6,
                    "cli_class": 8,
                    "learning_rate": 0.0001
                }

Each folder contains 10 iterations of tests with 100 communication rounds each