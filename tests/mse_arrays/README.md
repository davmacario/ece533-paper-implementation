these tests were done with the following parameters:
clients: 4
    - client 1:     "capabilities": {
                    "n_epochs": 10,
                    "batch_size": 64,
                    "cli_class": 5,
                    "learning_rate": 0.0001
                }
    - client 2:     "capabilities": {
                    "n_epochs": 15,
                    "batch_size": 12,
                    "cli_class": 8,
                    "learning_rate": 0.0001
                }
    - client 3:     "capabilities": {
                    "n_epochs": 25,
                    "batch_size": 24,
                    "cli_class": 3,
                    "learning_rate": 0.0001
                }
    - client 4:     "capabilities": {
                    "n_epochs": 50,
                    "batch_size": 12,
                    "cli_class": 2,
                    "learning_rate": 0.0001
                }

Each np array contains 50 iterations of tests with 100 communication rounds each
