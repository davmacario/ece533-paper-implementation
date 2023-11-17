these tests were done with the following parameters:
clients: 4
    - client 1:     "capabilities": {
                    "n_epochs": 1,
                    "batch_size": 6,
                    "cli_class": 5,
                    "learning_rate": 0.0001
                }
    - client 2:     "capabilities": {
                    "n_epochs": 2,
                    "batch_size": 6,
                    "cli_class": 5,
                    "learning_rate": 0.0001
                }
    - client 3:     "capabilities": {
                    "n_epochs": 3,
                    "batch_size": 6,
                    "cli_class": 5,
                    "learning_rate": 0.0001
                }
    - client 4:     "capabilities": {
                    "n_epochs": 4,
                    "batch_size": 6,
                    "cli_class": 5,
                    "learning_rate": 0.0001
                }

Each folder contains 20 iterations of tests with 150 communication rounds each

The goal of this test was to see if the global learning rate was affected by smaller n_epochs