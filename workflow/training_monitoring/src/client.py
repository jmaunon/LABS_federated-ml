import os
from collections import OrderedDict
from typing import Tuple, List, Dict

import torch
import numpy as np
import flwr as fl
from flwr.common import Scalar

np.random.seed(4567)

import model
from logger import Logger

# Init logger
job_id: str = os.getenv('JOB_ID')
node_id: str = os.getenv('NODE_ID')
logger: Logger = Logger(job_id, node_id)

USE_FEDBN: bool = True
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Flower Client
class Client(fl.client.NumPyClient):
    def __init__(
        self, 
        model: model.linearRegression,
        optim: torch.optim.Adam,
        data_train: torch.Tensor,
        target_train: torch.Tensor,
        data_test: torch.Tensor,
        target_test: torch.Tensor
    ) -> None:
        self.model = model
        self.optim = optim
        self.data_train = data_train
        self.target_train = target_train
        self.data_test = data_test
        self.target_test = target_test

    def get_parameters(self) -> List[np.ndarray]:
        self.model.train()
        if USE_FEDBN:
            # Return model parameters as a list of NumPy ndarrays, excluding parameters of BN layers when using FedBN
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
            # Return model parameters as a list of NumPy ndarrays
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.model.train()
        if USE_FEDBN:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)

        # Set number of iteration to 1 becouse we can pass whole dataset to the model 
        result_generator: Generator[int, float] = model.train(1, self.model, self.optim, self.data_train, self.target_train)
        for epoch, loss in result_generator:
            logger.log("loss", loss, {"epoch": epoch})

        return self.get_parameters(), 1, {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss_tensor = model.test(self.model, self.data_test, self.target_test)
        loss = float(loss_tensor)

        return loss, len(self.data_test), {}

def generate_data(): 
    """1.1. Generate synthetic dataset"""
    n_samples = 500
    x_ = np.sort(np.random.random(size=n_samples))

    # create the true signal to be learned
    coeff_ = 2
    bias_ = 1
    y_ = coeff_*x_ + bias_

    # y = a*x + b with noise
    wh_noise = np.random.normal(scale=10, size=n_samples)
    y = y_  * wh_noise

    """1.2. Split the dataset in training and validation"""
    val_pts = int(len(y)*0.25)

    sample_idx = np.random.choice(range(0, len(y)), len(y))
    training_idx, val_idx = sample_idx[val_pts:], sample_idx[:val_pts]

    x_val, x = x_[val_idx], x_[training_idx]
    y_val, y = y[val_idx], y[training_idx]

    return x, y, x_val, y_val

def load_train_data(x, y) -> Tuple[torch.Tensor, torch.Tensor]:
    """1.3. Create PyTorch Dataset"""
    return torch.from_numpy(x.reshape(-1,1).astype('float32')), torch.from_numpy(y.reshape(-1,1).astype('float32'))

def load_test_data(x_val, y_val) -> Tuple[torch.Tensor, torch.Tensor]:
    """1.3. Create PyTorch Dataset"""
    return torch.from_numpy(x_val.reshape(-1,1).astype('float32')), torch.from_numpy(y_val.reshape(-1,1).astype('float32'))

def main():
    # Load data
    x, y, x_val, y_val = generate_data()
    x_train, y_train = load_train_data(x, y)
    x_test, y_test = load_test_data(x_val, y_val)

    # Load model
    trained_model = model.linearRegression().to(DEVICE).train()

    # Get initial parameters and intance optim
    torch_model = model.linearRegression()
    params = torch_model.parameters()
    optim = torch.optim.Adam(params=params, lr=0.1)

    # Start client
    client = Client(trained_model, optim, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("server:8080", client)

if __name__ == "__main__":
    main()