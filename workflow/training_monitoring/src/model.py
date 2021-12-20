from typing import Generator, Tuple
import torch
import torch as Tensor
import torch.nn as nn

input_size = 1 # input features
output_size = 1 # output features

class linearRegression(nn.Module):
    def __init__(self) -> None:
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        return out

def train(
    iterations: int,
    model: linearRegression,
    optim: torch.optim.Adam,
    data: Tensor, 
    target: Tensor
) -> Generator[int, float, None]:
    for i in range(iterations):
        optim.zero_grad()
        output = model(data)
        loss = nn.functional.mse_loss(output, target)
        loss_item = loss.item()

        loss.backward()
        optim.step()
        yield i, float(loss_item)

def test(
    model: linearRegression,
    data: Tensor, 
    target: Tensor
) -> float:
    """Validate the network on the entire test set."""
    # Evaluate the network
    model.eval()
    output = model(data)
    return nn.functional.mse_loss(output, target)