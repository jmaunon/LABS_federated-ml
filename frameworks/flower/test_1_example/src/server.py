from typing import List, Optional, Tuple, Dict

import flwr as fl
from flwr.common import (
    EvaluateRes, 
    FitRes, 
    Parameters, 
    Scalar
)
from flwr.server.client_proxy import ClientProxy

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)
        
        """ To get weights as List[np.ndarray] call parameters_to_weights function"""
        # from flwr.common import parameters_to_weights
        # weights = parameters_to_weights(aggregated_parameters[0])

        return aggregated_parameters
    
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)

        # Call aggregate_evaluate from base class (FedAvg)
        loss_aggregated = super().aggregate_evaluate(rnd, results, failures)

        print(f"Test: Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")
        print(f"Test: Round {rnd} loss aggregated from client results: {loss_aggregated[0]}")

        return loss_aggregated


if __name__ == "__main__":
    strategy = AggregateCustomMetricStrategy()
    fl.server.start_server(config={"num_rounds": 3}, strategy=strategy)