import os
from typing import List, Optional, Tuple, Dict

import flwr as fl
from flwr.common import (
    EvaluateRes, 
    FitRes, 
    Parameters, 
    Scalar,
    # parameters_to_weights
)
from flwr.server.client_proxy import ClientProxy

from logger import Logger

# Init logger
job_id: str = os.getenv('JOB_ID')
node_id: str = os.getenv('NODE_ID')
logger: Logger = Logger(job_id, node_id)

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    @logger.logExecutionTime
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)
        
        """ To get weights as List[np.ndarray] call parameters_to_weights function"""
        
        # weights = parameters_to_weights(aggregated_parameters[0])

        return aggregated_parameters
    
    @logger.logExecutionTime
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Call aggregate_evaluate from base class (FedAvg)
        loss_aggregated = super().aggregate_evaluate(rnd, results, failures)

        # Send to metric to logger
        loss = float(loss_aggregated[0])
        logger.log("aggregated_loss_test", loss, {"rounds": rnd})

        return loss_aggregated


if __name__ == "__main__":
    strategy = AggregateCustomMetricStrategy()
    fl.server.start_server(config={"num_rounds": 10}, strategy=strategy)