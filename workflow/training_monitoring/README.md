# Flower.dev 3nd test: Training monitoring with Prometheus

## Description of experiment

The insights extracted from the experiment will be stored in the following [document](https://docs.google.com/document/d/1VIexWsHAtkV3dgBt2XMepSbB8IynaVcoNijvYCviJ54/edit#).

- Train a model with the data distributed between two silos.
- Deploy the experiment with Docker.
- Connect with Prometheus using [PushGateway](https://github.com/prometheus/pushgateway/).

## Requirements to run the experiment

- Git (used: version 2.30.1 (Apple Git-130))
- Docker (used: version 20.10.8, build 3967b7d).
- Docker Compose (used: version v2.0.0).

## Run the experient

1. Run a terminal and go to the current folder `workflow/training_monitoring/`.
2. Run `make` command and wait for debug prints.