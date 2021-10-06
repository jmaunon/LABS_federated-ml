# Flower.dev 1st test: Official PyTorch Example

## Description of experiment

The insights extracted from the experiment will be stored in the following [document](https://docs.google.com/document/d/1YoilYXbeijTbBq17qCPyiUJAZJHayoCQv9b_MVe4Lco/edit?usp=sharing).

- Train a model with the data distributed between two silos.
- Deploy the experiment with Docker.
- Take an [official example](https://flower.dev/docs/example-pytorch-from-centralized-to-federated.html) to test the published features.

## Requirements to run the experiment

- Git (used: version 2.30.1 (Apple Git-130))
- Docker (used: version 20.10.8, build 3967b7d).
- Docker Compose (used: version v2.0.0).

## Run the experient

1. Run a terminal and go to the current folder `flower/test_1_example`.
2. Run `make` command and wait for debug prints.
3. The expected output has been persisted in [`execution-with-aggregatos.log`](./execution-with-aggregatos.log) file.