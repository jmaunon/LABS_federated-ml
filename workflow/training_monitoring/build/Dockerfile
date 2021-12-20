FROM python:3.7-slim

WORKDIR /flower

RUN pip install flwr torch torchvision requests --no-cache-dir

EXPOSE 8080

ENTRYPOINT ["tail", "-f", "/dev/null"]
