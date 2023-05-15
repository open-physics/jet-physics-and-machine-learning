# syntax=docker/dockerfile:1

FROM ubuntu:22.04

ARG WORK_HOME=/usr/local/jet-physics-and-machine-learning
# python, gfortran, vim
RUN apt update \
  && apt install -y software-properties-common \
  && add-apt-repository ppa:deadsnakes/ppa \
  && apt update \
  && apt install -y gfortran vim \
  && apt install -y python3.10 python3.10-venv \
  && ln -s /usr/bin/python3 /usr/bin/python \
  && apt install -y python3-pip

WORKDIR ${WORK_HOME}
COPY pyproject.toml .
RUN pip3 install pip-tools
COPY requirements.txt .
RUN pip-compile --extra=dev --extra=linux pyproject.toml --upgrade --resolver=backtracking
RUN pip3 install --no-cache-dir -r requirements.txt
