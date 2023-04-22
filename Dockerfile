# syntax=docker/dockerfile:1

FROM ubuntu:22.04
# python, gfortran, vim
RUN apt update \
  && apt install -y software-properties-common \
  && add-apt-repository ppa:deadsnakes/ppa \
  && apt update \
  && apt install -y gfortran vim \
  && apt install -y python3.10 python3.10-venv \
  && ln -s /usr/bin/python3 /usr/bin/python \
  && apt install -y python3-pip

# RUN apt update && apt install -y gcc clang clang-tools cmake python3.10 python3-pip
# RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# RUN git clone https://github.com/Homebrew/brew ~/.linuxbrew/Homebrew \
# && mkdir ~/.linuxbrew/bin \
# && ln -s ../Homebrew/bin/brew ~/.linuxbrew/bin \
# && eval $(~/.linuxbrew/bin/brew shellenv) \
# && brew --version

# RUN brew install pyenv

WORKDIR /jet-physics
# COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
RUN pip3 install pip-tools
RUN pip-compile --extra=dev pyproject.toml --resolver=backtracking
RUN pip3 install --no-cache-dir -r requirements.txt
