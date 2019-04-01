FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
LABEL maintainer="Guillaume Raille <guillaume.raille@epfl.ch>"

# Install Requirements
RUN apt-get update && apt-get install -y build-essential \
                                         checkinstall \
                                         libreadline-gplv2-dev \
                                         libncursesw5-dev \
                                         libssl-dev \
                                         libsqlite3-dev \
                                         tk-dev \
                                         libgdbm-dev \
                                         libc6-dev \
                                         libbz2-dev \
                                         zlib1g-dev \
                                         openssl \
                                         libffi-dev \
                                         python3-dev \
                                         python3-setuptools \
                                         wget \
                                         git \
                                         default-jre \
                                         ipython \
                                         ipython-notebook

# Install python
RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update
RUN apt install -y python3.7
RUN apt-get install -y python3-pip
RUN python3.7 -m pip install --upgrade pip

# Install Package dependencies for the project
RUN python3.7 -m pip install torch==1.0.1.post2
RUN python3.7 -m pip install torchtext==0.3.1
RUN python3.7 -m pip install tensorboardX==1.6
RUN python3.7 -m pip install numpy==1.16.2  
RUN python3.7 -m pip install spacy==2.1.0
RUN python3.7 -m spacy download en_core_web_sm
RUN python3.7 -m spacy download en_core_web_lg
RUN python3.7 -m pip install pyhocon==0.3.51
RUN python3.7 -m pip install ipdb

ENV PYTHONPATH /code
ENV DATA /data

WORKDIR  /code

ENV TZ=Europe/Zurich
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# FIX local problem
RUN apt-get install -y locales
RUN locale-gen "en_US.UTF-8"
ENV LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 LANGUAGE=en_US.UTF-8

# Switch to local user
# NEEDS THE FOLLOWING ARGS: UNAME, UID, GID.
ARG UNAME
ARG UID
ARG GID
RUN groupadd -f -g $GID $UNAME
RUN useradd -m -u $UID -g $GID -s /bin/bash $UNAME
RUN chown $UNAME /code
USER $UNAME

# === USEFUL DOCKER COMMAND ===
# BUILD
# docker build --build-arg UNAME=$(whoami) --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t edit-transformer:0.0.2 .
# RUN INTERACTIVE MODE
# docker run -it --rm -v $(pwd):/code -v $(pwd)/../edit-transformer-data:/data -e INTERACTIVE_ENVIRONMENT=True edit-transformer:0.0.2 /bin/bash
# docker run -it --rm -v $(pwd):/code -v $(pwd)/../edit-transformer-data:/data -e INTERACTIVE_ENVIRONMENT=True edit-transformer:0.0.2 python3.7 edit_transformer/generation.py
# RUN TRAINING BACKGROUND MODE
# docker run -d --rm -v $(pwd):/code -v $(pwd)/../edit-transformer-data:/data -e CUDA_VISIBLE_DEVICES=0 edit-transformer:0.0.2 python3.7 -u edit_transformer/training.py
# docker run -d --rm -v $(pwd):/code -v $(pwd)/../edit-transformer-data:/data -e CUDA_VISIBLE_DEVICES=1 edit-transformer:0.0.2 python3.7 -u edit_transformer/training.py
