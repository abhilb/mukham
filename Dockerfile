FROM ubuntu:20.04

RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y git cmake g++ \
    libsdl2-dev \
    libopencv-dev \
    xterm python3 \
    python3-dev \
    python3-setuptools \
    gcc libtinfo-dev \
    zlib1g-dev \
    build-essential \
    libedit-dev \
    libxml2-dev \
    libopenblas-dev \
    liblapack-dev
ADD . /mukham/