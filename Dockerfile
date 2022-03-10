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

RUN mkdir mukham
ADD ./src mukham/src
ADD ./imgui mukham/imgui
ADD ./implot mukham/implot
ADD ./dlib mukham/dlib
ADD ./spdlog mukham/spdlog
ADD ./test mukham/test
ADD ./tvm mukham/tvm
ADD ./assets mukham/assets
ADD ./models mukham/models
ADD ./CMakeLists.txt mukham/CMakeLists.txt

RUN mkdir mukham/build

ADD ./build.sh mukham/build/build.sh

WORKDIR mukham/build

CMD ["/bin/bash", "-c", "./build.sh"]


