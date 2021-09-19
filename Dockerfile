FROM ubuntu:20.04

RUN apt-get update

RUN apt-get install -y git

ADD . /mukham/

RUN apt-get install cmake g++ libsdl2-dev libopencv-dev