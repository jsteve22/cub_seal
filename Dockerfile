FROM ubuntu

RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests wget vim  \ 
    libboost-all-dev ca-certificates software-properties-common \
    git sudo make gcc g++ valgrind cmake python3 

RUN mkdir /test
COPY seal_demo /test/seal_demo
WORKDIR /test
# COPY SEAL /test/SEAL
RUN git clone https://github.com/microsoft/SEAL.git
WORKDIR /test/SEAL
RUN cmake -S . -B build
RUN cmake --build build
RUN sudo cmake --install build

WORKDIR /test/seal_demo
RUN cmake -S . -B build
RUN cmake --build build
RUN make 

WORKDIR /test/seal_demo