#Download base image ubuntu
FROM ubuntu

#Adding the dependencies required for ABY and EzPC
RUN apt-get update && apt-get install -y apt-utils && apt-get install -y make
RUN apt-get install -y gcc
RUN apt-get install -y wget git-core
RUN apt-get install -y git
RUN apt-get install -y g++
RUN apt-get install -y cmake
RUN apt-get install -y libgmp-dev
RUN apt-get install -y libssl-dev
RUN apt-get install -y vim
RUN apt-get install -y libglib2.0-dev
RUN apt-get install -y ocaml
#Get opam
RUN apt-get install -y opam

#Adding the git cloned EzPC directory to the dockerimage
RUN mkdir -p /ezpc-workdir
#Setting workdir means that after every RUN, the directory resets to workdir
WORKDIR /ezpc-workdir

RUN cp /usr/lib/x86_64-linux-gnu/libcrypto.so /usr/local/lib/libcrypto.so
RUN cp /usr/lib/x86_64-linux-gnu/libcrypto.so.1.1 /usr/local/lib/libcrypto.so.1.1

#Clone latest ABY installation scripts and install ABY
RUN git clone https://github.com/mayank0403/ABY-Installation-Scripts
RUN cd ABY-Installation-Scripts && git checkout docker
RUN mkdir ABY-latest
RUN cp ABY-Installation-Scripts/* ABY-latest/.
WORKDIR /ezpc-workdir/ABY-latest
RUN ["bash", "install.sh"]
RUN ["bash", "install-cmake.sh"]
RUN ["bash", "install-after-cmake-check.sh"]
#To stick to a commit that works in ABY
WORKDIR /ezpc-workdir/ABY-latest/ABY
RUN git checkout -b docker_specific_branch 8aa003c2e145c6d43b6ec73ef75618be43951b1d
#29th April 2019 commit
#RUN git checkout -b docker_specific_branch d4ca22996967ba076b6ca378e231d72f9c35cf51
#Add docker-test dir and copy ezpc.h and other files
RUN cd src/examples/ && mkdir docker-test && echo "add_subdirectory(docker-test)" >> CMakeLists.txt

#Pull EzPC source from GitHub.
WORKDIR /ezpc-workdir
RUN git clone https://github.com/mpc-msri/EzPC

WORKDIR /ezpc-workdir/ABY-latest/ABY
RUN cd src/examples/docker-test && cp -r ../../../../../EzPC/EzPC/ABY_example/* .

WORKDIR /ezpc-workdir/EzPC
RUN git checkout -b docker_specific_branch c0ef6819a5e9bf25c36df4dc5181f2bdeac7a9cc

#EzPC specific ocaml packages
RUN opam init
RUN opam switch 4.06.1
RUN eval $(opam config env) && \
    opam depext conf-m4.1 && \
    opam install -y ocamlbuild && \
    opam install -y Stdint && \
    opam install -y menhir && \
    cd EzPC && \
    make
RUN echo 'eval "$(opam config env)"' >> /root/.bashrc

