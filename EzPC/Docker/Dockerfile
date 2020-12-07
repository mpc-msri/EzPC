# Authors: Mayank Rathee.

# Copyright:
# Copyright (c) 2020 Microsoft Research
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#Download base image ubuntu
FROM ubuntu

#Adding the dependencies required for ABY and EzPC
RUN apt-get update && apt-get install -y apt-utils && apt-get install -y make
RUN apt-get install -y gcc
RUN apt-get install -y wget git-core
RUN apt-get install -y git
RUN apt-get install -y g++
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install cmake
RUN apt-get install -y libgmp-dev
RUN apt-get install -y libssl-dev
RUN apt-get install -y vim
RUN apt-get install -y libglib2.0-dev
RUN apt-get install -y ocaml
RUN apt-get install -y python3
RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install ipython
RUN pip3 install matplotlib
RUN apt-get install -y graphviz

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
RUN cd ABY-Installation-Scripts && git checkout docker && git pull origin docker
RUN mkdir ABY-latest
RUN cp ABY-Installation-Scripts/* ABY-latest/.
WORKDIR /ezpc-workdir/ABY-latest
RUN ["bash", "install.sh"]
RUN ["bash", "install-cmake.sh"]
#RUN ["bash", "install-after-cmake-check.sh"]
#To stick to a commit that works in ABY
RUN cd ../ABY-Installation-Scripts && git pull
RUN cp ../ABY-Installation-Scripts/* .

WORKDIR /ezpc-workdir/ABY-latest/ABY
RUN git submodule update --init --recursive
RUN git checkout 8aa003c2e145c6d43b6ec73ef75618be43951b1d
RUN git checkout 08baa853de76a9070cb8ed8d41e96569776e4773 -- CMakeLists.txt 
RUN git checkout -b docker_specific_branch
RUN cd extern/ENCRYPTO_utils/ && git checkout 11cd5efa9be6d506c967d6e6835fa239e47c7207
#WORKDIR /ezpc-workdir/ABY-latest
#RUN ["bash", "install-after-cmake-check.sh"]
WORKDIR /ezpc-workdir/ABY-latest/ABY
RUN cd extern/OTExtension/ && git checkout e4bcb68f9e7eb8753bc6a6ab1c2e3b70ad40e54f
WORKDIR /ezpc-workdir/ABY-latest
RUN ["bash", "install-after-cmake-check.sh"]
WORKDIR /ezpc-workdir/ABY-latest/ABY

#29th April 2019 commit
#RUN git checkout -b docker_specific_branch d4ca22996967ba076b6ca378e231d72f9c35cf51
#Add docker-test dir and copy ezpc.h and other files
RUN cd src/examples/ && mkdir docker-test && echo "add_subdirectory(docker-test)" >> CMakeLists.txt

#Pull EzPC source from GitHub.
WORKDIR /ezpc-workdir
RUN git clone https://github.com/mpc-msri/EzPC
RUN cd EzPC && git checkout random_forest

WORKDIR /ezpc-workdir/ABY-latest/ABY
RUN cd src/examples/docker-test && cp -r ../../../../../EzPC/EzPC/EzPC/ABY_example/* .

WORKDIR /ezpc-workdir/EzPC/EzPC/EzPC
#RUN git checkout -b docker_specific_branch c8b45fd28fb41c46d3054f4fbe7c1b0dddb87ccd

#EzPC specific ocaml packages
RUN opam init --disable-sandboxing 
RUN opam switch create 4.06.1 
RUN opam switch 4.06.1
RUN opam install -y depext
RUN eval $(opam env) && \
    `opam depext conf-m4.1` && \
    opam install -y ocamlbuild && \
    opam install -y Stdint && \
    opam install -y menhir

RUN git fetch && git pull
RUN git checkout -b docker_specific_branch 8896358f139c5bf0c7c9519c69e259d56e094914 
RUN eval $(opam env) && make
RUN echo 'eval "$(opam env)"' >> /root/.bashrc

WORKDIR /ezpc-workdir/EzPC/EzPC
