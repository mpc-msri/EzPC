# Introduction 
This repository contains code for EzPC - a language for secure machine learning.  
Project webpage: https://aka.ms/ezpc  
Paper link: https://www.microsoft.com/en-us/research/publication/ezpc-programmable-efficient-and-scalable-secure-two-party-computation-for-machine-learning/  

# Developer Guide/Wiki
Refer to the repository's Wiki section for EzPC's developer guide to how to code in EzPC.

# Requirements
- [] unzip (Use apt-get install unzip for this)
- [] opam (Ocaml package manager) ( wget https://raw.github.com/ocaml/opam/master/shell/opam_installer.sh -O - | sh -s /usr/local/bin)
- [] OCaml 4.05+
- [] ABY (Read the readme for ABY here for instructions: https://github.com/encryptogroup/ABY)
 	- Required packages for ABY:
		- [] g++
		- [] make
		- [] libgmp-dev
		- [] libglib2.0-dev
		- [] libssl-dev
- [] Stdint (opam install Stdint)
- [] Menhir (opam install menhir)

# Compiling ABY
- Build ABY using the instructions on their repo https://github.com/encryptogroup/ABY.
- Use the flag -DABY_BUILD_EXE=On when compiling using cmake to make sure that examples are also built.
- Ideally this should succeed and any problems faced in this should be resolved using the issues page of the ABY repo.
- But in setting up ABY ourselves at the time of this release, we have faced certain problems for which we give probable tips which worked for us.
	- On installing Boost version 1.66 and compiling the examples folder results in error like :
```
	/usr/bin/ld: cannot find -lBoost::system
	/usr/bin/ld: cannot find -lBoost::thread
``` 
	- This requires a small fix to replace instances of Boost::system with boost_system in the file $ABY/extern/ENCRYPTO_utils/CMakeLists.txt

# Compiling EzPC
```
>> opam switch 4.06.1 # Run "ocaml" to make sure that the switch was successful
# If the switch was not successful meaning that you are running ocaml 4.06.1
# then run - "eval `opam config env`"
>> opam install menhir
# If you get errors in menhir installation do:
# run - "opam depext conf-m4.1"
# then again run - "opam install menhir"
>> opam install StdInt
>> cd ezpc/
>> make # Use make debug to get a debug build
>> ./ezpc.sh --help # Shows usage
```

Sample compilation of ezpc program to cpp
```
>> ./ezpc.sh dot_product.ezpc
```

# Running EzPC with ABY (first time setup) 
```
>> # In the following, we are assuming that ABY was successfully installed using the instructions given in 
#	their repo at : https://github.com/encryptogroup/ABY. Hereafter, we refer to the ABY root directory using
#	the environment variable $ABY and the ezpc root directory using $EZPC.
>> cd $EZPC; ls
# This should produce following output:
# EzPC  README.md
>> cd $ABY ; ls 
# This should produce following output:
# CMakeLists.txt  Doxyfile  LICENSE  README.md  bin  build  cmake  extern  runtest_scr.sh  src
>> cd src/examples/
>> mkdir ezpc
>> echo "add_subdirectory(ezpc)" >> CMakeLists.txt
>> cd ../../
>> cp -r $EZPC/EzPC/ABY_example/* $ABY/src/examples/ezpc/
>> cd $ABY/src/examples/ezpc/
>> touch CMakeLists.txt
>> echo "add_executable(ezpc_test millionaire_prob_test.cpp common/millionaire_prob.cpp common/millionaire_prob.h)" >> CMakeLists.txt
>> echo "target_link_libraries(ezpc_test ABY::aby ENCRYPTO_utils::encrypto_utils)" >> CMakeLists.txt
>> cd $ABY/build/
>> cmake .. -DABY_BUILD_EXE=On
>> make
>> # Now that the sample ezpc generated ABY program is in place, run it using the following command
>> cd $ABY/build/bin
>> ./ezpc_test -r 0 >/dev/null 2>&1 & 
>> ./ezpc_test -r 1 
# For the default checked in program at $EZPC/EzPC/ABY_Example/ this should output 2.

```

# Running EzPC generated ABY file (after above first time setup)
```
>> # In the following, we assume you generated a .cpp file from some ezpc program for ABY backend. 
# We denote the full path of the file (including its name) by $TEST_FILE.
>> ls $TEST_FILE
# This should produce something like this: 
# ~/ezpc/EzPC/dot_product0.cpp
>> cp $TEST_FILE $ABY/src/examples/ezpc/common/millionaire_prob.cpp
>> cd $ABY/build
>> make
>> cd $ABY/build/bin
>> # Now run the ezpc_test executable 
>> ./ezpc_test -r 0 & 
>> ./ezpc_test -r 1 
```

# Running test suite for ezpc
- Running “make runtest” will compile all the \*.ezpc programs in the test_suite folder, place their compiled output in gen/ folder and then do a diff with the expected cpp output placed in test_suite/precompiled_output/ folder. It hardly takes 5 sec to run and at the end as a summary outputs the list of files for which the diff didn’t match. The diff is done using proper flags to ignore differences due to change in spaces, newlines etc. 
- If no diff is found in the summary, everything is good to go.
- If there are some files for which the diff is non-zero, please use your favourite diff tool to check whats changed and either fix compiler for this or do the following to take the new compiled program as the ground truth. 
	- Run “make forceCopyTestAll” to compile all the \*.ezpc files placed in the test_suite folder, and copy/overwrite them in the test_suite/precompiled_output/ directory.
	- Run “make forceCopyTestFile FCFILE=./test_suite/output.ezpc” to do the same for a particular file. This will do the same action as above, but for a particular file and not all the files in the test_suite directory. 


# Running EzPC generated .oc file
- Copy the .oc generated file to `EzPC/OBLIVC_example` folder. 
- Clone the oblivc project from https://github.com/samee/obliv-c and follow the instruction in their github page to install dependancies
- Create a folder called `ezpc` in `obliv-c/test/oblivc/` (inside the cloned project)
- Copy the contents of `EzPC/OBLIVC_example` folder to `obliv-c/test/oblivc/ezpc` folder
- Change directory to `obliv-c/test/oblivc/ezpc` and do `make`
- Run the generated `a.out` file in `obliv-c/test/oblivc/ezpc` with -r 0 and -r 1 parameters to run client and server respectively. (`./a.out -r 0` and `./a.out -r 1`)

# Express Installation/Running with Docker (easier)
## NOTE: The Dockerfile in our master branch is outdated. To use our latest Dockerfile, refer to https://github.com/mpc-msri/EzPC/issues/15. 
EzPC is shipped with a Dockerfile which can be used to generate Docker image for the project. This Dockerfile will yeild a Docker image with EzPC project set up by default and all dependencies including ABY will be automatically installed.<br />
Follow the following instructions to set up the Docker image:<br />
- Install Docker on your system (works with all major OS).
- Go to Docker directory of EzPC project and run `sudo docker build -t ezpc_image .` (in you CLI). If you want to build the image without using cache then add `--no-cache` option (not required unless you want to use a different commit of EzPC or ABY).
- The above step might take 30-40 minutes. After it is done, spin the docker image by using `sudo docker run -it ezpc_image`.
- You are now inside the docker container with EzPC project already setup and ready to use. (Refer the image down below to get an idea of directory structure of the project)
- To run your first EzPC test program, go to Docker directory inside your running Docker container and run `./compile_docker_example.sh`. This will ask you to either compile a binary op example or an arithmetic one. 
- After compilation is done, you are now ready to run the example.
- Open another session of CLI/terminal on your computer and run `docker container ls`.
- Copy the value in the field CONTAINER_ID and then run `docker exec -it CONTAINER_ID /bin/bash` (replacing CONTAINER_ID in the command with the copied value).
- This will open another session of the same Docker container.
- Now that you have 2 sessions open, you can run client on one session and server on the other.
- Enter Docker directory in both the sessions and run `./run_docker_example_server.sh` in one and `./run_docker_example_client.sh` in the other.
- This will execute the example and you will get result on client and server.
- In order to see the EzPC files which were used in this example, you can find them in EzPC/docker_test directory.
- NOTE: Any changes that you make inside Docker container will be lost as soon as you `exit` the container. In order to save your changes before exiting, use `docker commit CONTAINER_ID ezpc_image`.<br /><br />
- NOTE: If you face any issues, please refer to https://github.com/mpc-msri/EzPC/issues/59.

![alt-text](https://github.com/mayank0403/mayank0403.github.io/blob/master/images/EzPC-Docker-Structure%20(3)%20(1).jpg)
 
