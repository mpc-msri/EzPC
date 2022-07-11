# Introduction 
This repository contains code for EzPC - a language for secure machine learning.  
Project webpage: https://aka.ms/ezpc  
Paper link: https://www.microsoft.com/en-us/research/publication/ezpc-programmable-efficient-and-scalable-secure-two-party-computation-for-machine-learning/  

# Developer Guide/Wiki
Refer to the repository's Wiki section for EzPC's developer guide to how to code in EzPC.

# Setup
You should have run ```setup_env_and_build.sh```. It will automatically build EzPC and also clone and build ABY.

# Compiling an EzPC file
Sample compilation of ezpc program to cpp
```
cd EzPC
eval `opam config env`   		 # Load ocaml environment
./ezpc.sh --help          		 # Shows usage
./ezpc.sh dot_product.ezpc --bitlen 32	 # Default backend is ABY to change add [ --codegen EMP/SCI/CPP/OBLIVC/PORTHOS/CPPRING ]
```
This generates a dot_product0.cpp file.

# Running EzPC with ABY
```
./compile_aby.sh dot_product0.cpp 
# Running the compiled code:
# In one terminal:
./dot_product0 -r 0
# In another terminal:
./dot_product0 -r 1 

# For the default checked in program at $EZPC/EzPC/ABY_Example/ this should output 2.
```
# Running EzPC with EMP
```
./compile_emp.sh dot_product0.cpp 
# Running the compiled code:
# In one terminal(Client):
./dot_product0 1 12345 [optional ip address]
# In another terminal(Server):
./dot_product0 2 12345 [optional ip address]

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

# Manually compiling EzPC [Developers]
If you have made any change to the EzPC compiler, you can manually run make:
```
cd EzPC
eval `opam config env`    # Load ocaml environment
make
```
