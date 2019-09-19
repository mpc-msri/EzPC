# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.


#Modified by Mayank, Microsoft Research India.
#mayankrathee.japan@gmail.com
 

SGX_SDK ?= /home/mayank/Downloads/sgxsdk
SSL_SGX ?= /home/mayank/Downloads/sgxssl

######## Server Settings ########

Server_Cpp_Files := server.cpp IAS_web_service/IAS_web_service.cpp
Server_Include_Paths := -I../ -Iaux_lib -Iaux_lib/cpp-base64 -I$(SGX_SDK)/include -IIAS_web_service
Server_C_Flags := $(Server_Include_Paths) -fPIC -Wno-attributes
Server_Cpp_Flags := $(Server_C_Flags) -std=c++11
Server_Link_Flags := -L. -lpthread -lcrypto -lcurl -ljsoncpp -Laux_lib -laux_lib #-Wl,-rpath=$(CURDIR)

Server_Cpp_Objects := $(Server_Cpp_Files:.cpp=.o)

Server_Name := truce_server

#ifdef SIMULATE_IAS
#ifeq ($(SIMULATE_IAS),1)
#Server_Cpp_Flags += -DSIMULATE_IAS
#endif
#endif

.PHONY: all run


all: $(Server_Name) 


######## Server Objects ########

%.o: %.cpp
	@$(CXX) $(Server_Cpp_Flags) -c $< -o $@
	@echo "CXX  <=  $<"

$(Server_Name): $(Server_Cpp_Objects)
	@$(CXX) $^ -o $@ $(Server_Link_Flags)
	@echo "LINK =>  $@"



.PHONY: clean

clean:
	@rm -f $(Server_Name) $(Server_Cpp_Objects)
