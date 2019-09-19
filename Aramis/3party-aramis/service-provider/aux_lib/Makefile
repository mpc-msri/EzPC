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

AUX_LIB ?= libaux_lib.a


Cpp_Files := aux_funcs.cpp cpp-base64/base64.cpp

Include_Paths := -I. -Icpp-base64
C_Flags := $(SGX_COMMON_CFLAGS) -fPIC -Wno-attributes $(App_Include_Paths)
Cpp_Flags := $(C_Flags) -std=c++11
Link_Flags := -lcrypto -lpthread -Wl,-rpath=$(CURDIR)

Cpp_Objects := $(Cpp_Files:.cpp=.o)

.PHONY: all

all: $(AUX_LIB)

%.o: %.cpp
	@$(CXX) $(Cpp_Flags) -c $< -o $@
	@echo "CXX  <=  $<"

$(AUX_LIB): $(Cpp_Objects)
	ar -cvq $@ $(Cpp_Objects)
	@echo "CONSTRUCT  =>  $@"

	
clean:
	@rm -f $(AUX_LIB) $(Cpp_Objects) 
	