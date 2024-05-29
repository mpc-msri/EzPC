CUDA_VERSION ?= $(value CUDA_VERSION)
ifeq ($(CUDA_VERSION),)
	CUDA_VERSION = 11.7
endif
CUTLASS_PATH=./ext/cutlass
SYTORCH_PATH=./ext/sytorch
SYTORCH_BUILD_PATH=$(SYTORCH_PATH)/build
LLAMA_PATH=$(SYTORCH_PATH)/ext/llama
CUDA_ARCH =$(GPU_ARCH)

CXX=/usr/local/cuda-$(CUDA_VERSION)/bin/nvcc
FLAGS := -O3 -gencode arch=compute_$(CUDA_ARCH),code=[sm_$(CUDA_ARCH),compute_$(CUDA_ARCH)] -std=c++17 -m64 -Xcompiler="-O3,-w,-std=c++17,-fpermissive,-fpic,-pthread,-fopenmp,-march=native" 
LIBS := -lsytorch -lcryptoTools -lLLAMA -lbitpack -lcuda -lcudart -lcurand
SECFLOAT_LIBS := -lSCI-FloatML -lSCI-FloatingPoint -lSCI-BuildingBlocks -lSCI-LinearOT -lSCI-GC -lcrypto

UTIL_FILES := ./utils/gpu_mem.cu ./utils/gpu_file_utils.cpp ./utils/sigma_comms.cpp
OBJ_INCLUDES := -I '$(CUTLASS_PATH)/include' -I '$(CUTLASS_PATH)/tools/util/include' -I '$(SYTORCH_PATH)/include' -I '$(LLAMA_PATH)/include' -I '$(SYTORCH_PATH)/ext/cryptoTools' -I '.'
INCLUDES := $(OBJ_INCLUDES) -L$(CUTLASS_PATH)/build/tools/library -L$(SYTORCH_BUILD_PATH) -L$(SYTORCH_BUILD_PATH)/ext/cryptoTools -L$(SYTORCH_BUILD_PATH)/ext/llama -L$(SYTORCH_BUILD_PATH)/ext/bitpack -L$(SYTORCH_BUILD_PATH)/lib

dpf: tests/fss/dpf.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/fss/dpf

dpf_eval_all: tests/fss/dpf_eval_all.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/fss/dpf_eval_all

dpf_drelu: tests/fss/dpf_drelu.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/fss/dpf_drelu

dpf_lut: tests/fss/dpf_lut.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/fss/dpf_lut

gelu: tests/fss/gelu.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/fss/gelu

relu: tests/fss/relu.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/fss/relu

rmsnorm: tests/fss/rmsnorm.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/fss/rmsnorm

softmax: tests/fss/softmax.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/fss/softmax

fc: tests/fss/fc.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/fss/fc

layernorm: tests/fss/layernorm.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/fss/layernorm

silu: tests/fss/silu.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/fss/silu

truncate: tests/fss/truncate.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/fss/truncate

mha: tests/fss/mha.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/fss/mha

rotary_embedding: tests/fss/rotary_embedding.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/fss/rotary_embedding

secfloat_softmax: tests/fss/secfloat_softmax.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) $(SECFLOAT_LIBS) -o tests/fss/secfloat_softmax

piranha_softmax: tests/fss/piranha_softmax.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/fss/piranha_softmax

orca_dealer: experiments/orca/orca_dealer.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) $(SECFLOAT_LIBS) -o experiments/orca/orca_dealer

orca_evaluator: experiments/orca/orca_evaluator.cu experiments/orca/datasets/mnist.cpp
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) $(SECFLOAT_LIBS) -o experiments/orca/orca_evaluator

dcf: tests/fss/dcf/dcf.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/fss/dcf/dcf

aes: tests/fss/dcf/aes.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/fss/dcf/aes

dcf_relu_extend: tests/fss/dcf/relu_extend.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/fss/dcf/relu_extend

dcf_stochastic_truncate: tests/fss/dcf/stochastic_truncate.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/fss/dcf/stochastic_truncate

dcf_relu: tests/fss/dcf/relu.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/fss/dcf/relu

orca_conv2d: tests/nn/orca/conv2d_test.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/nn/orca/conv2d

orca_maxpool: tests/nn/orca/maxpool_test.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/nn/orca/maxpool

orca_relu_extend: tests/nn/orca/relu_extend_test.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/nn/orca/relu_extend

orca_fc: tests/nn/orca/fc_test.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/nn/orca/fc

orca_relu: tests/nn/orca/relu_test.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o tests/nn/orca/relu

orca_inference: experiments/orca/orca_inference.cu 
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o experiments/orca/orca_inference

orca_inference_u32: experiments/orca/orca_inference.cu
	$(CXX) $(FLAGS) -DInfType=u32 $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o experiments/orca/orca_inference_u32

sigma: experiments/sigma/sigma.cu 
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o experiments/sigma/sigma

sigma_correctness: experiments/sigma/sigma.cu 
	$(CXX) $(FLAGS) -DCORRECTNESS=1 $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o experiments/sigma/sigma

piranha: experiments/orca/piranha.cu 
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o experiments/orca/piranha

share_data: experiments/orca/share_data.cpp experiments/orca/datasets/mnist.cpp
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o experiments/orca/share_data

model_accuracy: experiments/orca/model_accuracy.cu experiments/orca/datasets/mnist.cpp
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o experiments/orca/model_accuracy

orca: orca_dealer orca_evaluator orca_inference orca_inference_u32 piranha

clean:
	rm -rf ext/cutlass/build
	rm -rf ext/sytorch/build
	rm -rf orca/experiments/output
	rm -rf sigma/experiments/output
	rm experiments/orca/orca_dealer
	rm experiments/orca/orca_evaluator
	rm experiments/orca/orca_inference
	rm experiments/orca/orca_inference_u32
	rm experiments/orca/piranha
	rm experiments/sigma/sigma
	
