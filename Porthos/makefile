CXX=g++
SRC_CPP_FILES     := $(wildcard src/*.cpp)
EX_NN_CPP_FILES	  := $(wildcard src/example_neural_nets/*.cpp)
OBJ_CPP_FILES     := $(wildcard util/*.cpp)
OBJ_FILES         := $(patsubst src/example_neural_nets/%.cpp, src/example_neural_nets/%.o, $(EX_NN_CPP_FILES))
OBJ_FILES    	  += $(patsubst src/%.cpp, src/%.o, $(SRC_CPP_FILES))
OBJ_FILES    	  += $(patsubst util/%.cpp, util/%.o, $(OBJ_CPP_FILES))
HEADER_FILES       = $(wildcard src/*.h)

FLAGS := -w -std=c++14 -pthread -msse4.1 -maes -msse2 -mpclmul -fpermissive -fpic -fopenmp
LIBS := -lcrypto -lssl
OBJ_INCLUDES := -I 'lib_eigen/' -I 'src/'
UTIL_INCLUDES := $($(OBJ_INCLUDES), -L./)


all: FLAGS += -O3
all: Porthos.out

debug: FLAGS += -O0 -g
debug: Porthos.out

Porthos.out: $(OBJ_FILES)
	g++ $(FLAGS) -o $@ $(OBJ_FILES) $(UTIL_INCLUDES) $(LIBS)
%.o: %.cpp $(HEADER_FILES)
	$(CXX) $(FLAGS) -c $< -o $@ $(OBJ_INCLUDES)

clean:
	rm -rf Porthos.out
	rm -rf src/*.o util/*.o src/example_neural_nets/*.o

