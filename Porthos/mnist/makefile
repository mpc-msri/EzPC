
CXX=g++
RM=rm -f
FLAGS=-std=c++11

TARGET= build

$(TARGET): MNISTParse.cpp MNISTParse.h globals.h
	$(CXX) $(FLAGS) -o $(TARGET) MNISTParse.cpp

clean:
	$(RM) $(TARGET)

superclean:
	$(RM) mnist_*