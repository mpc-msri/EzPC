

/* Parse MNIST data from CSV into tab delimited data.

   Inputs: training file and testing file (label, data format)
   Outputs: training data, training label, testing data, testing label.

   Labels are in one hot encoding.
*/


#include "MNISTParse.h"
using namespace std;


int main(int argc, char** argv)
{
	if (argc != 3)
		ERROR("Syntax: <program> <path-to-training-file> <path-to-testing-file>");

    assert(0 == parse(argv[1], TRAINING));
    assert(0 == parse(argv[2], TESTING));

    return 0;
}