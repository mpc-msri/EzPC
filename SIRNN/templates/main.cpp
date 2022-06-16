// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <thread>
#include <algorithm>
#include "utils/ArgMapping/ArgMapping.h"
#include "defines.h"
#include "datatypes.h"
#include "predictors.h"

using namespace std;

int party = 1;
string address = "127.0.0.1";
int port = 8000;
int num_threads = 2;
int32_t bitlength = 32;
string inputDir;


enum Version
{
	Fixed,
	Float
};
enum DatasetType
{
	Training,
	Testing
};
enum ProblemType
{
	Classification,
	Regression
};

bool profilingEnabled = false;

// Split the CSV row into multiple values
vector<string> readCSVLine(string line)
{
	vector<string> tokens;

	stringstream stream(line);
	string str;

	while (getline(stream, str, ','))
		tokens.push_back(str);

	return tokens;
}

vector<string> getFeatures(string line)
{
	static int featuresLength = -1;

	vector<string> features = readCSVLine(line);

	if (featuresLength == -1)
		featuresLength = (int)features.size();

	if ((int)features.size() != featuresLength)
		throw "Number of row entries in X is inconsistent";

	return features;
}

vector<string> getLabel(string line)
{
	static int labelLength = -1;

	vector<string> labels = readCSVLine(line);

	if (labelLength == -1)
		labelLength = (int)labels.size();

	if ((int)labels.size() != labelLength)
		throw "Number of row entries in Y is inconsistent";

	return labels;
}

void populateFixedVector(MYINT *features_int, vector<string> features, int scale)
{
	int features_size = (int)features.size();

	for (int i = 0; i < features_size; i++)
	{
		double f = (double)(atof(features.at(i).c_str()));
		double f_int = ldexp(f, -scale);
		features_int[i] = (MYINT)(f_int);
	}

	return;
}

int main(int argc, char *argv[])
{
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
    amap.arg("addr", address, "Localhost Run?");
    amap.arg("port", port, "Port Number");
    amap.arg("nt", num_threads, "Number of threads");
    amap.arg("inputDir", inputDir, "The location of input CSV files");

    amap.parse(argc, argv);
    assert(party == 1 || party == 2);

    cout << "Party: " << party << endl;

    int numOutputs = 1;

    // Invoke the predictor function

    ifstream featuresFile(inputDir + "X.csv");
    ifstream lablesFile(inputDir + "Y.csv");

    if (featuresFile.good() == false || lablesFile.good() == false)
        throw "Input files doesn't exist";

    #ifdef CLEARTEXT_ONLY
    string line1, line2;
    int correctCounter = 0;
    int counter = 0;
    while (getline(featuresFile, line1) && getline(lablesFile, line2))
	{
    #endif
        int64_t* fixed_res = new int64_t;
        if (party == 1) { // SERVER
            sirnnFixed(nullptr, fixed_res);
        } else { // party == 2 (CLIENT)
            // Reading the dataset

            int features_size = -1;
            MYINT *features_int = NULL;

            vector<int32_t*> labelsInt;

            #ifndef CLEARTEXT_ONLY
            string line1, line2;
    
            getline(featuresFile, line1);
            getline(lablesFile, line2);
            #endif
           
            vector<string> features = getFeatures(line1);
            vector<string> labelString = getLabel(line2);
            int32_t labelInt;

            labelInt = atoi(labelString[0].c_str());

                // Allocate memory to store the feature vector as arrays
            features_size = (int)features.size();

            features_int = new MYINT[features_size];

            cout << "[CLIENT] Input Parsed" << endl;

            // Populate the array using the feature vector
            populateFixedVector(features_int, features, scaleForX);

            sirnnFixed(features_int, fixed_res);
            #ifndef CLEARTEXT_ONLY
            cout<<"Predicted Label:" << *fixed_res << ", Actual Label:" << labelInt << endl;
            #else
            if ((*fixed_res) == labelInt)
                correctCounter++;
            counter++;
            #endif
            // Deallocate memory

            delete[] features_int;
        }
    #ifdef CLEARTEXT_ONLY
    }
    cout<<float(correctCounter*100)/float(counter)<<endl;
    #endif
	return 0;
}
