
#pragma once
#include "globals.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>
using namespace std;


void ERROR(string str)
{
	cout << "ERROR: " << str << endl;
	exit(1); 
}


int parse(char* filename, string type)
{
	ifstream input_file(filename);
	ofstream data_file;
	ofstream label_file;
	
	string line;
	int data_size;
	int *inputs;

	if (type == TRAINING)
	{
		data_size = TRAINING_DATA_SIZE;
		data_file.open(TRAINING_DATA);
		label_file.open(TRAINING_LABEL);
	}
	else if (type == TESTING)
	{
		data_size = TEST_DATA_SIZE;
		data_file.open(TESTING_DATA);
		label_file.open(TESTING_LABEL);
	}
	else
		ERROR("parse() only accepts TRAINING/TESTING");

	//Reading
	inputs = new int [data_size * (INPUT_DIMENSION + 1)];
	for (int row = 0; row < data_size; ++row)
	{
		getline(input_file, line);
		stringstream lineStream(line);
		string cell;
								
		for (int column = 0; column < INPUT_DIMENSION + 1; ++column)
		{
			getline(lineStream, cell, ',');
			// if (!cell.empty())
			inputs[row*(INPUT_DIMENSION+1) + column] = stoi(cell);
		}
	}

	//Writing
	for (int row = 0; row < data_size; ++row)
	{
		for (int column = 0; column < INPUT_DIMENSION; ++column)
		{
			data_file << inputs[row*(INPUT_DIMENSION+1) + column + 1] << "\t";
		}
		data_file << endl;
	}

	for (int row = 0; row < data_size; ++row)
	{
		for (int column = 0; column < OUTPUT_DIMENSION; ++column)
		{
			if (column == inputs[row*(INPUT_DIMENSION + 1)])
				label_file << 1 << "\t";
			else
				label_file << 0 << "\t";
		}
		label_file << endl;
	}

	delete[] inputs;
	input_file.close();
	data_file.close();
	label_file.close();

	return 0;
}