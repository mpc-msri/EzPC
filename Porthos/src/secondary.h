#ifndef SECONDARY_H
#define SECONDARY_H

#pragma once
#include "basicSockets.h"
#include <sstream>
#include "../util/TedKrovetzAesNiWrapperC.h"
#include "tools.h"
#include "globals.h"
#include <thread>
#include <iomanip>
#include <fstream>

void parseInputs(int argc, 
		char* argv[]);

void initializeMPC();

void deleteObjects();
#endif
