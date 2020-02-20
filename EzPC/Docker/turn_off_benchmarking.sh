#!/bin/bash

python3 turn_off_mpc_benchmarking.py

cp updated_ABYconstants.h ../../../ABY-latest/ABY/src/abycore/ABY_utils/ABYconstants.h

echo "[SUCCESS] Turned off benchmarking.\nYou will not see runtime and communication numbers now."
