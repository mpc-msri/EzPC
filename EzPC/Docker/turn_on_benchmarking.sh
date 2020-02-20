#!/bin/bash

python3 turn_on_mpc_benchmarking.py

cp updated_ABYconstants.h ../../../ABY-latest/ABY/src/abycore/ABY_utils/ABYconstants.h

echo "[SUCCESS] Turned on benchmarking.\nYou can now see runtime and communication numbers."
