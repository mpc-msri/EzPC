#!/bin/bash

# Authors: Pratik Bhatu.

# Copyright:
# Copyright (c) 2021 Microsoft Research
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

MODEL_DIR=$1
MODEL_BINARY_PATH=$2
MODEL_INPUT_PATH=$3
MODEL_WEIGHT_PATH=$4

MODEL_NAME=$(basename ${MODEL_DIR})
SESSION_NAME=${MODEL_NAME}

tmux has-session -t "${SESSION_NAME}" > /dev/null 2>&1
if [ "$?" -eq 0 ]; then
	echo "Killing existing tmux ${SESSION_NAME} session"
    tmux kill-session -t "${SESSION_NAME}"
fi

tmux new-session -s "${SESSION_NAME}" -d
tmux split-window -v -t "${SESSION_NAME}:0.0"

tmux send-keys -t "${SESSION_NAME}:0.0" "clear" Enter
tmux send-keys -t "${SESSION_NAME}:0.1" "clear" Enter

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

TIME_CMD="/usr/bin/time --format \"%M %U %S %e\""
PARTY0_IP="127.0.0.1"
PORT=12345

PARTY0_RUN_CMD="${MODEL_BINARY_PATH} r=1 port=${PORT} < ${MODEL_WEIGHT_PATH}"
PARTY1_RUN_CMD="${MODEL_BINARY_PATH} r=2 ip=${PARTY0_IP} port=${PORT} < ${MODEL_INPUT_PATH}"

PARTY0_DUMP_CMD="> ${MODEL_DIR}/party0_mpc_output.out 2> ${MODEL_DIR}/party0_stats"
PARTY1_DUMP_CMD="> ${MODEL_DIR}/party1_mpc_output.out 2> ${MODEL_DIR}/party1_stats"


PARTY0_FINAL_CMD="${TIME_CMD} ${PARTY0_RUN_CMD} ${PARTY0_DUMP_CMD}"
PARTY1_FINAL_CMD="${TIME_CMD} ${PARTY1_RUN_CMD} ${PARTY1_DUMP_CMD}"

tmux send-keys -t "${SESSION_NAME}:0.0" "${PARTY0_FINAL_CMD}" Enter
tmux send-keys -t "${SESSION_NAME}:0.1" "${PARTY1_FINAL_CMD}" Enter

PARTY0_FINAL_CMD="clear; ${SCRIPT_DIR}/print_stats_2pc.sh 0 ${MODEL_DIR}"
PARTY1_FINAL_CMD="clear; ${SCRIPT_DIR}/print_stats_2pc.sh 1 ${MODEL_DIR}"

tmux send-keys -t "${SESSION_NAME}:0.0" "${PARTY0_FINAL_CMD}" Enter
tmux send-keys -t "${SESSION_NAME}:0.1" "${PARTY1_FINAL_CMD}" Enter