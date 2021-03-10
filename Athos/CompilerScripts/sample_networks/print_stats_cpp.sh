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
MODEL_NAME=$(basename $MODEL_DIR)

echo "-------------------------------------------------------"
echo "                  ${MODEL_NAME} results"
echo "-------------------------------------------------------"
echo "Model outputs:"
echo -e "MPC Cleartext (CPP) output:\t $(awk '$0==($0+0)' ${MODEL_DIR}/mpc_output.out)"
echo -e "Tensorflow output:\t\t $(cat ${MODEL_DIR}/tf_pred.float)"
echo ""

read -r peakmem_kb usertime_s systemtime_s walltime_s <<<$(cat ${MODEL_DIR}/party0_stats)
user_percent=$(echo "scale=2; 100*$usertime_s / ($usertime_s + $systemtime_s)" | bc)
system_percent=$(echo "scale=2; 100*$systemtime_s / ($usertime_s + $systemtime_s)" | bc)
wall_user_s=$(echo "scale=2; $user_percent * $walltime_s / 100" | bc)
wall_system_s=$(echo "scale=2; $system_percent * $walltime_s / 100" | bc)
peakmem_gb=$(echo "scale=2; $peakmem_kb/1024/1024" | bc)

echo "Execution summary:"
echo -e "Peak Memory Usage:\t\t ${peakmem_kb} KB (${peakmem_gb}GB)"
echo -e "Total time taken:\t\t ${walltime_s} seconds"
echo -e "Total work time:\t\t ${wall_user_s} seconds (${user_percent}%)"
echo -e "Time spent waiting:\t\t ${wall_system_s} seconds (${system_percent}%)"
echo -e "Time taken by tensorflow:\t $(cat ${MODEL_DIR}/tf_pred.time) seconds"