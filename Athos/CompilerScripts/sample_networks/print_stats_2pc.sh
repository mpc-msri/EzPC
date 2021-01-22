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

PARTY=$1
MODEL_DIR=$2
MODEL_NAME=$(basename $MODEL_DIR)

if [ $PARTY -eq 0 ];
then
	PARTYNAME="Server"
elif [ $PARTY -eq 1 ];
then
	PARTYNAME="Client"
fi

echo "-------------------------------------------------------"
echo "                  ${MODEL_NAME} results [${PARTYNAME}]"
echo "-------------------------------------------------------"
if [ $PARTY -eq 1 ];
then
	echo "Model outputs:"
	echo -e "MPC SCI (2PC) output:\t $(awk '$0==($0+0)' ${MODEL_DIR}/party${PARTY}_mpc_output.out)"
	echo -e "Tensorflow output:\t\t $(cat ${MODEL_DIR}/tf_pred.float)"
	echo ""
fi
read -r peakmem_kb usertime_s systemtime_s walltime_s <<<$(cat ${MODEL_DIR}/party${PARTY}_stats)
log_wall_t_ms=$(grep "Total time taken =" ${MODEL_DIR}/party${PARTY}_mpc_output.out | grep -o '[0-9]\+[.]*[0-9]\+')
log_wall_t_s=$(echo "scale=2; ${log_wall_t_ms}/1000" | bc)
log_wait_t_s=$(grep "Total wait time =" ${MODEL_DIR}/party${PARTY}_mpc_output.out | grep -o '[0-9]\+[.]*[0-9]\+')
log_wait_t_s=$(echo "scale=2; $log_wait_t_s" | bc)
log_work_t_s=$(echo "scale=2; $log_wall_t_s - $log_wait_t_s" | bc)

work_percent=$(echo "scale=2; 100 * $log_work_t_s/$log_wall_t_s" | bc)
wait_percent=$(echo "scale=2; 100 * $log_wait_t_s/$log_wall_t_s" | bc)

wall_m=$(echo "$log_wall_t_s/60" | bc)
wall_s=$(echo "$log_wall_t_s%60/1" | bc)
walltime_m_s="${wall_m}m${wall_s}s"

peakmem_gb=$(echo "scale=2; $peakmem_kb/1024/1024" | bc)
comm=$(grep "Total data sent" ${MODEL_DIR}/party${PARTY}_mpc_output.out)

echo "Execution summary for ${PARTYNAME}:"
echo -e "[Communication]:\t\t $comm"
echo -e "Peak Memory Usage:\t\t ${peakmem_kb}KB (${peakmem_gb}GB)"
echo -e "Total time taken:\t\t ${walltime_m_s} ($log_wall_t_s seconds)"
echo -e "Total work time:\t\t ${log_work_t_s} seconds (${work_percent}%)"
echo -e "Time spent waiting:\t\t ${log_wait_t_s} seconds (${wait_percent}%)"
if [ $PARTY -eq 1 ];
then
	echo -e "Time taken by tensorflow:\t $(cat ${MODEL_DIR}/tf_pred.time) seconds"
fi
