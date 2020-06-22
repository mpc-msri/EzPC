#!/bin/bash

inp1=$1
inp2=$2
temp_1=${inp1}_tmp_cmp
temp_2=${inp2}_tmp_cmp

awk '$0==($0+0)' $inp1 > $temp_1
awk '$0==($0+0)' $inp2 > $temp_2

python compare_output.py $temp_1 $temp_2 24

rm $temp_1 $temp_2
