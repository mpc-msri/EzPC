#!/bin/bash

./build/fptraining 0 > temp/ct1.txt
./build/fptraining 0 > temp/ct2.txt

./build/fptraining 1
./build/fptraining 2 > temp/lt.txt &
sleep 1
./build/fptraining 3 &> /dev/null
sed -i '1d' temp/lt.txt
sed -i '1d' temp/lt.txt
sed -i '1d' temp/lt.txt
sed -i '1d' temp/lt.txt
