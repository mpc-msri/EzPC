#!/bin/bash

#First take graphviz output and convert to EzPC input and also pad
#to make all the decision trees complete binary trees
params=$(cat decision_tree_stat.txt)
./parse_random_forest_model_to_ezpc.sh $params

#Now we have EzPC input in ezpc_parsed_tree.txt
#Now go to EzPC code and put in the right values
#of depth and no. of trees in RF.
python3 correct_ezpc_code_params.py

cd .. && ./ezpc.sh seclud_random_forest/random_forest_main.ezpc --bitlen 64 && cd seclud_random_forest

echo EzPC part done. Generated CPP are available now.
