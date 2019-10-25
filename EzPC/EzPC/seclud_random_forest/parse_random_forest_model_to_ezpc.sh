#!/bin/bash

> ezpc_parsed_tree.txt

counter=0
while [ $counter -lt $1 ]
do
	#echo tree$counter
	python parse_graphviz_to_ezpc_input.py tree$counter.txt
	((counter++))
done

echo $2

echo Parsed all trees in Random Forest
