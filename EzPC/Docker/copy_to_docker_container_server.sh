#!/bin/bash

# exit if a command fails
set -e

if [ $# -eq 0 ]
then
	echo "ERROR. 1 arg expected. Usage is <script>.sh server_pickle_file_path"
	exit 1
fi

input=$(echo `sudo docker container ls`)

echo $input > temp.txt

python3 extract_container_id.py

id=$(cat container_id.txt)

echo Extracted container ID is: $id
echo $1
sudo docker cp $1 $id:/ezpc-workdir/EzPC/.

echo "\n[STATUS] Success!"

