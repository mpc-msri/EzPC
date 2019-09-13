#!/bin/bash

NUMIMAGES=50000
NUMPROCESSES=32
PERPROCESSIMAGES=$((NUMIMAGES / NUMPROCESSES))
ScalesToTest=(12)
EXECUTABLENAME="Resnet_acc64.out"
RUNSERIAL=true #Change this to "true" to run the first NUMIMAGES.
				#	"false" means it will pick up a random subset.
				#	NOTE: In that case, NUMIMAGES should equal #(lines in the random subset idx file).
RANDOMSUBSETFILEIDX="../../../HelperScripts/ImageNet_ValData/Random_Img_Idx.txt"

# Load paths from config file
. ./Paths.config
declare -a pids

for curScale in "${ScalesToTest[@]}";
do
	echo -e "******************SCALE = $curScale************************\n\n"
	for ((curProcessNum=0;curProcessNum<$NUMPROCESSES;curProcessNum++)); do
		curImgStartNum=$(( curProcessNum*PERPROCESSIMAGES + 1 ))
		endImgNum=$(( curImgStartNum + PERPROCESSIMAGES ))
		if (( curProcessNum == $NUMPROCESSES-1 ));then
			endImgNum=$(( NUMIMAGES + 1 ))
		fi
		if [ "$RUNSERIAL" = false ];
		then
			"./$EXECUTABLENAME" "$curScale" "$curImgStartNum" "$endImgNum" "$preProcessedImgSaveDirName" "$NUMIMAGES" "$RANDOMSUBSETFILEIDX" < "ResNet_img_input_weights_float.inp" > "InferenceOutputs/output_${curScale}_${curProcessNum}.outp" 2> "InferenceErrors/cerr_${curScale}_${curProcessNum}.outp" &
		else
			"./$EXECUTABLENAME" "$curScale" "$curImgStartNum" "$endImgNum" "$preProcessedImgSaveDirName" < "ResNet_img_input_weights_float.inp" > "InferenceOutputs/output_${curScale}_${curProcessNum}.outp" 2> "InferenceErrors/cerr_${curScale}_${curProcessNum}.outp" &

		fi
		pids+=($!)
	done
	
	echo -e "--->>>All processes started. Now going into waiting for each process.<<<---"
	for pid in ${pids[*]}; do
	    wait $pid
	done

	echo -e "--->>> Done waiting for all started processes. Unsetting Pids and starting next loop. <<<---"
	unset pids
done

echo -e "--->>>All processes completed.<<<---"
