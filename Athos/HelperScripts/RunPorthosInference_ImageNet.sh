#!/bin/bash

NUMIMAGES=96
NUMPROCESSES=16
PERPROCESSIMAGES=$((NUMIMAGES / NUMPROCESSES))
ScalesToTest=(18 19 20 21 22 23 24 25 26 27)
NetworkName="ResNet"
RANDOMSUBSETFILEIDX="./ImageNet_ValData/Random_Img_Idx.txt"
PreProcessedFilesDirectory="../Networks/$NetworkName/AccuracyAnalysisHelper/PreProcessedImages"
PorthosDir="../../SecureNN"

mapfile -t randomIndices < "$RANDOMSUBSETFILEIDX"
mkdir -p InferenceOutputs
mkdir -p InferenceErrors
declare -a pids

for curScale in "${ScalesToTest[@]}";
do
	echo -e "******************SCALE = $curScale************************\n\n"
	echo -e "Replacing this scale factor in globals.h."
	sed -i -r 's/#define FLOAT_PRECISION .*/#define FLOAT_PRECISION '"${curScale}"'/' src/globals.h
	echo -e "Doing make..."
	make -j > /dev/null
	echo -e "make done. Continuing..."
	ModelFilePath="../Athos/Networks/$NetworkName/AccuracyAnalysisHelper/$NetworkName"'_'"img_input_weights_float_scaled"'_'"$curScale.inp"
	echo -e "Using model path = $ModelFilePath."
	imgCounter=0
	while [ "${imgCounter}" -ne "${NUMIMAGES}" ];
	do
		echo -e "########Starting imgCounter = $imgCounter."
		for ((curProcessNum=0;curProcessNum<$NUMPROCESSES;curProcessNum++)); do
			echo -e "Running Porthos for imgCounter = $imgCounter, actual img idx = ${randomIndices[${imgCounter}]}."
			imgFileName="$PreProcessedFilesDirectory/ImageNum"'_'"${randomIndices[${imgCounter}]}"'_'"scaled"'_'"${curScale}.inp"
			stdoutFilesSuffix="./InferenceOutputs/stdout"'_'"${curScale}"'_'"${imgCounter}"'_'"proc"'_'
			stderrFilesSuffix="./InferenceErrors/stderr"'_'"${curScale}"'_'"${imgCounter}"'_'"proc"'_'
			./Porthos.out 3PC 0 files/parties_localhost files/keyA files/keyAB "$curProcessNum" "${imgCounter}" < "$ModelFilePath" > "${stdoutFilesSuffix}0.outp" 2> "${stderrFilesSuffix}0.outp" &
			pids+=($!)
			./Porthos.out 3PC 1 files/parties_localhost files/keyB files/keyAB "$curProcessNum" "${imgCounter}" < "$imgFileName" > "${stdoutFilesSuffix}1.outp" 2> "${stderrFilesSuffix}1.outp" &
			pids+=($!)
			./Porthos.out 3PC 2 files/parties_localhost files/keyB files/keyAB "$curProcessNum" "${imgCounter}" > "${stdoutFilesSuffix}2.outp" 2> "${stderrFilesSuffix}2.outp" &
			pids+=($!)
			((imgCounter=imgCounter+1))
			if [ "${imgCounter}" -eq "${NUMIMAGES}" ]; then
				break
			fi
		done
		echo -e "--->>>All processes started. Now going into waiting for each process.<<<---"
		for pid in ${pids[*]}; do
		    wait $pid
		done

		echo -e "--->>> Done waiting for all started processes. Unsetting Pids and starting next loop. <<<---"
		unset pids
	done
done

echo -e "--->>>All processes completed.<<<---"
