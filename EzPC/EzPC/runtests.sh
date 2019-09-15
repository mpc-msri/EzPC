#!/bin/bash

timeout=1m
testDir="./test_suite"
precompiledCodeDir="./test_suite/precompiled_output"
compiledCodeExtn="cpp"

declare genDir

generateAndDiffCode () {
	compileErrorFiles=()
	diffFailedFiles=()
	genDir=$1
	for fullFileName in $testDir/*.ezpc 
	do
		baseFileName=$(basename -- $fullFileName)
		actualFileName="${baseFileName%.*}"
		if ! [[ $actualFileName == *__temp* ]];
		then
			echo -e "\nProcessing file "$fullFileName"."
			fileNameWithExtn=$(basename -- "$fullFileName")
			actualFileName="${fileNameWithExtn%.*}"
			outputFullFileName=""$genDir"/"$actualFileName"0."$compiledCodeExtn""
			expectedCodeFullFileName=""$precompiledCodeDir"/"$actualFileName"0."$compiledCodeExtn""
			timeout $timeout ./ezpc.sh "$fullFileName" --o_prefix "$genDir"/"$actualFileName"
			exitCode=$?
			if [ $exitCode -eq 0 ]; 
			then 
				echo -e "Successfully generated compiled code for "$fullFileName"."
				echo -e "Comparing with pre-compiled code."
				if cmp -s "$outputFullFileName" "$expectedCodeFullFileName" 
				then
					echo -e "No diff. Output of compiler is as expected."
				else
					echo -e "Diff non-zero. Please check for diff between "$outputFullFileName" and "$expectedCodeFullFileName"."
					diffFailedFiles+=("$fullFileName")
				fi
			else
				echo -e "There was some error while running compiler. FullFileName="$fullFileName". ExitCode="$exitCode""
				compileErrorFiles+=("$fullFileName")
				break
			fi
			echo -e "\n--------------------------------------------------------\n"
		fi
	done

	echo -e "\n\n------------------------------SUMMARY-----------------------------"

	if [ ${#compileErrorFiles[@]} -eq 0 ];
	then
		echo -e "All files successfully compiled."
	else
		echo -e "Compilation failed for following files."
		printf '%s\n' "${compileErrorFiles[@]}"
	fi
	
	if [ ${#diffFailedFiles[@]} -eq 0 ];
	then
		echo -e "Diff for successfully compiled files successful."
	else
		echo -e "Diff failed for following files."
		printf '%s\n' "${diffFailedFiles[@]}"
	fi
}

generateCodeAndForceCopyForFile () {
	genDir=$1
	fullFileName=$2
	echo -e "\nProcessing file "$fullFileName"."
	fileNameWithExtn=$(basename -- "$fullFileName")
	actualFileName="${fileNameWithExtn%.*}"
	timeout $timeout ./ezpc.sh "$fullFileName" --o_prefix "$genDir"/"$actualFileName"
	outputFullFileName=""$genDir"/"$actualFileName"0."$compiledCodeExtn""
	expectedCodeFullFileName=""$precompiledCodeDir"/"$actualFileName"0."$compiledCodeExtn""
	exitCode=$?
	if [ $exitCode -eq 0 ]; 
	then 
		echo -e "Successfully generated compiled code for "$fullFileName"."
		echo -e "Copying to pre-compiled code."
		if cp "$outputFullFileName" "$expectedCodeFullFileName" 
		then
			echo -e "Copied output of compiler to "$expectedCodeFullFileName"."
		else
			echo -e "There was some problem in copying from "$outputFullFileName" to "$expectedCodeFullFileName"."
			copyFailedFiles+=("$fullFileName")
		fi
	else
		echo -e "There was some error while running compiler. FullFileName="$fullFileName". ExitCode="$exitCode""
		break
	fi
	echo -e "\n--------------------------------------------------------\n"
}

generateCodeAndForceCopy () {
	localGenDir=$1
	copyFailedFiles=()
	for fullFileName in $testDir/*.ezpc 
	do
		baseFileName=$(basename -- $fullFileName)
		actualFileName="${baseFileName%.*}"
		if ! [[ $actualFileName == *__temp* ]];
		then
			generateCodeAndForceCopyForFile $localGenDir $fullFileName
		fi
	done
	echo -e "------------------------------SUMMARY-----------------------------"
	if [ ${#copyFailedFiles[@]} -eq 0 ];
	then
		echo -e "Copy for all files successful."
	else
		echo -e "Copy failed for following files."
		printf '%s\n' "${copyFailedFiles[@]}"
	fi
}

while test $# -gt 0; do 
	case "$1" in 
		-genDir)
			genDir=$2
			shift #for key
			shift #for value
			;;
		-forceCopyFile)
			curFile=$2
			if [ -z ${genDir+x} ];
			then
				echo -e "GenDir is not specified. Exiting."
			fi
			generateCodeAndForceCopyForFile $genDir $curFile
			shift #for key
			shift #for value
			;;
		-genDiff)
			if [ -z ${genDir+x} ];
			then
				echo -e "GenDir is not specified. Exiting."
			fi
			generateAndDiffCode $genDir
			shift
			;;
		-genForceCopyAll)
			if [ -z ${genDir+x} ];
			then
				echo -e "GenDir is not specified. Exiting."
			fi
			generateCodeAndForceCopy $genDir
			shift
			;;
		*)
			echo -e "Incorrect usage."
			break
			;;
	esac
done

