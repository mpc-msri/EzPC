#!/bin/bash

# Authors: Nishant Kumar.

# Copyright:
# Copyright (c) 2020 Microsoft Research
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

timeout=1m
testDir="./test_suite"
precompiledCodeDir="./test_suite/precompiled_output"
compiledCodeExtn="cpp"

declare genDir

generateAndDiffCode () {
	declare checks 
	checks="Allgood"
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
			if [[ $baseFileName == random_* ]];
			then
				echo "Bitlen : 64"
				timeout $timeout ./ezpc.sh "$fullFileName" --o_prefix "$genDir"/"$actualFileName" --bitlen 64
			else
				echo "Bitlen : 32"
				timeout $timeout ./ezpc.sh "$fullFileName" --o_prefix "$genDir"/"$actualFileName" --bitlen 32
			fi
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
    		checks="Failed"

	fi
	
	if [ ${#diffFailedFiles[@]} -eq 0 ];
	then
		echo -e "Diff for successfully compiled files successful."
	else
		echo -e "Diff failed for following files."
		printf '%s\n' "${diffFailedFiles[@]}"
    		checks="Failed"

	fi

	echo -e "\n\n------------------------------Match Random-Forest Results-----------------------------"

	./compile_aby.sh gen/random_forest0.cpp
	./random_forest0 -r 0 &
	./random_forest0 -r 1 > gen_val.txt &
	wait

	./compile_aby.sh test_suite/precompiled_output/random_forest0.cpp
	./random_forest0 -r 0 &
	./random_forest0 -r 1 > pre_val.txt &
	wait

	if cmp -s gen_val.txt pre_val.txt 
	then
		echo -e "No diff. Output of compiler is as expected."
	else
		echo -e "Diff non-zero. Please check for diff."
		checks="Failed"
	fi

	echo -e "\n\n------------------------------Match Random-Forest-Polish Results-----------------------------"

	./compile_aby.sh gen/random_forest_polish0.cpp
	./random_forest_polish0 -r 0 &
	./random_forest_polish0 -r 1 > gen_val.txt &
	wait

	./compile_aby.sh test_suite/precompiled_output/random_forest_polish0.cpp
	./random_forest_polish0 -r 0 &
	./random_forest_polish0 -r 1 > pre_val.txt &
	wait

	if cmp -s gen_val.txt pre_val.txt 
	then
		echo -e "No diff. Output of compiler is as expected."
	else
		echo -e "Diff non-zero. Please check for diff."
		checks="Failed"
	fi
	echo $checks
	if [[ "${checks}" == "Failed" ]]
	then
		exit 1
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

