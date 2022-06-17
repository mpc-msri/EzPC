#!/bin/bash

# Authors: Saksham Gupta.
#This is adapted from runtests.

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
precompiledCodeDir="./test_suite/precompiled_output_emp"
compiledCodeExtn="cpp"

declare genDir
rm -rf gen_emp
mkdir gen_emp
declare checks 
checks="Allgood"
compileErrorFiles=()
diffFailedFiles=()
genDir="gen_emp"
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
            ./ezpc.sh "$fullFileName" --codegen EMP --o_prefix "$genDir"/"$actualFileName" --bitlen 64
        else
            echo "Bitlen : 32"
            ./ezpc.sh "$fullFileName" --codegen EMP --o_prefix "$genDir"/"$actualFileName" --bitlen 32
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

./compile_emp.sh gen_emp/random_forest0.cpp
./random_forest0 1 12345 &
./random_forest0 2 12345 > gen_val.txt &
wait

./compile_emp.sh test_suite/precompiled_output_emp/random_forest0.cpp
./random_forest0 1 12345 &
./random_forest0 2 12345 > pre_val.txt &
wait

if cmp -s gen_val.txt pre_val.txt 
then
    echo -e "No diff. Output of compiler is as expected."
else
    echo -e "Diff non-zero. Please check for diff."
    checks="Failed"
fi

echo -e "\n\n------------------------------Match Random-Forest-Polish Results-----------------------------"

./compile_emp.sh gen_emp/random_forest_polish0.cpp
./random_forest_polish0 1 12345 &
./random_forest_polish0 2 12345 > gen_val.txt &
wait

./compile_emp.sh test_suite/precompiled_output_emp/random_forest_polish0.cpp
./random_forest_polish0 1 12345 &
./random_forest_polish0 2 12345 > pre_val.txt &
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
