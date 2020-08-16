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

##########################################################################################
# This is the CrypTFlow compilation script.
# 	Use this on a network to compile to MPC protocol.
#	By default, this assumes there is a ezpc repo one level up - if you want to change it,
#		please use Paths.config to override the default paths.
#		Same goes for Porthos repository.
# 	NOTE : When overriding paths in Paths.config, assumption is there is no '/' at the end.
##########################################################################################

# Load overriden paths from config file
. Paths.config

echo -e "Loaded paths: EzPCDir - $EzPCDir, PorthosDir - $PorthosDir"

usage() {
	echo -e "CrypTFlow compilation script. Options:";
	echo -e "<-b|--bitlen> <bitlen> :: Bit length to compile for. Defaults to 64";
	echo -e "<-s|--scaling-fac> <sf> :: Scaling factor to compile for. Defaults to 12.";
	echo -e "<-t|--target> <target> :: Compilation target. Possible options: ABY/CPP/CPPRING/PORTHOS/PORTHOS2PC. Defaults to CPP.";
	echo -e "<-f|--filename> :: Python tensorflow file to compile."
	echo -e "<--modulo> :: Modulo to be used for shares. Applicable for CPPRING/PORTHOS2PC backend. For PORTHOS2PC, for backend type OT, this should be power of 2 and for backend type HE, this should be a prime."
	echo -e "<--backend> :: Backend to be used - OT/HE (default OT). Applicable for PORTHOS2PC backend."
	echo -e "<--disable-hlil-all-opti> :: Disable all optimizations in HLIL."
	echo -e "<--disable-rmo> :: Disable Relu-Maxpool optimization."
	echo -e "<--disable-liveness-opti> :: Disable Liveness Optimization."
	echo -e "<--disable-trunc-opti> :: Disable truncation placement optimization."
	echo -e "<--exec-python> <num of args for python script> <args for python script>... :: Execute the python script which is passed for compilation.";
	echo -e "<-h|--help> :: help options.";
	exit 1;
}

BITLEN="64"
SCALINGFACTOR="12"
COMPILATIONTARGET="CPP"
EXECPYTHONARGS=""
while [[ $# -gt 0 ]]
do
	key="$1"
	case $key in
		-b|--bitlen)
		BITLEN="$2"
		shift # past argument
		shift # past value
		;;
		-s|--scaling-fac)
		SCALINGFACTOR="$2"
		shift # past argument
		shift # past value
		;;
		-t|--target)
		COMPILATIONTARGET="$2"
		shift # past argument
		shift # past value
		;;
		-f|--filename)
		FILENAME="$2"
		shift
		shift
		;;
		--modulo)
		MODULO="$2"
		shift
		shift
		;;
		--backend)
		BACKEND="$2"
		shift
		shift
		;;
		--exec-python)
		numargs="$2"
		shift
		shift
		for ((curArgNum=0;curArgNum<$numargs;curArgNum++)); do
			EXECPYTHONARGS="${EXECPYTHONARGS} $1"
			shift #past this argument
		done
		EXECPYTHON=Y
		;;
		-h|--help)
		HELP=Y
		shift # past one arg
		;;
		--disable-hlil-all-opti)
		DisableHLILAllOpti=Y
		shift # past one arg
		;;
		--disable-rmo)
		DisableRMO=Y
		shift # past one arg
		;;
		--disable-liveness-opti)
		DisableLivenessOpti=Y
		shift # past one arg
		;;
		--disable-trunc-opti)
		DisableTruncOpti=Y
		shift # past one arg
		;;
		*)    # unknown option
		usage
		;;
	esac
done

if [ ! -z "$HELP" ] || [ -z "$FILENAME" ] ; then 
	usage 
fi

if [ ! -z "$EXECPYTHON" ]; then
	echo -e "Exec-python parameter passed. EXECPYTHONARGS=$EXECPYTHONARGS."
fi

ACTUALBITLEN="${BITLEN}"
if [ "$ACTUALBITLEN" -gt 32 ]; then
	BITLEN="64"
else
	BITLEN="32"
fi

compilationTargetLower=$(echo "$COMPILATIONTARGET" | awk '{print tolower($0)}')
compilationTargetHigher=$(echo "$COMPILATIONTARGET" | awk '{print toupper($0)}')
givenDirPath=$(dirname "$FILENAME")
fullDirPath=$(realpath "$givenDirPath")
porthosFullDirPath=$( realpath "$PorthosDir")
baseFileName=$(basename -- "$FILENAME")
extension="${baseFileName##*.}"
actualFileName="${baseFileName%.*}" #without extension
fullFilePath=$(realpath "$FILENAME")
ezpcOutputFileName=${actualFileName}'_'${BITLEN}'_'${compilationTargetLower}
ezpcOutputFullFileName=${fullDirPath}'/'${ezpcOutputFileName}'.ezpc'
finalCodeOutputFileName=${ezpcOutputFileName}'0.cpp'
if [ "$extension" != "py" ]; then
	echo -e "Error: Provide a python file to compile."
	usage
fi
cd "$fullDirPath"
if [ ! -z "$EXECPYTHON" ] ; then
	echo -e "********* Executing python script passed to compile. *********"
	if [ "$EXECPYTHONARGS" = "" ]; then
		python3 "$baseFileName"
	else
		python3 "$baseFileName" "$EXECPYTHONARGS"
	fi
	echo -e "********* Python script compilation done. *********"
fi

cd - > /dev/null
cd ./TFCompiler
python3 ProcessTFGraph.py "$fullFilePath"
cd ../SeeDot
seedotArgs="--astFile ${fullDirPath}/astOutput.pkl --consSF ${SCALINGFACTOR} --bitlen ${ACTUALBITLEN} --outputFileName ${ezpcOutputFullFileName}"
if [ ! -z "$DisableHLILAllOpti" ]; then
	seedotArgs="${seedotArgs} --disableAllOpti True"
fi
if [ ! -z "$DisableRMO" ]; then
	seedotArgs="${seedotArgs} --disableRMO True"
fi
if [ ! -z "$DisableLivenessOpti" ]; then
	seedotArgs="${seedotArgs} --disableLivenessOpti True"
fi
if [ ! -z "$DisableTruncOpti" ]; then
	seedotArgs="${seedotArgs} --disableTruncOpti True"
fi
python3 SeeDot.py $seedotArgs
cd ..
libraryFile="$compilationTargetLower"
if [ "$compilationTargetLower" == "aby" ] || [ "$compilationTargetLower" == "cppring" ] ; then 
	libraryFile="cpp"
fi
if [ "$libraryFile" == "cpp" ];then
	# CPP/ABY backend
	cat "./TFEzPCLibrary/Library${BITLEN}_${libraryFile}_pre.ezpc" "./TFEzPCLibrary/Library${BITLEN}_common.ezpc" "./TFEzPCLibrary/Library${BITLEN}_${libraryFile}_post.ezpc" "$ezpcOutputFullFileName" > temp
else
	cat "./TFEzPCLibrary/Library${BITLEN}_${libraryFile}.ezpc" "./TFEzPCLibrary/Library${BITLEN}_common.ezpc" "$ezpcOutputFullFileName" > temp
fi
mv temp "$ezpcOutputFullFileName"
cp "$ezpcOutputFullFileName" "$EzPCDir/EzPC"
cd "$EzPCDir/EzPC"
eval `opam config env`
ezpcArgs="--bitlen ${ACTUALBITLEN} --codegen ${compilationTargetHigher} --disable-tac"
if [ ! -z "$MODULO" ]; then
	ezpcArgs="${ezpcArgs} --modulo ${MODULO}"
fi
if [ ! -z "$BACKEND" ]; then
	backendUpper=$(echo "$BACKEND" | awk '{print toupper($0)}')
	ezpcArgs="${ezpcArgs} --backend ${backendUpper}"
	finalCodeOutputFileName=${ezpcOutputFileName}_${backendUpper}'0.cpp'
fi
if [ "$compilationTargetLower" == "porthos" ] ; then
	ezpcArgs="${ezpcArgs} --sf ${SCALINGFACTOR}"
fi
./ezpc.sh "$ezpcOutputFullFileName" ${ezpcArgs}
if [ "$compilationTargetLower" == "cpp" ] || [ "$compilationTargetLower" == "cppring" ] ; then
	cd "$fullDirPath"
	g++ -O3 "$finalCodeOutputFileName" -o "$actualFileName.out"
	echo -e "All compilation done."
else
	cd - > /dev/null
	echo -e "All compilation done."
	if hash clang-format 2> /dev/null; then
		clang-format -style=LLVM $fullDirPath/$finalCodeOutputFileName > tmp_clang
		mv tmp_clang $fullDirPath/$finalCodeOutputFileName
	fi
fi

