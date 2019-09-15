#!/bin/bash

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
	echo -e "<-b|--bitlen> <bitlen> :: Bit length to compile for. Possible options: 32/64. Defaults to 64";
	echo -e "<-s|--scaling-fac> <sf> :: Scaling factor to compile for. Defaults to 12.";
	echo -e "<-t|--target> <target> :: Compilation target. Possible options: CPP/PORTHOS. Defaults to CPP.";
	echo -e "<-f|--filename> :: Python tensorflow file to compile."
	echo -e "<--disable-hlil-all-opti> :: Disable all optimizations in HLIL."
	echo -e "<--disable-rmo> :: Disable Relu-Maxpool optimization."
	echo -e "<--disable-liveness-opti> :: Disable Liveness Optimization."
	echo -e "<--exec-python> <num of args for python script> <args for python script>... :: Execute the python script which is passed for compilation.";
	echo -e "<--help> :: help options.";
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
		--help)
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
seedotArgs="--astFile ${fullDirPath}/astOutput.pkl --consSF ${SCALINGFACTOR} --bitlen ${BITLEN} --outputFileName ${ezpcOutputFullFileName}"
if [ ! -z "$DisableHLILAllOpti" ]; then
	seedotArgs="${seedotArgs} --disableAllOpti True"
fi
if [ ! -z "$DisableRMO" ]; then
	seedotArgs="${seedotArgs} --disableRMO True"
fi
if [ ! -z "$DisableLivenessOpti" ]; then
	seedotArgs="${seedotArgs} --disableLivenessOpti True"
fi
python3 SeeDot.py $seedotArgs
cd ..
libraryFile="$compilationTargetLower"
if [ "$compilationTargetLower" == "aby" ];then 
	libraryFile="cpp"
fi
cat "./TFEzPCLibrary/Library${BITLEN}_$libraryFile.ezpc" "./TFEzPCLibrary/Library${BITLEN}_common.ezpc" "$ezpcOutputFullFileName" > temp
mv temp "$ezpcOutputFullFileName"
cp "$ezpcOutputFullFileName" "$EzPCDir/EzPC"
cd "$EzPCDir/EzPC"
eval `opam config env`
./ezpc.sh "$ezpcOutputFullFileName" --bitlen "$BITLEN" --codegen "$compilationTargetHigher" --disable-tac
if [ "$compilationTargetLower" == "cpp" ]; then
	cd "$fullDirPath"
	g++ -O3 "$finalCodeOutputFileName" -o "$actualFileName.out"
	echo -e "All compilation done."
elif [ "$compilationTargetLower" == "porthos" ]; then
	cd - > /dev/null
	python3 ./HelperScripts/AthosToPorthosTemp.py "$fullDirPath/$finalCodeOutputFileName"
	# cd "$fullDirPath"
	# cp "$finalCodeOutputFileName" "$porthosFullDirPath/src/main.cpp"
	# cd "$porthosFullDirPath"
	# make -j
	echo -e "All compilation done."
fi
