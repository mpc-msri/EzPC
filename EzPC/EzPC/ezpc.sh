#!/bin/bash

POSITIONAL=()
allArgs=""
while [[ $# -gt 0 ]]
do
	key="$1"

	case $key in
		--bitlen)
		BITLEN="$2"
		allArgs="${allArgs} $1 $2"
		shift # past argument
		shift # past value
		;;
		--o_prefix)
		OPREFIX="$2"
		allArgs="${allArgs} $1 $2"
		shift # past argument
		shift # past value
		;;
		--help)
		HELP=Y
		shift # past one arg
		;;
		--codegen)
		CODEGEN="$2"
		allArgs="${allArgs} $1 $2"
		shift # past argument
		shift # past value
		;;
		--bool_sharing| --shares_dir)
		allArgs="${allArgs} $1 $2"
		shift # past argument
		shift # past value
		;;
		--disable-tac| --disable-cse| --dummy_inputs| --debug_partitions)
		allArgs="${allArgs} $1"
		shift # past one arg
		;;
		*)    # unknown option
		# allArgs="${allArgs} $1"
		POSITIONAL+=("$1") # save it in an array for later
		shift # past argument
		;;
	esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

if [ -z ${HELP} ]; then
	# Set bitlen to 32 if not specified in command line args
	if [ -z ${BITLEN} ]; then BITLEN=32; fi
	FILENAME="$1"
	if [ -z ${FILENAME} ]; then
		echo "Error::FileName not specified."
		exit 1
	fi

	# Now that bitlen and fileName are known, run C++ preprocessor.
	### 	First find the library name to #include.
	currentDirPath=$(pwd)
	LIBRARYNAME="${currentDirPath}/Library/Library${BITLEN}.ezpc"
	###		Next find the filename, extension and thus the new file names
	fullFilePath=$(dirname "$FILENAME")
	baseFileName=$(basename -- "$FILENAME")
	extension="${baseFileName##*.}"
	actualFileName="${baseFileName%.*}"
	newFileNamePrefix="${fullFilePath}/${actualFileName}"
	newFileName1="${newFileNamePrefix}__temp1.${extension}"
	newFileName2="${newFileNamePrefix}.${extension}"
	addEzPCLib=false

	if [ -z ${CODEGEN} ]; then
		addEzPCLib=true
	else
		codegenUpperCase=$(echo "${CODEGEN}" | awk '{print toupper($0)}')
		if [ "$codegenUpperCase" == "ABY" ] || [ "$codegenUpperCase" == "CPP" ] ; then
			addEzPCLib=true
		fi
	fi

	if [ "$addEzPCLib" = true ]; then
		newFileName2="${newFileNamePrefix}__temp2.${extension}"
		### 	Next echo the #include line to the top of the file.
		echo -e "#include \"${LIBRARYNAME}\"\n" | cat - ${FILENAME} > ${newFileName1}
		###		Finally run the C++ preprocessor
		echo -e "Running cpp preprocessor to include library files :::"
		cpp -E -P "${newFileName1}" > "${newFileName2}"
	fi

	# Now run the actual ezpc compiler on newFileName2
	echo -e "Running ezpc compiler on generated file :::"
	###		One more thing before running the compiler -- if the output file prefix is specified, use that, else use orig file name prefix
	if [ -z ${OPREFIX} ]; then 
		OPREFIX="${newFileNamePrefix}"
	fi
	command="./ezpc "${allArgs}" --o_prefix "${OPREFIX}" "${newFileName2}""
	echo "./ezpc $allArgs --o_prefix "${OPREFIX}" "${newFileName2}""
	./ezpc $allArgs --o_prefix "${OPREFIX}" "${newFileName2}"
	rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
	#rm "${newFileName1}" "${newFileName2}"
else
	./ezpc --help
fi
