SCRIPT_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
inp1=$1
bitlen=$2
if [ -z "$bitlen" ]
then
	echo "Please pass bitlen."

	exit 1
fi

temp_1=${inp1}_tmp_cmp

awk '$0==($0+0)' $inp1 > $temp_1

python3 ${SCRIPT_DIR}/convert_to_signed.py $temp_1 ${inp1}_signed $bitlen

rm $temp_1
