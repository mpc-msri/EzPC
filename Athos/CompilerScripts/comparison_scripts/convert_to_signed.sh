SCRIPT_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
inp1=$1
temp_1=${inp1}_tmp_cmp

awk '$0==($0+0)' $inp1 > $temp_1

python3 ${SCRIPT_DIR}/convert_to_signed.py $temp_1 ${inp1}_signed

rm $temp_1
