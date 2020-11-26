# Usage: tf_output.float(floatingpt) party0_output(fixedpt) BITLEN SCALING_FACTOR PRECISION(upto how many points to compare?)
# This first converts unsigned fixedpt to signed
SCRIPT_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
echo "Comparing output with tensorflow output upto $5 decimal points."
$SCRIPT_DIR/convert_to_signed.sh $2 $3
#Then runs the comparison script on it.
python3 $SCRIPT_DIR/compare_output.py $1 $2_signed $4 $5
if [ "$?" -eq 0 ]; then
	echo "Output matches upto ${5} decimal points"
else
	echo "Output does not match upto ${5} decimal points"
fi
