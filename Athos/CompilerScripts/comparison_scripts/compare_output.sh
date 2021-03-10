# Authors: Pratik Bhatu.

# Copyright:
# Copyright (c) 2021 Microsoft Research
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
