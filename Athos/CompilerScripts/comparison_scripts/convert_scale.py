"""

Authors: Pratik Bhatu.

Copyright:
Copyright (c) 2021 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import sys

if __name__ == "__main__":
    assert len(sys.argv) == 4
    file_name = sys.argv[1]
    input_scale = int(sys.argv[2])
    output_scale = int(sys.argv[3])
    output_file_name = file_name + "_" + str(output_scale)
    output_file = open(output_file_name, "w")
    with open(file_name, "r") as a_file:
        for line in a_file:
            output = int(
                (float(int(line.strip())) / (2 ** input_scale)) * (2 ** output_scale)
            )
            output_file.write(str(output) + "\n")
    output_file.close()
