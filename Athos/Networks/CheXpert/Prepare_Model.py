"""

Authors: Saksham Gupta.

Copyright:
Copyright (c) 2020 Microsoft Research
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

import os
import sys
import Util
import numpy


def main():

    (
        all_testing_features,
        all_testing_labels,
        all_testing_ids,
    ) = Util.load_preprocess_validation_data()
    for ii, curFeature in enumerate(all_testing_features):
        imgFileName = "./PreProcessedImages/model_input_{}.inp".format(ii)

        # Passing scale as 0 because the data being loaded is already scaled.
        print("Dumping image data...")
        with open(imgFileName, "w") as ff:
            for xx in numpy.nditer(curFeature, order="C"):
                ff.write(str(int(xx)) + "\n")


if __name__ == "__main__":
    main()
