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
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

tf.enable_eager_execution()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(
            "Usage: python convert_saved_model_to_frozen_graph.py saved_model_directory"
        )
    model_dir = sys.argv[1]
    saved_model = tf.saved_model.load_v2(model_dir)
    model = saved_model.signatures["serving_default"]
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
    )
    frozen_function = convert_variables_to_constants_v2(full_model)
    gd = frozen_function.graph.as_graph_def()
    with tf.io.gfile.GFile(model_dir + "/frozen_graph.pb", "wb") as f:
        f.write(gd.SerializeToString())
    print("Frozen graph saved in " + model_dir + "/frozen_graph.pb")
