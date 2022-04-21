#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""

Authors: Saksham Gupta.

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

This was originally taken from https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon/examples/08_replacing_a_subgraph
and modified as below for our needs.

"""

import onnx_graphsurgeon as gs
import onnx


@gs.Graph.register()
def replace_with_hardsigmoid(self, inputs, outputs):

    # Disconnet input nodes of all output tensors
    for out in outputs:
        out.inputs.clear()

    # Insert the new node.
    return self.layer(op="HardSigmoid", inputs=inputs, outputs=outputs)


# Now we'll do the actual replacement
graph = gs.import_onnx(onnx.load("mobilenet_v3_large.onnx"))

tmap = graph.tensors()
# You can figure out the input and output tensors using Netron. In our case:
# Inputs: [inp1,inp2]
# Outputs: [output]
# Using dictonary to replace multiple subgraphs
dict = {
    "604": "319",
    "658": "394",
    "661": "402",
    "667": "412",
    "670": "420",
    "676": "431",
    "679": "439",
    "685": "450",
    "688": "458",
    "694": "469",
    "697": "477",
    "703": "493",
    "706": "501",
    "712": "518",
    "715": "526",
    "721": "542",
    "724": "550",
    "730": "567",
    "733": "575",
    "739": "592",
    "596": "601",
}
for key in dict:

    inputs = [tmap[key]]  # name of inputs of the subgraph to be removed
    outputs = [tmap[dict[key]]]  # name of output of the subgraph to be removes

    graph.replace_with_hardsigmoid(inputs, outputs)

    # Remove the now-dangling subgraph.
    graph.cleanup().toposort()

# That's it!
onnx.save(gs.export_onnx(graph), "replaced_mobile.onnx")
