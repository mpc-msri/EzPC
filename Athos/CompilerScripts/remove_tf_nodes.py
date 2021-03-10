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

import tensorflow as tf
import sys
import json
from json import JSONDecodeError
from tf_graph_io import load_graph_def_pb, dump_graph_def_pb


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            """Usage: python remove_node.py config.json
config.json should have the following fields.
{
  "model_name" : "model.pb",
  "nodes_to_remove" : ["loss", "model_outputs"]
}

"""
        )
        sys.exit()
    config_path = sys.argv[1]
    with open(config_path) as f:
        try:
            config = json.load(f)
        except JSONDecodeError as e:
            sys.exit(
                "Error while parsing the config json:\n"
                + e.msg
                + " at line no. "
                + str(e.lineno)
            )
    model_name = config["model_name"]
    nodes_to_remove = config["nodes_to_remove"]
    gd = load_graph_def_pb(model_name)
    to_remove = [n for n in gd.node if n.name in nodes_to_remove]
    for i in to_remove:
        gd.node.remove(i)
    removed = nodes_to_remove

    while removed:
        removed, prev_removed = set(), removed
        nodes = list(gd.node)
        for node in nodes:
            if any(inp in prev_removed for inp in node.input):
                gd.node.remove(node)
                removed.add(node.name)

    new_graph_name = "processed_" + model_name
    dump_graph_def_pb(gd, new_graph_name)
    print("Pruned graph is dumped in {}".format(new_graph_name))
