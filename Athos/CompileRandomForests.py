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
import argparse
from argparse import RawTextHelpFormatter
import os
import os.path
import sys
import json

from RandomForests.convert_pickle_to_graphviz import convert_pickle_to_graphviz
from RandomForests.parse_graphviz_to_ezpc_input import parse_graphviz_to_ezpc_input
from RandomForests.patch_ezpc_code_params import patch_ezpc_code_params


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--task",
        required=False,
        type=str,
        choices=["cla", "reg"],
        help="""Choose cla for classificatin.
Choose reg for regression.
""",
    )
    parser.add_argument(
        "--no_features",
        required=False,
        type=int,
        help="Number of features in the dataset.",
    )
    parser.add_argument(
        "--model_type",
        required=False,
        type=str,
        choices=["tree", "forest"],
        help="""Choose tree for decision tree.
Choose forest for random forest.
""",
    )
    parser.add_argument(
        "--pickle",
        required=False,
        type=str,
        help="Path to the pickle file",
    )
    parser.add_argument(
        "--scale",
        required=False,
        type=int,
        default=10,
        help="Scaling factor for float -> fixedpt.",
    )
    parser.add_argument(
        "--bitlen",
        required=False,
        type=int,
        default=64,
        choices=[32, 64],
        help="Bit length to compile for.",
    )
    parser.add_argument(
        "--role",
        required=True,
        type=str,
        choices=["server", "client"],
        default="server",
        help="Pickle file owner is server, data owner is client",
    )
    parser.add_argument(
        "--config",
        required=False,
        type=str,
        help="Path to the client config file",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.role == "server":
        if args.pickle is None:
            print("Path to pickle file not specified. See --help for options")
        if args.model_type is None:
            print("Model type not specified. See --help for options")
        if args.no_features is None:
            print("Number of features not specified. See --help for options.")
        if args.task is None:
            print("Task is not specified. See --help for options.")
        if None in [args.pickle, args.model_type, args.no_features, args.task]:
            sys.exit()
    else:
        if args.config is None:
            print(
                "Path to the client config file not specified. See --help for options"
            )
            sys.exit()

    # args.task, args.model_type, args.no_features, args.pickle, args.scale, args.bitlen, args.config

    if args.role == "server":
        if not os.path.isfile(args.pickle):
            sys.exit("Pickle file (" + args.pickle + ") specified does not exist")

        pickle_dir = os.path.dirname(os.path.abspath(args.pickle))
        build_dir = os.path.join(pickle_dir, "ezpc_build_dir")
        os.system("rm -rf {build_dir}".format(build_dir=build_dir))
        os.mkdir(build_dir)

        # Dumps tree0, tree1, ..treeN.txt
        no_of_estim = convert_pickle_to_graphviz(
            args.pickle, args.task, args.model_type, build_dir
        )

        max_tree_depth = -1
        for i in range(0, no_of_estim):
            tree_file_path = os.path.join(build_dir, "tree" + str(i) + ".txt")
            max_depth = parse_graphviz_to_ezpc_input(
                tree_file_path, args.task, args.scale
            )
            max_tree_depth = max(max_tree_depth, max_depth)
        print("Parsed all trees in Random Forest")

        no_features = args.no_features
        scale = args.scale
        bitlen = args.bitlen

        client_json = {
            "no_of_trees": no_of_estim,
            "depth": max_tree_depth,
            "no_of_features": no_features,
            "scale": scale,
            "bitlen": bitlen,
        }
        json_path = os.path.join(build_dir, "client.json")
        with open(json_path, "w") as f:
            json.dump(client_json, f)
    else:
        if not os.path.isfile(args.config):
            sys.exit("Config file (" + args.config + ") specified does not exist")

        with open(args.config) as f:
            client_json = json.load(f)
        no_of_estim = client_json["no_of_trees"]
        max_tree_depth = client_json["depth"]
        no_features = client_json["no_of_features"]
        scale = client_json["scale"]
        bitlen = client_json["bitlen"]

        config_dir = os.path.dirname(os.path.abspath(args.config))
        build_dir = os.path.join(config_dir, "ezpc_build_dir")
        os.system("rm -rf {build_dir}".format(build_dir=build_dir))
        os.mkdir(build_dir)

    ezpc_file_name = "random_forest.ezpc"
    output_path = os.path.join(build_dir, ezpc_file_name)
    patch_ezpc_code_params(no_of_estim, max_tree_depth, no_features, scale, output_path)

    athos_dir = os.path.dirname(os.path.abspath(__file__))
    ezpc_dir = os.path.join(athos_dir, "../EzPC/EzPC/")
    os.system('cp "{ezpc}" "{ezpc_dir}"'.format(ezpc=output_path, ezpc_dir=ezpc_dir))
    os.chdir(ezpc_dir)
    ezpc_args = ""
    ezpc_args = "--bitlen {bl} --codegen {target} ".format(bl=bitlen, target="ABY")
    output_name = "random_forest0.cpp"
    os.system(
        'eval `opam config env`; ./ezpc.sh "{}" '.format(ezpc_file_name) + ezpc_args
    )
    os.system("./compile_aby.sh {}".format(output_name))
    output_binary_path = os.path.join(build_dir, "random_forest")
    os.system(
        'mv "{bin}" "{op_bin}"'.format(bin="random_forest0", op_bin=output_binary_path)
    )
    print("\n\n")
    print("Compiled binary: " + output_binary_path)

    if args.role == "server":
        model_weights = "weight_sf_" + str(scale) + ".inp"
        weights_path = os.path.join(build_dir, model_weights)
        print("Model weights dumped in " + weights_path)
        print("Send client.json to the client machine. Path: ", json_path)
    print("\n\n")
