import math
import os


def patch_ezpc_code_params(
    no_of_trees, depth, no_of_features, scaling_factor, output_path
):
    no_of_nodes = pow(2, depth) - 1
    script_path = os.path.dirname(os.path.abspath(__file__))
    rf_ezpc_file_path = os.path.join(script_path, "random_forest_base_file.ezpc")
    with open(rf_ezpc_file_path, "r") as f:
        lines = f.readlines()

    lines[3] = lines[3][:-1] + str(no_of_features) + ";\n"
    lines[5] = lines[5][:-1] + str(depth) + ";\n"
    lines[6] = lines[6][:-1] + str(depth) + ";\n"
    lines[7] = lines[7][:-1] + str(no_of_trees) + ";\n"
    lines[8] = lines[8][:-1] + str(no_of_nodes) + ";\n"

    with open(output_path, "w") as f:
        for i in range(len(lines)):
            f.write(lines[i])

    print("Generated ezpc code: " + output_path)
