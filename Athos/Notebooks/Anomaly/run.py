import os
import sys

import argparse
from argparse import RawTextHelpFormatter


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--name",
        required=True,
        type=str,
        choices=["dense", "pw", "dw", "dwpw", "halfshuffle", "fullshuffle"],
    )

    parser.add_argument("--dump", required=True, type=str, choices=["y", "n"])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    weight_name = f"anom_{args.name}_normal_weights23.inp"
    exe_name = f"{args.name}"

    weight_path = os.path.join("Weights", weight_name)
    exe_path = os.path.join("CPP", "build", exe_name)

    dump_cmd = ""
    if args.dump == "y":
        dump_cmd = f"> {exe_path}.out"

    os.system(
        f"({exe_path} r=1 < {weight_path}) & ({exe_path} r=2 > /dev/null) {dump_cmd}"
    )
