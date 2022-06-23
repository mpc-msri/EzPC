#! env python
from __future__ import annotations
import json
import os
from time import sleep
import urllib
from enum import Enum
from urllib import request, parse
import subprocess
import numpy as np
import argparse
import os
import errno

args = {}


def make_executable(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2  # copy R bits to X
    os.chmod(path, mode)


def server_loop():
    print("Starting server loop", args.num_testcases, "times")
    print("Resetting old secret shares")
    with open("secret_shares.txt", "w") as secret_shares_file:
        secret_shares_file.write(f"{args.num_testcases}\n{args.num_classes}\n")
    for idx in range(args.num_testcases):
        print("Serving ", idx)
        with open(args.model_weights_file) as inp:
            with open(f"output_{idx}.txt", "w") as output:
                subprocess.run(
                    ["./model_SCI_OT.out", "r=1", f"port={args.port}"],
                    stdin=inp,
                    stdout=output,
                )


def compare_labels():
    print("Starting label comparisons")
    bin_url = f"{args.website_url}/static/contests/objects/compare_labels_nm"
    print("Downloading label comparison binary from website", bin_url)
    subprocess.run(["curl", "--output", "compare_labels", bin_url])
    make_executable("./compare_labels")
    print("Starting compare_labels script")
    with open("secret_shares.txt") as inp:
        with open("accuracy.txt", "w") as output:
            # compare_labels 2(server)/1(client) port [ip]
            subprocess.run(
                ["./compare_labels", "2", f"{args.port}"], stdin=inp, stdout=output
            )


if __name__ == "__main__":
    print("Starting participant script")
    parser = argparse.ArgumentParser(description="Script for contest participant.")
    parser.add_argument(
        "--website_url",
        type=str,
        required=True,
        help="Complete URL/IP address with port of the  contest website",
    )
    parser.add_argument(
        "--model_weights_file",
        type=str,
        required=True,
        help="Model input weights scaled fixedpoint file",
    )
    parser.add_argument(
        "--num_testcases", type=int, required=True, help="Number of testcases"
    )
    parser.add_argument(
        "--num_classes", type=int, required=True, help="Number of label classes"
    )
    parser.add_argument("--port", type=int, required=True, help="Port of the server")
    args = parser.parse_args()

    server_loop()
    compare_labels()
    print("Finished participant script.")

# python /home/dagrawal/mlcomp/contests/static/contests/scripts/participant_script.py --website_url http://127.0.0.1:8000 --num_testcases 3 --num_classes 5 --port 32000 --model_weights_file=model_input_weights_fixedpt_scale_15.inp
