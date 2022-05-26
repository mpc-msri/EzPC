import os
import sys

import argparse
from argparse import RawTextHelpFormatter

def parse_args() :
	parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

	parser.add_argument(
		"--filt",
		required=True,
		type=int,
		choices=[3, 5]
	)

	parser.add_argument(
		"--dg",
		required=True,
		type=str,
		choices=["dense", "group"]
	)

	parser.add_argument(
		"--bench",
		required=True,
		type=int,
		choices=[1, 2, 3, 4]
	)

	parser.add_argument(
		"--nw",
		required=True,
		type=str,
		choices=["normal", "winograd"]
	)

	parser.add_argument(
		"--dump",
		required=False,
		type=str,
		choices=['y', 'n']
	)

	return parser.parse_args()

if __name__ == "__main__" :
	args = parse_args()

	bench_str = f"conv{args.filt}_{args.dg}_bench{args.bench}"
	weight_name = f"{bench_str}_{args.nw}_weights23.inp"
	exe_name = f"{bench_str}_{args.nw}_secure"

	weight_path = os.path.join(f"Microbenches", "Weights", weight_name)
	exe_path = os.path.join(f"Microbenches", "CPP", "build", exe_name)

	dump_cmd = ""
	if args.dump == 'y' :
		dump_cmd = f"> {exe_path}.out"

	os.system(f"({exe_path} r=1 < {weight_path}) & ({exe_path} r=2 > /dev/null) {dump_cmd}")