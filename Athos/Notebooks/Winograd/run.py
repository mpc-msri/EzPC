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
		"--sm",
		required=False,
		type=str,
		choices=["single", "multi"]
	)

	parser.add_argument(
		"--nw",
		required=True,
		type=str,
		choices=["normal", "winograd"]
	)

	parser.add_argument(
		"--exec",
		required=True,
		type=str,
		choices=["clear", "secure"]
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
	is_dense = args.dg == "dense"
	is_single = args.sm == "single"
	is_normal = args.nw == "normal"
	is_clear = args.exec == "clear"
	is_dump = args.dump == 'y'

	suff = str(args.dg)
	if is_dense :
		suff = suff + f"_{args.sm}"

	input_name = f"conv{args.filt}_{suff}_input.inp"
	weight_name = f"conv{args.filt}_{suff}_{args.nw}_weights23.inp"
	exe_name = f"conv{args.filt}_{suff}_{args.nw}_{args.exec}"

	input_path = os.path.join(f"Conv{args.filt}", "Inputs/", input_name)
	weight_path = os.path.join(f"Conv{args.filt}", "Weights/", weight_name)
	exe_path = os.path.join(f"Conv{args.filt}", "CPP", "build", exe_name)

	dump_cmd = ""
	if is_dump :
		dump_cmd = f"> {exe_path}.out"

	if is_clear :
		os.system(f"cat {input_path} {weight_path} | {exe_path} {dump_cmd}")
		# print(f"cat {input_path} {weight_path} | {exe_path}")
	else :
		os.system(f"({exe_path} r=1 < {weight_path}) & ({exe_path} r=2 < {input_path} > /dev/null) {dump_cmd}")

		# os.system(f"({exe_path} r=1 < {weight_path} > /dev/null) & ({exe_path} r=2 < {input_path}) {dump_cmd}")
		# print(f"({exe_path} r=1 < {weight_path}) & ({exe_path} r=2 < {input_path} > /dev/null)")
