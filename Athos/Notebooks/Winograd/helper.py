import json
import subprocess
import os
import sys

import argparse
from argparse import RawTextHelpFormatter

def generate_config(net_path, net_name, input_shape, use_winograd=False, secure=False) :
	config_file = open(os.path.join(net_path, "config.json"), 'w')
	config_dict = {
		"model_name" : net_name + ".onnx",
		"input_tensors" : {"input" : "{},{},{},{}".format(*input_shape)},
		"target" : "SCI" if secure else "CPP",
		"scale" : 23,
		"bitlength" : 60,
		"output_tensors" : ["output"],
		"winograd" : use_winograd
	}
	json.dump(config_dict, config_file)

def parse_args() :
	parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

	parser.add_argument(
		"--root",
		required=True,
		type=str
	)

	parser.add_argument(
		"--filter",
		required=True,
		type=int
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

	return parser.parse_args()

if __name__ == "__main__" :
	args = parse_args()
	is_dense = args.dg == "dense"
	is_single = args.sm == "single"

	input_shape = None
	if is_single :
		input_shape = (1, 1, 10, 10)
	else :
		input_shape = (1, 4, 10, 10)

	net_suff = ""
	if is_dense :
		net_suff = f"_{args.dg}_{args.sm}"
	else :
		net_suff = f"_{args.dg}"


	generate_config(
		os.path.join(args.root, f"Conv{args.filter}"),
		# f"Conv{args.filter}", 
		f"conv{args.filter}" + net_suff, 
		input_shape, 
		args.nw == "winograd", 
		args.exec == "secure"
	)