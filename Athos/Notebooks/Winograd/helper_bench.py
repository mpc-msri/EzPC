import json
import subprocess
import os
import sys

import argparse
from argparse import RawTextHelpFormatter

def generate_config(net_path, net_name, input_shape, use_winograd) :
	config_file = open(os.path.join(net_path, "config.json"), 'w')
	config_dict = {
		"model_name" : net_name + ".onnx",
		"input_tensors" : {"input" : "{},{},{},{}".format(*input_shape)},
		"target" : "SCI",
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

	return parser.parse_args()

if __name__ == "__main__" :
	args = parse_args()
	is_dense = args.dg == "dense"
	is_winograd = args.nw == "winograd"

	shapes = {
	    "conv3_dense_bench1" : (1, 1, 6, 6),
	    "conv3_dense_bench2" : (1, 3, 46, 46),
	    "conv3_dense_bench3" : (1, 16, 128, 128),
	    "conv3_dense_bench4" : (1, 64, 256, 256),
	    
	    "conv5_dense_bench1" : (1, 1, 8, 8),
	    "conv5_dense_bench2" : (1, 3, 46, 46),
	    "conv5_dense_bench3" : (1, 16, 128, 128),
	    "conv5_dense_bench4" : (1, 64, 256, 256),
	    
	    "conv3_group_bench1" : (1, 3, 6, 6),
	    "conv3_group_bench2" : (1, 16, 46, 46),
	    "conv3_group_bench3" : (1, 32, 128, 128),
	    "conv3_group_bench4" : (1, 64, 256, 256),
	    
	    "conv5_group_bench1" : (1, 3, 8, 8),
	    "conv5_group_bench2" : (1, 16, 46, 46),
	    "conv5_group_bench3" : (1, 32, 128, 128),
	    "conv5_group_bench4" : (1, 64, 256, 256),
	}

	bench_str = f"conv{args.filter}_{args.dg}_bench{args.bench}"
	input_shape = shapes[bench_str]

	generate_config(
		os.path.join(args.root, f"Microbenches"),
		bench_str, 
		input_shape, 
		is_winograd
	)