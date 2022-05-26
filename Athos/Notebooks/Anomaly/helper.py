import json
import subprocess
import os
import sys

import argparse
from argparse import RawTextHelpFormatter

def generate_config(net_name) :
	config_file = open("config.json", 'w')
	config_dict = {
		"model_name" : "anom_" + net_name + ".onnx",
		"input_tensors" : {"input" : "1,64,80,80"},
		"target" : "SCI",
		"scale" : 23,
		"bitlength" : 60,
		"output_tensors" : ["output"],
		"winograd" : False
	}
	json.dump(config_dict, config_file)

def parse_args() :
	parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
	parser.add_argument(
		"--name",
		required=True,
		type=str,
		choices=["dense", "dw", "pw", "dwpw", "halfshuffle", "fullshuffle"]
	)

	return parser.parse_args()

if __name__ == "__main__" :
	args = parse_args()

	generate_config(
		args.name
	)