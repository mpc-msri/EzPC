import os
import sys

if __name__ == "__main__" :					
	for name in ["dense", "dw", "pw", "dwpw", "halfshuffle", "fullshuffle"] :
		os.system(f"python3 helper.py --name {name}")
		os.system(f"python3 ../../CompileONNXGraph.py --config config.json --role server")

	os.system(f"mv *.ezpc EzPC/")
	os.system(f"mv *.inp Weights/")
	os.system(f"mv *.cpp CPP/")