import os
import sys

if __name__ == "__main__" :
	for filt in [3, 5] :
		for dg in ["dense", "group"] :
			for bench in range(1, 5) :
				for nw in ["normal", "winograd"] :
					# print(f"python3 ../helper_bench.py --root .. --filter {filt} --dg {dg} --bench {bench} --nw {nw}")
					os.system(f"python3 ../helper_bench.py --root .. --filter {filt} --dg {dg} --bench {bench} --nw {nw}")
					os.system(f"python3 ../../../CompileONNXGraph.py --config config.json --role server")

					# sys.exit(0)


	os.system(f"mv *.ezpc EzPC/")
	os.system(f"mv *.inp Weights/")
	os.system(f"mv *.cpp CPP/")