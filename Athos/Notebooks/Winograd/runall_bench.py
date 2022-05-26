import os
import sys

if __name__ == "__main__" :
	dump = sys.argv[1]

	for filt in [3, 5] :
		for dg in ["dense", "group"] :
			for bench in range(1, 5) :
				for nw in ["normal", "winograd"] :
					cmd = f"python3 run_bench.py --filt {filt} --dg {dg} --nw {nw} --bench {bench} --dump {dump}"
					print(cmd)
					# os.system(f"python3 ../helper_bench.py --root .. --filter {filt} --dg {dg} --bench {bench} --nw {nw}")