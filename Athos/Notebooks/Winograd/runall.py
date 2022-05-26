import os
import sys

if __name__ == "__main__" :
	filt = int(sys.argv[1])
	dump = sys.argv[2]

	for dg in ["dense", "group"] :
		for sm in ["single", "multi"] :
			if dg == "group" and sm == "multi" :
				continue

			for nw in ["normal", "winograd"] :
				for ex in ["clear", "secure"] :
					cmd = f"python3 run.py --filt {filt} --dg {dg} --sm {sm} --nw {nw} --exec {ex} --dump {dump}"
					print(cmd)
					os.system(cmd)
