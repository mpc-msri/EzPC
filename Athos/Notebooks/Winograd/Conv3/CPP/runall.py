import os

if __name__ == "__main__" :
	for dg in ["dense", "group"] :
		for sm in ["single", "multi"] :
			if dg == "group" and sm == "multi" :
				continue

			for nw in ["normal", "winograd"] :
				for ex in ["clear", "secure"] :
					cmd = f"python3 run.py --dg {dg} --sm {sm} --nw {nw} --exec {ex} --dump y"
					print(cmd)
					os.system(cmd)