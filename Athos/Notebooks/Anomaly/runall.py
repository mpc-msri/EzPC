import os
import sys

if __name__ == "__main__":
    dump = sys.argv[1]

    for name in ["dense", "dwpw", "halfshuffle", "fullshuffle"]:
        cmd = f"python3 run.py --name {name} --dump {dump}"
        # print(cmd)
        os.system(cmd)
