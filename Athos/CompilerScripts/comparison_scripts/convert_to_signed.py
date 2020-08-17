import sys

if __name__ == "__main__":
        assert(len(sys.argv) == 3)
        inp_fname = sys.argv[1]
        out_fname = sys.argv[2]
        f = open(inp_fname, 'r')
        op = [(int(line.rstrip())) for line in f]
        f.close()
        f = open(out_fname, 'w')
        for i in op:
            f.write(str( i if (i<2**63) else i - 2**64) + '\n')
        f.close()
