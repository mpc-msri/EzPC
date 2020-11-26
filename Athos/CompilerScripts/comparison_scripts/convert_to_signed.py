import sys

if __name__ == "__main__":
        assert(len(sys.argv) == 4)
        inp_fname = sys.argv[1]
        out_fname = sys.argv[2]
        bitlen = int(sys.argv[3])
        f = open(inp_fname, 'r')
        op = [(int(line.rstrip())) for line in f]
        f.close()
        f = open(out_fname, 'w')
        for i in op:
            f.write(str( i if (i<2**(bitlen-1)) else i - 2**bitlen) + '\n')
        f.close()
