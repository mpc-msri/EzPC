import sys
from enum import Enum
from functools import reduce

BITS_64 = 8
BITS_8 = 1

max_relu_mem = 0.0
max_conv_mem = 0.0
max_relu_idx = -1
max_conv_idx = -1


class FunctionCallType(Enum):
    _RELU = 1
    _CONV3D = 2


def extract_call_params(call_type, call_tokens):
    params = []
    bound = 0
    if call_type is FunctionCallType._RELU:
        bound = 6
    if call_type is FunctionCallType._CONV3D:
        bound = 19
    for i in range(len(call_tokens)):
        if i > 0 and i < bound:
            call_tokens[i] = call_tokens[i][call_tokens[i].find(")") + 1 : -1]
            params.append(int(call_tokens[i]))
    return params


def get_footprint(call_type, call_tokens, idx):
    call_params = []
    global max_relu_mem, max_conv_mem, max_relu_idx, max_conv_idx
    memory_primary = 0.0  # Bytes
    memory_helper = 0.0  # Bytes
    if call_type is FunctionCallType._RELU:
        call_params = extract_call_params(call_type, call_tokens)
        relu_size = reduce(lambda x, y: x * y, call_params)
        memory_primary += 4 * relu_size * BITS_64
        memory_primary += relu_size * BITS_64
        memory_primary += relu_size * BITS_64
        memory_helper = memory_primary
        # ShareConvert/ComputeMSB
        memory_primary += (
            4 * relu_size * BITS_64 + 3 * relu_size * BITS_8 + 64 * relu_size * BITS_8
        )
        memory_helper += relu_size * BITS_8
        # Coming from DotProduct in ComputeMSB
        memory_primary += 9 * relu_size * BITS_64
        # considering porthos optimizations
        memory_helper += (
            5 * relu_size * BITS_64 + relu_size * BITS_8 + 2 * 64 * relu_size * BITS_8
        )
        temp = max_relu_mem
        max_relu_mem = max(max_relu_mem, 2 * memory_primary + memory_helper)
        if temp != max_relu_mem:
            max_relu_idx = idx

    if call_type is FunctionCallType._CONV3D:
        (
            n,
            d,
            h,
            w,
            ci,
            fd,
            fh,
            fw,
            co,
            zpadDleft,
            zpadDright,
            zpadHleft,
            zpadHright,
            zpadWleft,
            zpadWright,
            strideD,
            strideH,
            strideW,
        ) = extract_call_params(call_type, call_tokens)
        reshapedfilterrows = co
        reshapedfiltercols = fd * fh * fw * ci
        reshapedinputrows = reshapedfiltercols
        newD = int(((d + (zpadDleft + zpadDright)) - fd) / strideD) + 1
        newH = int(((h + (zpadHleft + zpadHright)) - fh) / strideH) + 1
        newW = int(((w + (zpadWleft + zpadWright)) - fw) / strideW) + 1
        reshapedinputcols = n * newD * newH * newW
        rows = reshapedfilterrows
        common_dim = reshapedfiltercols
        cols = reshapedinputcols
        size_left = rows * common_dim * BITS_64
        size_right = common_dim * cols * BITS_64
        size = rows * cols * BITS_64

        memory_primary += size + size_left + size_right
        # Enter MatMulCSF2D
        memory_primary += size + size_left + size_right
        # Enter funcMatMulMPC
        memory_primary += size + size_left + size_right
        memory_helper = memory_primary
        memory_helper += 2 * (size + size_left + size_right)
        memory_primary += 2 * (size_left + size_right) + size
        # EigenMult
        memory_helper += size + size_left + size_right
        # Send2Vectors and Recv2Vectors in parallel
        memory_primary += 2 * (size + size_left + size_right)
        temp = max_conv_mem
        max_conv_mem = max(max_conv_mem, 2 * memory_primary + memory_helper)
        if temp != max_conv_mem:
            max_conv_idx = idx

    return (
        memory_primary / 10.0 ** 6,
        memory_primary / 10.0 ** 6,
        memory_helper / 10.0 ** 6,
        "MB",
    )


if len(sys.argv) != 2:
    print("Usage: python3 get_memory_footprint.py <cpp file relative path>")
    exit(0)
fname = sys.argv[1]

f = open(fname, "r")
print("Parsing cpp file: " + fname + "...")
lines = f.readlines()

conv_calls = []
relu_calls = []

for i in range(len(lines)):
    tokens = lines[i][:-1].split()
    if len(tokens) == 0:
        continue
    if tokens[0] == "Conv3DCSF(":
        conv_calls.append(list(tokens))
    if tokens[0] == "Relu5(":
        relu_calls.append(list(tokens))

print("\nReLU5 calls:")
for i in range(len(relu_calls)):
    print(get_footprint(FunctionCallType._RELU, relu_calls[i], i))

print("\nConv3D calls:")
for i in range(len(conv_calls)):
    print(get_footprint(FunctionCallType._CONV3D, conv_calls[i], i))

print("\n================== Statistics =================")
print("No. of Convolution3D calls:", len(conv_calls))
print("No. of ReLU calls:         ", len(relu_calls))
print("NOTE: the memory does not include incoming vector sizes")
print(
    "Maximum ReLU mem combined :",
    "{:0.3f}".format(max_relu_mem / 10.0 ** 6),
    "MB | ReLU #",
    max_relu_idx,
)
print(
    "Maximum Conv mem combined :",
    "{:0.3f}".format(max_conv_mem / 10.0 ** 6),
    "MB | Conv #",
    max_conv_idx,
)
print(
    "Trivially Conv mem can be :",
    "{:0.3f}".format((17.0 / 20.0) * max_conv_mem / 10.0 ** 6),
    "MB | Conv #",
    max_conv_idx,
)
