import sys

spaces = 4


def get_signature(N, D):
    sig = "def void Concat" + str(N) + "T" + (str(D) * (N + 1)) + "("
    # Output shape
    for i in range(D):
        sig += "int32_pl s" + str(i + 1) + ", "

    for t in range(N):
        # int32_pl inp1s1, int32_pl inp1s2,
        for i in range(D):
            sig += "int32_pl inp" + str(t + 1) + "s" + str(i + 1) + ", "
        # int64_al[inp1s1][inp1s2] inp1
        sig += "int64_al"
        for i in range(D):
            sig += "[inp" + str(t + 1) + "s" + str(i + 1) + "]"
        sig += " inp" + str(t + 1) + ", "

    # int32_pl axis, int64_al[s1][s2][s3][s4] outp
    sig += "int32_pl axis, int64_al"
    for i in range(D):
        sig += "[s" + str(i + 1) + "]"
    sig += " outp){\n"
    return sig


def ind(indent):
    return indent * spaces * " "


def generate_for_loop_prolog(D, indent):
    prolog = ""
    for i in range(D):
        prolog += ind(indent) + "for i" + str(i + 1) + "=[0:s" + str(i + 1) + "]{\n"
        indent += 1
    return (prolog, indent)


def generate_epilog(D, indent):
    epilog = ""
    for i in range(D):
        epilog += ind(indent) + "};\n"
        indent -= 1
    return (epilog, indent)


def get_output_access(D):
    access = ""
    for i in range(D):
        access += "[i" + str(i + 1) + "]"
    return access


# axis is 0 indexed
# tensor is 0 indexed
def get_input_access(tensor, axis, D):
    access = ""
    for i in range(D):
        if i != axis:
            access += "[i" + str(i + 1) + "]"
        else:
            access += "[i" + str(i + 1)
            for t in range(tensor):
                access += "-inp" + str(t + 1) + "s" + str(axis + 1)
            access += "]"
    return access


# tensor is 0 indexed

# tensor=0: inp1s1
# tensor=1: inp1s1 + inp2s1
def generate_bound(tensor, axis):
    sN = "s" + str(axis + 1)
    inp1 = "inp1"
    code = inp1 + sN
    # loop starts from second tensor(1)
    for i in range(1, tensor + 1):
        code += " + inp" + str(i + 1) + sN
    return code


def get_assgn_stmt(tensor, axis, D):
    code = "outp" + get_output_access(D) + " = "
    code += "inp" + str(tensor + 1) + get_input_access(tensor, axis, D) + ";\n"
    return code


def generate_concat_axis(axis, indent, N, D):
    code = ""
    loop_idx = "i" + str(axis + 1)
    for i in range(N - 1):
        code += ind(indent) + "if (" + loop_idx + " < ("
        code += generate_bound(i, axis)
        code += ")) {\n"
        indent += 1
        code += ind(indent) + get_assgn_stmt(i, axis, D)
        indent -= 1
        code += ind(indent) + "}\n"
        code += ind(indent) + "else {\n"
        indent += 1

    # Trailing else block
    code += ind(indent) + get_assgn_stmt(N - 1, axis, D)
    indent -= 1
    # code += ind(indent) + "};\n"
    # indent -= 1

    # close all branches
    # code+="CONCAT START\n"
    (close, indent) = generate_epilog(N - 1, indent)

    code += close
    # code+="CONCAT END\n"
    return (code, indent)


def generate_code(N, D):
    code = get_signature(N, D)
    indent = 1
    (prolog, indent) = generate_for_loop_prolog(D, indent)
    code += prolog
    for i in range(D - 1):
        code += ind(indent) + "if (axis == " + str(i) + "){\n"
        indent += 1
        (concat, indent) = generate_concat_axis(i, indent, N, D)
        code += concat
        code += ind(indent) + "}\n"
        code += ind(indent) + "else {\n"
        indent += 1

    # Trailing else block, last axis
    (concat, indent) = generate_concat_axis(D - 1, indent, N, D)
    code += concat
    # code+="AXIS START\n"

    # CLose all axis else conditions
    (epilog, indent) = generate_epilog(D - 1, indent)
    code += epilog
    # code+="AXIS END\n"

    # CLose all loops
    # code+="LOOP START\n"
    (epilog, indent) = generate_epilog(D, indent)
    code += epilog
    # code+="LOOP END\n"
    indent -= 1
    code += ind(indent) + "}\n"

    return code


#  indent
#  for i in range(N):

if __name__ == "__main__":
    N = int(sys.argv[1])  # num_tensors
    D = int(sys.argv[2])  # dim_tensors

    code = generate_code(N, D)
    print(code)
