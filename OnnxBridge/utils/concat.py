ob, cb = "{", "}"


def get_axis_body_(i, n, axis):
    ax1 = axis + 1

    sub_ind = f"i{ax1}"
    for j in range(1, i):
        sub_ind += f"-inp{j}s{ax1}"

    if ax1 == 1:
        sq_braks = f"[{sub_ind}][i2][i3][i4]"
    elif ax1 == 2:
        sq_braks = f"[i1][{sub_ind}][i3][i4]"
    elif ax1 == 3:
        sq_braks = f"[i1][i2][{sub_ind}][i4]"
    else:
        sq_braks = f"[i1][i2][i3][{sub_ind}]"

    assgn_stmt = f"outp[i1][i2][i3][i4] = inp{i}{sq_braks} ;"

    if i == n:
        return assgn_stmt
    else:
        summ = f"inp1s{ax1}"
        for j in range(2, i + 1):
            summ += f"+inp{j}s{ax1}"

        else_body = get_axis_body_(i + 1, n, axis)
        if_else = f"\
if (i{ax1} < ({summ})) {ob}\n\
{assgn_stmt}\n\
{cb}\n\
else {ob}\n\
{else_body}\n\
{cb}\n"
        return if_else


def get_axis_body(n, axis):
    return get_axis_body_(1, n, axis)


def get_function(n, backend=None):
    dtype = "FPArray" if backend == "SECFLOAT" else "float"

    # Function arguments
    inps = ""
    for i in range(n):
        st = ""
        for j in range(4):
            st += f"int32_t inp{i+1}s{j+1}, "
        st += f"vector<vector<vector<vector<{dtype}>>>> &inp{i+1}, "

        inps += st
    args = f"int32_t s1, int32_t s2, int32_t s3, int32_t s4, {inps}int32_t axis, vector<vector<vector<vector<{dtype}>>>> &outp"

    # Function signature
    func_name = f"Concat{n}T{'4'*(n+1)}"
    func_signature = f"void {func_name} ({args})"

    # Conditional body
    axis_body = [get_axis_body(n, ax) for ax in range(4)]
    cond_body = f"\
if (axis==0) {ob}\n\
    {axis_body[0]}\n\
{cb}\n\
    else if (axis==1) {ob}\n\
    {axis_body[1]}\n\
{cb}\n\
    else if (axis==2) {ob}\n\
    {axis_body[2]}\n\
{cb}\n\
    else {ob}\n\
    {axis_body[3]}\n\
{cb}\n\
"
    # loop body
    func_body = f"\
for (uint32_t i1 = 0; i1 < s1 ; i1++) {ob}\n\
for (uint32_t i2 = 0; i2 < s2 ; i2++) {ob}\n\
for (uint32_t i3 = 0; i3 < s3 ; i3++) {ob}\n\
for (uint32_t i4 = 0; i4 < s4 ; i4++) {ob}\n\
{cond_body}\n\
{cb} \n\
{cb} \n\
{cb} \n\
{cb} \n\
"

    # Total body
    total = f"\
{func_signature} {ob}\n\
{func_body}\n\
{cb}\n\
"

    return total


def write_concat_implementations(concat_list, backend, file_path):
    with open(file_path, "w") as fp:
        for n in concat_list:
            fp.write(get_function(n, backend))
