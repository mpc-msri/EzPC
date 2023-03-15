from utils import VariableGen, Party


# Contains Functions to generate cpp format code in string Data Type.


def comment(string, indent):
    return str(f"{'   ' * indent}//{string}")


def decl(name, data_type, shape, indent, party=Party.ALICE):
    comma = ","
    return str(
        f"{'   ' * indent}auto {name} = make_vector_float({party.name},{comma.join(str(sh) for sh in shape)});"
    )


def if_stmnt(stmnt, indent, party=Party.ALICE):
    return str(
        f"{'   ' * indent}if(party == {party.name}){'{'}\n"
        f"{'   ' * (indent + 1)}{stmnt}\n"
        f"{'   ' * indent}{'}'}\n"
    )


def decl_multiple_int(variables, indent):
    need = ""
    return str(
        f"{'   ' * indent}int32_t {need.join(f'{v}=0,' for v in variables[:-1])}{variables[-1]}=0;\n"
    )


def generate_loop_vars(number):
    var_list = []
    [var_list.append(VariableGen.get_loop_var()) for i in range(0, number)]
    # VariableGen.reset_loop_var_counter()
    return var_list


def generate_reshape_vars(number):
    var_list = []
    [var_list.append(VariableGen.get_reshape_var()) for i in range(0, number)]
    return var_list


def nested_for_input_loop(counter, shape, variables, name, indent):
    need = ""
    open_braces = "{"
    close_braces = "}"
    loop = (
        f"{'   ' * indent}for(int {variables[counter]} = 0; "
        f"{variables[counter]} < {shape[counter]}; {variables[counter]}++)"
        f"{open_braces}\n"
    )
    loop += (
        f"{'   ' * (indent + 1)}cin>>{name}{need.join(f'[{v}]' for v in variables)}"
        if counter + 1 == len(shape)
        else nested_for_input_loop(counter + 1, shape, variables, name, indent + 1)
    )
    loop += f"\n{'   ' * indent}{close_braces}"
    return loop


def reshape_helper(counter, shape, variables, indent):
    l = len(shape)
    open_braces = "{"
    close_braces = "}"
    code = f"{'   ' * indent}{variables[l - counter - 1]} += 1;\n\n"
    if len(shape) != counter + 1:
        code += (
            f"{'   ' * indent}if({variables[l - counter - 1]} == {shape[l - counter - 1]})\n"
            f"{'   ' * indent}{open_braces}\n"
        )
        code += f"{'   ' * (indent + 1)}{variables[l - counter - 1]} = 0;\n"
        code += reshape_helper(counter + 1, shape, variables, indent + 1)
        code += f"\n{'   ' * indent}{close_braces}"
    return code


def nested_for_reshape_loop(
    counter1, shape1, variables1, counter2, shape2, variables2, name1, name2, indent
):
    need = ""
    open_braces = "{"
    close_braces = "}"
    loop = (
        f"{'   ' * indent}for(int {variables2[counter2]} = 0; "
        f"{variables2[counter2]} < {shape2[counter2]}; {variables2[counter2]}++)\n"
        f"{'   ' * indent}{open_braces}\n"
    )
    if counter2 + 1 == len(shape2):
        loop += (
            f"{'   ' * (indent + 1)}{name2}{need.join(f'[{v}]' for v in variables2)} "
            f"= {name1}{need.join(f'[{v}]' for v in variables1)};\n\n"
        )
        loop += reshape_helper(counter1, shape1, variables1, indent + 1)
    else:
        loop += nested_for_reshape_loop(
            counter1,
            shape1,
            variables1,
            counter2 + 1,
            shape2,
            variables2,
            name1,
            name2,
            indent + 1,
        )
    loop += f"\n{'   ' * indent}{close_braces}\n"
    return loop


def take_input(name, shape, party, indent):
    variables = generate_loop_vars(len(shape))
    VariableGen.reset_loop_var_counter()
    dim = len(shape)
    comma = ","
    statement = str(
        f"{'   ' * indent}if(__party=={party.name})cout<<\"Input {name}:\"<<endl;\n"
    )
    # return nested_for_loop(0, shape, variables, name, indent)
    return (
        statement
        + f"{'   ' * indent}auto {name} = input{dim}({comma.join(str(sh) for sh in shape)},{party.name});"
    )


def give_output(name, shape, party, indent):
    dim = len(shape)
    comma = ","
    return f"{'   ' * indent}output{dim}({name},{comma.join(str(sh) for sh in shape)},{party.name});"


def iterate_list(var_list):
    comma = ", "
    return f"{comma.join((str(var) for var in var_list))}"


def iterate_concat_list(num_list):
    comma = ""
    return f"{comma.join((str(n) for n in num_list))}"


def concat_list(value_info, var_dict, input):
    return f"{iterate_list(value_info[input][1])}, {var_dict[input]}"


def iterate_dict(var_dict):
    comma = ", "
    return f"{comma.join(((iterate_list(var_dict[key]) if isinstance(var_dict[key], list) else str(var_dict[key])) for key in var_dict))} "


def fun_call(attributes, inputs, output, value_info, var_dict, indent):
    return str(
        f"{'   ' * indent}name({(iterate_dict(attributes) if attributes else '')}{',' if attributes else ''}"
        f"{iterate_list(inputs)},{iterate_list(output)});"
    )


def delete_variable(name, indent):
    return str(
        f"{'   ' * indent}{name}.clear();\n"
        f"{'   ' * indent}{name}.shrink_to_fit();\n"
    )


def get_n_h_c_w(listt):
    if len(listt) == 4:
        n, c, h, w = listt
        return [n, h, w, c]
    elif len(listt) == 2:
        n, c = listt
        return [n, c, 1, 1]
    else:
        exit()
