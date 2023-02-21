from onnx.backend.base import BackendRep

from utils import logger
from utils.backend_helper import decl, comment, take_input, delete_variable, give_output
from Secfloat.func_calls import Operator
from utils.nodes import Node, Input, Output
from utils.onnx_nodes import OnnxNode


def process_delete_list(program):
    """
    Prepares a list of variables to be deleted after each function call if they are not needed in the program ahead.
    :param program: Program List
    :return: Variables Delete Order List
    """

    logger.debug("Processing Delete Variables Order List.")
    delete_order_list = [None] * len(program)
    counter = len(program) - 1
    deleted = []

    def update(dummy_list, variable):
        dummy_list.append(variable)
        deleted.append(variable)

    for node in reversed(program):
        if isinstance(node, Node):
            tmp_list = []
            [
                update(tmp_list, variable) if variable not in deleted else None
                for variable in node.inputs
            ]
            delete_order_list[counter] = tmp_list
        counter -= 1

    return delete_order_list


def check_variables_to_delete(delete_order_list, code_list, counter, var_dict, indent):
    """
    Checks and adds code to delete any variable if not needed any further.
    :param delete_order_list: Variables Delete Order List
    :param code_list: Code-List in CPP Format.
    :param counter: Counter for the Last Node processed
    :param var_dict: Variable Dictionary
    :param indent: Space Indentation
    :return: NA
    """
    if delete_order_list[counter] is not None:
        code_list.append(
            comment(
                f"Deleting Variable {delete_order_list[counter]} as they are not needed further.",
                indent + 1,
            )
        )
        [
            code_list.append(delete_variable(var_dict[variable], indent + 1))
            for variable in delete_order_list[counter]
        ]
        code_list.append("\n\n")


def prepare_input(code_list, node, var_dict, input_taken, indent):
    """
    Adds code for Input Nodes in Code-List in CPP Format.
    :param code_list: Code-List in CPP Format.
    :param node: Input Node to be processed.
    :param var_dict: Variable Dictionary.
    :param input_taken: List of variables already input to update it with new inputs.
    :param indent: Space Indentation.
    :return: NA
    """
    if isinstance(node, Input):
        code_list.append(
            comment(
                f"Declaration and Input for variable {node.name} of shape {node.shape} as {var_dict[node.name]}",
                indent + 1,
            )
        )
        # code_list.append(decl(var_dict[node.name], node.data_type, node.shape, indent + 1))
        code_list.append(
            take_input(var_dict[node.name], node.shape, node.party, indent + 1)
        )
        code_list.append("\n\n")
        input_taken.append(node.name)


def prepare_func(code_list, node, var_dict, value_info, input_taken, indent):
    """
    Adds code for Operator Nodes in Code-List in CPP Format.
    :param code_list: Code-List in CPP Format.
    :param node: Input Node to be processed.
    :param var_dict: Variable Dictionary.
    :param value_info: Dictionary {var}->(data-type,shape)
    :param input_taken: List of variables already input to check if arguments still need to be input.
    :param indent: Space Indentation.
    :return: NA
    """
    code_list.append(
        comment(
            f"Function Call to {node.op_type} with inputs {node.inputs} and gives output {node.outputs}",
            indent + 1,
        )
    )
    if node.outputs is not None:
        [
            code_list.append(
                decl(
                    var_dict[output],
                    value_info[str(output)][0],
                    value_info[str(output)][1],
                    indent + 1,
                )
            )
            for output in node.outputs
        ]
        input_taken += node.outputs

    operator = getattr(Operator, node.op_type)
    code_list.append(str(f'{"   " * (indent+1)}cout<<"Inside {node.op_type}"<<endl;'))
    code_list.append(
        operator(
            node.attrs, node.inputs, node.outputs, value_info, var_dict, indent + 1
        )
    )
    code_list.append("\n\n")


def prepare_output(code_list, node, var_dict, indent):
    """
    Adds code for Input Nodes in Code-List in CPP Format.
    :param code_list: Code-List in CPP Format.
    :param node: Input Node to be processed.
    :param var_dict: Variable Dictionary.
    :param indent: Space Indentation.
    :return: NA
    """
    if isinstance(node, Output):
        code_list.append(
            comment(
                f"Output of variable '{node.name}' of shape {node.shape} as {var_dict[node.name]} to {node.party.name}",
                indent + 1,
            )
        )
        code_list.append(
            give_output(var_dict[node.name], node.shape, node.party, indent + 1)
        )
        code_list.append("\n\n")


def prepare_export(program, var_dict, value_info, backend, file_path):
    """
    Prepares the Program List for export by converting it into cpp format.
    :param program: Program List having a list of Input, Nodes and Output nodes classes.
    :param var_dict: Variable Dictionary.
    :param value_info: Dictionary {var}->(data-type,shape).
    :return: Code-List in CPP Format.
    """
    code_list = []
    indent = 1
    input_taken = []  # list of variables already input
    input_dict = dict()

    if backend == "SECFLOAT":
        code_list.append(f'#include "{file_path}/lib_secfloat/common.cpp" \n\n\n')
        code_list.append(
            "int main(int __argc, char **__argv)\n{\n\n      __init(__argc, __argv);\n"
        )
    elif backend == "SECFLOAT_CLEARTEXT":
        code_list.append(
            f'#include "{file_path}/lib_cleartext/cleartext_common.cpp" \n\n\n'
        )
        code_list.append(
            "int main(int __argc, char **__argv)\n{\n\n     int __party=0;\n"
        )

    for node in program:

        func = getattr(OnnxNode, node.op_type)
        func(node)

    delete_order_list = process_delete_list(program)
    counter = 0

    logger.info("Starting Export...")
    for node in program:

        if isinstance(node, Input):
            input_dict[node.name] = node
        elif isinstance(node, Node):
            [
                prepare_input(
                    code_list, input_dict[input_var], var_dict, input_taken, indent
                )
                if input_var not in input_taken
                else None
                for input_var in node.inputs
            ]
            prepare_func(code_list, node, var_dict, value_info, input_taken, indent)
            if backend == "SECFLOAT_CLEARTEXT":
                check_variables_to_delete(
                    delete_order_list, code_list, counter, var_dict, indent
                )
        elif isinstance(node, Output):
            prepare_output(code_list, node, var_dict, indent)

        counter += 1

    code_list.append("      return 0;\n")
    code_list.append("}")
    logger.info("Completed Export.")

    return code_list


class FzpcBackendRep(BackendRep):
    """
    This is FzpcBackendRep Class for representing model in a particular backend rather than general onnx.
    Provides functionalities to export the model currently, can be extended to run models directly in future versions.
    """

    def __init__(self, program, value_info, var_dict, path, file_name, backend):
        self.program_AST = program
        self.value_info = value_info
        self.var_dict = var_dict
        self.path = path
        self.file_name = file_name
        self.backend = backend

    def export_model(self, file_path):
        """
        Exports the FzpcBackendRep to Secfloat Backend in .cpp format following the crypto protocols.
        :return: NA
        """
        logger.info("Preparing to export Model.")
        ct = "" if self.backend == "SECFLOAT" else "_ct"
        code_list = prepare_export(
            self.program_AST, self.var_dict, self.value_info, self.backend, file_path
        )

        with open(self.path + f"/{self.file_name}_secfloat{ct}.cpp", "w") as fp:
            fp.write("\n".join(code_list))

        logger.info(
            f"Secure Model File Saved in Secfloat format as {self.file_name}_secfloat{ct}.cpp"
        )


export_model = FzpcBackendRep.export_model
