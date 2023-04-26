from LLAMA.sytorch_func_calls import Operator
from onnx.backend.base import BackendRep
from utils import logger
from utils.backend_helper import iterate_list
from utils.nodes import Node
from utils.onnx_nodes import OnnxNode


def func_call(node, value_info):
    """
    Maps the onnx node to the corresponding function call in the backend.
    """
    func_map = {
        "Relu": "ReLU",
        "Conv": f"{'Conv3D' if len(value_info[node.inputs[0]][1]) == 5 else 'Conv2D'}",
        "MaxPool": "MaxPool2D",
        "Flatten": "Flatten",
        "Gemm": "FC",
        "Concat": "concat",
        "BatchNormalization": "BatchNorm2dInference",
        "AveragePool": "AvgPool2D",
        "GlobalAveragePool": "GlobalAvgPool2D",
        "Add": "add",
        "ConvTranspose": "ConvTranspose3D",
    }
    return func_map[node.op_type]


non_sequential = ["Concat", "Add"]
tab_space = "     "


def create_func_names(program):
    """
    Creates name mapping for every node, as the node names may not be unique and may not even exist.
    :param program: The node list from onnx file.
    :return: node name list index wise.
    """
    size = len(program)
    node_names = [""] * size
    count = 0
    for idx, node in enumerate(program):
        if isinstance(node, Node):
            node_names[idx] = (node.op_type).lower() + str(count)
            count += 1
    # print(node_names)
    return node_names


def inputs_to_take(node):
    """
    Tells how many inputs are to be used during the forward pass
    :param node: node op type
    :return:
    """
    tmp_dict = {
        "Conv": 1,
        "Relu": 1,
        "MaxPool": 1,
        "Gemm": 1,
        "Flatten": 1,
        "AveragePool": 1,
        "Concat": -1,
        "Add": -1,
        "BatchNormalization": 1,
        "GlobalAveragePool": 1,
        "ConvTranspose": 1,
    }
    return tmp_dict[node]


def prepare_func(node, var_dict, value_info, input_taken, mode, indent):
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
    operator = getattr(Operator, node.op_type)
    return operator(
        node.attrs,
        node.inputs,
        node.outputs,
        value_info,
        var_dict,
        mode,
        indent + 1,
    )


def cleartext_pre(code_list, program, scale, mode, indent):
    code_list.append("#include <sytorch/layers/layers.h>")
    code_list.append("#include <sytorch/module.h>")
    code_list.append("#include <sytorch/utils.h>\n\n")


def cleartext_post(code_list, program, scale, mode, indent):
    # Input
    n = program[0].shape[0]
    c = program[0].shape[1]
    dims = program[0].shape[2:]
    # n, c, h, w = program[0].shape
    code_list.append(
        f"""

int main(int argc, char**__argv){'{'}

    prngWeights.SetSeed(osuCrypto::toBlock(0, 0));
    prngStr.SetSeed(osuCrypto::toBlock(time(NULL)));

    int party = atoi(__argv[1]);
    std::string ip = "127.0.0.1";

    srand(time(NULL));
    
    const u64 scale = {scale};

    if (party == 0) {'{'}
        Net<i64> net;
        net.init(scale);
        std::string weights_file = __argv[3];
        net.load(weights_file);
        Tensor<i64> input({'{'}{iterate_list([n]+ dims +[c])}{'}'});
        input.input_nchw(scale);
        print_dot_graph(net.root);
        net.forward(input);
        print(net.activation, scale, 64);
        return 0;
    {'}'}

{'}'}
        """
    )


def llama_pre(code_list, program, scale, mode, bitlength, indent):
    code_list.append("#include <sytorch/backend/llama_extended.h>")
    code_list.append("#include <sytorch/layers/layers.h>")
    code_list.append("#include <sytorch/module.h>")
    code_list.append("#include <sytorch/utils.h>\n\n")


def llama_post(code_list, program, scale, mode, bitlength, indent):
    # Input
    n = program[0].shape[0]
    c = program[0].shape[1]
    dims = program[0].shape[2:]
    # n, c, h, w = program[0].shape
    code_list.append(
        f"""
    
int main(int __argc, char**__argv){'{'}
    
    prngWeights.SetSeed(osuCrypto::toBlock(0, 0));
    prngStr.SetSeed(osuCrypto::toBlock(time(NULL)));

    int party = atoi(__argv[1]);
    std::string ip = "127.0.0.1";

    using LlamaVersion = LlamaExtended<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    
    const u64 scale = {scale};

    if (party == 0) {'{'}
        Net<i64> net;
        net.init(scale);
        std::string weights_file = __argv[3];
        net.load(weights_file);
        Tensor<i64> input({'{'}{iterate_list([n]+ dims +[c])}{'}'});
        input.input_nchw(scale);
        print_dot_graph(net.root);
        net.forward(input);
        print(net.activation, scale, 64);
        return 0;
    {'}'}

    LlamaConfig::bitlength = {bitlength};
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    LlamaConfig::num_threads = 4;
    if(__argc > 2){'{'}
        ip = __argv[2];
    {'}'}
    llama->init(ip, true);

    Net<u64> net;
    net.init(scale);
    net.setBackend(llama);
    net.optimize();
    if(party == SERVER){'{'}
        std::string weights_file = __argv[3];
        net.load(weights_file);
    {'}'}
    else if(party == DEALER){'{'}
        net.zero();
    {'}'}
    llama->initializeInferencePartyA(net.root);

    Tensor<u64> input({'{'}{iterate_list([n]+ dims +[c])}{'}'});
    if(party == CLIENT){'{'}
         input.input_nchw(scale);
    {'}'}
    llama->initializeInferencePartyB(input);

    llama::start();
    net.forward(input);
    llama::end();

    auto &output = net.activation;
    llama->outputA(output);
    if (party == CLIENT) {'{'}
        print(output, scale, LlamaConfig::bitlength);
    {'}'}
    llama->finalize();
{'}'}
    """
    )


def prepare_export(program, var_dict, value_info, mode, scale, bitlength, backend):
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
    logger.info("Starting Export...")

    # Check nodes for assertions and modifications
    for node in program:
        func = getattr(OnnxNode, node.op_type)
        func(node)

    # Start CPP program
    number_of_nodes = 0
    if backend == "CLEARTEXT_LLAMA":
        cleartext_pre(code_list, program, scale, mode, indent)
    elif backend == "LLAMA":
        llama_pre(code_list, program, scale, mode, bitlength, indent)

    node_names = create_func_names(program)

    # Start Class
    code_list.append(f"template <typename T>")
    code_list.append(f"class Net: public SytorchModule<T> {'{'}")
    code_list.append(f"{tab_space * (indent)}using SytorchModule<T>::add;")
    code_list.append(f"{tab_space * (indent)}using SytorchModule<T>::concat;")
    code_list.append(f"public:")

    # 1st Pass
    for idx, node in enumerate(program):
        if isinstance(node, Node) and node.op_type not in non_sequential:
            number_of_nodes += 1
            code_list.append(
                f"{tab_space * (indent)}{func_call(node, value_info)}<T> *{node_names[idx]};"
            )
    code_list.append(f"{tab_space * (indent)}\n\n")

    # 2nd Pass
    code_list.append(f"public:")
    code_list.append(f"{tab_space * (indent)}Net()")
    code_list.append(f"{tab_space * (indent)}{'{'}")
    for idx, node in enumerate(program):
        if isinstance(node, Node) and node.op_type not in non_sequential:
            code_list.append(
                f"{tab_space * (indent + 1)}{node_names[idx]} = {prepare_func(node, var_dict, value_info, input_taken, mode, 0)}"
            )
    code_list.append(f"{tab_space * (indent)}{'}'}\n")

    # 3rd Pass
    code_list.append(f"{tab_space * (indent)}Tensor<T>& _forward(Tensor<T> &input)")
    code_list.append(f"{tab_space * (indent)}{'{'}")
    for idx, node in enumerate(program):
        if isinstance(node, Node):
            if node.op_type in non_sequential:
                code_list.append(
                    f"{tab_space * (indent + 1)}auto &{var_dict[node.outputs[0]]} = {func_call(node, value_info)}({iterate_list([var_dict[x] for x in node.inputs])});"
                )
            else:
                code_list.append(
                    f"{tab_space * (indent + 1)}auto &{var_dict[node.outputs[0]]} = {node_names[idx]}->forward({iterate_list([var_dict[x] for x in node.inputs[:inputs_to_take(node.op_type)]])});"
                )
    code_list.append(f"{tab_space * (indent + 1)}return {var_dict[program[-1].name]};")
    code_list.append(f"{tab_space * (indent)}{'}'}\n")

    # End Class
    code_list.append("};\n")

    if backend == "CLEARTEXT_LLAMA":
        cleartext_post(code_list, program, scale, mode, indent)
    elif backend == "LLAMA":
        llama_post(code_list, program, scale, mode, bitlength, indent)

    logger.info("Completed Export.")

    return code_list


class SytorchBackendRep(BackendRep):
    """
    This is FzpcBackendRep Class for representing model in a particular backend rather than general onnx.
    Provides functionalities to export the model currently, can be extended to run models directly in future versions.
    """

    def __init__(self, program, value_info, var_dict, path, file_name):
        self.program_AST = program
        self.value_info = value_info
        self.var_dict = var_dict
        self.path = path
        self.file_name = file_name

    def export_model(self, mode, scale, bitlength, backend):
        """
        Exports the FzpcBackendRep to Secfloat Backend in .cpp format following the crypto protocols.
        :return: NA
        """
        logger.info(f"Preparing to export Model to {backend}")
        self.var_dict[self.program_AST[0].name] = "input"
        code_list = prepare_export(
            self.program_AST,
            self.var_dict,
            self.value_info,
            mode,
            scale,
            bitlength,
            backend,
        )
        logger.info(
            f"Secure Model File Saved in Secfloat format as {self.file_name}_{backend}_{scale}.cpp"
        )

        with open(self.path + f"/{self.file_name}_{backend}_{scale}.cpp", "w") as fp:
            fp.write("\n".join(code_list))


export_model = SytorchBackendRep.export_model
