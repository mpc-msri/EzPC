import os, sys

import onnx.checker
from onnx.backend.base import Backend

from LLAMA.sytorchBackendRep import SytorchBackendRep
from Secfloat.backendRep import FzpcBackendRep
from utils import logger, support_device, optimizations, VariableGen
from utils.nodes import (
    Node,
    process_input_nodes,
    process_func_nodes,
    process_output_nodes,
    Input,
    print_nodes,
)
from utils.onnx2IR_helper import get_node_metadata


def create_dict(program):
    """
    Creates a dictionary for the variable names in onnx file since they are not always a valid identifier.
    :param program: The node list from onnx file.
    :return: Variable dictionary.
    """
    var_dict = dict()
    for node in program:
        if isinstance(node, Input):
            if node.name not in var_dict:
                var_dict[node.name] = VariableGen.get_var()
        elif isinstance(node, Node):
            for output in node.outputs:
                if output not in var_dict:
                    var_dict[output] = VariableGen.get_var()
    return var_dict


class IR(Backend):
    """
    This is Intermediate Representation for Onnx.
    This Class gives api methods to prepare a model and run it.
    """

    @classmethod
    def preprocess_model(cls, model_fname, logging_level, backend):
        """
        Preprocesses the onnx file, which includes:
        Optimising
        Shape Inference
        Save Model Weights to a file
        Strip Model of its weights.
        :param model_fname: Model path
        :param logging_level: Logging Level
        :return: Stripped Model
        """
        logger.setLevel(logging_level)
        logger.handlers[0].setLevel(logging_level)
        logger.info("Application Started")

        model_name = os.path.basename(model_fname)[:-5]
        model_abs_dir = os.path.dirname(os.path.abspath(model_fname))
        logger.info(f"Loading onnx graph: {model_name}")
        model = onnx.load(model_fname)

        Node.opset_version = model.opset_import[0].version
        logger.info(f"Model Received : opset version : {Node.opset_version}")

        batch_size = optimizations.check_batch_size(model)
        logger.info(f"Batch Size : {batch_size}")
        if batch_size == 0:
            logger.error("Batch Size 0 is not supported")
            sys.exit()

        model = optimizations.optimise(model)
        logger.info("Model Optimized")

        model = optimizations.infer_shapes(model)
        logger.info("Shape Inference Done")

        is_compatible, unsupported_nodes = cls.is_compatible(model, backend)

        if is_compatible:
            logger.info("Model is OK!")
        else:
            newline = "\n  "
            logger.error(
                f"\nUnsupported Nodes: \n  {newline.join(node for node in unsupported_nodes)}\n "
            )
            logger.error("Model Not Supported")
            sys.exit()

        if backend in ["CLEARTEXT_LLAMA", "LLAMA"]:
            weights_path = optimizations.dump_model_weights_as_dat(
                model, model_abs_dir, model_name
            )
        elif backend in ["SECFLOAT", "SECFLOAT_CLEARTEXT"]:
            weights_path = optimizations.dump_model_weights_as_inp(
                model, model_abs_dir, model_name
            )

        logger.info(f"Dumping model weights in:\n {weights_path}")
        logger.info(f"These are to be used as input for party owning the model.")

        stripped_model = optimizations.strip_weights(model)
        pruned_model_path = os.path.join(
            model_abs_dir, "optimised_" + model_name + ".onnx"
        )
        onnx.save(stripped_model, pruned_model_path)

        model = onnx.load(pruned_model_path)
        return model

    @classmethod
    def is_compatible(cls, model, backend, device: str = "2PC", **kwargs):
        """
        Checks whether the model is compatible with the backend.
        :param model: The model to br checked.
        :param device: 2PC by default for secure MPC.
        :param kwargs: any extra params.
        :return: bool.
        """
        not_supported = []
        implemented_sytorch = [
            "Relu",
            "Softmax",
            "Conv",
            "MaxPool",
            "AveragePool",
            "Flatten",
            "Gemm",
            "BatchNormalization",
            "Concat",
            "GlobalAveragePool",
            "Add",
            "ConvTranspose",
        ]
        implemented_secfloat = [
            "Relu",
            "Sigmoid",
            "Softmax",
            "Conv",
            "MaxPool",
            "Concat",
            "BatchNormalization",
            "AveragePool",
            "GlobalAveragePool",
            "Flatten",
            "Reshape",
            "Gemm",
            "Tanh",
        ]
        if backend in ["SECFLOAT", "SECFLOAT_CLEARTEXT"]:
            implemented = implemented_secfloat
        elif backend in ["CLEARTEXT_LLAMA", "LLAMA"]:
            implemented = implemented_sytorch
        for node in model.graph.node:
            if node.op_type not in implemented:
                not_supported.append(node.op_type)
        not_supported = [*set(not_supported)]
        return (True, []) if len(not_supported) == 0 else (False, not_supported)

    @classmethod
    def prepare(
        cls,
        model,
        backend,
        device: str = "2PC",
        strict=True,
        logging_level="INFO",
        # logging_level='DEBUG',
        **kwargs,
    ):
        """

        :param model: The onnx model to be converted.
        :param device: 2PC by default for secure MPC.
        :param strict: for model semantics will see what todo
        :param logging_level: The logging level, default is INFO. Change it to DEBUG
        to see more conversion details or to WARNING to see less
        :param kwargs: will see what todo-Doc
        :return: Returns a Internal Representation of Onnx model called FzpcRep.
        """

        path = os.path.abspath(model)
        path = os.path.dirname(path)
        file_name = os.path.basename(model)
        model = cls.preprocess_model(model, logging_level, backend)
        super(IR, cls).prepare(model, device, **kwargs)
        logger.info("Optimised Stripped Model Loaded")

        if cls.supports_device(device):
            logger.info("Device Supported")
        else:
            logger.exception("Device not supported")

        program = None
        value_info = get_node_metadata(model)
        var_dict = dict()

        logger.info("Reading Onnx file to IR Nodes.")
        program = process_input_nodes(program, model.graph, var_dict)
        program = process_func_nodes(program, model.graph, var_dict)
        program = process_output_nodes(program, model.graph, var_dict)
        logger.info("Reading Onnx file completed.")

        program = optimizations.relu_maxpool_optimiser(program)
        logger.info("Relu Maxpool Optimisation Done.")

        # Works only if debugging is on
        if logger.getEffectiveLevel() == "DEBUG":
            print_nodes(program)

        var_dict = create_dict(program)
        logger.info("Onnx Variable -> IR variable Dictionary Created.")

        if backend in ["SECFLOAT", "SECFLOAT_CLEARTEXT"]:
            backend_rep = FzpcBackendRep(
                program, value_info, var_dict, path, file_name[:-5], backend
            )
        elif backend in ["CLEARTEXT_LLAMA", "LLAMA"]:
            backend_rep = SytorchBackendRep(
                program, value_info, var_dict, path, file_name[:-5]
            )
        logger.info("BackendRep Created.")
        return backend_rep

    @classmethod
    def supports_device(cls, device: str):
        """
        Checks whether the backend is compiled with 2PC device support.
        """
        return support_device(device)


prepare = IR.prepare
