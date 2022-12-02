import math
from utils import logger, VariableGen
from utils.backend_helper import (
    iterate_list,
    iterate_dict,
    decl,
    comment,
    generate_reshape_vars,
    decl_multiple_int,
    nested_for_reshape_loop,
    iterate_concat_list,
    concat_list,
)


def get_padding(attributes, inputs, output, value_info, var_dict):
    if "auto_pad" in attributes.keys():
        if (
            str(attributes["auto_pad"], "UTF-8") == "NOTSET"
            or str(attributes["auto_pad"], "UTF-8") == "VALID"
        ):
            return attributes["pads"] if "pads" in attributes.keys() else [0, 0, 0, 0]
        else:
            stride_h = attributes["strides"][0]
            stride_w = attributes["strides"][1]
            out_h = value_info[output[0]][1][2]
            out_w = value_info[output[0]][1][3]
            in_h = value_info[inputs[0]][1][2]
            in_w = value_info[inputs[0]][1][3]
            ker_h = (
                value_info[inputs[1]][1][2]
                if "kernel_shape" not in attributes.keys()
                else attributes["kernel_shape"][0]
            )
            ker_w = (
                value_info[inputs[1]][1][3]
                if "kernel_shape" not in attributes.keys()
                else attributes["kernel_shape"][0]
            )
            pad_h = math.ceil(((out_h - 1) * stride_h + ker_h - in_h) / 2)
            pad_w = math.ceil(((out_w - 1) * stride_w + ker_w - in_w) / 2)
            pads = [pad_h, pad_w, pad_h, pad_w]
            return pads
    else:
        return attributes["pads"]
    pass


class Operator:
    """
    Class preparing the Function Calls specific for each function.
    """

    @classmethod
    def Relu(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Relu function call.")
        cmmnt = comment("Call  Relu(shape,input,output)\n", indent)
        return str(
            cmmnt + f"{'   ' * indent}Relu("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )

    @classmethod
    def LeakyRelu(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Relu function call.")
        cmmnt = comment("Call  Relu(shape,input,output)\n", indent)
        return str(
            cmmnt + f"{'   ' * indent}Relu("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{attributes['alpha']}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )

    @classmethod
    def Sigmoid(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Sigmoid function call.")
        cmmnt = comment("Call  Sigmoid(shape,input,output)\n", indent)
        return str(
            cmmnt + f"{'   ' * indent}Sigmoid("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )

    @classmethod
    def Softmax(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Softmax function call.")
        cmmnt = comment("Call  Softmax(shape,input,output)\n", indent)
        return str(
            cmmnt + f"{'   ' * indent}Softmax("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )

    @classmethod
    def Conv(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Conv function call.")
        pads = get_padding(attributes, inputs, outputs, value_info, var_dict)

        spatial_size = len(value_info[inputs[0]][1]) - 2
        if spatial_size == 2:
            assert len(inputs) == 2 or len(inputs) == 3
            assert len(attributes["strides"]) == 2
            assert value_info[inputs[1]][1][2:] == tuple(attributes["kernel_shape"])
            filterShape = value_info[inputs[1]][1]
            N, CI, H, W = value_info[inputs[0]][1]
            convadd = ""
            if len(inputs) == 3:
                convadd = str(
                    f"{'   ' * indent}ConvAdd("
                    f"{iterate_list(value_info[outputs[0]][1])}, "
                    f"{var_dict[outputs[0]]}, {var_dict[inputs[2]]}, {var_dict[outputs[0]]}"
                    f");"
                )
                pass
            return (
                str(
                    f"{'   ' * indent}Conv2DGroupWrapper("
                    f"{N}, {CI}, {H}, {W}, "
                    f"{filterShape[2]}, {filterShape[3]}, {value_info[inputs[1]][1][0]}, "
                    f"{(iterate_list(pads))}, "
                    f"{(iterate_list(attributes['strides']))}, "
                    f"{(attributes['group'] if 'group' in attributes else 1)}, "
                    f"{iterate_list([var_dict[x] for x in inputs[:2]])}, "
                    f"{iterate_list([var_dict[x] for x in outputs])}"
                    f");\n"
                )
                + convadd
            )

    @classmethod
    def MaxPool(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside MaxPool function call.")
        pads = get_padding(attributes, inputs, outputs, value_info, var_dict)
        return str(
            f"{'   ' * indent}MaxPool("
            f"{iterate_list(value_info[outputs[0]][1])}, "
            f"{attributes['kernel_shape'][0]}, {attributes['kernel_shape'][1]}, "
            f"{iterate_list(pads)}, "
            f"{attributes['strides'][0]}, {attributes['strides'][1]}, "
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )

    @classmethod
    def Concat(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Concat function call.")
        return str(
            f"{'   ' * indent}Concat{len(inputs)}T{iterate_concat_list([len(value_info[x][1]) for x in inputs])}{len(value_info[outputs[0]][1])}("
            f"{iterate_list(value_info[outputs[0]][1])}, "
            f"{iterate_list([concat_list(value_info, var_dict, x) for x in inputs])}, "
            f"{attributes['axis']}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )

    @classmethod
    def BatchNormalization(
        cls, attributes, inputs, outputs, value_info, var_dict, indent
    ):
        logger.debug("Inside BatchNormalization function call.")
        return str(
            f"{'   ' * indent}BatchNormalization("
            f"{iterate_list(value_info[outputs[0]][1])}, "
            # f"{(iterate_dict(attributes) if attributes else '')}{',' if attributes else ''}"
            # f"{iterate_list(value_info[outputs[0]][1])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )

    @classmethod
    def AveragePool(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside AveragePool function call.")
        pads = get_padding(attributes, inputs, outputs, value_info, var_dict)
        return str(
            f"{'   ' * indent}AvgPool("
            f"{iterate_list(value_info[outputs[0]][1])}, "
            f"{attributes['kernel_shape'][0]}, {attributes['kernel_shape'][1]}, "
            f"{iterate_list(pads)}, "
            f"{attributes['strides'][0]}, {attributes['strides'][1]}, "
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )

    @classmethod
    def GlobalAveragePool(
        cls, attributes, inputs, outputs, value_info, var_dict, indent
    ):
        logger.debug("Inside GloablAveragePool function call.")
        return str(
            f"{'   ' * indent}AvgPool("
            f"{iterate_list(value_info[outputs[0]][1])}, "
            f"{value_info[inputs[0]][1][2]}, {value_info[inputs[0]][1][3]}, 0, 0, 0, 0, 1, 1, "
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )

    @classmethod
    def Flatten(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Flatten function call.")
        return cls.Reshape(attributes, inputs, outputs, value_info, var_dict, indent)
        # return str(
        #     f"{'   ' * indent}Flatten("
        #     # f"{iterate_list(value_info[inputs[0]][1])}, "
        #     # f"{(iterate_dict(attributes) if attributes else '')}{',' if attributes else ''}"
        #     f"{iterate_list(value_info[outputs[0]][1])}, "
        #     f"{iterate_list([var_dict[x] for x in inputs])}, "
        #     f"{iterate_list([var_dict[x] for x in outputs])}"
        #     f");")

    @classmethod
    def Reshape(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Reshape function call.")
        # counter1, counter2 = len(value_info(inputs[0])[1]), len(value_info(outputs[0])[1])
        variables1 = generate_reshape_vars(len(value_info[inputs[0]][1]))
        variables2 = generate_reshape_vars(len(value_info[outputs[0]][1]))
        code = decl_multiple_int(variables1, indent)
        code += nested_for_reshape_loop(
            0,
            value_info[inputs[0]][1],
            variables1,
            0,
            value_info[outputs[0]][1],
            variables2,
            var_dict[inputs[0]],
            var_dict[outputs[0]],
            indent,
        )
        return code

    @classmethod
    def Gemm(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Gemm function call.")
        code = ""
        if len(value_info[inputs[2]][1]) == 2:
            code += decl("bias_mod", "fparray2d", [value_info[inputs[2]][1][1]], indent)
            code += "\n"
            value_info["bias_mod"] = ("fparray", [value_info[inputs[2]][1][1]])
            var_dict["bias_mod"] = "bias_mod"
            code += cls.Reshape(
                attributes, [inputs[2]], ["bias_mod"], value_info, var_dict, indent
            )
            inputs[2] = "bias_mod"
        code += str(
            f"{'   ' * indent}Gemm("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list(value_info[inputs[1]][1])}, "
            f"{attributes['alpha']}, {attributes['beta']}, "
            f"{attributes['transA']}, {attributes['transB']}, "
            f"{iterate_list(value_info[outputs[0]][1])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )
        return code

    @classmethod
    def Tanh(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Tanh function call.")
        cmmnt = comment("Call  Tanh(shape,input,output)\n", indent)
        return str(
            cmmnt + f"{'   ' * indent}Tanh("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )
