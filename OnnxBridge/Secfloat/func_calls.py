import math
import struct
from utils import logger, VariableGen
from utils.backend_helper import (
    iterate_list,
    iterate_list_singleton,
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
            cmmnt + f"{'  ' * indent}Relu("
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
            cmmnt + f"{'  ' * indent}Relu("
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
            cmmnt + f"{'  ' * indent}Sigmoid("
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
            cmmnt + f"{'  ' * indent}Softmax("
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
                    f"{'  ' * indent}ConvAdd("
                    f"{iterate_list(value_info[outputs[0]][1])}, "
                    f"{var_dict[outputs[0]]}, {var_dict[inputs[2]]}, {var_dict[outputs[0]]}"
                    f");"
                )
                pass
            return (
                str(
                    f"{'  ' * indent}Conv2DGroupWrapper("
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
            f"{'  ' * indent}MaxPool("
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
            f"{'  ' * indent}Concat{len(inputs)}T{iterate_concat_list([len(value_info[x][1]) for x in inputs])}{len(value_info[outputs[0]][1])}("
            f"{iterate_list(value_info[outputs[0]][1])}, "
            f"{iterate_list([concat_list(value_info, var_dict, x) for x in inputs])}, "
            f"{attributes['axis']}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )

    @classmethod
    def BatchNormalization(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside BatchNormalization function call.")
        return str(
            f"{'  ' * indent}BatchNormalization("
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
            f"{'  ' * indent}AvgPool("
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
    def GlobalAveragePool(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside GloablAveragePool function call.")
        return str(
            f"{'  ' * indent}AvgPool("
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
        #     f"{'  ' * indent}Flatten("
        #     # f"{iterate_list(value_info[inputs[0]][1])}, "
        #     # f"{(iterate_dict(attributes) if attributes else '')}{',' if attributes else ''}"
        #     f"{iterate_list(value_info[outputs[0]][1])}, "
        #     f"{iterate_list([var_dict[x] for x in inputs])}, "
        #     f"{iterate_list([var_dict[x] for x in outputs])}"
        #     f");")

    @classmethod
    def Reshape(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Reshape function call.")
        # print(inputs)
        # print(outputs)
        # print(value_info[inputs[0]][1])
        # print(value_info[inputs[1]][1])
        # print(value_info[outputs[0]][1])
        # print(value_info)
        # print(attributes)
        # counter1, counter2 = len(value_info(inputs[0])[1]), len(value_info(outputs[0])[1])
        return "    // Reshape() function not implemented"
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
            f"{'  ' * indent}Gemm("
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
        cmmnt = comment("Call Tanh(shape,input,output)\n", indent)
        return str(
            cmmnt + f"{'  ' * indent}Tanh("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )

    @classmethod
    def Add(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Add function call.")
        # TODO: Add broadcasting support.
        # print(inputs)
        # print(outputs)
        # print(value_info[inputs[0]][1])
        # print(value_info[inputs[1]][1])
        # print(value_info[outputs[0]][1])
        # print(attributes)
        # assert(value_info[inputs[0]][1] == value_info[inputs[1]][1])
        # assert(value_info[inputs[0]][1] == value_info[outputs[0]][1])
        # return "    // Add() function not implemented"
        cmmnt = comment("Call Add(shape,input1,input2,output)\n", indent)
        return str(
            cmmnt + f"{'  ' * indent}ElemWiseAdd("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )

    @classmethod
    def Sub(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Sub function call.")
        # TODO: Add broadcasting support.
        # print(inputs)
        # print(outputs)
        # print(value_info[inputs[0]][1])
        # print(value_info[outputs[0]][1])
        # print(attributes)
        assert(value_info[inputs[0]][1] == value_info[inputs[1]][1])
        assert(value_info[inputs[0]][1] == value_info[outputs[0]][1])
        cmmnt = comment("Call Sub(shape,input1,input2,output)\n", indent)
        return str(
            cmmnt + f"{'  ' * indent}ElemWiseSub("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )

    @classmethod
    def Mul(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Mul function call.")
        # TODO: Add broadcasting support.
        # print(inputs)
        # print(outputs)
        # print(value_info[inputs[0]][1])
        # print(value_info[outputs[0]][1])
        # print(attributes)
        cmmnt = comment("Call Mul(shape,input1,input2,output)\n", indent)
        return str(
            cmmnt + f"{'  ' * indent}ElemWiseMul("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )

    @classmethod
    def Div(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Div function call.")
        # TODO: Add broadcasting support.
        # print(inputs)
        # print(outputs)
        # print(value_info[inputs[0]][1])
        # print(value_info[outputs[0]][1])
        # print(attributes)
        assert(value_info[inputs[0]][1] == value_info[inputs[1]][1])
        assert(value_info[inputs[0]][1] == value_info[outputs[0]][1])
        cmmnt = comment("Call Div(shape,input1,input2,output)\n", indent)
        return str(
            cmmnt + f"{'  ' * indent}ElemWiseDiv("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )

    @classmethod
    def Sqrt(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Sqrt function call.")
        # print(inputs)
        # print(outputs)
        # print(value_info[inputs[0]][1])
        # print(value_info[outputs[0]][1])
        # print(attributes)
        cmmnt = comment("Call Sqrt(shape,input,output)\n", indent)
        return str(
            cmmnt + f"{'  ' * indent}ElemWiseSqrt("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )
    
    @classmethod
    def Pow(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Pow function call.")
        # TODO: Add broadcasting support.
        # print(inputs)
        # print(outputs)
        # print(value_info[inputs[0]][1])
        # print(value_info[outputs[0]][1])
        # print(attributes)
        assert value_info[inputs[0]][1] == value_info[outputs[0]][1]
        assert value_info[inputs[1]][1] == ()
        cmmnt = comment("Call Pow(shape,input1,input2,output)\n", indent)
        return str(
            cmmnt + f"{'  ' * indent}ElemWisePow("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )

    @classmethod
    def Shape(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Shape function call.")
        # TODO: See if we should merge Shape and Gather
        # print(inputs)
        # print(outputs)
        # print(value_info[inputs[0]][1])
        # print(value_info[outputs[0]][1])
        # print(attributes)
        cmmnt = comment("Call Shape(shape,input,output)\n", indent)
        return str(
            cmmnt + f"{'  ' * indent}Shape("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )
    
    @classmethod
    def Gather(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Gather function call.")
        # TODO: See if we should merge Shape and Gather.
        # print(inputs)
        # print(outputs)
        # print(value_info[inputs[0]][1])
        # print(value_info[inputs[1]][1])
        # print(value_info[outputs[0]][1])
        # print(attributes)
        cmmnt = comment("Call Gather(input_shape,indices_shape,input,indices,output)\n", indent)
        if 'axes' in attributes:
            # TODO: Implement support for multiple axes.
            raise NotImplementedError
        else:
            if 'axis' in attributes:
                # TODO: Implement support for non-zero axis.
                return str(
                    cmmnt + f"{'  ' * indent}Gather{attributes['axis']}("
                    f"{iterate_list(value_info[inputs[0]][1])}, "
                    f"{iterate_list(value_info[inputs[1]][1])}, "
                    f"{iterate_list([var_dict[x] for x in inputs])}, "
                    f"{iterate_list([var_dict[x] for x in outputs])}"
                    f");"
                )
            else:
                return str(
                    cmmnt + f"{'  ' * indent}Gather0("
                    f"{iterate_list(value_info[inputs[0]][1])}, "
                    f"{iterate_list(value_info[inputs[1]][1])}, "
                    f"{iterate_list([var_dict[x] for x in inputs])}, "
                    f"{iterate_list([var_dict[x] for x in outputs])}"
                    f");"
                )

    @classmethod
    def Unsqueeze(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Unsqueeze function call.")
        # TODO: Make Unsqueeze more generalized.
        # print(inputs)
        # print(outputs)
        # print(value_info[inputs[0]])
        # print(value_info[outputs[0]])
        # print(attributes)
        cmmnt = comment("Call Unsqueeze(shape,axes,input,output)\n", indent)
        return str(
            cmmnt + f"{'  ' * indent}Unsqueeze("
            f"{iterate_list(value_info[outputs[0]][1])}, "
            f"{iterate_list(attributes['axes'])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )
    
    @classmethod
    def ConstantOfShape(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside ConstantOfShape function call.")
        # TODO: Add support for additional datatypes and multiple axes.
        # print(inputs)
        # print(outputs)
        # print(value_info[inputs[0]][1])
        # print(value_info[outputs[0]][1])
        assert(len(value_info[inputs[0]][1]) == 1)
        value = int.from_bytes(attributes['value'].raw_data, byteorder="little", signed=True)
        cmmnt = comment("Call ConstantOfShape(shape,value,input,output)\n", indent)
        return str(
            cmmnt + f"{'  ' * indent}ConstantOfShapeI64("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{value}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )


    @classmethod
    def NonZero(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside NonZero function call.")
        # TODO: Add support for multiple axes.
        # print(inputs)
        # print(outputs)
        # print(value_info[inputs[0]][1])
        # print(value_info[outputs[0]][1])
        # print(attributes)
        assert(len(value_info[inputs[0]][1]) == 1)
        cmmnt = comment("Call NonZero(shape,input,output)\n", indent)
        return str(
            cmmnt + f"{'  ' * indent}NonZero1D("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )


    @classmethod
    def Transpose(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Transpose function call.")
        # TODO; Add support for more permutations and input shapes as needed.
        # print(inputs)
        # print(outputs)
        # print(value_info[inputs[0]][1])
        # print(value_info[outputs[0]][1])
        # print(attributes)
        cmmnt = comment("Call Transpose(shape,perm,input,output)\n", indent)
        return str(
            cmmnt + f"{'  ' * indent}Transpose{len(attributes['perm'])}T{iterate_concat_list(attributes['perm'])}("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )

    @classmethod
    def Cast(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Cast function call.")
        print(inputs)
        print(outputs)
        print(value_info[inputs[0]][1])
        print(value_info[outputs[0]][1])
        print(attributes)
        raise NotImplementedError

    @classmethod
    def Split(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        # TODO: Add support for multiple axes and outputs.
        logger.debug("Inside Split function call.")
        # assert(sum(attributes['split']) == value_info[inputs[0]][1][attributes['axis']])
        cmmnt = comment("Call Split(shape,splits,input,outputs)\n", indent)
        return str(
            cmmnt + f"{'  ' * indent}Split{attributes['axis']}T{len(attributes['split'])}("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list(attributes['split'])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )

    @classmethod
    def Slice(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        # TODO: Add support for more general slice.
        logger.debug("Inside Slice function call.")
        assert(len(value_info[inputs[1]][1]) == 1)
        assert(len(value_info[inputs[2]][1]) == 1)
        assert(len(value_info[inputs[3]][1]) == 1)
        assert(len(value_info[inputs[4]][1]) == 1)
        assert(value_info[inputs[1]][1][0] == 1)
        assert(value_info[inputs[2]][1][0] == 1)
        assert(value_info[inputs[3]][1][0] == 1)
        assert(value_info[inputs[4]][1][0] == 1)
        cmmnt = comment("Call Slice(shape,starts,ends,axes,steps,input,output)\n", indent)
        return str(
            cmmnt + f"{'  ' * indent}Slice("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list_singleton([var_dict[x] for x in inputs[1:]])}, "
            f"{iterate_list([var_dict[inputs[0]]])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )

    @classmethod
    def Squeeze(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Squeeze function call.")
        # TODO: Add support for multiple axes and input shapes.
        # print(inputs)
        # print(outputs)
        # print(value_info[inputs[0]][1])
        # print(value_info[outputs[0]][1])
        # print(attributes)
        cmmnt = comment("Call Squeeze(shape,axes,input,output)\n", indent)
        return str(
            cmmnt + f"{'  ' * indent}Squeeze("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list(attributes['axes'])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )

    @classmethod
    def ReduceMean(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside ReduceMean function call.")
        # TODO: Add support for non-keepdims and multiple axes.
        # print(inputs)
        # print(outputs)
        # print(value_info[inputs[0]][1])
        # print(value_info[outputs[0]][1])
        # print(attributes)
        cmmnt = comment("Call ReduceMean(shape,axes,input,output)\n", indent)
        return str(
            cmmnt + f"{'  ' * indent}ReduceMean("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list(attributes['axes'])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )

    @classmethod
    def MatMul(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside MatMul function call.")
        # print(inputs)
        # print(outputs)
        # print(value_info[inputs[0]][1])
        # print(value_info[outputs[0]][1])
        # print(attributes)
        # assert(len(value_info[inputs[0]][1]) == 2)
        # assert(len(value_info[inputs[1]][1]) == 2)
        # assert(value_info[inputs[0]][1][1] == value_info[inputs[1]][1][0])
        # assert(value_info[inputs[0]][1][0] == value_info[outputs[0]][1][0])
        # assert(value_info[inputs[1]][1][1] == value_info[outputs[0]][1][1])
        cmmnt = comment("Call MatMul(shape,inputs,output)\n", indent)
        return ""
        # return str(
        #     cmmnt + f"{'  ' * indent}MatMul2D("
        #     f"{iterate_list(value_info[inputs[0]][1])}, "
        #     f"{value_info[inputs[1]][1][1]}, "
        #     f"{iterate_list([var_dict[x] for x in inputs])}, "
        #     f"{iterate_list([var_dict[x] for x in outputs])}"
        #     f");"
        # )