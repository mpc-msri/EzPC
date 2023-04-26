import math

from utils import logger
from utils.backend_helper import iterate_list


def get_padding(attributes, inputs, output, value_info, var_dict):
    if "pads" in attributes.keys():
        return attributes["pads"]
    elif "auto_pad" in attributes.keys() and (
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


def get_padding_3d(attributes, inputs, output, value_info, var_dict):
    if "pads" in attributes.keys():
        return attributes["pads"]
    elif "auto_pad" in attributes.keys() and (
        str(attributes["auto_pad"], "UTF-8") == "NOTSET"
        or str(attributes["auto_pad"], "UTF-8") == "VALID"
    ):
        return attributes["pads"] if "pads" in attributes.keys() else [0, 0, 0, 0, 0, 0]
    else:
        stride_d = attributes["strides"][0]
        stride_h = attributes["strides"][1]
        stride_w = attributes["strides"][2]
        out_d = value_info[output[0]][1][2]
        out_h = value_info[output[0]][1][3]
        out_w = value_info[output[0]][1][4]
        in_d = value_info[inputs[0]][1][2]
        in_h = value_info[inputs[0]][1][3]
        in_w = value_info[inputs[0]][1][4]
        ker_d = (
            value_info[inputs[1]][1][2]
            if "kernel_shape" not in attributes.keys()
            else attributes["kernel_shape"][0]
        )
        ker_h = (
            value_info[inputs[1]][1][3]
            if "kernel_shape" not in attributes.keys()
            else attributes["kernel_shape"][1]
        )
        ker_w = (
            value_info[inputs[1]][1][4]
            if "kernel_shape" not in attributes.keys()
            else attributes["kernel_shape"][2]
        )
        pad_d = math.ceil(((out_d - 1) * stride_d + ker_d - in_d) / 2)
        pad_h = math.ceil(((out_h - 1) * stride_h + ker_h - in_h) / 2)
        pad_w = math.ceil(((out_w - 1) * stride_w + ker_w - in_w) / 2)
        pads = [pad_d, pad_h, pad_w, pad_d, pad_h, pad_w]
        return pads


def get_dilation(attributes, inputs, output, value_info, var_dict):
    return (
        attributes["dilations"]
        if "dilations" in attributes.keys()
        else [1 for _ in range(len(value_info[inputs[0]][1]) - 2)]
    )


class Operator:
    """
    Class preparing the Function Calls specific for each function.
    """

    @classmethod
    def Relu(cls, attributes, inputs, outputs, value_info, var_dict, mode, indent):
        logger.debug("Inside Relu function call.")
        return str(f"{'   ' * indent}new ReLU<T>();")

    @classmethod
    def BatchNormalization(
        cls, attributes, inputs, outputs, value_info, var_dict, mode, indent
    ):
        logger.debug("Inside BatchNorm function call.")
        shape = value_info[inputs[1]][1][0]
        return str(f"{'   ' * indent}new BatchNormInference<T>({shape});")

    @classmethod
    def Concat(cls, attributes, inputs, outputs, value_info, var_dict, mode, indent):
        logger.debug("Inside Concat function call.")
        return str(f"{'   ' * indent}new Concat<T>();")

    @classmethod
    def GlobalAveragePool(
        cls, attributes, inputs, outputs, value_info, var_dict, mode, indent
    ):
        logger.debug("Inside GlobalAveragePool function call.")
        return str(f"{'   ' * indent}new GlobalAvgPool2D<T>();")

    @classmethod
    def Add(cls, attributes, inputs, outputs, value_info, var_dict, mode, indent):
        logger.debug("Inside Add function call.")
        return str(f"{'   ' * indent}new Add<T>();")

    @classmethod
    def Truncate(cls, attributes, inputs, outputs, value_info, var_dict, mode, indent):
        logger.debug("Inside Truncate function call.")
        return str(f"{'   ' * indent}new Truncate<T>(scale);")

    @classmethod
    def Softmax(cls, attributes, inputs, outputs, value_info, var_dict, mode, indent):
        logger.debug("Inside Softmax function call.")
        # todo: check format

    @classmethod
    def Conv(cls, attributes, inputs, outputs, value_info, var_dict, mode, indent):
        logger.debug("Inside Conv function call.")

        spatial_size = len(value_info[inputs[0]][1]) - 2
        if spatial_size == 2:
            assert (
                len(inputs) == 2 or len(inputs) == 3
            )  # todo: bias is always there or not
            pads = get_padding(attributes, inputs, outputs, value_info, var_dict)
            assert len(attributes["strides"]) == 2
            assert value_info[inputs[1]][1][2:] == tuple(attributes["kernel_shape"])
            CI = value_info[inputs[0]][1][1]
            CO = value_info[outputs[0]][1][1]
            filterShape = value_info[inputs[1]][1][2]
            pad = pads[0]
            stride = attributes["strides"][0]
            isBias = ", true" if len(inputs) == 3 else ""
            return str(
                f"{'   ' * indent}new Conv2D<T>("
                f"{CI}, {CO}, {filterShape}, {pad}, {stride}{isBias}"
                f");"
            )
        elif spatial_size == 3:
            assert len(inputs) == 2 or len(inputs) == 3
            assert len(attributes["strides"]) == 3
            assert value_info[inputs[1]][1][2:] == tuple(attributes["kernel_shape"])
            pads = get_padding_3d(attributes, inputs, outputs, value_info, var_dict)
            CI = value_info[inputs[0]][1][1]
            CO = value_info[outputs[0]][1][1]
            filterShape = value_info[inputs[1]][1]
            pad = pads[0]
            strides = attributes["strides"]
            dilations = get_dilation(attributes, inputs, outputs, value_info, var_dict)
            isBias = ", true" if len(inputs) == 3 else ""
            return str(
                f"{'   ' * indent}new Conv3D<T>("
                f"{CI}, {CO}, {'{'}{iterate_list(filterShape[2:])}{'}'}, {'{'}{iterate_list(pads)}{'}'}, {'{'}{iterate_list(strides)}{'}'},{'{'}{iterate_list(dilations)}{'}'}{isBias}"
                f");"
            )

    @classmethod
    def ConvTranspose(
        cls, attributes, inputs, outputs, value_info, var_dict, mode, indent
    ):
        logger.debug("Inside ConvTranspose function call.")
        pads = get_padding_3d(attributes, inputs, outputs, value_info, var_dict)
        spatial_size = len(value_info[inputs[0]][1]) - 2
        if spatial_size == 3:
            assert len(inputs) == 2 or len(inputs) == 3
            assert len(attributes["strides"]) == 3
            assert value_info[inputs[1]][1][2:] == tuple(attributes["kernel_shape"])
            CI = value_info[inputs[0]][1][1]
            CO = value_info[outputs[0]][1][1]
            filterShape = value_info[inputs[1]][1]
            pad = pads[0]
            strides = attributes["strides"]
            dilations = get_dilation(attributes, inputs, outputs, value_info, var_dict)
            isBias = ", true" if len(inputs) == 3 else ""
            return str(
                f"{'   ' * indent}new ConvTranspose3D<T>("
                f"{CI}, {CO}, {'{'}{iterate_list(filterShape[2:])}{'}'}, {'{'}{iterate_list(pads)}{'}'}, {'{'}{iterate_list(strides)}{'}'}, {'{'}{iterate_list(dilations)}{'}'}{isBias}"
                f");"
            )

    @classmethod
    def MaxPool(cls, attributes, inputs, outputs, value_info, var_dict, mode, indent):
        logger.debug("Inside MaxPool function call.")
        pads = get_padding(attributes, inputs, outputs, value_info, var_dict)
        filter_shape = attributes["kernel_shape"][0]
        pad = pads[0]
        stride = attributes["strides"][0]
        return str(
            f"{'   ' * indent}new MaxPool2D<T>("
            f"{filter_shape}, {pad}, {stride}"
            f");"
        )

    @classmethod
    def AveragePool(
        cls, attributes, inputs, outputs, value_info, var_dict, mode, indent
    ):
        logger.debug("Inside AveragePool function call.")
        pads = get_padding(attributes, inputs, outputs, value_info, var_dict)
        filter_shape = attributes["kernel_shape"][0]
        pad = pads[0]
        stride = attributes["strides"][0]
        return str(
            f"{'   ' * indent}new AvgPool2D<T>("
            f"{filter_shape}, {pad}, {stride}"
            f");"
        )

    @classmethod
    def Flatten(cls, attributes, inputs, outputs, value_info, var_dict, mode, indent):
        logger.debug("Inside Flatten function call.")
        return str(f"{'   ' * indent}new Flatten<T>();")

    @classmethod
    def Reshape(cls, attributes, inputs, outputs, value_info, var_dict, mode, indent):
        logger.debug("Inside Reshape function call.")
        return str(f"{'   ' * indent}new Reshape<T>();")
        # todo : check format

    @classmethod
    def Gemm(cls, attributes, inputs, outputs, value_info, var_dict, mode, indent):
        logger.debug("Inside Gemm function call.")
        inn = value_info[inputs[0]][1][1]
        out = value_info[outputs[0]][1][1]
        isBias = ", true" if len(inputs) == 3 else ""

        return str(f"{'   ' * indent}new FC<T>(" f"{inn}, {out}{isBias}" f");")
        # ) + cls.Truncate(
        #     attributes, inputs, outputs, value_info, var_dict, mode, indent
        # )
