from numbers import Number

from onnx import TensorProto

# dictionary [onnx data type in number] -> [cpp data-type]
TENSOR_TYPE_TO_FZPC_TYPE = {
    int(TensorProto.FLOAT): "float32",
    int(TensorProto.UINT8): "uint8",
    int(TensorProto.INT8): "int8",
    int(TensorProto.UINT16): "uint16",
    int(TensorProto.INT16): "int16",
    int(TensorProto.INT32): "int32",
    int(TensorProto.INT64): "int64",
    int(TensorProto.BOOL): "bool",
    int(TensorProto.FLOAT16): "float16",
    int(TensorProto.DOUBLE): "float64",
    int(TensorProto.COMPLEX64): "complex64",
    int(TensorProto.COMPLEX128): "complex128",
    int(TensorProto.UINT32): "uint32",
    int(TensorProto.UINT64): "uint64",
    int(TensorProto.STRING): "string",
}


def get_node_metadata(model):
    """
    Gives the MetaData to the variables i.e their data-type and their shape.
    :param model: ModelProto
    :return: Dictionary {var}->(data-type,shape)
    """
    value_info = {}
    for val in model.graph.value_info:
        value_info[val.name] = (
            onnx2ir(val.type.tensor_type.elem_type),
            proto_val_to_dimension_tuple(val),
        )
    return value_info


def _onnx_dtype(dtype):
    """
    Gives the onnx data-type in number format
    :param dtype: data-type
    :return: onnx format data type in number
    """
    if isinstance(dtype, Number):
        onnx_dype = dtype
    elif isinstance(dtype, str):
        onnx_dype = TensorProto.DataType.Value(dtype)
    else:
        raise RuntimeError("dtype should be number or str.")
    return onnx_dype


def onnx2ir(dtype):
    """
    Converts Onnx Data-Type to Cpp Data-Type
    :param dtype:
    :return:
    """
    return TENSOR_TYPE_TO_FZPC_TYPE[_onnx_dtype(dtype)]


__onnx_attr_translator = {
    "axis": lambda x: int(x),
    "axes": lambda x: [int(a) for a in x],
    "dtype": lambda x: onnx2ir(x),
    "keepdims": lambda x: bool(x),
    "to": lambda x: onnx2ir(x),
}


def translate_onnx(key, val):
    return __onnx_attr_translator.get(key, lambda x: x)(val)


def convert_attribute_proto(onnx_arg):
    """
    Convert an ONNX AttributeProto into an appropriate Python object
    for the type.

    NB: Tensor attribute gets returned as the straight proto.
    """
    if onnx_arg.HasField("f"):
        return onnx_arg.f
    elif onnx_arg.HasField("i"):
        return onnx_arg.i
    elif onnx_arg.HasField("s"):
        return onnx_arg.s
    elif onnx_arg.HasField("t"):
        return onnx_arg.t  # this is a proto!
    elif len(onnx_arg.floats):
        return list(onnx_arg.floats)
    elif len(onnx_arg.ints):
        return list(onnx_arg.ints)
    elif len(onnx_arg.strings):
        return list(onnx_arg.strings)
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(onnx_arg))


def proto_val_to_dimension_tuple(proto_val):
    """
    Gives the dimensions of the Proto Value in Tuple form.
    """
    return tuple([dim.dim_value for dim in proto_val.type.tensor_type.shape.dim])
