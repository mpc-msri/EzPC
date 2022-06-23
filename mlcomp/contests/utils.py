import json
import onnx


def mb_to_bytes(mb: int) -> int:
    return mb * 1000000


def validate_config(data: str) -> bool:
    try:
        obj = json.loads(data)
        if obj["target"] != "SCI":
            return False
        if obj["backend"] != "OT":
            return False
    except:
        return False
    return True


def check_model_valid(path: str):
    onnx_model = path.read()
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        return str(e)
    else:
        return True
