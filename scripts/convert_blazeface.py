"""
Convert Blazeface model
"""

from pathlib import Path
import platform
import tflite
import tvm
from tvm import relay, transform
from typing import NamedTuple, Tuple


class ConversionParams(NamedTuple):
    model_path: Path
    shape: Tuple[int]
    input_name: str
    dtype: str


def convert(convert_params: ConversionParams):
    """
    Conerts blazeface model to shared object
    """

    if not convert_params.model_path.exists():
        print(f"Model path {convert_params.model_path} doesn't exist")
        return

    tflite_model_buf = None
    with open(convert_params.model_path, "rb") as f:
        tflite_model_buf = f.read()

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={convert_params.input_name: convert_params.shape},
        dtype_dict={convert_params.input_name: convert_params.dtype},
    )

    target = 'llvm'
    with transform.PassContext(opt_level=3):
        #lib = relay.build(mod, tvm.target.Target(target="llvm", host="llvm"), params)
        lib = relay.build(mod, target, params=params)
        if platform.system() == "Windows":
            output_file = convert_params.model_path.with_suffix(".dll")
        else:
            output_file = convert_params.model_path.with_suffix(".so")
        lib.export_library(str(output_file))


if __name__ == "__main__":
    model_path = (
        Path(__file__).absolute().parents[1]
        / "models"
        / "blazeface"
        / "face_detection_front.tflite"
    )
    params = ConversionParams(model_path, (1, 128, 128, 3), "input", "float32")
    convert(params)
