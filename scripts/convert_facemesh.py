"""
Converts FaceMesh TFLite model to ShareObject
"""

from pathlib import Path
import tflite
import tvm
from tvm import relay, transform
import platform

def convert(model_path: Path, batch: int, convert) -> None:
    """
    Converts the facemesh tflite model to
    shared object that can then be used in
    apache tvm runtime

    @params:
    model_name : path to the tflite model

    @returns:
    None
    """

    # check whether the model exists
    if not model_path.exists():
        print(f"Model path {model_path} doesn't exist")
        return

    tflite_model_buf = None
    with open(model_path, "rb") as f:
        tflite_model_buf = f.read()

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

    input_tensor = "input_1"
    input_shape  = (batch, 192, 192, 3)
    input_dtype  = "float32"

    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict = {input_tensor: input_shape},
        dtype_dict = {input_tensor: input_dtype}
    )

    desired_layouts = {
        'nn.conv2d': ['NCHW', 'default'],
        'nn.depthwise_conv2d': ['NCHW', 'default']
    }
    seq = tvm.transform.Sequential( [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ConvertLayout(desired_layouts)
        ])

    target = "llvm"
    with transform.PassContext(opt_level=3):
        if convert:
            mod = seq(mod)
        lib = relay.build(mod, target, params=params)
        if platform.system() == 'Windows':
            output_file = model_path.with_suffix(".dll")
        else:
            output_file = model_path.with_suffix(".so")
        lib.export_library(str(output_file))


if __name__ == '__main__':
    model_path = (
        Path(__file__).absolute().parents[1]
        / "models"
        / "facemesh"
        / "face_landmark.tflite"
    )
    convert(model_path, 1, False)
