"""
Converts FaceMesh TFLite model to ShareObject
"""

from pathlib import Path
import tflite
import tvm
from tvm import relay, transform
from tvm import te
import argparse

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
        lib.export_library(str(model_path.with_suffix(".so")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="Path to the model file", type=str, required=True)
    parser.add_argument('-b', '--batch', help="Batch size", type=int, required=True)
    parser.add_argument('-c', '--convert', action="store_true", help="Covert layout")

    args = vars(parser.parse_args())
    print(args)
    convert(Path(args['input']), int(args['batch']), args['convert'])
    
