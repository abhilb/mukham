from pathlib import Path

# tflite
import tflite

# tvm
import tvm
from tvm import relay, transform
from tvm import te


def convert(model_name: str) -> None:
    model_path = Path.cwd().parent / "models"
    model_full_path = model_path / model_name

    tflite_model_buf = None
    with open(model_full_path, "rb") as f:
        tflite_model_buf = f.read()

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

    input_tensor = "input"
    input_shape = (1, 128, 128, 3)
    input_dtype = "float32"
    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={input_tensor: input_shape},
        dtype_dict={input_tensor: input_dtype},
    )

    with transform.PassContext(opt_level=3):
        lib = relay.build(mod, tvm.target.create("llvm"), params=params)
        lib.export_library(str(model_full_path.with_suffix(".tar")))


if __name__ == "__main__":
    convert("face_detection_short_range.tflite")
