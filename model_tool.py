from ultralytics.utils.torch_utils import model_info
from ultralytics import YOLO
import onnx_tool
import numpy as np
ptpath = 'qat.pt'
# onnxpath = 'test_light/finetune_light_Adam_5e-5_0.01_e350_b256_f0_250212_mc9_box6_swish_addbgdatax3_4482/weights/best.onnx'
# Load the YOLOv8 model
model = YOLO(ptpath)
# import ipdb; ipdb.set_trace()
# Export the model to ONNX format
# model.export(format="onnx",imgsz=(192,288))  # creates 'yolov8n.onnx'
model.export(format='onnx', imgsz=(288,448),opset=12)  # opset=12 否则ncnn无法识别 export注册
# model.export(format='onnx', imgsz=(256,384),opset=12)
# model.export(format='ncnn')
# Load the exported ONNX model
# onnx_model = YOLO(onnxpath)

# Run inference
# results=onnx_model("1.jpg",imgsz=[256,384])
# print(results)
# onnx_tool.model_profile(onnxpath)