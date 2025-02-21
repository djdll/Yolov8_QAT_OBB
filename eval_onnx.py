from ultralytics.utils.torch_utils import model_info
from ultralytics import YOLO
import onnx_tool
import numpy as np
onnxpath = 'qat.onnx'
onnx_model = YOLO(onnxpath)
# Run inference
results=onnx_model("datasets/coco128/images/train2017/000000000009.jpg",
                    imgsz=(288,448),
                    # save=Trues
                    )
print(results)