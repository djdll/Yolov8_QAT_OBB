import cv2
import math
import numpy as np
import onnxruntime
import argparse
import os
import sys
import cv2
import numpy as np
import math
# infer onnx obb model
class_names = ["class names"]      
input_shape = (640, 640)  # h w
input_height = 640
input_width = 640
score_threshold = 0.45  
nms_threshold = 0.2

def _get_covariance_matrix(boxes):
    a, b, c = boxes[2], boxes[3], boxes[4]
    a = (a ** 2) / 12
    b = (b ** 2) / 12
    cos = np.cos(c)
    sin = np.sin(c)
    cos2 = cos ** 2
    sin2 = sin ** 2
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

def probiou(obb1, obb2, CIoU=False, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (np.ndarray): A numpy array of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (np.ndarray): A numpy array of shape (N, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.ndarray): A numpy array of shape (N, ) representing obb similarities.
    """
    # Extract center coordinates from obb1 and obb2
    x1, y1 = obb1[0], obb1[1]
    x2, y2 = obb2[0], obb2[1]
    
    # Get the covariance matrix components
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    # Compute the terms t1, t2, and t3
    t1 = (((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2) /
          ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)) * 0.25
    
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) /
          ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)) * 0.5
    
    t3 = (((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2) /
          (4 * np.sqrt(np.maximum(0, (a1 * b1 - c1 ** 2)) *
                       np.maximum(0, (a2 * b2 - c2 ** 2))) + eps)) + eps
    t3 = np.log(t3) * 0.5
    
    # Calculate bd and hd
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(np.maximum(0, 1.0 - np.exp(-bd) + eps))
    
    # Compute IoU
    iou = 1 - hd
    return iou

def nms_rotated(boxes, nms_thresh):
    pred_boxes = []
    import ipdb; ipdb.set_trace()
    sort_boxes = boxes # sorted(boxes, key=lambda x: x.score, reverse=True)
    for i in range(len(sort_boxes)):
        if sort_boxes[i][6]!= -1:
            for j in range(i + 1, len(sort_boxes), 1):
                ious = probiou(sort_boxes[i][:5], sort_boxes[j][:5])
                if ious > nms_thresh:
                    sort_boxes[j][6] = -1
            pred_boxes.append(sort_boxes[i])
    return pred_boxes

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
#过滤掉无用的框
def filter_box(outputs):  
    outputs = np.squeeze(outputs)
    rotated_boxes = []
    scores = []
    class_ids = []
    classes_scores = outputs[4:(4+len(class_names)), ...]  
    angles = outputs[-1, ...]   
    for i in range(outputs.shape[1]):              
        class_id = np.argmax(classes_scores[...,i])
        score = classes_scores[class_id][i]
        angle = angles[i]
        angle = angle * math.pi / 180 
        if score > score_threshold:
            rotated_boxes.append(np.concatenate([outputs[:4, i], np.array([angle,score, class_id, angle * 180 / math.pi])]))
            scores.append(score)
            class_ids.append(class_id)
    sorted_indices = np.argsort(scores)[::-1]
    rotated_boxes = [rotated_boxes[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    class_ids = [class_ids[i] for i in sorted_indices]
    indices = nms_rotated(rotated_boxes, nms_threshold)
    import ipdb;ipdb.set_trace()
    output = indices
    return output

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))    
    dw, dh = (new_shape[1] - new_unpad[0])/2, (new_shape[0] - new_unpad[1])/2  # wh padding 
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im

def rotate_box(x, y, w, h, r):
    """
    计算旋转框的四个角点坐标。
    Args:
        x, y: 旋转框的中心坐标
        w, h: 旋转框的宽度和高度
        r: 旋转角度（弧度）
    Returns:
        四个角点的坐标，按照顺时针顺序排列
    """
    # 计算框的四个角点
    cos_r = math.cos(r)
    sin_r = math.sin(r)
    
    # 相对于框中心的四个角点坐标
    corners = np.array([
        [-w / 2, -h / 2],  # 左上角
        [w / 2, -h / 2],   # 右上角
        [w / 2, h / 2],    # 右下角
        [-w / 2, h / 2],   # 左下角
    ])
    
    # 旋转矩阵
    rotation_matrix = np.array([
        [cos_r, -sin_r],
        [sin_r, cos_r]
    ])
    
    # 旋转并平移到(x, y)
    rotated_corners = np.dot(corners, rotation_matrix.T) + np.array([x, y])
    
    return rotated_corners

def draw_rotated_boxes(image, boxes, input_size, original_size, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制多个旋转框，并将坐标映射到原图大小。
    Args:
        image: 输入图像
        boxes: 一个包含多个框的列表，每个框是 (x, y, w, h, r)
        input_size: 输入图像的尺寸 (input_width, input_height)
        original_size: 原始图像的尺寸 (original_width, original_height)
        color: 绘制框的颜色 (B, G, R)
        thickness: 框的线条宽度
    """
    input_width, input_height = input_size
    original_width, original_height = original_size

    for box in boxes:
        x, y, w, h, r,_,_,_ = box  # 提取每个框的参数
        corners = rotate_box(x, y, w, h, r)
        
        # 映射到原图大小
        corners[:, 0] = (corners[:, 0] / input_width) * original_width
        corners[:, 1] = (corners[:, 1] / input_height) * original_height
        
        # 将四个角点连接成一个多边形
        corners = corners.astype(int)
        for i in range(4):
            pt1 = tuple(corners[i])
            pt2 = tuple(corners[(i + 1) % 4])
            cv2.line(image, pt1, pt2, color, thickness)

    return image

if __name__=="__main__":
    image = cv2.imread('20240229092058567.bmp')
    input = letterbox(image, input_shape)

    input = input[:, :, ::-1].transpose(2, 0, 1).astype(dtype=np.float32)  #BGR2RGB和HWC2CHW
    input = input / 255.0
    input_tensor = []
    input_tensor.append(input) # 1,3,640,640 need to be 1,1,640,640 if you 3 channel image input_tensor.append(input[np.newaxis,:,:,:])
    
    onnx_session = onnxruntime.InferenceSession('output/qat.onnx', providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
        
    input_name = []
    for node in onnx_session.get_inputs():
        input_name.append(node.name)

    output_name = []
    for node in onnx_session.get_outputs():
        output_name.append(node.name)

    inputs = {}
    for name in input_name:
        inputs[name] =  np.array(input_tensor)
  
    outputs = onnx_session.run(None, inputs)
    print(outputs[0].shape)
    outputs = np.concatenate([outputs[0], outputs[1], outputs[2]], axis=1)
    boxes = filter_box(outputs)

    input_size = (640, 640) # w h
    original_size = (1280, 1280) # w h
    draw_rotated_boxes(image, boxes,input_size, original_size)
    cv2.imwrite('result.jpg', image)
