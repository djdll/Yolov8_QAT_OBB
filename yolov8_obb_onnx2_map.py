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
from sklearn.metrics import precision_recall_curve, average_precision_score
# calculate onnx P R 
class_names = ["your class names"]      
input_shape = (640, 640) 
score_threshold = 0.45  
nms_threshold = 0.2 
input_height = 640
input_width = 640

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
    sort_boxes = boxes
    for i in range(len(sort_boxes)):
        if sort_boxes[i][6]!= -1:
            for j in range(i + 1, len(sort_boxes), 1):
                ious = probiou(sort_boxes[i][:5], sort_boxes[j][:5])
                if ious > nms_thresh:
                    sort_boxes[j][6] = -1
            pred_boxes.append(sort_boxes[i])
    return pred_boxes

def xyxyxyxy2xywhr(corners):
    """
    Convert Oriented Bounding Box (OBB) from [xy1, xy2, xy3, xy4] to [cx, cy, w, h, rotation].
    Rotation values are expected in degrees from 0 to 180.

    Args:
        corners (list or numpy.ndarray): Input corners of shape (8,), representing
                                          the coordinates [x1, y1, x2, y2, x3, y3, x4, y4].

    Returns:
        (numpy.ndarray): Converted data in [cx, cy, w, h, rotation] format.
                          The rotation is in radians.
    """
    # Ensure corners is a numpy array
    corners = np.array(corners, dtype=np.float32)
    corners = corners.reshape(4, 2)  # Reshape to (4, 2) representing (x1, y1), (x2, y2), (x3, y3), (x4, y4)

    # Use cv2.minAreaRect to get the rotated bounding box properties
    (x, y), (w, h), angle = cv2.minAreaRect(corners)

    # Convert angle from degrees to radians
    angle_rad = angle / 180 * np.pi

    # Return as [cx, cy, w, h, rotation (radians)]
    return np.array([x, y, w, h, angle_rad], dtype=np.float32)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

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
        angle = angle * math.pi / 180 # 角度->弧度
        if score > score_threshold:
            # x,y,h,w,score,id,angle
            rotated_boxes.append(np.concatenate([outputs[:4, i], np.array([angle,score, class_id, angle * 180 / math.pi])]))
            scores.append(score)
            class_ids.append(class_id)
    sorted_indices = np.argsort(scores)[::-1]
    rotated_boxes = [rotated_boxes[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    class_ids = [class_ids[i] for i in sorted_indices]

    indices = nms_rotated(rotated_boxes, nms_threshold)
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

def read_label_file(label_path):
    """读取标签文件，返回格式为 class_id x1 y1 x2 y2 x3 y3 x4 y4"""
    with open(label_path, 'r') as file:
        lines = file.readlines()
    
    labels = []
    if len(lines) >= 1:
        for line in lines:
            parts = line.strip().split()  # 假设每行格式是 class_id x1 y1 x2 y2 x3 y3 x4 y4
            label = list(map(float, parts))
            if label!=[]:
                labels.append(label)
        return labels
    else:
        print(f"Warning: {label_path} is empty or contains only empty lines.")
        return []  # 如果是空文件，返回空列表

def calculate_mean_precision_recall(precisions, recalls):
    """
    计算多个图像的平均精确率和召回率。

    Args:
        precisions (list): 每张图片的精确率
        recalls (list): 每张图片的召回率

    Returns:
        tuple: 平均精确率和召回率
    """
    mean_precision = np.mean(precisions) if precisions else 0.0
    mean_recall = np.mean(recalls) if recalls else 0.0
    return mean_precision, mean_recall

def calculate_precision_recall_2(pred_boxes, gt_boxes, iou_thresh=0.45):
    tp, fp, fn = 0, 0, 0
    
    if len(gt_boxes) == 0:
        # 如果没有标签框，所有的检测框都算作误检（FP）
        fp = len(pred_boxes)
    elif len(pred_boxes) == 0:
        # 如果没有检测框，所有的标签框都算作漏检（FN）
        fn = len(gt_boxes)
    else:
        matched_gt = [False] * len(gt_boxes)  # 用于标记标签框是否匹配
        for pred in pred_boxes:
            matched = False
            for i, gt in enumerate(gt_boxes):
                if not matched_gt[i]:
                    iou_value = probiou(pred, gt)  # 计算预测框与标签框的IoU
                    if iou_value >= iou_thresh:
                        tp += 1  # 正确检测
                        matched_gt[i] = True  # 标记该标签框已匹配
                        matched = True
                        break  # 找到一个匹配的框后，跳出标签框的循环
            if not matched:
                fp += 1  # 如果没有匹配的框，则为误检

        # 计算漏检：即标签框中没有被匹配的框
        fn = len([1 for matched in matched_gt if not matched])

    # 计算精确率和召回率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall, tp, fp, fn


def calculate_precision_recall(pred_boxes, gt_boxes, iou_thresh=0.5):
    tp, fp, fn = 0, 0, 0
    matched_gt_indices = []  # 用于标记哪些GT框已经匹配过

    # 比较预测框和标签框，计算TP, FP, FN
    for pred in pred_boxes:
        iou_max = 0
        best_gt_index = -1
        
        # 遍历所有标签框，找到最佳匹配的标签框
        for i, gt in enumerate(gt_boxes):
            iou = probiou(pred, gt)  # 计算IoU，可以用你之前的probiou方法
            if iou > iou_max:
                iou_max = iou
                best_gt_index = i  # 记录最好的匹配标签框的索引
        
        # 如果最佳匹配的IoU大于阈值，则算作TP，标记该标签框为已匹配
        if iou_max > iou_thresh and best_gt_index != -1 and best_gt_index not in matched_gt_indices:
            tp += 1
            matched_gt_indices.append(best_gt_index)  # 该GT已被匹配
        else:
            fp += 1  # 如果没有匹配的框，则为误检
    
    # 剩下的未匹配的GT框算作FN
    fn = len(gt_boxes) - len(matched_gt_indices)

    # 计算精确率和召回率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall, tp, fp, fn


def save_predictions_to_txt(predictions, output_file):
    """
    Save the list of predictions to a text file.

    Args:
        predictions (list): List of bounding boxes, each in [x, y, w, h, angle] format.
        output_file (str): The path to the output text file.
    """
    with open(output_file, 'w') as f:
        for pred in predictions:
            # Convert the prediction to a space-separated string and write to file
            pred_str = ' '.join(map(str, pred))
            f.write(pred_str + '\n')  # Write each prediction in a new line

def normalize_bounding_box(filtered_results, img_width, img_height):
    """
    Normalize the bounding box coordinates (x, y, w, h) based on the image dimensions.

    Args:
        filtered_results (numpy.ndarray): Array of shape (8,) containing [x, y, w, h, angle, ...].
        img_width (int): The width of the image.
        img_height (int): The height of the image.

    Returns:
        numpy.ndarray: The normalized bounding box with [x, y, w, h, angle] values.
    """
    # Extract the first 5 values: [x, y, w, h, angle]
    x, y, w, h, angle = filtered_results[:5]

    # Normalize x, y, w, h
    normalized_x = x / img_width
    normalized_y = y / img_height
    normalized_w = w / img_width
    normalized_h = h / img_height
    normalized_angle = angle  # Angle does not require normalization here

    # Return the normalized values
    return np.array([normalized_x, normalized_y, normalized_w, normalized_h, normalized_angle])


def predict_images_from_folder(image_folder, label_folder, input_shape, model_path,nms_thresh, iou_thresh):
    # 获取文件夹内所有图片文件路径
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.bmp', '.jpg', '.png'))]
    
    # 获取文件夹内所有标签文件路径
    label_paths = {os.path.splitext(f)[0]: os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith('.txt')}
    # 加载ONNX模型
    onnx_session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
    
    input_name = [node.name for node in onnx_session.get_inputs()]
    output_name = [node.name for node in onnx_session.get_outputs()]
    all_predictions = []
    all_gt_labels = []
    precisions, recalls = [], []
    from tqdm import tqdm
    for image_path in tqdm(image_paths):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        all_gt_label=[]
        # 读取对应的标签文件
        if image_name in label_paths:
            label_path = label_paths[image_name]
            gt_boxes = read_label_file(label_path)  # 读取标签
            
            for label in gt_boxes:
                # 假设 label[1:] 是该框的 xyxyxyxy 格式 (去掉 classid)
                gt_boxes_xywhr = xyxyxyxy2xywhr(label[1:])
                
                # 将转换后的结果加入到 all_gt_labels 中
                all_gt_label.append(gt_boxes_xywhr)
            all_gt_labels.append(all_gt_label)
            # print(f"Labels for {image_name}: {labels}")
            # all_gt_labels.append(gt_boxes)
        else:
            print(f"No label found for {image_name}")
        
        # 读取每一张图片
        image = cv2.imread(image_path)
        input_image = letterbox(image, input_shape)
        
        # 预处理输入：BGR2RGB 和 HWC2CHW
        input_image = input_image[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)  # BGR2RGB 和 HWC2CHW
        input_image = input_image / 255.0  # 归一化
        
        # 扩展维度以匹配模型的输入形状
        input_tensor = []
        input_tensor.append(input_image)
        # 创建输入字典
        inputs = {name: np.array(input_tensor) for name in input_name}

        # 执行预测
        outputs = onnx_session.run(None, inputs)
        
        # 合并输出结果
        print(f"Prediction for {image_name}:")
        results = np.concatenate([outputs[0], outputs[1], outputs[2]], axis=1)  # 合并输出

        # 过滤掉无用的框
        filtered_results = filter_box(results)
        all_prediction=[]
        for filtered_result in filtered_results:
            normalized_results = normalize_bounding_box(filtered_result[:5], input_shape[1], input_shape[0]) # 归一化
            all_prediction.append(normalized_results)
        all_predictions.append(all_prediction)

    # 初始化TP, FP, FN
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for i, gt_boxes in enumerate(all_gt_labels):
        pred_boxes = all_predictions[i]
        precision, recall, tp, fp, fn = calculate_precision_recall(pred_boxes, gt_boxes )
        total_tp += tp
        total_fp += fp
        total_fn += fn
    print(total_tp, total_fp, total_fn)
    # 计算总体的精确率和召回率
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    print(f"Overall Precision: {overall_precision}")
    print(f"Overall Recall: {overall_recall}")


    save_predictions_to_txt(all_predictions, "output_file.txt")
    return precisions, recalls#, ap_80
if __name__ == "__main__":
    image_folder = "images/val"  # 图片文件夹路径
    label_folder = "labels/val"  # 手动指定标签文件夹路径
    input_shape = (640, 640)  # 假设输入尺寸是640x640
    model_path = 'output/obb/qat.onnx'  # ONNX模型路径
    nms_thresh = 0.2  # NMS阈值
    iou_thresh = 0.45  # IoU阈值
    predict_images_from_folder(image_folder, label_folder, input_shape, model_path,nms_thresh, iou_thresh)
