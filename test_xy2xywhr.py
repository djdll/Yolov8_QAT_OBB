import numpy as np
import cv2
def xyxyxyxy2xywhr(corners):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xy1, xy2, xy3, xy4] to [xywh, rotation].
    Rotation values are expected in degrees from 0 to 90.

    Args:
        corners (list or numpy.ndarray): Input corners of shape (8,) or (n, 8).

    Returns:
        numpy.ndarray: Converted data in [cx, cy, w, h, rotation] format of shape (n, 5).
    """
    if len(corners) == 0:
        raise ValueError("Input corners array is empty. Cannot proceed with the conversion.")
    import ipdb;ipdb.set_trace() 
    # Convert the input corners to a numpy array and reshape to (4, 2)
    # corners = np.array(corners, dtype=np.float32).reshape(4, 2)
    corners = np.array(corners, dtype=np.float32).reshape(4, 2)
    print(f"Input corners: {corners}")  # 添加调试信息，查看输入数据
    
    if corners.size == 8:
        corners = corners.reshape(4, 2)  # Reshape to (4, 2) for a single bounding box
    elif corners.size % 8 == 0:
        # Handle batch case (n, 8)
        corners = corners.reshape(-1, 4, 2)
    else:
        raise ValueError("Input corners array has an invalid size. Expected size should be a multiple of 8.")

    # Prepare the list to store the converted boxes
    rboxes = []
    (x, y), (w, h), angle = cv2.minAreaRect(corners)
    # Iterate through the corner points
    for pts in corners:
        print(f"Processing points: {pts}")  # 查看每个处理的点
        # Use cv2.minAreaRect to get accurate xywhr (center, width, height, angle)
        (x, y), (w, h), angle = cv2.minAreaRect(pts)

        # Adjust the angle to be within 0 to 180 degrees
        if angle < -45:
            angle += 90

        # Append the converted [x, y, w, h, angle] to the result list
        rboxes.append([x, y, w, h, angle])

    # Convert result back to numpy array and return
    return np.array(rboxes)

# 示例：如果你传入一个有效的 corners 数组
corners = [0.1967948717948718, 0.2621794871794872, 0.7519230769230769, 0.2621794871794872, 
           0.7519230769230769, 0.8173076923076923, 0.1967948717948718, 0.8173076923076923]
result = xyxyxyxy2xywhr(corners)
print(result)