from sort import Sort
import numpy as np

# 初始化 SORT 追蹤器
tracker = Sort()

# 示例檢測數據（通常從 YOLO 或其他檢測模型獲得）
# [x_min, y_min, x_max, y_max, confidence] 格式的邊界框數據
detections = np.array([[100, 200, 300, 400, 0.9]])

# 使用 SORT 進行追蹤
trackers = tracker.update(detections)

# trackers 返回的數據 [x_min, y_min, x_max, y_max, track_id]
for track in trackers:
    print(f'Track ID: {int(track[4])}, Bounding Box: {track[:4]}')
