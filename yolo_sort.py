import torch
import cv2
import numpy as np
import sys
sys.path.append('yolov5-6.0')  # 添加 YOLOv5 模型路徑
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from sort import Sort  # 引入 SORT 追蹤器

# 設定裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 載入 YOLOv5 模型
model = attempt_load('yolov5s.pt', map_location=device)
model.eval()

# 初始化 SORT 追蹤器
#tracker = Sort()
tracker = Sort(max_age=10, min_hits=3)
# max_age：如果場景中人物可能短暫遮擋，可以適當提高這個值，比如設置為 10 或 15，允許追蹤器更耐心等待物件重新出現。
# min_hits：如果需要減少短暫的偽檢測或偶然檢測，可以增加這個值，比如設置為 3，這樣追蹤器會在物件穩定檢測數次後才正式追蹤。

# 打開影片
cap = cv2.VideoCapture('monitor.mp4')  # 或使用攝像頭：cv2.VideoCapture(0) 沒測過

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 從影像中讀取每一幀。如果成功讀取，ret 會是 True，並且 frame 會包含讀取的圖像資料。如果失敗（例如，播放結束），ret 會是 False。

    # 使用 letterbox 調整圖像尺寸，使其符合 YOLOv5 要求的尺寸
    img = letterbox(frame, new_shape=640)[0]

    # YOLOv5 模型接受 RGB 格式，因此需要將 BGR 轉換為 RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 圖像預處理：轉換為張量並正規化到 [0, 1] 區間
    img = torch.from_numpy(img_rgb).to(device).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)
    # 轉換為 (batch_size, 3, height, width)

    # 推理
    pred = model(img)[0]
    # letterbox 函數通常返回一個tuple，這些元素分別是：
    # 調整過大小的圖像（主要是這個）。
    # 填充區域的資訊。
    # 圖像尺寸變換的比例因子。

    # 後處理：進行非極大值抑制（NMS）來過濾多餘的邊界框
    # conf_thres 置信度閾值
    # iou_thres 適當調整 iou_thres（IOU 閾值）可以提高邊界框的去重效果，避免多重框重疊。
    pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.45)

    # 用於儲存 SORT 追蹤所需的邊界框資料
    dets_to_sort = []

    # 將結果映射回原始圖像
    for det in pred:  # 遍歷每個檢測
        if len(det):
            # 重新縮放到原始圖像尺寸
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            # 遍歷每個邊界框
            for *xyxy, conf, class_id in det:
                if int(class_id) == 0:  # 如果檢測到的是 "person"（類別 0）
                    # 添加邊界框到 dets_to_sort，格式：x1, y1, x2, y2, conf
                    dets_to_sort.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), conf.cpu().item()])

    # 轉換成 NumPy 陣列供 SORT 使用
    if len(dets_to_sort) > 0:
        dets_to_sort = np.array(dets_to_sort)

        # 使用 SORT 進行追蹤
        tracked_objects = tracker.update(dets_to_sort)

        # 繪製邊界框和追蹤 ID
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 顯示影像
    cv2.imshow('YOLOv5 Human Detection with SORT', frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
