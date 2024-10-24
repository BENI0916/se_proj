import torch
import cv2
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox  # 引入 letterbox 函數

# 設定裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 載入 YOLOv5 模型
model = attempt_load('yolov5s.pt', map_location=device)
model.eval()

# 打開視頻文件或攝像頭
cap = cv2.VideoCapture('monitor.mp4')  # 或使用攝像頭：cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用 letterbox 調整圖像尺寸，使其符合 YOLOv5 要求的尺寸
    img = letterbox(frame, new_shape=640)[0]

    # YOLOv5 模型接受 RGB 格式，因此需要將 BGR 轉換為 RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 圖像預處理：轉換為張量並正規化到 [0, 1] 區間
    img = torch.from_numpy(img_rgb).to(device).float() / 255.0  # 正規化
    img = img.permute(2, 0, 1).unsqueeze(0)  # 轉換為 (batch_size, 3, height, width)

    # 推理
    pred = model(img)[0]

    # 後處理：進行非極大值抑制（NMS）來過濾多餘的邊界框
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # 將結果映射回原始圖像
    for det in pred:  # 遍歷每個檢測
        if len(det):
            # 重新縮放到原始圖像尺寸
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, class_id in det:
                # 如果檢測到的是 "person"（COCO 中類別 0 代表 "person"）
                if int(class_id) == 0:
                    # 繪製邊界框
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person: {conf:.2f}', (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 顯示視頻幀
    cv2.imshow('YOLOv5 Human Detection', frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()