import torch
import cv2
from pathlib import Path
from models.experimental import attempt_load  # 導入 attempt_load

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 使用 attempt_load 加載模型權重
model = attempt_load('yolov5s.pt', map_location=device)  # 直接加載權重文件

# 設置模型為評估模式
model.eval()

# 打開視頻文件或監控視頻流
cap = cv2.VideoCapture('monitor.mp4')  # 可以替換為視頻文件如 'video.mp4'

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 將影像調整為 640x640
    img_resized = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # 將調整後的影像轉換為 Tensor
    img_tensor = torch.from_numpy(img_rgb).to(device)
    img_tensor = img_tensor.float()
    img_tensor = img_tensor.permute(2, 0, 1)  # 轉換維度為 (C, H, W)
    img_tensor = img_tensor.unsqueeze(0)  # 增加批量維度 (1, C, H, W)

    # 使用 YOLO 模型進行推理
    results = model(img_tensor)

    # 打印 results 以確認結構
    print(results)

    # 根據結果結構提取 detections
    detections = results[0].xyxy[0].cpu().numpy()  # 根據實際的結果結構進行調整
    for detection in detections:
        x_min, y_min, x_max, y_max, confidence, class_id = detection
        if int(class_id) == 0:  # YOLO 的 COCO 數據集中，類別 0 代表 "person"
            # 繪製邊界框
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(frame, f'Person: {confidence:.2f}', (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 顯示每一幀畫面
    cv2.imshow('YOLOv5 Human Detection', frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
