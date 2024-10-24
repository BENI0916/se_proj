import torch
import cv2
import sys
sys.path.append('yolov5-6.0')
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox  # 引入 letterbox 函數

# 設定裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 載入 YOLOv5 模型
model = attempt_load('yolov5s.pt', map_location=device)
model.eval()
print('end')