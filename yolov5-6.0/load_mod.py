import torch
from models import Model  # 替換為你的模型類的導入

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 創建模型實例
model = Model().to(device)  # 確保使用正確的模型類

# 載入檢查點
checkpoint = torch.load('yolov5s.pt', map_location=device)

# 從檢查點中提取模型權重並載入
model.load_state_dict(checkpoint['model'], strict=False)  # 設置 strict=False 以避免不匹配問題

# 設置模型為評估模式
model.eval()

# 測試模型（例如，對一個樣本進行預測）
# input_tensor = torch.randn(1, 3, 640, 640).to(device)  # 使用合適的尺寸
# output = model(input_tensor)
# print(output)
