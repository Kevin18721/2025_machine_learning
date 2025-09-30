# -*- coding: utf-8 -*-
"""
氣象格點資料處理與機器學習示範程式
功能：
1. 讀取 XML 格點溫度資料
2. 生成分類與回歸資料集
3. 訓練簡單 RandomForest 模型
4. 評估模型並畫圖
"""

import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# ===============================
# 1. 讀取 XML 資料
# ===============================

# 讀 XML 檔案
xml_file = "Week_4/O-A0038-003.xml"
tree = ET.parse(xml_file)
root = tree.getroot()

# 因為 XML 有 namespace，需要加上命名空間
ns = {'ns': 'urn:cwa:gov:tw:cwacommon:0.1'}

# 抓 <Content>
content_elem = root.find('.//ns:Content', ns)
if content_elem is None:
    raise ValueError("找不到 <Content> 標籤！")

content_text = content_elem.text.strip()

# 使用 regex 抓出所有科學記號浮點數
data_list = [float(x) for x in re.findall(r'-?\d+\.\d+E[+-]\d+', content_text)]

# 將資料轉為 2D 網格 (120 列 × 67 欄)
grid_data = np.array(data_list).reshape((120, 67))

# ===============================
# 2. 生成經緯度座標
# ===============================
lon_start, lat_start = 120.00, 21.88
lon_res, lat_res = 0.03, 0.03
num_lon, num_lat = 67, 120

lons = lon_start + np.arange(num_lon) * lon_res
lats = lat_start + np.arange(num_lat) * lat_res

lon_grid, lat_grid = np.meshgrid(lons, lats)  # shape (120, 67)

# ===============================
# 3. 生成監督式學習資料集
# ===============================

# --- 分類資料集 ---
labels = np.where(grid_data == -999, 0, 1)
classification_data = np.column_stack((
    lon_grid.flatten(),
    lat_grid.flatten(),
    labels.flatten()
))

# --- 回歸資料集 (僅保留有效值) ---
mask = grid_data != -999
regression_data = np.column_stack((
    lon_grid[mask],
    lat_grid[mask],
    grid_data[mask]
))

# ===============================
# 4. 分類模型訓練
# ===============================
X_cls = classification_data[:, :2]  # 經緯度
y_cls = classification_data[:, 2]

X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

cls_model = RandomForestClassifier(n_estimators=100, random_state=42)
cls_model.fit(X_cls_train, y_cls_train)
y_cls_pred = cls_model.predict(X_cls_test)
cls_acc = accuracy_score(y_cls_test, y_cls_pred)
print("Classification Accuracy:", cls_acc)

# ===============================
# 5. 回歸模型訓練
# ===============================
X_reg = regression_data[:, :2]
y_reg = regression_data[:, 2]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_model.predict(X_reg_test)
mse = np.mean((y_reg_test - y_reg_pred) ** 2)
rmse = np.sqrt(mse)
print("Regression RMSE:", rmse)

# ===============================
# 6. 畫圖：資料分布與預測
# ===============================
# 原始溫度熱圖
plt.figure(figsize=(10,6))
plt.imshow(grid_data, origin='lower', extent=[lons[0], lons[-1], lats[0], lats[-1]], cmap='coolwarm')
plt.colorbar(label='Temperature (°C)')
plt.title('Original Temperature Grid')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# 分類資料散點圖
plt.figure(figsize=(10,6))
plt.scatter(X_cls_test[:,0], X_cls_test[:,1], c=y_cls_pred, cmap='bwr', s=20)
plt.title('Classification Prediction (0=invalid, 1=valid)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# 回歸資料預測散點圖
plt.figure(figsize=(10,6))
plt.scatter(X_reg_test[:,0], X_reg_test[:,1], c=y_reg_pred, cmap='coolwarm', s=20)
plt.colorbar(label='Predicted Temperature (°C)')
plt.title('Regression Prediction')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()