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
xml_file = "O-A0038-003.xml"
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
# 對整個經緯度網格做預測
grid_points = np.column_stack((lon_grid.flatten(), lat_grid.flatten()))
grid_pred = cls_model.predict(grid_points).reshape(lon_grid.shape)

plt.figure(figsize=(10,6))
# 用 contourf 畫有效/無效區域
plt.contourf(lon_grid, lat_grid, grid_pred, levels=[-0.5,0.5,1.5], colors=['lightgray','skyblue'], alpha=0.5)
# 可以疊加測試集的散點
plt.scatter(X_cls_test[:,0], X_cls_test[:,1], c=y_cls_test, cmap='bwr', s=20, edgecolor='k', alpha=0.7)
plt.title('Classification Boundary (0=invalid, 1=valid)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.axis("equal")
plt.show()

# ===============================
# 6.2 用回歸模型畫預測熱圖
# ===============================
# 對整個網格做預測
masked_reg_pred = np.full_like(grid_data, np.nan)  # 先填 nan
mask = grid_data != -999
masked_reg_pred[mask] = reg_model.predict(np.column_stack((lon_grid[mask], lat_grid[mask])))


plt.figure(figsize=(10,6))
plt.imshow(masked_reg_pred, origin='lower', extent=[lons[0], lons[-1], lats[0], lats[-1]],
           cmap='coolwarm')
plt.colorbar(label='Predicted Temperature (°C)')
plt.title('Regression Prediction (masked invalid values)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# ===============================
# 6.3 結合分類與回歸：h(x)
# ===============================
# 對整個網格做分類預測
C_pred = cls_model.predict(grid_points).reshape(lon_grid.shape)  # 0=無效, 1=有效

# 對有效點做回歸預測
R_pred = np.full_like(grid_data, -999, dtype=float)  # 預設無效值為 -999
mask_valid = C_pred == 1
R_pred[mask_valid] = reg_model.predict(np.column_stack((lon_grid[mask_valid], lat_grid[mask_valid])))

h_grid = R_pred  # h(x)

# 畫圖
plt.figure(figsize=(10,6))
# 使用 cmap='coolwarm'，無效值 -999 可以用 nan 轉透明
masked_h_grid = np.where(h_grid==-999, np.nan, h_grid)
plt.imshow(masked_h_grid, origin='lower', extent=[lons[0], lons[-1], lats[0], lats[-1]],
           cmap='coolwarm')
plt.colorbar(label='Predicted Temperature h(x) (°C)')
plt.title('Combined Model Prediction h(x) = R(x) if C(x)=1 else -999')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()