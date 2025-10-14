# -*- coding: utf-8 -*-
"""
氣象格點資料處理與機器學習最終整合示範程式
功能：
1. 讀取 XML 格點溫度資料
2. 生成分類與回歸資料集
3. 訓練 RandomForest 分類與回歸模型
4. 定義並使用 h(X) 整合模型
5. 評估模型並畫圖
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
xml_file = "O-A0038-003.xml"
tree = ET.parse(xml_file)
root = tree.getroot()

ns = {'ns': 'urn:cwa:gov:tw:cwacommon:0.1'}
content_elem = root.find('.//ns:Content', ns)
if content_elem is None:
    raise ValueError("❌ 找不到 <Content> 標籤！")

content_text = content_elem.text.strip()
data_list = [float(x) for x in re.findall(r'-?\d+\.\d+E[+-]\d+', content_text)]

grid_data = np.array(data_list).reshape((120, 67))

# ===============================
# 2. 生成經緯度網格
# ===============================
lon_start, lat_start = 120.00, 21.88
lon_res, lat_res = 0.03, 0.03
num_lon, num_lat = 67, 120

lons = lon_start + np.arange(num_lon) * lon_res
lats = lat_start + np.arange(num_lat) * lat_res
lon_grid, lat_grid = np.meshgrid(lons, lats)

# ===============================
# 3. 生成監督式資料集
# ===============================
X = np.column_stack((lon_grid.flatten(), lat_grid.flatten()))
y = grid_data.flatten()

valid_mask = (y != -999)
X_valid, y_valid = X[valid_mask], y[valid_mask]
X_invalid = X[~valid_mask]

print(f"總格點數: {len(y)}")
print(f"有效格點數: {len(X_valid)}")
print(f"無效格點數: {len(X_invalid)}\n")

# --- 分類資料集: 有效=1, 無效=0 ---
y_cls = (y != -999).astype(int)

# ===============================
# 4. 模型訓練
# ===============================
# 4.1 分類模型
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
    X, y_cls, test_size=0.2, random_state=42
)
cls_model = RandomForestClassifier(n_estimators=100, random_state=42)
cls_model.fit(X_cls_train, y_cls_train)
y_cls_pred = cls_model.predict(X_cls_test)
cls_acc = accuracy_score(y_cls_test, y_cls_pred)
print("Classification Accuracy:", cls_acc)

# 4.2 回歸模型（僅有效格點）
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_valid, y_valid, test_size=0.2, random_state=42
)
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_model.predict(X_reg_test)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
print("Regression RMSE:", rmse)

# ===============================
# 5. 定義 h(X)
# ===============================
def h(X_input):
    """
    h(X) = 回歸預測 R(X) 如果 C(X)=1，否則 -999
    """
    cls_pred = cls_model.predict(X_input)
    reg_pred = reg_model.predict(X_input)
    return np.where(cls_pred == 1, reg_pred, -999)

# ===============================
# 6. 視覺化
# ===============================
# 6.1 原始溫度格點分布
plt.figure(figsize=(10,6))
plt.scatter(X_valid[:,0], X_valid[:,1], c=y_valid, cmap='coolwarm', s=20)
plt.title("Original Valid Temperature Grid")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.axis("equal")
plt.colorbar(label="Temperature (°C)")
plt.show()

# 6.2 分類結果邊界圖
plt.figure(figsize=(10,6))
grid_points = np.column_stack((lon_grid.flatten(), lat_grid.flatten()))
grid_cls_pred = cls_model.predict(grid_points).reshape(lon_grid.shape)
plt.contourf(lon_grid, lat_grid, grid_cls_pred, levels=[-0.5,0.5,1.5],
             colors=['lightgray','yellow'], alpha=0.5)
plt.contour(lon_grid, lat_grid, grid_cls_pred, levels=[0.5], colors='gray', linewidths=2)
plt.scatter(X_cls_test[:,0], X_cls_test[:,1], c=y_cls_test, cmap='bwr', s=20, edgecolor='k', alpha=0.7)
plt.title("Classification Boundary (0=invalid, 1=valid)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.axis("equal")
plt.show()

# 6.3 回歸熱圖（僅有效格點）
masked_reg_pred = np.full_like(grid_data, np.nan, dtype=float)
masked_reg_pred[valid_mask.reshape(grid_data.shape)] = reg_model.predict(X_valid)
plt.figure(figsize=(10,6))
plt.imshow(masked_reg_pred, origin='lower',
           extent=[lons[0], lons[-1], lats[0], lats[-1]],
           cmap='coolwarm')
plt.colorbar(label='Predicted Temperature (°C)')
plt.title('Regression Prediction (Valid Only)')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.axis("equal")
plt.show()

# 6.4 h(X) 最終整合模型
h_pred = h(grid_points).reshape(lon_grid.shape)
plt.figure(figsize=(10,6))
plt.imshow(h_pred, origin='lower',
           extent=[lons[0], lons[-1], lats[0], lats[-1]],
           cmap='coolwarm',vmin=-10, vmax=30)
plt.colorbar(label='Predicted Temperature (°C)')
plt.title('Combined Model h(X) Prediction')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.axis("equal")
plt.show()
