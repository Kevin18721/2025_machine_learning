# -*- coding: utf-8 -*-
"""
GDA 氣象格點分類示範程式
功能：
1. 讀取 XML 格點溫度資料
2. 建立分類標籤（有效=1, 無效=0）
3. 訓練 GDA 模型
4. 評估分類準確度
5. 畫決策邊界圖
"""

import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as GDA
from sklearn.metrics import accuracy_score

# ===============================
# 1. 讀取 XML
# ===============================
xml_file = "O-A0038-003.xml"
tree = ET.parse(xml_file)
root = tree.getroot()
ns = {'ns': 'urn:cwa:gov:tw:cwacommon:0.1'}

content_elem = root.find('.//ns:Content', ns)
if content_elem is None:
    raise ValueError("找不到 <Content> 標籤！")

content_text = content_elem.text.strip()
data_list = [float(x) for x in re.findall(r'-?\d+\.\d+E[+-]\d+', content_text)]
grid_data = np.array(data_list).reshape((120, 67))

# ===============================
# 2. 經緯度網格
# ===============================
lon_start, lat_start = 120.00, 21.88
lon_res, lat_res = 0.03, 0.03
num_lon, num_lat = 67, 120

lons = lon_start + np.arange(num_lon) * lon_res
lats = lat_start + np.arange(num_lat) * lat_res
lon_grid, lat_grid = np.meshgrid(lons, lats)

X = np.column_stack((lon_grid.flatten(), lat_grid.flatten()))

# 分類標籤：有效=1, 無效=0
y_cls = (grid_data.flatten() != -999).astype(int)

print(f"總格點數: {len(y_cls)}")
print(f"有效格點數: {np.sum(y_cls==1)}")
print(f"無效格點數: {np.sum(y_cls==0)}\n")

# ===============================
# 3. GDA 分類模型
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cls, test_size=0.2, random_state=42
)

gda_model = GDA(store_covariance=True)
gda_model.fit(X_train, y_train)
y_pred = gda_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("GDA Classification Accuracy:", acc)

# ===============================
# 4. 畫決策邊界
# ===============================
plt.figure(figsize=(10,6))

# 網格上預測
xx, yy = np.meshgrid(np.linspace(lons[0], lons[-1], 200),
                     np.linspace(lats[0], lats[-1], 200))
Z = gda_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=[-0.5,0.5,1.5], colors=['lightgray','yellow'], alpha=0.3)
plt.contour(xx, yy, Z, levels=[0.5], colors='gray', linewidths=2)
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='bwr', s=20, edgecolor='k', alpha=0.7)

plt.title("(GDA/QDA) Classification Boundary")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.axis("equal")
plt.show()
