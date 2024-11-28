# Iris 分類模型專案

## 專案概述
這個專案使用 K-最近鄰居(KNN)算法對著名的 Iris（鳶尾花）數據集進行分類分析。專案包含了數據載入、模型訓練、評估和視覺化等完整的機器學習流程。

## 功能特點
- 使用 scikit-learn 的 KNN 分類器
- 包含數據預處理和分割
- 模型性能評估
- 使用 PCA 進行數據視覺化

## 技術棧
- Python
- scikit-learn
- pandas
- matplotlib

## 程式碼說明

### 1. 數據準備
```python
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
```
- 載入 scikit-learn 內建的 Iris 數據集
- X 包含了鳶尾花的四個特徵
- y 包含了鳶尾花的類別標籤

### 2. 數據分割
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- 將數據集分為訓練集(80%)和測試集(20%)
- random_state=42 確保結果可重現

### 3. 模型訓練
```python
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
```
- 創建 KNN 分類器，設定 K=3
- 使用訓練數據訓練模型

### 4. 模型評估
```python
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)
```
- 使用測試集進行預測
- 計算準確率
- 生成詳細的分類報告

### 5. 視覺化
```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
```
- 使用 PCA 將 4 維數據降為 2 維以便視覺化
- 創建散點圖顯示不同類別的分布情況

## 執行結果
- 模型準確率會被輸出
- 分類報告包含每個類別的精確度、召回率和 F1 分數
- 生成 PCA 視覺化圖，顯示三種鳶尾花品種的分布情況

## 使用說明
1. 確保已安裝所需的 Python 套件
2. 直接運行程式碼即可看到分類結果和視覺化圖表

## 注意事項
- PCA 降維可能會損失部分數據信息
- KNN 的 K 值可以根據需求調整
- 數據集分割的比例可以根據實際需求修改
