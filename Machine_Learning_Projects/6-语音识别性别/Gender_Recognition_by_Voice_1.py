# 支持向量机算法试图在两个类之间画一个超平面来分离它们。超平面可以有很多，但是支持向量机的目的是找出每个类之间的数据点之间的最大距离。这些数据点是从每个类到超平面最近的数据点。
# SVM使用称为核技巧的技术来转换数据。如果数据点具有较小的维度空间并且无法绘制超平面，则会尝试向数据添加新维度。
# https://www.kaggle.com/primaryobjects/voicegender
# Gender Recognition by Voice
# Identify a voice as male or female

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Read Data
df = pd.read_csv("voice.csv")
# First 5 Rows of Data
df.head()
df.columns
df.info()

sns.pairplot(df, hue='label', vars=['skew', 'kurt',
 'sp.ent', 'sfm', 'mode','meanfun',
 'meandom','dfrange'])
plt.show()

sns.countplot(df.label)
plt.show()

sns.scatterplot(x = 'skew', y = 'kurt', hue = 'label', data = df)
plt.show()

plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), annot=True, linewidth=.5, fmt='.2f', linecolor = 'grey')
plt.show()

X = df.drop(['label'],axis=1)
y = df.label


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
# Import SVM
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Paired_r", linewidth=2, linecolor='w', fmt='.0f')
plt.xlabel('Predicted Value')
plt.ylabel('True Value')
plt.show()

# Normalization 将数据归一化
X = (X - np.min(X)) / (np.max(X) - np.min(X)).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt='.0f', cmap='brg_r')
plt.xlabel('Predicted Value')
plt.ylabel('True Value')
plt.show()

# 为模型找到最佳参数
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1, 1, 10, 100], 'gamma':[1, 0.1, 0.01, 0.001], 'kernel' : ['rbf', 'poly', 'sigmoid', 'linear']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=4)
grid.fit(X_train, y_train)
print("Best Parameters: ",grid.best_params_)

grid_pred = grid.predict(X_test)
cmNew = confusion_matrix(y_test, grid_pred)
sns.heatmap(cmNew, annot=True, fmt='.0f', cmap='gray_r')
plt.xlabel('Predicted Value')
plt.ylabel('True Value')
plt.show()

print(classification_report(y_test, grid_pred))
