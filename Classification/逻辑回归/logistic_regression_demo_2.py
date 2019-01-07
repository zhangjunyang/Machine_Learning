# 导入模型。调用逻辑回归LogisticRegression()函数。
from sklearn.linear_model import LogisticRegression 
clf = LogisticRegression()
print clf

# fit()训练。调用fit(x,y)的方法来训练模型，其中x为数据的属性，y为所属类型。
clf.fit(train_feature,label)

# predict()预测。利用训练得到的模型对数据集进行预测，返回预测结果。
predict['label'] = clf.predict(predict_feature)