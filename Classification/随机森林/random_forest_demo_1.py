#随机森林
from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestRegressor  
import numpy as np  
from sklearn.datasets import load_iris 

iris=load_iris()
print(iris['target'].shape)
#这里使用默认的参数设置
rf=RandomForestRegressor()
#进行模型的训练  
rf.fit(iris.data[:150],iris.target[:150])
#随机挑选两个预测不相同的样本  
instance=iris.data[[100,109]]  
print(instance)
predict_instance[0] = rf.predict(instance[[0]])
predict_instance[1] = rf.predict(instance[[1]])
print('instance 0 prediction；', predict_instance[0])
print(iris.target[100])
print('instance 1 prediction；', predict_instance[0])
print(iris.target[109])

# Classification and Regression Tree Algorithm
# def decision_tree(train, test, max_depth, min_size):
#     tree = build_tree(train, max_depth, min_size)
#     predictions = list()
#     for row in test:
#         prediction = predict(tree, row)
#         predictions.append(prediction)
#     return(predictions)