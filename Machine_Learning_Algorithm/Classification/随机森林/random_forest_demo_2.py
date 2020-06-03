# 随机森林分类器、决策树、extra树分类器的比较
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = make_blobs(n_samples=10000, n_features=10, centers=100,random_state=0)

clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
scores = cross_val_score(clf, X, y)
print(scores)
print(scores.mean())                             

clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y)
print(scores)
print(scores.mean())                             

clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y)
print(scores)
print(scores.mean())

# Classification and Regression Tree Algorithm
# def decision_tree(train, test, max_depth, min_size):
#     tree = build_tree(train, max_depth, min_size)
#     predictions = list()
#     for row in test:
#         prediction = predict(tree, row)
#         predictions.append(prediction)
#     return(predictions)