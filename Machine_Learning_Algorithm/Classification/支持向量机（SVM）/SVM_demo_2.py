# 少量训练集
# -*- coding:utf-8 -*- 
from sklearn import svm

X = [[2,0],[1,1],[2,3]]#平面上的三个点
y = [0,0,1]#标记,第一个点和第二点属于第0类，第三个点属于第一类

clf = svm.SVC(kernel='linear')#linear为小写，线性核函数

clf.fit(X,y)#创建向量机模型

print(clf)
print(clf.support_vectors_)#打印出支持向量
print(clf.support_)#打印出给出的向量中属于支持向量的向量的index下标

print(clf.n_support_)#打印出两个类中各自找出的支持向量的个数

print(clf.predict([[2,0],]))#必须传入一个嵌套列表
