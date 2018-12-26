# 训练集多的支持向量机
# -*- coding:utf-8 -*-

print(__doc__)

import numpy as np
import matplotlib.pyplot as pl  #python中的绘图模块
from pylab import show

from sklearn import svm

np.random.seed(0)  #随机固定随机值
X = np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]]  #随机生成左下方20个点，右上方20个点
Y = [0]*20+[1]*20  #将前20个归为标记0，后20个归为标记1


#建立模型
clf = svm.SVC(kernel='linear')
clf.fit(X,Y)  #传入参数

#画出建立的超平面
w = clf.coef_[0]  #取得w值，w中是二维的
a = -w[0]/w[1]  #计算直线斜率
xx = np.linspace(-5,5)   #随机产生连续x值
yy = a*xx-(clf.intercept_[0])/w[1]   #根据随机x得到y值

#计算与直线相平行的两条直线
b = clf.support_vectors_[0]
yy_down = a*xx+(b[1]-a*b[0])
b = clf.support_vectors_[-1]
yy_up = a*xx+(b[1]-a*b[0])

print('w:',w)
print('a:',a)
print('support_vectors:',clf.support_vectors_)
print('clf.coef_',clf.coef_)


#画出三条直线
pl.plot(xx,yy,'k-')  
pl.plot(xx,yy_down,'k--')
pl.plot(xx,yy_up,'k--')

pl.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=80,facecolors='none')
pl.scatter(X[:,0],X[:,1],c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()