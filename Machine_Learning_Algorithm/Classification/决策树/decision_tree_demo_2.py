# -*- coding: utf-8 -*-
from math import log

#创建数据的函数
def createDataSet():  
    dataSet = [[1,1,'yes'],  
               [1,1, 'yes'],  
               [1,0,'no'],  
               [0,1,'no'],  
               [0,1,'no']]  
    labels = ['no surfacing','flippers']  
    return dataSet, labels

#计算给定数据shangnon熵的函数：
def calcShannonEnt(dataSet):  
    #calculate the shannon value  
    numEntries = len(dataSet)  
    labelCounts = {}  
    for featVec in dataSet:      #create the dictionary for all of the data  获取样本中各类别数目
        currentLabel = featVec[-1]  
        if currentLabel not in labelCounts.keys():  
            labelCounts[currentLabel] = 0  
        labelCounts[currentLabel] += 1  
    shannonEnt = 0.0  
    for key in labelCounts:  
        prob = float(labelCounts[key])/numEntries  
        shannonEnt -= prob*log(prob,2) #get the log value  
    return shannonEnt


#划分数据集，获得特征取某一值的样本，即获得子集
def splitDataSet(dataSet, axis, value):  
    retDataSet = []  
    for featVec in dataSet:  
        if featVec[axis] == value:      #abstract the fature  
            reducedFeatVec = featVec[:axis]  
            reducedFeatVec.extend(featVec[axis+1:])  
            retDataSet.append(reducedFeatVec)  
    return retDataSet 

#选择最大信息增益的特征
def chooseBestFeatureToSplit(dataSet):  
    numFeatures = len(dataSet[0])-1  #获取特征数目
    baseEntropy = calcShannonEnt(dataSet)  #在样本集合中确定任一样本类别所需的信息熵
    bestInfoGain = 0.0; bestFeature = -1  #初始化最大信息增益和测试特征
    for i in range(numFeatures):  #逐个特征计算信息增益，并选出最大信息增益和确定测试特征
        featList = [example[i] for example in dataSet]  #获取所有样本的某个特征值
        uniqueVals = set(featList)  
        newEntropy = 0.0  
        for value in uniqueVals:  #按特征不同取值划分样本并计算加权信息熵
            subDataSet = splitDataSet(dataSet, i , value)  #获得特征取某一值的样本，即获得子集
            prob = len(subDataSet)/float(len(dataSet))  #计算该子集在总样本中的权重
            newEntropy +=prob * calcShannonEnt(subDataSet)  #计算子集中确定分类所需的信息熵，并求加权平均
        infoGain = baseEntropy - newEntropy  #计算信息增益
        if(infoGain > bestInfoGain):  
            bestInfoGain = infoGain  #获取最大信息增益，从而获取测试特征
            bestFeature = i  #测试特征的索引
    return bestFeature  

#递归创建树用于找出出现次数最多的类别（少数服从多数）
def majorityCnt(classList):  
    classCount = {}  
    for vote in classList:  
        if vote not in classCount.keys(): classCount[vote] = 0  
        classCount[vote] += 1  
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  
    return sortedClassCount[0][0]

#创建树
def createTree(dataSet, labels): 
    labels=list(labels)#python中使用"="产生的新变量，虽然名称不一样，但都指向同一个内存地址，实际上两者是一样的，为了产生真正的新变量，避免后面del()函数对原变量值产生影响，这里使用变量类型转换来赋值
    classList = [example[-1] for example in dataSet]  #获取分类标签
    # the type is the same, so stop classify  
    if classList.count(classList[0]) == len(classList):  #所有样本类别相同，则停止分支
        return classList[0]  
    # traversal all the features and choose the most frequent feature  
    if (len(dataSet[0]) == 1):  #特征列表为空，强制生成叶节点，并标记类别
        return majorityCnt(classList)  
    bestFeat = chooseBestFeatureToSplit(dataSet)  #产生测试特征
    bestFeatLabel = labels[bestFeat]  
    myTree = {bestFeatLabel:{}}  #初始化树，树通过字典存放
    del(labels[bestFeat])  #逐步缩减特征规模
    #get the list which attain the whole properties  
    featValues = [example[bestFeat] for example in dataSet]  
    uniqueVals = set(featValues)  
    for value in uniqueVals:  
        subLabels = labels[:]  
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)  #递归进行分支，决策树逐步生长，该决策树是存放在字典中，字典有键值对并很方便地实现层层嵌套
    return myTree  

#利用决策树进行样本类别预测
def classify(inputTree, featLabels, testVec):  #这里只能对单个样本进行预测
    for ky in inputTree.keys():
        #firstStr = inputTree.keys()[0]  
        secondDict = inputTree[ky]  
        #featIndex = featLabels.index(firstStr) 
        featIndex = featLabels.index(ky) #ky是根节点代表的特征，featIndex是取根节点特征在特征列表的索引，方便后面对输入样本逐变量判断
        for key in secondDict.keys():  #这里每一个key值对应的是根节点特征的不同取值
            if testVec[featIndex] == key:  #找到输入样本在决策树中的由根节点往下走的路径
                if type(secondDict[key]).__name__ == 'dict':  #该分支产生了一个内部节点，则在决策树中继续同样的操作查找路径
                    classLabel = classify(secondDict[key], featLabels, testVec)  
                else: classLabel = secondDict[key]    #该分支产生是叶节点，直接取值就得到类别     

    return classLabel 



#使用决策树
myDat, labels = createDataSet()  #产生训练数据和特征列表
myTree = createTree(myDat,labels) #创建决策树
print(myTree)
input_type=classify(myTree,labels,[1,0])#利用决策树对输入样本类别进行预测
print(input)