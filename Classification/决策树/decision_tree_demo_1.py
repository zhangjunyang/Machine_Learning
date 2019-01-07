#!/usr/bin/python
import sys
import copy
import math
import getopt

def usage():
    print '''Help Information:
    -h, --help: show help information;
    -r, --train: train file;
    -t, --test: test file;
    '''

def getparamenter():
    try:
      opts, args = getopt.getopt(sys.argv[1:], "hr:t:k:", ["help", "train=","test=","kst="])
    except getopt.GetoptError, err:
      print str(err)
      usage()
      sys.exit(1)

    sys.stderr.write("\ntrain.py : a python script for perception training.\n")
    sys.stderr.write("Copyright 2016 sxron, search, Sogou. \n")
    sys.stderr.write("Email: shixiang08abc@gmail.com \n\n")

    train = ''
    test = ''
    for i, f in opts:
      if i in ("-h", "--help"):
        usage()
        sys.exit(1)
      elif i in ("-r", "--train"):
        train = f
      elif i in ("-t", "--test"):
        test = f
      else:
        assert False, "unknown option"
  
    print "start trian parameter \ttrain:%s\ttest:%s" % (train,test)

    return train,test

def loaddata(file):
    fin = open(file,'r')
    data = []
    while 1:
        dataline = []
        line = fin.readline()
        if not line:
            break
        ts = line.strip().split('\t')
        for temp in ts:
            dataline.append(temp.strip())
        data.append(dataline)
    return data

def majorityCnt(classList):
    classCnt = {}
    for cls in classList:
        if not classCnt.has_key(cls):
            classCnt[cls] = 0
        classCnt[cls] += 1

    SortClassCnt = sorted(classCnt.iteritems(),key=lambda d:d[1],reverse=True)
    return SortClassCnt[0][0]

def calcShannonEnt(trainData):
    numEntries = len(trainData)
    labelDic = {}
    for trainLine in trainData:
        currentLabel = trainLine[-1]
        if not labelDic.has_key(currentLabel):
            labelDic[currentLabel] = 0
        labelDic[currentLabel] += 1

    shannonEnt = 0.0
    for key,value in labelDic.items():
        prob = float(value)/numEntries
        shannonEnt -= prob * math.log(prob,2)
    return shannonEnt

def splitData(trainData,index,value):
    subData = []
    for trainLine in trainData:
        if trainLine[index]==value:
            reducedFeatVec = []
            for i in range(0,len(trainLine),1):
                if i==index:
                    continue
                reducedFeatVec.append(trainLine[i])
            subData.append(reducedFeatVec)
    return subData

def chooseBestFeature(trainData):
    numFeatures = len(trainData[0])-1
    baseEntropy = calcShannonEnt(trainData)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(0,numFeatures,1):
        currentFeature = [temp[i] for temp in trainData]
        uniqueValues = set(currentFeature)
        newEntropy = 0.0
        splitInfo = 0.0
        for value in uniqueValues:
            subData = splitData(trainData,i,value)
            prob = float(len(subData))/len(trainData)
            newEntropy += prob * calcShannonEnt(subData)
            splitInfo -= prob * math.log(prob,2)
        infoGain = (baseEntropy - newEntropy) / splitInfo
        if infoGain > bestInfoGain :
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def CreateTree(trainData):
    classList = [temp[-1] for temp in trainData]
    classListSet = set(classList)
    if len(classListSet)==1:
        return classList[0]
    if len(trainData[0])==1:
        return majorityCnt(classList)

    bestFeature = chooseBestFeature(trainData)
    myTree = {bestFeature:{}}
    featureValues = [example[bestFeature] for example in trainData]
    uniqueValues = set(featureValues)
    for value in uniqueValues:
        myTree[bestFeature][value] = CreateTree(splitData(trainData, bestFeature, value))
    return myTree

def classify(testData,dTrees):
    index = int(dTrees.keys()[0])
    secondDict = dTrees[index]
    testValue = testData[index]
    for key in secondDict.keys():
        if testValue==key:
            if type(secondDict[key]).__name__=='dict':
                secondTest = copy.deepcopy(testData)
                del secondTest[index]
                classLabel = classify(secondTest,secondDict[key])
            else:
                classLabel = secondDict[key]
    return classLabel

def TestFunc(testData,dTrees):
    for temp in testData:
        classLabel = classify(temp,dTrees)
        print "%s\t%s" % (temp,classLabel)

def main():
    #set parameter
    train,test = getparamenter()

    #load train data
    trainData = loaddata(train)
    testData = loaddata(test)

    #create Decision Tree
    dTrees = CreateTree(trainData)
    print dTrees

    #test Decision Tree
    TestFunc(testData,dTrees) 

if __name__=="__main__":
    main()